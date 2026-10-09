"""系统装配工厂。

把「灌库 → 两路召回 → 融合 → 护栏 → 编排」的拼装集中到一个地方，
demo / 评测 / API 共用一套构造逻辑，避免每个入口各写一遍。
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from src.core.graph import Graph
from src.config import Config
from src.guards.input import InputGuard
from src.guards.output import OutputGuard
from src.ingest.pipeline import DocSpec, ingest
from src.models import LLM, Embedding, HashingEmbedding, SentenceTransformerEmbedding
from src.retriever.hybrid import HeuristicReranker, HybridRetriever, Reranker
from src.retriever.lexical import BM25Retriever
from src.retriever.vector import VectorRetriever
from src.store.db import connect, load_visible_chunks

DEFAULT_WEIGHTS = {"vector": 0.5, "bm25": 0.5}


@dataclass
class SystemBundle:
    conn: sqlite3.Connection
    retriever: HybridRetriever
    graph: Graph
    input_guard: InputGuard
    output_guard: OutputGuard
    llm: LLM
    chunk_count: int


def make_embedding(kind: str = "hashing", **kw) -> Embedding:
    if kind == "hashing":
        return HashingEmbedding(**kw)
    if kind == "sentence-transformers":
        return SentenceTransformerEmbedding(**kw)
    if kind == "ollama":
        from src.models import OllamaEmbedding

        return OllamaEmbedding(**kw)
    raise ValueError(f"unknown embedding: {kind}")


def make_llm(name: str = "mock", **kw) -> LLM:
    if name == "mock":
        from src.models import MockLLM

        return MockLLM()
    if name == "ollama":
        from src.models import OllamaLLM

        return OllamaLLM(**kw)
    if name == "openai-compat":
        from src.models import OpenAICompatLLM

        return OpenAICompatLLM(**kw)
    raise ValueError(f"unknown llm: {name}")


def build_bundle(
    docs: list[DocSpec],
    *,
    db_path: str = "data/index.db",
    embedding_kind: str = "hashing",
    embedding_kwargs: dict | None = None,
    llm_name: str = "mock",
    llm_kwargs: dict | None = None,
    weights: dict[str, float] | None = None,
    reranker: Reranker | None = None,
    rrf_k: int = 60,
    sanitize_at: float = 0.35,
    block_at: float = 0.7,
    min_grounding: float = 0.25,
    top_k: int = 5,
    tenants: dict[str, tuple[str, ...]] | None = None,
) -> SystemBundle:
    """装配完整系统。

    tenants 用于告诉 BM25 内存索引有哪些 (tenant, groups) 组合需要建可见集；
    生产环境里这一步应该改成按需加载，而不是全量。
    """
    from src.pipeline.rag_graph import build_graph

    conn = connect(db_path)
    ingest(conn, docs)

    if tenants is None:
        tenants = {}
        for d in docs:
            merged = set(tenants.get(d.tenant_id, ())) | {d.acl_group}
            tenants[d.tenant_id] = tuple(sorted(merged))

    chunks: list = []
    for tenant, groups in tenants.items():
        chunks.extend(load_visible_chunks(conn, tenant, groups))

    embedding = make_embedding(embedding_kind, **(embedding_kwargs or {}))
    vector = VectorRetriever(embedding=embedding, conn=conn)
    lexical = BM25Retriever()
    lexical.index(chunks)

    retriever = HybridRetriever(
        vector=vector, lexical=lexical, rrf_k=rrf_k,
        weights=weights or DEFAULT_WEIGHTS, reranker=reranker,
    )
    input_guard = InputGuard(sanitize_at=sanitize_at, block_at=block_at)
    output_guard = OutputGuard(min_grounding=min_grounding)
    llm = make_llm(llm_name, **(llm_kwargs or {}))
    graph = build_graph(
        retriever=retriever, input_guard=input_guard,
        output_guard=output_guard, llm=llm, top_k=top_k,
    )
    return SystemBundle(
        conn=conn, retriever=retriever, graph=graph,
        input_guard=input_guard, output_guard=output_guard,
        llm=llm, chunk_count=len(chunks),
    )


def _llm_kwargs(cfg: Config) -> dict:
    """把配置里的 llm 字段整理成构造参数。不同实现的参数名不一样，这里做一次映射。"""
    kw: dict = {}
    if cfg.llm.model:
        kw["model"] = cfg.llm.model
    if cfg.llm.base_url:
        kw["base_url"] = cfg.llm.base_url
    return kw


def build_from_config(cfg: Config, docs: list[DocSpec] | None = None,
                      **overrides) -> SystemBundle:
    """按 config.json 装配。命令行传进来的参数优先于配置文件。"""
    from src.config import docs_from_config

    return build_bundle(
        docs if docs is not None else docs_from_config(cfg),
        db_path=cfg.db_path,
        embedding_kind=cfg.embedding.kind,
        embedding_kwargs=cfg.embedding.extra or None,
        llm_name=cfg.llm.kind,
        llm_kwargs=_llm_kwargs(cfg),
        weights=dict(cfg.retrieval.weights),
        reranker=HeuristicReranker() if cfg.retrieval.rerank else None,
        rrf_k=cfg.retrieval.rrf_k,
        sanitize_at=cfg.guards.sanitize_at,
        block_at=cfg.guards.block_at,
        min_grounding=cfg.guards.min_grounding,
        top_k=cfg.retrieval.top_k,
        **overrides,
    )
