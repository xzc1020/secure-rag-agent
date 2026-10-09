"""配置加载。

用 JSON 而不是 YAML：核心链路要保持零第三方依赖，而 PyYAML 不在标准库里。
这个取舍本身也值得记录——很多项目默认上 YAML，其实 JSON + 注释足够，
还能省掉一层依赖。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = "config.json"


@dataclass(frozen=True)
class EmbeddingConfig:
    kind: str = "hashing"
    dim: int = 512
    model: str = "BAAI/bge-small-zh-v1.5"
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LLMConfig:
    kind: str = "mock"
    model: str = ""
    base_url: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int = 5
    rrf_k: int = 60
    weights: dict[str, float] = field(default_factory=lambda: {"vector": 0.5, "bm25": 0.5})
    rerank: bool = True


@dataclass(frozen=True)
class GuardConfig:
    sanitize_at: float = 0.35
    block_at: float = 0.7
    min_grounding: float = 0.25
    strict: bool = False


@dataclass(frozen=True)
class Config:
    db_path: str = "data/index.db"
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    guards: GuardConfig = field(default_factory=GuardConfig)
    docs: list[dict[str, Any]] = field(default_factory=list)


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    emb = raw.get("embedding", {})
    llm = raw.get("llm", {})
    ret = raw.get("retrieval", {})
    gi = raw.get("guards", {}).get("input", {})
    go = raw.get("guards", {}).get("output", {})
    return Config(
        db_path=raw.get("db_path", "data/index.db"),
        embedding=EmbeddingConfig(
            kind=emb.get("kind", "hashing"),
            dim=emb.get("dim", 512),
            model=emb.get("model", "BAAI/bge-small-zh-v1.5"),
            extra=emb.get("extra", {}),
        ),
        llm=LLMConfig(
            kind=llm.get("kind", "mock"),
            model=llm.get("model", ""),
            base_url=llm.get("base_url", ""),
            extra=llm,
        ),
        retrieval=RetrievalConfig(
            top_k=ret.get("top_k", 5),
            rrf_k=ret.get("rrf_k", 60),
            weights=ret.get("weights", {"vector": 0.5, "bm25": 0.5}),
            rerank=ret.get("rerank", True),
        ),
        guards=GuardConfig(
            sanitize_at=gi.get("sanitize_at", 0.35),
            block_at=gi.get("block_at", 0.7),
            min_grounding=go.get("min_grounding", 0.25),
            strict=go.get("strict", False),
        ),
        docs=raw.get("docs", []),
    )


def docs_from_config(cfg: Config):
    """把配置里的文档列表转成 DocSpec 列表。"""
    from src.ingest.pipeline import DocSpec

    return [
        DocSpec(
            doc_id=d["doc_id"], path=d["path"], title=d["title"],
            tenant_id=d["tenant_id"], acl_group=d["acl_group"],
            sensitivity=d.get("sensitivity", 0),
        )
        for d in cfg.docs
    ]
