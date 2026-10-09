"""混合召回：RRF 融合 + 可选重排。

为什么用 RRF 而不是加权求和（面试高频问题）：
  BM25 与余弦相似度输出的分数量纲不可比，加权前必须先归一化，
  而归一化系数对数据分布敏感。RRF 只用排名信息，天然免疫量纲问题。
  代价是丢掉了绝对相关性差异 —— 所以后面接一层 rerank 把它补回来。
"""

from __future__ import annotations

from typing import Sequence

from src.core.types import Chunk, ScoredChunk
from src.retriever.base import ACL, Retriever, tokenize


class Reranker:
    """重排接口。Identity 版什么都不做，用于验证融合本身的效果。"""

    name = "identity"

    def rerank(self, query: str, hits: Sequence[ScoredChunk],
               top_k: int) -> list[ScoredChunk]:
        return list(hits)[:top_k]


class HeuristicReranker(Reranker):
    """离线可用的轻量重排：按查询词覆盖率 + 位置信息给片段重新打分。

    真机上建议换成 BGE-M3 cross-encoder 或商业 rerank API，
    实现 rerank() 即可，调用方不变。
    """

    name = "heuristic"

    def rerank(self, query: str, hits: Sequence[ScoredChunk],
               top_k: int) -> list[ScoredChunk]:
        """重打分 = 词项覆盖 + 融合分 + 排名位置。

        三项必须先各自归一化再加权：融合分本身是 1/(k+rank) 量级（约 0.01），
        覆盖率是 0~1，直接加权等于让融合分彻底失效。这是写这段时踩过的坑。
        """
        q_terms = set(tokenize(query))
        hits = list(hits)
        if not hits:
            return []
        if not q_terms:
            return hits[:top_k]

        covers, scores, positions = [], [], []
        for h in hits:
            terms = set(tokenize(h.chunk.text))
            covers.append(len(q_terms & terms) / len(q_terms))
            scores.append(h.score)
            positions.append(1.0 / h.rank)
        c_max, s_max, p_max = max(covers) or 1.0, max(scores) or 1.0, max(positions) or 1.0

        rescored = []
        for i, h in enumerate(hits):
            final = (0.6 * covers[i] / c_max
                     + 0.3 * scores[i] / s_max
                     + 0.1 * positions[i] / p_max)
            rescored.append((final, h))
        rescored.sort(key=lambda t: t[0], reverse=True)
        return [
            ScoredChunk(chunk=h.chunk, score=s, source="rerank", rank=i + 1)
            for i, (s, h) in enumerate(rescored[:top_k])
        ]


class HybridRetriever(Retriever):
    """把两路召回结果按 RRF 融合。

    weights 用于给不同通道加权，取值由 eval/recall.py 的网格搜索给出，
    不要凭感觉填 —— 这也是本项目里"用数据做决策"的一个具体体现。
    """

    name = "hybrid"

    def __init__(self, vector: Retriever, lexical: Retriever,
                 rrf_k: int = 60, weights: dict[str, float] | None = None,
                 reranker: Reranker | None = None) -> None:
        self.vector = vector
        self.lexical = lexical
        self.rrf_k = rrf_k
        self.weights = weights or {"vector": 0.5, "bm25": 0.5}
        self.reranker = reranker

    def index(self, chunks: Sequence[Chunk]) -> None:
        self.vector.index(chunks)
        self.lexical.index(chunks)

    def search(self, query: str, top_k: int = 20, *, acl: ACL) -> list[ScoredChunk]:
        v_hits = self.vector.search(query, top_k=top_k, acl=acl)
        l_hits = self.lexical.search(query, top_k=top_k, acl=acl)

        fused: dict[str, float] = {}
        meta: dict[str, ScoredChunk] = {}
        for source, hits in ((self.vector.name, v_hits), (self.lexical.name, l_hits)):
            w = self.weights.get(source, 1.0)
            for h in hits:
                key = h.chunk.chunk_id
                contrib = w / (self.rrf_k + h.rank)
                fused[key] = fused.get(key, 0.0) + contrib
                meta.setdefault(key, h)

        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
        merged = [
            ScoredChunk(chunk=meta[k].chunk, score=s, source="hybrid", rank=i + 1)
            for i, (k, s) in enumerate(ranked[:top_k])
        ]
        if self.reranker is not None:
            merged = self.reranker.rerank(query, merged, top_k)
        return merged
