"""BM25 召回实现。

关键设计：ACL 过滤发生在**倒排表遍历阶段**，不可见的 chunk 既不参与打分
也不会进结果集，而不是先全量召回再剔除。这一点和向量侧的 SQL WHERE 下推
属于同一种处理，答辩时可以一起讲。
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Sequence

from src.core.types import Chunk, ScoredChunk
from src.retriever.base import ACL, Retriever, tokenize


class BM25Retriever(Retriever):
    name = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b

        self._chunks: list[Chunk] = []
        self._by_acl: dict[ACL, list[int]] = {}  # ACL -> 可见 chunk 下标

        self._tf: list[dict[str, int]] = []
        self._df: dict[str, int] = defaultdict(int)
        self._postings: dict[str, list[int]] = defaultdict(list)
        self._avgdl: float = 0.0

    # ------------------------------------------------------------------
    def index(self, chunks: Sequence[Chunk]) -> None:
        self._chunks = list(chunks)
        self._by_acl.clear()
        self._tf.clear()
        self._df.clear()
        self._postings.clear()

        total_len = 0
        for i, c in enumerate(self._chunks):
            tokens = tokenize(c.text)
            tf_row: dict[str, int] = defaultdict(int)
            for t in tokens:
                tf_row[t] += 1
            self._tf.append(dict(tf_row))
            total_len += len(tokens)
            for t in tf_row:
                self._df[t] += 1
                self._postings[t].append(i)
        self._avgdl = (total_len / len(self._chunks)) if self._chunks else 0.0

    # ------------------------------------------------------------------
    def _visible_indices(self, acl: ACL) -> list[int]:
        """同一 ACL 的可见集合会被缓存，避免每次请求重复扫全表。"""
        cached = self._by_acl.get(acl)
        if cached is not None:
            return cached
        tenant_id, groups = acl
        visible = [
            i for i, c in enumerate(self._chunks)
            if c.tenant_id == tenant_id and c.acl_group in groups
        ]
        self._by_acl[acl] = visible
        return visible

    def search(self, query: str, top_k: int = 20, *, acl: ACL) -> list[ScoredChunk]:
        visible = self._visible_indices(acl)
        if not visible:
            return []
        visible_set = set(visible)
        n_docs = len(visible)

        scores: dict[int, float] = {}
        for term in set(tokenize(query)):
            postings = self._postings.get(term)
            if not postings:
                continue
            df = self._df[term]
            # 逆文档频率；注意 df 是全局统计，这里用可见集合规模做平滑
            idf = math.log(1 + (n_docs - min(df, n_docs) + 0.5) / (min(df, n_docs) + 0.5))
            for i in postings:
                if i not in visible_set:  # ACL 下推：不可见直接跳过打分
                    continue
                tf = self._tf[i].get(term, 0)
                if not tf:
                    continue
                dl = sum(self._tf[i].values()) or self._avgdl or 1.0
                denom = tf + self.k1 * (1 - self.b + self.b * dl / (self._avgdl or 1.0))
                scores[i] = scores.get(i, 0.0) + idf * (tf * (self.k1 + 1)) / (denom or 1.0)

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [
            ScoredChunk(chunk=self._chunks[i], score=s, source=self.name, rank=r + 1)
            for r, (i, s) in enumerate(ranked)
        ]
