"""向量召回实现（暴力余弦）。

关于为什么会在这里用暴力检索：
  - 原型阶段数据量在万级以内，暴力余弦的延迟完全够用，且零依赖、可确定复现
  - 换成 sqlite-vec / pgvector 时**只需要新增一个实现类**，上层通过 Retriever
    协议调用，不受影响 —— 这是接口抽象的实际价值，也是应对"数据量大了怎么办"的答案

ACL 仍然在 SQL 层下推（见 src/store/db.py 的 load_visible_chunks）。
"""

from __future__ import annotations

import pickle
import sqlite3
from typing import Sequence

from src.core.types import Chunk, ScoredChunk
from src.retriever.base import ACL, Embedding, Retriever, cosine
from src.store.db import load_visible_chunks


class VectorRetriever(Retriever):
    name = "vector"

    def __init__(self, embedding: Embedding, conn: sqlite3.Connection | None = None,
                 db_path: str = "data/index.db") -> None:
        self.embedding = embedding
        if conn is None:
            conn = sqlite3.connect(db_path)
        self.conn = conn
        self._chunks: list[Chunk] = []
        self._vecs: list[list[float]] = []
        self._cache: dict[ACL, tuple[list[Chunk], list[list[float]]]] = {}

    # ------------------------------------------------------------------
    def refresh(self, acl: ACL) -> tuple[list[Chunk], list[list[float]]]:
        """从 SQLite 加载该 ACL 可见的切片。SQL WHERE 已经在 storage 层做过租户/权限过滤。"""
        cached = self._cache.get(acl)
        if cached is not None:
            return cached

        tenant_id, groups = acl
        chunks = load_visible_chunks(self.conn, tenant_id, groups)
        if not chunks:
            self._cache[acl] = ([], [])
            return [], []

        rows = {
            r["chunk_id"]: r["embedding"]
            for r in self.conn.execute("SELECT chunk_id, embedding FROM chunks")
        }
        missing = [c for c in chunks if rows.get(c.chunk_id) is None]
        if missing:  # 增量编码：只为还没有向量的 chunk 计算 embedding
            vecs = self.embedding.encode([c.text for c in missing])
            self.conn.executemany(
                "UPDATE chunks SET embedding = ? WHERE chunk_id = ?",
                [(pickle.dumps(v), c.chunk_id) for v, c in zip(vecs, missing)],
            )
            self.conn.commit()
            for c, v in zip(missing, vecs):
                rows[c.chunk_id] = pickle.dumps(v)

        vectors = [pickle.loads(rows[c.chunk_id]) for c in chunks]
        self._cache[acl] = (chunks, vectors)
        return chunks, vectors

    def invalidate(self) -> None:
        """灌库后调用，让可见集与向量缓存失效。"""
        self._cache.clear()

    # ------------------------------------------------------------------
    def index(self, chunks: Sequence[Chunk]) -> None:
        """向量检索的索引物化在 SQLite 里（见 refresh），这里主动失效缓存即可。"""
        self.invalidate()

    def search(self, query: str, top_k: int = 20, *, acl: ACL) -> list[ScoredChunk]:
        chunks, vectors = self.refresh(acl)
        if not chunks:
            return []
        qv = self.embedding.encode([query])[0]
        scored = [(cosine(qv, v), i) for i, v in enumerate(vectors)]
        scored.sort(key=lambda t: t[0], reverse=True)
        return [
            ScoredChunk(chunk=chunks[i], score=s, source=self.name, rank=r + 1)
            for r, (s, i) in enumerate(scored[:top_k])
        ]
