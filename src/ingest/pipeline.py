"""文档灌库：加载 → 切片 → 写库。

切片策略用「按段落边界优先、超长再按字数切」而不是固定长度硬切：
硬切会把表格和条款从中间断开，直接影响后面的引用溯源效果。
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from src.core.types import Chunk
from src.store.db import insert_chunks, upsert_document


@dataclass(frozen=True)
class DocSpec:
    """一个待灌库文档。ACL 元数据在这里指定，落地后成为检索过滤的依据。"""

    doc_id: str
    path: str
    title: str
    tenant_id: str
    acl_group: str
    sensitivity: int = 0


def load_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def chunk_text(text: str, *, size: int = 320, overlap: int = 60) -> list[str]:
    """先按空行切段，再把过长的段落按字数滑窗切分。"""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    out: list[str] = []
    for para in paragraphs:
        if len(para) <= size:
            out.append(para)
            continue
        start = 0
        while start < len(para):
            piece = para[start:start + size]
            out.append(piece)
            if start + size >= len(para):
                break
            start += max(1, size - overlap)
    return out or [text[:size]]


def build_chunks(spec: DocSpec, *, size: int = 320, overlap: int = 60) -> list[Chunk]:
    pieces = chunk_text(load_text(spec.path), size=size, overlap=overlap)
    return [
        Chunk(
            chunk_id=f"{spec.doc_id}#{i:04d}",
            doc_id=spec.doc_id,
            ordinal=i,
            text=piece,
            tenant_id=spec.tenant_id,
            acl_group=spec.acl_group,
            sensitivity=spec.sensitivity,
            title=spec.title,
        )
        for i, piece in enumerate(pieces)
    ]


def ingest(conn: sqlite3.Connection, specs: list[DocSpec]) -> int:
    """写入 documents / chunks 表，返回切片总数。"""
    total = 0
    for spec in specs:
        chunks = build_chunks(spec)
        upsert_document(
            conn, doc_id=spec.doc_id, title=spec.title, source=spec.path,
            tenant_id=spec.tenant_id, acl_group=spec.acl_group,
            sensitivity=spec.sensitivity,
        )
        insert_chunks(conn, chunks)
        total += len(chunks)
    conn.commit()
    return total
