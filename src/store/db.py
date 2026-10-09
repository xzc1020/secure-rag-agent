"""SQLite 存储层。

选 SQLite 的理由见 README：原型阶段优先可复现性与零运维，且把 ACL 过滤下推到
SQL WHERE 里，和 pgvector 的做法一致，迁移时只需换 VectorBackend 实现。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable, Sequence

from src.core.types import Chunk

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id       TEXT PRIMARY KEY,
    title        TEXT NOT NULL,
    source       TEXT,
    tenant_id    TEXT NOT NULL,
    acl_group    TEXT NOT NULL,
    sensitivity  INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    TEXT PRIMARY KEY,
    doc_id      TEXT NOT NULL,
    ordinal     INTEGER NOT NULL,
    text        TEXT NOT NULL,
    tenant_id   TEXT NOT NULL,
    acl_group   TEXT NOT NULL,
    sensitivity INTEGER NOT NULL DEFAULT 0,
    title       TEXT NOT NULL DEFAULT '',
    embedding   BLOB
);

-- ACL 下推的关键索引：tenant + acl_group 前缀匹配 WHERE 条件
CREATE INDEX IF NOT EXISTS idx_chunks_acl   ON chunks(tenant_id, acl_group);
CREATE INDEX IF NOT EXISTS idx_chunks_doc   ON chunks(doc_id, ordinal);

CREATE TABLE IF NOT EXISTS audit_log (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts         TEXT NOT NULL DEFAULT (datetime('now')),
    trace_id   TEXT NOT NULL,
    stage      TEXT NOT NULL,
    payload    TEXT
);
"""


def connect(db_path: str | Path = "data/index.db") -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def upsert_document(conn: sqlite3.Connection, *, doc_id: str, title: str,
                    source: str = "", tenant_id: str = "default",
                    acl_group: str = "public", sensitivity: int = 0) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO documents VALUES (?,?,?,?,?,?)",
        (doc_id, title, source, tenant_id, acl_group, sensitivity),
    )


def insert_chunks(conn: sqlite3.Connection, chunks: Sequence[Chunk],
                  embeddings: Sequence[bytes | None] | None = None) -> None:
    rows = [
        (
            c.chunk_id, c.doc_id, c.ordinal, c.text,
            c.tenant_id, c.acl_group, c.sensitivity, c.title,
            (embeddings[i] if embeddings else None),
        )
        for i, c in enumerate(chunks)
    ]
    conn.executemany(
        "INSERT OR REPLACE INTO chunks "
        "(chunk_id,doc_id,ordinal,text,tenant_id,acl_group,sensitivity,title,embedding) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        rows,
    )


def load_visible_chunks(conn: sqlite3.Connection, tenant_id: str,
                        groups: Iterable[str]) -> list[Chunk]:
    """按 ACL 拉取可见切片。

    注意：ACL 条件写在 SQL WHERE 里，不可见的数据根本不会被读进内存，
    而不是拉回来之后再过滤。这是本项目安全模型的基础。
    """
    groups = list(groups)
    if not groups:
        return []
    placeholders = ",".join("?" for _ in groups)
    sql = (
        "SELECT chunk_id,doc_id,ordinal,text,tenant_id,acl_group,sensitivity,title "
        f"FROM chunks WHERE tenant_id = ? AND acl_group IN ({placeholders}) "
        "ORDER BY doc_id, ordinal"
    )
    return [Chunk(**dict(r)) for r in conn.execute(sql, [tenant_id, *groups])]


def count_chunks(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
