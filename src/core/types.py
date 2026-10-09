"""核心数据结构。

设计原则：所有跨模块的契约都定义在这里，上层不依赖任何具体实现的类型，
这样替换检索后端（SQLite -> pgvector）或模型（hash embedding -> BGE）时不需要改上层。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# --------------------------------------------------------------------------
# 权限模型
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Principal:
    """调用主体。检索层拿它做过滤下推。"""

    user_id: str
    tenant_id: str
    groups: frozenset[str]

    def to_filter(self) -> dict[str, Any]:
        """转成可直接进 SQL WHERE 的过滤条件。"""
        return {"tenant_id": self.tenant_id, "groups": tuple(sorted(self.groups))}


@dataclass(frozen=True)
class Chunk:
    """一段文档切片。ACL 字段冗余落到这里，是为了让检索能在存储层先过滤。"""

    chunk_id: str
    doc_id: str
    ordinal: int
    text: str
    tenant_id: str
    acl_group: str
    sensitivity: int = 0
    title: str = ""

    def is_visible_to(self, tenant_id: str, groups: frozenset[str]) -> bool:
        return self.tenant_id == tenant_id and self.acl_group in groups


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float
    source: str  # vector | lexical | hybrid
    rank: int = 0


# --------------------------------------------------------------------------
# 编排上下文
# --------------------------------------------------------------------------

@dataclass
class TurnContext:
    """一次请求在全链路上的共享状态。每个节点读写自己关心的字段。"""

    query: str
    tenant_id: str
    groups: frozenset[str]
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    # 护栏产物
    safe_query: str = ""
    blocked: bool = False
    block_reason: str = ""

    # 检索产物
    vector_hits: list[ScoredChunk] = field(default_factory=list)
    lexical_hits: list[ScoredChunk] = field(default_factory=list)
    candidates: list[ScoredChunk] = field(default_factory=list)

    # 生成产物
    answer: str = ""
    citations: list[str] = field(default_factory=list)
    grounded: bool | None = None  # 引用溯源校验结果

    # 观测
    audit: list[dict[str, Any]] = field(default_factory=list)

    def log(self, stage: str, **payload: Any) -> None:
        self.audit.append({"stage": stage, **payload})

    @property
    def acl(self) -> tuple[str, frozenset[str]]:
        return self.tenant_id, self.groups


# --------------------------------------------------------------------------
# 护栏判定
# --------------------------------------------------------------------------

class Action(str, Enum):
    ALLOW = "allow"        # 放行
    SANITIZE = "sanitize"  # 清洗后放行
    ESCALATE = "escalate"  # 需要人工确认（工具调用场景）
    BLOCK = "block"        # 直接拒绝


@dataclass(frozen=True)
class Verdict:
    action: Action
    reason: str
    risk: float = 0.0
    payload: Any = None
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def is_blocked(self) -> bool:
        return self.action is Action.BLOCK
