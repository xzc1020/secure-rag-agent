"""检索层抽象。

面试时被问"数据量到十万级怎么扩"时的答案，落点就在这里的接口：
上层只依赖 Retriever 协议，换 pgvector / Milvus 只要新写一个实现类。

所有 search 方法都强制接收 acl 参数 —— 权限过滤是检索契约的一部分，
不会因为换了实现而被遗漏。
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Iterable, Sequence

from src.core.types import Chunk, ScoredChunk

ACL = tuple[str, frozenset[str]]  # (tenant_id, groups)


def _visible(chunk: Chunk, acl: ACL) -> bool:
    tenant_id, groups = acl
    return chunk.tenant_id == tenant_id and chunk.acl_group in groups


class Retriever(ABC):
    """召回后端统一协议。"""

    name: str = "base"

    @abstractmethod
    def index(self, chunks: Sequence[Chunk]) -> None:
        """构建索引。"""

    @abstractmethod
    def search(self, query: str, top_k: int = 20, *, acl: ACL) -> list[ScoredChunk]:
        """返回按相关性降序的结果，且保证全部对该 acl 可见。"""


# --------------------------------------------------------------------------
# 分词：不引入 jieba，用「CJK 单字 + bigram」保证中文召回
# --------------------------------------------------------------------------

_TOKEN_RE = None


def _compile_token_re():
    global _TOKEN_RE
    import re

    _TOKEN_RE = re.compile(r"[a-z0-9]+|[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
    return _TOKEN_RE


def _is_cjk(ch: str) -> bool:
    if len(ch) != 1:
        return False
    o = ord(ch)
    return 0x3400 <= o <= 0x4DBF or 0x4E00 <= o <= 0x9FFF or 0xF900 <= o <= 0xFAFF


def tokenize(text: str, *, bigram: bool = True) -> list[str]:
    """中英混合分词。中文用 单字+相邻bigram 近似词粒度，避免引入分词依赖。"""
    global _TOKEN_RE
    if _TOKEN_RE is None:
        _TOKEN_RE = _compile_token_re()
    units = _TOKEN_RE.findall(text.lower())
    if not bigram:
        return units
    out: list[str] = []
    for i, u in enumerate(units):
        out.append(u)
        if _is_cjk(u) and i + 1 < len(units) and _is_cjk(units[i + 1]):
            out.append(u + units[i + 1])
    return out


# --------------------------------------------------------------------------
# Embedding 抽象
# --------------------------------------------------------------------------

class Embedding(ABC):
    """向量编码抽象。离线降级用 HashingEmbedding，真机上换成 BGE / 各家 API。"""

    name: str = "base"
    dim: int = 0

    @abstractmethod
    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        ...


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError("dim mismatch")
    dot = sa = sb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        sa += x * x
        sb += y * y
    if sa == 0.0 or sb == 0.0:
        return 0.0
    return dot / (math.sqrt(sa) * math.sqrt(sb))
