"""轻量 DAG 编排引擎。

为什么不直接用 LangGraph：不是为了重复造轮子，而是这个项目要在执行路径上插入
安全卡点（护栏），需要精确控制每个节点的输入契约与终止条件。自写约 200 行即可，
代价很小，换来的是每个节点都能讲清楚。

特性：
  - 节点就是一个接受 TurnContext 的函数，可返回 next-label 决定路由
  - 边可带 label 或 when 条件谓词
  - 每个节点自动记录耗时与审计日志
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

from src.core.types import TurnContext

# 注意：这里必须写 Optional[str] 而不是 str | None。
# `from __future__ import annotations` 只让**注解**延迟求值，
# 类型别名是普通赋值语句，会立即执行；而 `X | Y`（PEP 604）要 Python 3.10+ 才支持。
NodeFn = Callable[[TurnContext], Optional[str]]


@dataclass(frozen=True)
class Edge:
    to: str
    label: str | None = None
    when: Callable[[TurnContext], bool] | None = None

    def matches(self, ctx: TurnContext, route: str | None) -> bool:
        if self.when is not None:
            return self.when(ctx)
        if self.label is not None:
            return route == self.label
        return route is None or route == ""


@dataclass
class Graph:
    name: str = "graph"
    _nodes: dict[str, NodeFn] = field(default_factory=dict)
    _edges: dict[str, list[Edge]] = field(default_factory=lambda: defaultdict(list))
    _entry: str | None = None

    def node(self, fn: NodeFn | None = None, *, name: str | None = None):
        """注册节点，支持装饰器用法。"""

        def wrap(f: NodeFn) -> NodeFn:
            n = name or f.__name__
            if n in self._nodes:
                raise ValueError(f"duplicate node: {n}")
            self._nodes[n] = f
            return f

        return wrap(fn) if fn is not None else wrap

    def add_node(self, name: str, fn: NodeFn) -> "Graph":
        self._nodes[name] = fn
        return self

    def edge(self, src: str, dst: str, *, label: str | None = None,
             when: Callable[[TurnContext], bool] | None = None) -> "Graph":
        if src not in self._nodes or dst not in self._nodes:
            raise KeyError(f"unknown node in edge {src} -> {dst}")
        self._edges[src].append(Edge(dst, label, when))
        return self

    def entry(self, name: str) -> "Graph":
        self._entry = name
        return self

    def _next(self, current: str, ctx: TurnContext, route: str | None) -> str | None:
        """按优先级选下一条出边：显式 when 条件 > label 匹配 > 默认边。"""
        edges = self._edges.get(current, [])
        if not edges:
            return None
        for e in edges:
            if e.when is not None and e.when(ctx):
                return e.to
        if route:
            for e in edges:
                if e.label == route:
                    return e.to
        fallbacks = [e for e in edges if e.when is None and e.label is None]
        return fallbacks[0].to if fallbacks else None

    def run(self, ctx: TurnContext, *, start: str | None = None,
            max_steps: int = 64) -> TurnContext:
        current = start or self._entry
        if current is None:
            raise RuntimeError("entry node not set")

        steps = 0
        while current is not None:
            if steps >= max_steps:
                raise RuntimeError(f"max steps exceeded at node={current}")
            fn = self._nodes[current]
            n_before = len(ctx.audit)
            t0 = time.perf_counter()
            route = fn(ctx)
            cost_ms = round((time.perf_counter() - t0) * 1000, 2)

            # 节点内部自己落的日志保留业务字段，这里补上耗时与所属节点，
            # 保证一条 trunk_audit 记录既能知道发生了什么也能知道花了多久。
            produced = ctx.audit[n_before:]
            if not produced:
                ctx.log(current, cost_ms=cost_ms)
            else:
                produced[0]["cost_ms"] = cost_ms
                produced[0]["node"] = current

            current = self._next(current, ctx, route)
            steps += 1
        return ctx

    def ascii_map(self, *, start: str | None = None) -> str:
        """生成流程图的 ASCII 表示，可直接贴进 README。"""
        start = start or self._entry or ""
        lines: list[str] = [f"[{start}]"]
        seen: set[str] = set()

        def walk(node: str, depth: int, visited: Iterable[str]) -> None:
            if node in seen:
                return
            seen.add(node)
            for e in self._edges.get(node, []):
                tag = e.label or ("#" + getattr(e.when, "__name__", "cond") if e.when else "")
                suffix = f"  ({tag})" if tag else ""
                lines.append("  " + "  " * depth + "-> " + e.to + suffix)
                walk(e.to, depth + 1, visited)

        walk(start, 0, seen)
        return "\n".join(lines)
