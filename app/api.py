"""FastAPI 服务层。

用 FastAPI 而不是自己搓 HTTP 的理由很简单：OpenAPI 文档白送，
健康检查、请求体校验、错误码都是现成的。这个项目的重点在检索与安全，
HTTP 这层没必要重复劳动。

启动： PYTHONPATH=. python -m app.api
文档： http://localhost:8000/docs
"""

from __future__ import annotations

from typing import Any

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "app/api.py 需要 fastapi：pip install -r requirements-real.txt\n"
        "若只是要跑链路，用 `python demo.py`（零依赖，不需要 fastapi）。"
    ) from exc

from src.core.types import Principal, TurnContext
from src.pipeline.factory import build_from_config
from src.config import load_config

APP_TITLE = "Secure RAG Agent"
APP_DESC = (
    "面向不可信输入的企业文档问答 Agent。"
    "核心设计：ACL 在检索层下推，输入/输出双层护栏，全生命周期审计。"
)

app = FastAPI(title=APP_TITLE, description=APP_DESC, version="0.1.0")


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000,
                          description="用户提问")
    user_id: str = Field(..., description="调用者 ID，用于审计")
    tenant_id: str = Field(..., description="租户")
    groups: list[str] = Field(default_factory=list,
                              description="该用户所属权限组；为空则看不到任何文档")


class QueryResponse(BaseModel):
    trace_id: str
    answer: str
    blocked: bool
    block_reason: str | None
    citations: list[str] = Field(default_factory=list)
    grounded: bool | None
    sources: list[str] = Field(default_factory=list)
    timeline: list[dict[str, Any]] = Field(default_factory=list)


# --------------------------------------------------------------------------
# 应用级单例。生产上这里应该换成依赖注入 + 连接池，原型阶段够用。
# --------------------------------------------------------------------------

_STATE: dict[str, Any] = {}


@app.on_event("startup")
def startup() -> None:
    cfg = load_config()
    bundle = build_from_config(cfg)
    _STATE["bundle"] = bundle
    _STATE["config"] = cfg


def _bundle():
    bundle = _STATE.get("bundle")
    if bundle is None:  # 兜底：允许不经过 startup 事件直接调用（测试场景）
        cfg = load_config()
        bundle = build_from_config(cfg)
        _STATE["bundle"] = bundle
    return bundle


@app.get("/health")
def health() -> dict[str, Any]:
    b = _bundle()
    return {"status": "ok", "chunks": b.chunk_count, "llm": b.llm.name}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    bundle = _bundle()
    principal = Principal(
        user_id=req.user_id, tenant_id=req.tenant_id, groups=frozenset(req.groups)
    )
    ctx = TurnContext(
        query=req.question,
        tenant_id=principal.tenant_id,
        groups=principal.groups,
    )
    bundle.graph.run(ctx)

    return QueryResponse(
        trace_id=ctx.trace_id,
        answer=ctx.answer,
        blocked=ctx.blocked,
        block_reason=ctx.block_reason or None,
        citations=ctx.citations,
        grounded=ctx.grounded,
        sources=sorted({h.chunk.doc_id for h in ctx.candidates}),
        timeline=ctx.audit,
    )


@app.get("/trace/{trace_id}")
def trace(trace_id: str) -> dict[str, Any]:
    """按 trace_id 回溯一条完整请求链路。

    目前是内存占位实现：这里本应落 SQLite 的 audit_log 表（schema 已经建好）。
    """
    raise HTTPException(status_code=501, detail="审计日志落表尚未实现，见 README Roadmap")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
