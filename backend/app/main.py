"""FastAPI 应用工厂：挂载中间件、路由与 SSE 流式端点。

启动时自动把 seed_corpus 灌入检索索引（离线 mock 嵌入，秒级完成），
保证 `uvicorn backend.app.main:app` 一条命令即可演示全链路。
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from backend.app.api.deps import (Principal, RateLimitMiddleware, record_audit,
                                  verify_access)
from backend.app.api.routes_admin import router as admin_router
from backend.app.api.routes_grading import router as grading_router
from backend.app.api.routes_lesson import router as lesson_router
from backend.app.api.routes_profile import router as profile_router
from backend.app.api.routes_qa import router as qa_router
from backend.app.config import get_settings
from backend.app.graphs.router_graph import get_router_graph
from backend.app.safety.pii_masker import get_pii_masker

SEED_DIR = Path(__file__).resolve().parent / "data" / "seed_corpus"


def _bootstrap_corpus() -> Dict[str, Any]:
    """启动灌库：seed_corpus -> 管道 -> 混合检索索引。"""
    from backend.app.data.pipeline import CorpusPipeline

    if not SEED_DIR.exists():
        return {"docs": 0, "note": "seed_corpus 缺失"}
    return CorpusPipeline().ingest_dir(str(SEED_DIR))


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.corpus_stats = _bootstrap_corpus()
    yield


class ChatRequest(BaseModel):
    """统一入口请求体。"""

    utterance: str = Field(min_length=1, max_length=500)
    session_id: str = "default"
    clarify_rounds: int = 0
    result_input: Dict[str, Any] = Field(default_factory=dict)


def create_app() -> FastAPI:
    """应用工厂。"""
    settings = get_settings()
    app = FastAPI(title="慧学·中小学多智能体教学辅助平台", version="0.1.0",
                  lifespan=lifespan)
    app.add_middleware(RateLimitMiddleware, qps=20.0, burst=40)
    app.include_router(qa_router)
    app.include_router(lesson_router)
    app.include_router(grading_router)
    app.include_router(profile_router)
    app.include_router(admin_router)

    @app.get("/healthz")
    def healthz() -> Dict[str, Any]:
        return {"status": "ok", "mock_mode": settings.mock_mode,
                "corpus": getattr(app.state, "corpus_stats", {})}

    @app.post("/api/v1/chat")
    def chat(req: ChatRequest, principal: Principal = Depends(verify_access)) -> Dict[str, Any]:
        """统一入口：走顶层路由图（角色收口→辨意→派发闸值）。"""
        masker = get_pii_masker()
        masked = masker.mask(req.utterance, req.session_id)
        if not masked.ok:
            return {"answer": "为了保护你的隐私，这条消息需要老师当面帮你。",
                    "escalate_review": True}
        final = get_router_graph().run({
            "session_id": req.session_id, "user_role": principal.role,
            "utterance": masked.masked_text, "clarify_rounds": req.clarify_rounds,
            "result_input": req.result_input})
        answer = masker.restore(str(final.get("clarify_question") or
                                   json.dumps(final.get("result", {}), ensure_ascii=False)),
                                req.session_id)
        record_audit("chat", principal.actor, req.session_id,
                     f"intent={final.get('intent')} target={final.get('dispatch_target')}")
        return {
            "intent": final.get("intent", ""), "confidence": final.get("intent_confidence", 0),
            "dispatch_target": final.get("dispatch_target", ""),
            "clarify_question": final.get("clarify_question", ""),
            "escalate_review": final.get("escalate_review", False),
            "result": final.get("result", {}), "answer": answer,
        }

    @app.get("/api/v1/chat/stream")
    async def chat_stream(q: str, session_id: str = "default",
                          principal: Principal = Depends(verify_access)):
        """SSE 流式端点：先推路由决策事件，再逐块推送答案。"""
        masker = get_pii_masker()
        masked = masker.mask(q, session_id)

        async def event_gen():
            if not masked.ok:
                yield {"event": "refuse", "data": "输入包含无法安全脱敏的信息"}
                return
            final = get_router_graph().run({
                "session_id": session_id, "user_role": principal.role,
                "utterance": masked.masked_text})
            yield {"event": "route", "data": json.dumps(
                {"intent": final.get("intent"), "target": final.get("dispatch_target"),
                 "confidence": final.get("intent_confidence")}, ensure_ascii=False)}
            text = final.get("clarify_question") or str(
                (final.get("result") or {}).get("answer") or
                (final.get("result") or {}).get("report") or
                (final.get("result") or {}).get("comment") or "已完成处理。")
            restored = masker.restore(text, session_id)
            for i in range(0, len(restored), 12):
                yield {"event": "token", "data": restored[i:i + 12]}
            yield {"event": "done", "data": "end"}

        return EventSourceResponse(event_gen())

    return app


app = create_app()
