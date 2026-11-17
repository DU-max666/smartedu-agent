"""答疑路由：脱敏 → 敏感词 → 答疑子图 → 风险分级 → 回填还原。"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from backend.app.api.deps import Principal, record_audit, verify_access
from backend.app.graphs.qa import build_qa_graph
from backend.app.safety.pii_masker import get_pii_masker
from backend.app.safety.risk_policy import classify, enforce
from backend.app.safety.sensitive_words import SensitiveWordScanner

router = APIRouter(prefix="/api/v1/qa", tags=["qa"])
_graph = None
_scanner = SensitiveWordScanner()


def _get_graph():
    global _graph
    if _graph is None:
        _graph = build_qa_graph()
    return _graph


class AskRequest(BaseModel):
    """答疑请求。"""

    question: str = Field(min_length=2, max_length=500)
    grade: str = ""
    subject: str = ""
    session_id: str = "default"


class AskResponse(BaseModel):
    answer: str
    refuse_reason: str = ""
    cached: bool = False
    review_status: str = "none"


@router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, principal: Principal = Depends(verify_access)) -> AskResponse:
    """同步答疑端点（流式见 main.py 的 SSE 端点）。"""
    masker = get_pii_masker()
    masked = masker.mask(req.question, req.session_id)
    if not masked.ok:
        record_audit("qa_refuse", principal.actor, "pii", masked.reason)
        raise HTTPException(status_code=422, detail="输入包含无法安全脱敏的信息，已转教师处理")

    hits = _scanner.scan(masked.masked_text)
    if _scanner.should_block(hits):
        record_audit("qa_refuse", principal.actor, "sensitive", ",".join(h.word for h in hits))
        return AskResponse(answer="这个话题需要你的班主任来帮助你，我已经悄悄告诉老师了。",
                           refuse_reason="sensitive_topic", review_status="escalated")

    final = _get_graph().invoke({
        "question": masked.masked_text, "grade": req.grade, "subject": req.subject,
        "session_id": req.session_id, "student_ref": principal.actor,
    })
    answer = masker.restore(final.get("answer", ""), req.session_id)

    level = classify("qa_answer")
    decision = enforce(level, {
        "pii_ok": True,
        "citation_ok": bool(final.get("citation_valid")),
        "consistency_ok": bool(final.get("consistency_ok")),
        "sensitive_ok": True,
    })
    review_status = "none"
    if not decision.allowed:
        review_status = "teacher_review"
        record_audit("qa_downgrade", principal.actor, "risk", ";".join(decision.reasons))
    record_audit("qa_ask", principal.actor, req.session_id,
                 f"cached={final.get('cached')} refused={bool(final.get('refuse_reason'))}")
    safe_answer = answer if decision.allowed else "老师正在核实这个答案，稍后给你准确回复。"
    return AskResponse(answer=safe_answer, refuse_reason=final.get("refuse_reason", ""),
                       cached=bool(final.get("cached")), review_status=review_status)


class ReviewAnswerRequest(BaseModel):
    """教师复核定稿：定稿答案按镜像缓存纪律写回。"""

    question: str
    answer: str
    grade: str = ""
    subject: str = ""


@router.post("/review-finalize")
def finalize(req: ReviewAnswerRequest, principal: Principal = Depends(verify_access)) -> dict:
    if principal.role != "teacher":
        raise HTTPException(status_code=403, detail="仅教师可定稿")
    from backend.app.db.cache import get_mirror_cache

    written = get_mirror_cache().put(req.question, req.answer, req.grade, req.subject,
                                    reviewed=True)
    record_audit("qa_finalize", principal.actor, "cache", f"written={written}")
    return {"cached": written}
