"""备课路由：仅教师角色可用（白名单在路由图与端点双重校验）。"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from backend.app.api.deps import Principal, record_audit, verify_access
from backend.app.graphs.lesson_prep import build_lesson_graph

router = APIRouter(prefix="/api/v1/lesson", tags=["lesson"])
_graph = None


def _require_teacher(principal: Principal = Depends(verify_access)) -> Principal:
    if principal.role != "teacher":
        raise HTTPException(status_code=403, detail="备课功能仅对教师开放")
    return principal


class DraftRequest(BaseModel):
    subject: str = Field(min_length=1)
    grade: str = ""
    topic: str = Field(min_length=1)


@router.post("/draft")
def create_draft(req: DraftRequest, principal: Principal = Depends(_require_teacher)) -> dict:
    """生成备课草稿（大纲/课件/组卷），产物为教师草稿级。"""
    global _graph
    if _graph is None:
        _graph = build_lesson_graph()
    final = _graph.invoke({"subject": req.subject, "grade": req.grade,
                          "topic": req.topic, "teacher_id": principal.actor})
    record_audit("lesson_draft", principal.actor, final.get("draft_id", ""),
                 f"stages={len(final.get('outline', []))} quiz={len(final.get('quiz_items', []))}")
    return {
        "draft_id": final.get("draft_id", ""),
        "draft_status": final.get("draft_status", "draft"),
        "outline": final.get("outline", []),
        "slides": final.get("slides_content", []),
        "quiz": final.get("quiz_items", []),
    }
