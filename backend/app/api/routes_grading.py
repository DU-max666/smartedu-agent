"""批改路由：提交批改任务与学生申诉。"""
from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from backend.app.api.deps import Principal, record_audit, verify_access
from backend.app.graphs.grading import build_grading_graph
from backend.app.safety.schema_validator import is_valid

router = APIRouter(prefix="/api/v1/grading", tags=["grading"])
_graph = None

ITEM_SCHEMA: Dict[str, Any] = {
    "type": "array",
    "items": {
        "type": "object",
        "required": ["qid", "type", "student_answer"],
        "properties": {
            "qid": {"type": "string"},
            "type": {"type": "string", "enum": ["objective", "subjective"]},
            "stem": {"type": "string"},
            "student_answer": {"type": "string"},
            "ref_answer": {"type": "string"},
            "full_score": {"type": "number", "minimum": 0},
            "points": {"type": "array", "items": {"type": "string"}},
            "appeal": {"type": "boolean"},
        },
    },
}


class GradingRequest(BaseModel):
    grade: str = ""
    subject: str = ""
    items: List[Dict[str, Any]] = Field(min_length=1)


def _get_graph():
    global _graph
    if _graph is None:
        _graph = build_grading_graph()
    return _graph


@router.post("/run")
def run_grading(req: GradingRequest, principal: Principal = Depends(verify_access)) -> dict:
    """执行批改：客观题规则判分 + 主观题要点评分 + 低置信分流。"""
    if not is_valid(req.items, ITEM_SCHEMA):
        raise HTTPException(status_code=422, detail="题目结构不符合 schema")
    final = _get_graph().invoke({
        "items": req.items, "grade": req.grade, "subject": req.subject,
        "student_ref": principal.actor, "task_id": f"T-{principal.actor}"})
    record_audit("grading_run", principal.actor, req.subject or "",
                 f"review={final.get('review_status')} pending={final.get('low_confidence_items')}")
    return {
        "objective_results": final.get("objective_results", []),
        "subjective_results": final.get("subjective_results", []),
        "low_confidence_items": final.get("low_confidence_items", []),
        "review_status": final.get("review_status", "auto_done"),
        "comment_suggestion": final.get("comment", ""),
        "comment_note": "评语为建议稿，须教师签发后方可下发",
    }


class AppealRequest(BaseModel):
    qid: str
    reason: str = Field(min_length=2)


@router.post("/appeal")
def appeal(req: AppealRequest, principal: Principal = Depends(verify_access)) -> dict:
    """学生申诉：直接进教师复核队列，机器不改判。"""
    from backend.app.tools.academic_affairs import run as academic_run

    pushed = academic_run("push_review_task", task={
        "kind": "appeal", "student_ref": principal.actor,
        "qid": req.qid, "reason": req.reason})
    record_audit("grading_appeal", principal.actor, req.qid, req.reason[:100])
    return {"accepted": True, "review_task_id": pushed.get("task_id", ""),
            "message": "申诉已提交，将由任课老师复核判分。"}
