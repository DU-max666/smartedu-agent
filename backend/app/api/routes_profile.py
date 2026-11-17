"""学情路由：教师/家长可查看，学生仅可查看本人（由上游网关保证，此处校验角色）。"""
from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from backend.app.api.deps import Principal, record_audit, verify_access
from backend.app.graphs.profile import build_profile_graph

router = APIRouter(prefix="/api/v1/profile", tags=["profile"])
_graph = None


class ProfileRequest(BaseModel):
    student_ref: str
    grade: str = ""
    subject: str = ""
    knowledge_tree: Dict[str, Any] | None = None


@router.post("/report")
def report(req: ProfileRequest, principal: Principal = Depends(verify_access)) -> dict:
    """生成学情简报（教师草稿级，需教师复核后下发）。"""
    if principal.role == "student" and req.student_ref != principal.actor:
        raise HTTPException(status_code=403, detail="学生仅可查看本人学情")
    global _graph
    if _graph is None:
        _graph = build_profile_graph()
    final = _graph.invoke({
        "student_ref": req.student_ref, "grade": req.grade, "subject": req.subject,
        "knowledge_tree": req.knowledge_tree or {}})
    record_audit("profile_report", principal.actor, req.student_ref,
                 f"lagging={final.get('lagging_nodes')}")
    return {
        "mastery": final.get("mastery", {}),
        "lagging_nodes": final.get("lagging_nodes", []),
        "attribution": final.get("attribution", ""),
        "report": final.get("report", ""),
        "risk_level": "teacher_draft",
    }
