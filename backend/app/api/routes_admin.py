"""管理路由：教师复核队列与审计查询（仅教师/管理员）。"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from backend.app.api.deps import Principal, record_audit, recent_audit, verify_access
from backend.app.tools.academic_affairs import run as academic_run

router = APIRouter(prefix="/api/v1/admin", tags=["admin"])


def _require_teacher(principal: Principal = Depends(verify_access)) -> Principal:
    if principal.role != "teacher":
        raise HTTPException(status_code=403, detail="仅教师可访问")
    return principal


@router.get("/review-queue")
def review_queue(principal: Principal = Depends(_require_teacher)) -> dict:
    """待复核任务列表（路由升级/低置信批改/申诉）。"""
    return academic_run("list_review_tasks", task={"status": "pending"})


class ReviewDecision(BaseModel):
    task_id: str
    decision: str  # approve / reject
    final_answer: str = ""


@router.post("/review-decision")
def review_decision(req: ReviewDecision,
                    principal: Principal = Depends(_require_teacher)) -> dict:
    """教师复核签发：approve 且带定稿答案时按镜像缓存纪律写回。"""
    if req.decision not in ("approve", "reject"):
        raise HTTPException(status_code=422, detail="decision 仅支持 approve/reject")
    from backend.app.db.cache import get_mirror_cache

    cached = False
    if req.decision == "approve" and req.final_answer:
        cached = get_mirror_cache().put(req.task_id, req.final_answer, "", "", reviewed=True)
    record_audit("review_decision", principal.actor, req.task_id,
                 f"{req.decision} cached={cached}")
    return {"task_id": req.task_id, "decision": req.decision, "written_to_cache": cached}


@router.get("/audit")
def audit(limit: int = 50, principal: Principal = Depends(_require_teacher)) -> dict:
    """审计查询：优先数据库，无库时回退进程内环形缓冲。"""
    import os

    if os.getenv("SKIP_DB", "0") != "1":
        try:
            from sqlalchemy import select
            from sqlalchemy.orm import Session as SASession

            from backend.app.db.models import AuditLog, get_engine, init_db
            from backend.app.config import get_settings

            engine = get_engine(get_settings().database_url)
            init_db(engine)
            with SASession(engine) as db:
                rows = db.scalars(select(AuditLog).order_by(AuditLog.id.desc())
                                  .limit(min(limit, 200))).all()
                if rows:
                    return {"source": "db", "items": [
                        {"actor": r.actor, "action": r.action, "subject": r.subject,
                         "detail": r.detail, "created_at": r.created_at} for r in rows]}
        except Exception:
            pass
    return {"source": "memory", "items": recent_audit(min(limit, 200))}


@router.get("/cache-stats")
def cache_stats(principal: Principal = Depends(_require_teacher)) -> dict:
    """镜像缓存运行指标（用于观察写回纪律生效情况）。"""
    from backend.app.db.cache import get_mirror_cache

    return get_mirror_cache().stats()
