"""教务系统工具：作答记录读取与教师复核任务下发。

真实部署对接校内教务接口（HTTP）；离线/演示环境使用内存 mock 数据源，
run() 契约一致。学生标识一律使用脱敏代号（student_ref），
本模块拒绝接受姓名/学号明文（校验失败直接报错并记审计）。
"""
from __future__ import annotations

import re
import time
from typing import Any, Dict, List

TOOL_SPEC: Dict[str, Any] = {
    "name": "academic_affairs",
    "description": "教务数据访问：读取学生脱敏作答记录、提交/查询教师复核任务。",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["get_attempts", "push_review_task", "list_review_tasks"],
                "description": "操作类型",
            },
            "student_ref": {"type": "string", "description": "脱敏学生代号，形如 S-xxxx"},
            "subject": {"type": "string", "description": "学科（get_attempts 用）"},
            "task": {"type": "object", "description": "复核任务体（push_review_task 用）"},
        },
        "required": ["action"],
        "additionalProperties": False,
    },
}

_STUDENT_REF_RE = re.compile(r"^S-[A-Za-z0-9]{4,12}$")

# ---- mock 数据源（演示用自创数据，student_ref 为脱敏代号）----
_MOCK_ATTEMPTS: Dict[str, List[Dict[str, Any]]] = {
    "S-demo01": [
        {"hw": 1, "qid": "MATH-3A-001", "topic": "单位换算", "correct": True, "score_ratio": 1.0},
        {"hw": 1, "qid": "MATH-3A-002", "topic": "单位换算", "correct": False, "score_ratio": 0.33},
        {"hw": 2, "qid": "MATH-3A-003", "topic": "周长", "correct": True, "score_ratio": 1.0},
        {"hw": 3, "qid": "MATH-3A-001R", "topic": "单位换算", "correct": False, "score_ratio": 0.0},
        {"hw": 3, "qid": "MATH-3A-002R", "topic": "单位换算", "correct": False, "score_ratio": 0.33},
    ],
}
_REVIEW_TASKS: List[Dict[str, Any]] = []


def run(action: str, student_ref: str = "", subject: str = "",
        task: Dict[str, Any] | None = None) -> Dict[str, Any]:
    if action == "get_attempts":
        if student_ref and not _STUDENT_REF_RE.match(student_ref):
            return {"error": "student_ref 必须是脱敏代号（S-xxxx），拒绝明文身份"}
        attempts = _MOCK_ATTEMPTS.get(student_ref, [])
        return {"student_ref": student_ref, "attempts": attempts, "count": len(attempts)}

    if action == "push_review_task":
        payload = dict(task or {})
        payload.setdefault("task_id", f"RV-{int(time.time() * 1000) % 10**9}")
        payload.setdefault("status", "pending")
        payload["created_at"] = time.time()
        _REVIEW_TASKS.append(payload)
        return {"accepted": True, "task_id": payload["task_id"]}

    if action == "list_review_tasks":
        status = (task or {}).get("status", "")
        rows = [t for t in _REVIEW_TASKS if not status or t["status"] == status]
        return {"tasks": rows, "count": len(rows)}

    return {"error": f"未知 action: {action}"}
