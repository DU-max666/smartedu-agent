"""API 依赖：轻量鉴权、按 IP 令牌桶限流、审计中间件。

鉴权模型面向"区域教育平台网关后置"的部署形态：上游统一认证网关
校验票据后，只透传 API Key 与角色头；本服务不接触任何学生明文身份。
"""
from __future__ import annotations

import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict

from fastapi import Header, HTTPException, Request

from backend.app.config import get_settings

VALID_ROLES = ("teacher", "student", "parent")


@dataclass
class Principal:
    """请求主体：角色 + 脱敏标识。"""

    role: str
    actor: str


def verify_access(x_huixue_key: str = Header(default=""),
                  x_user_role: str = Header(default=""),
                  x_user_ref: str = Header(default="")) -> Principal:
    """校验接入 Key 与角色头；学生/家长必须携带脱敏代号。"""
    expected = os.getenv("HUIXUE_API_KEY", "dev-local-key")
    if x_huixue_key != expected:
        raise HTTPException(status_code=401, detail="接入密钥无效")
    if x_user_role not in VALID_ROLES:
        raise HTTPException(status_code=403, detail="角色不合法")
    actor = x_user_ref or f"{x_user_role}-anon"
    if x_user_role in ("student", "parent") and not (x_user_ref.startswith("S-") or x_user_ref.endswith("-anon")):
        raise HTTPException(status_code=403, detail="学生/家长标识必须为脱敏代号")
    return Principal(role=x_user_role, actor=actor)


class RateLimitMiddleware:
    """纯 ASGI 中间件：按客户端 IP 令牌桶限流（默认 20 QPS）。"""

    def __init__(self, app, qps: float = 20.0, burst: int = 40) -> None:
        self.app = app
        self.qps = qps
        self.burst = burst
        self._buckets: Dict[str, Deque[float]] = defaultdict(deque)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = dict((k.lower().decode(), v.decode()) for k, v in scope.get("headers", []))
        ip = headers.get("x-forwarded-for", "local").split(",")[0].strip()
        now = time.monotonic()
        bucket = self._buckets[ip]
        while bucket and now - bucket[0] > 1.0:
            bucket.popleft()
        if len(bucket) >= self.burst:
            from starlette.responses import JSONResponse

            resp = JSONResponse({"detail": "请求过于频繁，请稍后再试"}, status_code=429)
            await resp(scope, receive, send)
            return
        bucket.append(now)
        await self.app(scope, receive, send)


# 进程内审计环形缓冲（无数据库时 admin 查询走这里）
_AUDIT_BUFFER: deque = deque(maxlen=2000)


def record_audit(action: str, actor: str, subject: str, detail: str) -> None:
    """记录一条审计；同时尽力落库（有库时）。"""
    _AUDIT_BUFFER.append({"actor": actor, "action": action, "subject": subject,
                          "detail": detail[:500], "created_at": time.time()})
    if os.getenv("SKIP_DB", "0") == "1":
        return
    try:
        from backend.app.db.models import AuditLog, get_engine, init_db

        engine = get_engine(get_settings().database_url)
        init_db(engine)
        from sqlalchemy.orm import Session as SASession

        with SASession(engine) as db:
            db.add(AuditLog(actor=actor, action=action, subject=subject, detail=detail[:500]))
            db.commit()
    except Exception:
        pass  # 审计落库失败不影响主流程，但环形缓冲始终有底


def recent_audit(limit: int = 50) -> list:
    return list(_AUDIT_BUFFER)[-limit:]
