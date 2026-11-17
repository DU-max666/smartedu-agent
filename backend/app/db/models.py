"""SQLAlchemy 业务模型：学校 / 用户 / 班级 / 任务 / 产物版本 / 审计。

设计取舍：
- 学生身份只存脱敏代号（student_ref），姓名列一律不设——从 schema 层面
  杜绝明文身份落库；
- 产物（备课稿/批改结果/学情报告）走 ArtifactVersion 版本链，
  教师复核针对具体版本签发，历史版本可回溯；
- AuditLog 记录路由派发与复核动作，只追加不修改。
"""
from __future__ import annotations

import time
from typing import Optional

from sqlalchemy import (
    Float, ForeignKey, Integer, String, Text, create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


class Base(DeclarativeBase):
    """ORM 声明基类。"""


class School(Base):
    """接入慧学平台的学校（区域教育平台成员校）。"""

    __tablename__ = "schools"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128))
    region_code: Mapped[str] = mapped_column(String(32), index=True)


class User(Base):
    """平台用户；学生仅以脱敏代号入库。"""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    school_id: Mapped[int] = mapped_column(ForeignKey("schools.id"), index=True)
    role: Mapped[str] = mapped_column(String(16))          # teacher / student / parent
    display_name: Mapped[str] = mapped_column(String(64), default="")  # 教师可用真名，学生留空
    student_ref: Mapped[Optional[str]] = mapped_column(String(32), unique=True, nullable=True)
    class_id: Mapped[Optional[int]] = mapped_column(ForeignKey("classes.id"), nullable=True)


class Clazz(Base):
    """班级。"""

    __tablename__ = "classes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    school_id: Mapped[int] = mapped_column(ForeignKey("schools.id"), index=True)
    grade: Mapped[str] = mapped_column(String(32))
    label: Mapped[str] = mapped_column(String(32))


class AgentTask(Base):
    """一次智能体任务（备课/答疑/批改/学情）。"""

    __tablename__ = "agent_tasks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    agent: Mapped[str] = mapped_column(String(16))          # lesson/qa/grading/profile
    status: Mapped[str] = mapped_column(String(16), default="pending")
    input_digest: Mapped[str] = mapped_column(String(64), default="")  # 只存输入哈希，不存原文
    created_at: Mapped[float] = mapped_column(Float, default=lambda: time.time())


class ArtifactVersion(Base):
    """产物版本链：教师复核签发绑定到具体版本。"""

    __tablename__ = "artifact_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    task_id: Mapped[int] = mapped_column(ForeignKey("agent_tasks.id"), index=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    content_json: Mapped[str] = mapped_column(Text)
    review_state: Mapped[str] = mapped_column(String(16), default="draft")  # draft/reviewed/rejected
    reviewer_id: Mapped[Optional[int]] = mapped_column(ForeignKey("users.id"), nullable=True)
    created_at: Mapped[float] = mapped_column(Float, default=lambda: time.time())


class AuditLog(Base):
    """审计日志：路由派发、复核动作、缓存写回。"""

    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    actor: Mapped[str] = mapped_column(String(64))          # user 标识或 system
    action: Mapped[str] = mapped_column(String(32))
    subject: Mapped[str] = mapped_column(String(128), default="")
    detail: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[float] = mapped_column(Float, default=lambda: time.time())


def get_engine(database_url: str):
    """按配置创建引擎（SQLite 下自动允许多线程访问）。"""
    connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
    return create_engine(database_url, connect_args=connect_args, future=True)


def make_session_factory(engine) -> sessionmaker:
    return sessionmaker(bind=engine, future=True)


def init_db(engine) -> None:
    Base.metadata.create_all(engine)
