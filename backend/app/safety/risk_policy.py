"""内容风险分级与复核策略：教师草稿 / 学生可见 / 成绩评语三级。

分级语义：
- TEACHER_DRAFT：仅教师可见的工作草稿，允许不确定性存在，但禁止缓存下发；
- STUDENT_VISIBLE：面向学生/家长的答案，必须通过引用回查、一致性校验、
  敏感词三项检查，任一失败自动降级为教师草稿并进入复核队列；
- GRADE_COMMENT：成绩类评语，属最高风险级，机器只能产出"建议稿"，
  签发权永远在教师，系统层面不存在自动下发路径。
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List


class RiskLevel(str, Enum):
    """三级风险分级。"""

    TEACHER_DRAFT = "teacher_draft"
    STUDENT_VISIBLE = "student_visible"
    GRADE_COMMENT = "grade_comment"


@dataclass
class Decision:
    """策略判定结果。"""

    allowed: bool
    downgrade_to: RiskLevel | None
    need_teacher_review: bool
    reasons: List[str]


# 每一级放行所需通过的检查项名称
_REQUIRED_CHECKS: Dict[RiskLevel, List[str]] = {
    RiskLevel.TEACHER_DRAFT: ["pii_ok"],
    RiskLevel.STUDENT_VISIBLE: ["pii_ok", "citation_ok", "consistency_ok", "sensitive_ok"],
    RiskLevel.GRADE_COMMENT: ["pii_ok", "sensitive_ok"],
}

# 检查失败后的降级目标（成绩评语失败直接卡住，不降级不展示）
_DOWNGRADE: Dict[RiskLevel, RiskLevel | None] = {
    RiskLevel.TEACHER_DRAFT: None,
    RiskLevel.STUDENT_VISIBLE: RiskLevel.TEACHER_DRAFT,
    RiskLevel.GRADE_COMMENT: None,
}


def classify(content_type: str) -> RiskLevel:
    """按内容类型归级。"""
    mapping = {
        "lesson_draft": RiskLevel.TEACHER_DRAFT,
        "qa_answer": RiskLevel.STUDENT_VISIBLE,
        "grading_result": RiskLevel.STUDENT_VISIBLE,
        "profile_report": RiskLevel.TEACHER_DRAFT,
        "score_comment": RiskLevel.GRADE_COMMENT,
    }
    return mapping.get(content_type, RiskLevel.TEACHER_DRAFT)


def enforce(level: RiskLevel, checks: Dict[str, bool],
            teacher_signed: bool = False) -> Decision:
    """执行分级复核策略。

    checks: 检查项名 -> 是否通过；缺项按未通过处理（fail-closed）。
    teacher_signed: 仅对 GRADE_COMMENT 有意义（教师签发标记）。
    """
    reasons: List[str] = []
    for name in _REQUIRED_CHECKS[level]:
        if not checks.get(name, False):
            reasons.append(f"检查项 {name} 未通过")

    if level == RiskLevel.GRADE_COMMENT and not teacher_signed:
        reasons.append("成绩评语未经教师签发，禁止下发")

    if not reasons:
        return Decision(allowed=True, downgrade_to=None,
                        need_teacher_review=False, reasons=[])

    downgrade = _DOWNGRADE[level]
    # 学生可见内容失败后降级为草稿，同时必然进入复核队列
    need_review = level in (RiskLevel.STUDENT_VISIBLE, RiskLevel.GRADE_COMMENT)
    return Decision(allowed=False, downgrade_to=downgrade,
                    need_teacher_review=need_review, reasons=reasons)
