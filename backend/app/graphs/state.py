"""共享状态定义（LangGraph TypedDict 状态）。

约定：
- 所有状态字段 total=False，节点只写增量，LangGraph 自动合并；
- ``messages`` 一律为 {"role","content"} 字典列表，避免绑定特定消息类；
- 审计信息（谁、以什么角色、走了哪条派发路径）统一放 ``audit_trail``。
"""
from __future__ import annotations

from typing import Any, Dict, List, TypedDict


class EvidenceItem(TypedDict, total=False):
    """一条检索证据：正文 + 可回查的引用坐标。"""

    doc_id: str          # 语料文档 ID
    chunk_id: str        # 切片 ID（子块）
    parent_id: str       # 父块 ID（供上下文扩展）
    text: str            # 证据正文
    page: str            # 教材页码，如 "第12页"
    question_no: str     # 题号（题库证据时）
    score: float         # 融合重排后得分
    source_type: str     # textbook / curriculum / qa_bank / dialogue


class AuditEntry(TypedDict, total=False):
    """审计链单条记录。"""

    node: str
    actor_role: str
    decision: str
    detail: Dict[str, Any]
    ts: float


class RouterState(TypedDict, total=False):
    """顶层路由图状态。"""

    session_id: str
    user_role: str                                  # teacher / student / parent
    utterance: str                                  # 本轮用户输入
    allowed_agents: List[str]                       # 角色收口后的候选 Agent 白名单
    intent: str                                     # lesson / qa / grading / profile / chitchat
    intent_confidence: float
    clarify_rounds: int                             # 已进行的澄清轮数
    clarify_question: str                           # 澄清话术（返回给用户）
    escalate_review: bool                           # 是否转教师复核
    dispatch_target: str                            # 最终派发的子图名
    result_input: Dict[str, Any]                    # 子图业务入参（题目列表/课题等结构化输入）
    audit_trail: List[AuditEntry]
    result: Dict[str, Any]                          # 子图执行结果汇总


class LessonState(TypedDict, total=False):
    """备课子图状态。"""

    teacher_id: str
    subject: str
    grade: str                                      # 学段+年级，如 "小学三年级"
    topic: str                                      # 课题
    curriculum_points: List[str]                    # 课标要点（检索所得）
    outline: List[Dict[str, str]]                   # 教学大纲环节
    slides_content: List[Dict[str, str]]            # 课件页内容
    quiz_items: List[Dict[str, Any]]                # 组卷题目
    draft_id: str
    draft_status: str                               # draft / reviewed / rejected
    evidence: List[EvidenceItem]
    messages: List[Dict[str, str]]
    audit_trail: List[AuditEntry]


class QAState(TypedDict, total=False):
    """答疑子图状态。"""

    student_ref: str                                # 脱敏后的学生标识
    grade: str
    subject: str
    question: str
    rewritten_query: str                            # 问题改写结果
    evidence: List[EvidenceItem]
    answer: str
    revise_attempts: int                            # 收紧重生成次数
    citation_valid: bool                            # 引用回查是否通过
    consistency_ok: bool                            # 答案与证据一致性校验
    refuse_reason: str                              # 非空表示拒答
    cached: bool                                    # 是否命中镜像缓存
    messages: List[Dict[str, str]]
    audit_trail: List[AuditEntry]


class GradingState(TypedDict, total=False):
    """批改子图状态。"""

    task_id: str
    student_ref: str
    grade: str
    subject: str
    items: List[Dict[str, Any]]                     # 题目列表（含题型/学生作答/参考答案/评分要点）
    objective_results: List[Dict[str, Any]]         # 客观题规则判分结果
    subjective_results: List[Dict[str, Any]]        # 主观题评分结果（含置信度）
    low_confidence_items: List[str]                 # 低置信题号 -> 转教师复核
    appeal_items: List[str]                         # 学生申诉题号 -> 转教师复核
    review_status: str                              # auto_done / need_teacher_review
    comment: str                                    # 成绩评语（三级风险：必须教师签发）
    messages: List[Dict[str, str]]
    audit_trail: List[AuditEntry]


class ProfileState(TypedDict, total=False):
    """学情子图状态。"""

    student_ref: str
    grade: str
    subject: str
    knowledge_tree: Dict[str, Any]                  # 知识点树（嵌套 dict）
    attempt_log: List[Dict[str, Any]]               # 作答记录（已脱敏）
    mastery: Dict[str, float]                       # 知识点 -> 掌握度 0~1
    lagging_nodes: List[str]                        # 掉队预警知识点
    attribution: str                                # LLM 归因说明
    report: str                                     # 面向教师/家长的学情报告
    messages: List[Dict[str, str]]
    audit_trail: List[AuditEntry]
