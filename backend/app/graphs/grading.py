"""批改子图：客观题规则判分 + 主观题按评分要点 LLM 评分 → 低置信/申诉转教师复核。

判分纪律：
1. 客观题零模型：归一化精确匹配（全半角/空格/标点折叠），准确率上限即规则上限；
2. 主观题逐要点判分，输出必须命中 schema（score/confidence/matched_points/missed_points），
   校验失败自动纠错重试一次，仍失败按低置信转人工；
3. 置信度 < 0.75 或学生申诉 → 打包进教师复核队列（系统只建议，不终审）；
4. 成绩评语属最高风险级：本图只产出"建议稿"，签发权在教师。
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List

from langgraph.graph import END, START, StateGraph

from backend.app.config import get_settings
from backend.app.gateway.model_gateway import get_gateway
from backend.app.graphs.state import GradingState
from backend.app.safety.schema_validator import validate
from backend.prompts import grading as P

SUBJECTIVE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["score", "confidence", "reason"],
    "properties": {
        "score": {"type": "number", "minimum": 0},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string"},
        "matched_points": {"type": "array", "items": {"type": "string"}},
        "missed_points": {"type": "array", "items": {"type": "string"}},
    },
    "additionalProperties": False,
}

LOW_CONFIDENCE = 0.75


def _audit(node: str, decision: str, detail: Dict[str, Any], state: GradingState) -> Dict[str, Any]:
    trail = list(state.get("audit_trail", []))
    trail.append({"node": node, "actor_role": "system", "decision": decision,
                  "detail": detail, "ts": time.time()})
    return {"audit_trail": trail}


def _normalize(text: str) -> str:
    """判分用文本归一化：全角转半角、去空白与标点。"""
    table = {ord(c): ord(c) - 0xFEE0 for c in "０１２３４５６７８９ＡＢＣＤＥａｂｃｄｅ"
             if 0xFF01 <= ord(c) <= 0xFF5E}
    folded = (text or "").translate(table).lower()
    return re.sub(r"[\s，。、；：！？!?,.;:]+", "", folded)


def grade_objective(state: GradingState) -> Dict[str, Any]:
    """客观题规则判分。"""
    results: List[Dict[str, Any]] = []
    for item in state.get("items", []):
        if item.get("type") != "objective":
            continue
        student, ref = _normalize(item.get("student_answer", "")), _normalize(item.get("ref_answer", ""))
        full = float(item.get("full_score", 1))
        correct = bool(student) and student == ref
        results.append({"qid": item.get("qid", ""), "correct": correct,
                        "score": full if correct else 0.0, "method": "rule_match"})
    return _audit("grade_objective", "done", {"count": len(results)},
                  state) | {"objective_results": results}


def _mock_subjective(item: Dict[str, Any]) -> Dict[str, Any]:
    """离线确定性主观题评分：按评分要点关键词覆盖率给分。"""
    answer = _normalize(item.get("student_answer", ""))
    points = item.get("points", [])
    full = float(item.get("full_score", 10))
    if not points:
        matched_ratio = 1.0 if answer == _normalize(item.get("ref_answer", "")) else 0.0
        return {"score": round(full * matched_ratio, 1),
                "confidence": 0.9 if matched_ratio in (0.0, 1.0) else 0.6,
                "reason": "按参考答案比对", "matched_points": [], "missed_points": []}
    matched, missed = [], []
    for i, point in enumerate(points, 1):
        keywords = [w for w in re.split(r"[，,、；;（）()\s]+", point) if len(w) >= 2]
        hit = any(_normalize(k) in answer for k in keywords)
        if not hit:
            # 字符重合度兜底：近似"要点语义被学生表述覆盖"
            point_chars = set(_normalize(point))
            if point_chars and len(point_chars & set(answer)) / len(point_chars) >= 0.4:
                hit = True
        (matched if hit else missed).append(f"要点{i}")
    ratio = len(matched) / len(points)
    score = round(full * ratio, 1)
    # 边界情形（部分满足）刻意压低置信，逼出教师复核路径
    confidence = 0.95 if ratio in (0.0, 1.0) else 0.6
    return {"score": score, "confidence": confidence,
            "reason": f"满足 {len(matched)}/{len(points)} 个评分要点",
            "matched_points": matched, "missed_points": missed}


def grade_subjective(state: GradingState) -> Dict[str, Any]:
    """主观题按评分要点 LLM 评分（JSON schema 强校验）。"""
    results: List[Dict[str, Any]] = []
    gateway = get_gateway()
    for item in state.get("items", []):
        if item.get("type") != "subjective":
            continue
        points_block = "\n".join(f"- 要点{i}: {p}" for i, p in enumerate(item.get("points", []), 1))
        prompt = P.SUBJECTIVE_TEMPLATE.format(
            stem=item.get("stem", ""), full_score=item.get("full_score", 10),
            points_block=points_block or "（无要点，按参考答案比对）",
            student_answer=item.get("student_answer", ""))
        messages = [{"role": "system", "content": P.SYSTEM}, {"role": "user", "content": prompt}]
        try:
            data = gateway.chat_json(messages, scene="grading",
                                     mock_example=_mock_subjective(item))
        except Exception:
            data = {"score": 0.0, "confidence": 0.0, "reason": "模型输出异常，转人工",
                    "matched_points": [], "missed_points": []}
        errors = validate(data, SUBJECTIVE_SCHEMA)
        if errors:
            data = {"score": 0.0, "confidence": 0.0,
                    "reason": f"输出结构非法: {errors[0]}", "matched_points": [],
                    "missed_points": []}
        full = float(item.get("full_score", 10))
        data["score"] = max(0.0, min(full, float(data["score"])))
        data["qid"] = item.get("qid", "")
        results.append(data)
    return _audit("grade_subjective", "done", {"count": len(results)},
                  state) | {"subjective_results": results}


def triage(state: GradingState) -> Dict[str, Any]:
    """分流：低置信与申诉题转教师复核队列。"""
    low = [r["qid"] for r in state.get("subjective_results", [])
           if float(r.get("confidence", 1.0)) < LOW_CONFIDENCE]
    appeals = [item.get("qid", "") for item in state.get("items", []) if item.get("appeal")]
    need = sorted(set(low) | set(appeals))
    if need:
        from backend.app.tools.academic_affairs import run as academic_run

        academic_run("push_review_task", task={
            "kind": "grading_review", "student_ref": state.get("student_ref", ""),
            "qids": need,
            "payload": json.dumps({"objective": state.get("objective_results", []),
                                   "subjective": state.get("subjective_results", [])},
                                  ensure_ascii=False)[:2000]})
    status = "need_teacher_review" if need else "auto_done"
    return _audit("triage", status, {"pending": need}, state) | {
        "low_confidence_items": need, "review_status": status}


def draft_comment(state: GradingState) -> Dict[str, Any]:
    """成绩评语建议稿（永远需教师签发，risk_policy 层面强制）。"""
    obj = state.get("objective_results", [])
    subj = state.get("subjective_results", [])
    total_correct = sum(1 for r in obj if r.get("correct"))
    mock_text = (f"客观题 {total_correct}/{len(obj)} 正确；主观题平均得分率 "
                 f"{round(sum(r.get('score', 0) for r in subj) / max(len(subj), 1) * 10):.0f} 分档。"
                 "建议关注错题对应知识点的变式练习。")
    result = get_gateway().chat(
        [{"role": "system", "content": P.SYSTEM},
         {"role": "user", "content": "请为本次作业写一段面向家长的评语建议稿（不超过80字）。"}],
        scene="grading", mock_example=mock_text)
    return _audit("draft_comment", "suggestion_only", {"signed": False}, state) | {
        "comment": result.text}


def build_grading_graph():
    """编译批改子图。"""
    builder = StateGraph(GradingState)
    builder.add_node("grade_objective", grade_objective)
    builder.add_node("grade_subjective", grade_subjective)
    builder.add_node("triage", triage)
    builder.add_node("draft_comment", draft_comment)
    builder.add_edge(START, "grade_objective")
    builder.add_edge("grade_objective", "grade_subjective")
    builder.add_edge("grade_subjective", "triage")
    builder.add_edge("triage", "draft_comment")
    builder.add_edge("draft_comment", END)
    return builder.compile()
