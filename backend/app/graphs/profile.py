"""学情子图：作答记录读取 → 知识树掌握度计算 → 掉队预警规则 → LLM 归因 → 报告。

掌握度模型（原创简化设计）：
- 叶子知识点掌握度 = 近期作答得分率的时间衰减加权平均（越近权重越大），
  父节点 = 子节点加权均值（权重为样本量）；
- 掉队预警完全由规则引擎触发（掌握度 < 0.45 且样本 ≥ 3，
  或近两次均分较前期下滑 > 0.2），LLM 只解释成因、不制造标签——
  保证预警可复现、可审计。
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List

from langgraph.graph import END, START, StateGraph

from backend.app.gateway.model_gateway import get_gateway
from backend.app.graphs.state import ProfileState
from backend.app.tools import academic_affairs
from backend.prompts import profile as P

DECAY = 0.85  # 越早的作业权重衰减系数


def _audit(node: str, decision: str, detail: Dict[str, Any], state: ProfileState) -> Dict[str, Any]:
    trail = list(state.get("audit_trail", []))
    trail.append({"node": node, "actor_role": "system", "decision": decision,
                  "detail": detail, "ts": time.time()})
    return {"audit_trail": trail}


def load_attempts(state: ProfileState) -> Dict[str, Any]:
    """读取脱敏作答记录。"""
    data = academic_affairs.run("get_attempts", student_ref=state.get("student_ref", ""),
                                subject=state.get("subject", ""))
    return _audit("load_attempts", "done", {"count": data.get("count", 0)}, state) | {
        "attempt_log": data.get("attempts", [])}


def _leaf_nodes(tree: Dict[str, Any]) -> List[str]:
    """知识树叶子名称（嵌套 dict 约定：值仍为 dict 则是分支）。"""
    leaves: List[str] = []
    for name, child in (tree or {}).items():
        if isinstance(child, dict) and child:
            leaves.extend(_leaf_nodes(child))
        else:
            leaves.append(name)
    return leaves


def compute_mastery(state: ProfileState) -> Dict[str, Any]:
    """按知识树计算掌握度（含父节点聚合）。"""
    attempts = state.get("attempt_log", [])
    leaves = _leaf_nodes(state.get("knowledge_tree", {})) or \
        sorted({a.get("topic", "") for a in attempts if a.get("topic")})
    mastery: Dict[str, float] = {}
    samples: Dict[str, int] = {}
    ordered = sorted(attempts, key=lambda a: a.get("hw", 0))
    n = len(ordered)
    for topic in leaves:
        rows = [(i, a) for i, a in enumerate(ordered) if a.get("topic") == topic]
        if not rows:
            continue
        weights = [DECAY ** (n - 1 - i) for i, _ in rows]
        total_w = sum(weights) or 1.0
        mastery[topic] = round(sum(w * float(a.get("score_ratio", 0.0))
                                   for w, (_, a) in zip(weights, rows)) / total_w, 3)
        samples[topic] = len(rows)
    # 父节点聚合
    def aggregate(tree: Dict[str, Any]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for name, child in (tree or {}).items():
            if isinstance(child, dict) and child:
                sub = aggregate(child)
                out.update(sub)
                vals = [v for k, v in sub.items() if k != name] or list(sub.values())
                out[name] = round(sum(vals) / len(vals), 3) if vals else 0.0
            elif name in mastery:
                out[name] = mastery[name]
        return out

    mastery = aggregate(state.get("knowledge_tree", {})) or mastery
    return _audit("compute_mastery", "done", {"nodes": len(mastery)}, state) | {
        "mastery": mastery}


def lagging_rules(state: ProfileState) -> Dict[str, Any]:
    """掉队预警规则引擎。"""
    mastery = state.get("mastery", {})
    attempts = state.get("attempt_log", [])
    lagging: List[str] = []
    for topic, score in mastery.items():
        samples = sum(1 for a in attempts if a.get("topic") == topic)
        if score < 0.45 and samples >= 3:
            lagging.append(topic)
            continue
        rows = sorted([a for a in attempts if a.get("topic") == topic], key=lambda a: a.get("hw", 0))
        if len(rows) >= 4:
            early = sum(a.get("score_ratio", 0) for a in rows[:2]) / 2
            late = sum(a.get("score_ratio", 0) for a in rows[-2:]) / 2
            if early - late > 0.2 and topic not in lagging:
                lagging.append(topic)
    return _audit("lagging_rules", "warn" if lagging else "clear",
                  {"lagging": lagging}, state) | {"lagging_nodes": lagging}


def attribute(state: ProfileState) -> Dict[str, Any]:
    """LLM 归因：只解释规则引擎给出的预警，不新增标签。"""
    lagging = state.get("lagging_nodes", [])
    if not lagging:
        return _audit("attribute", "skip", {}, state) | {"attribution": "未发现掉队风险知识点。"}
    mastery = state.get("mastery", {})
    trend = {t: mastery.get(t, 0.0) for t in lagging}
    snippet = json.dumps([a for a in state.get("attempt_log", [])
                          if a.get("topic") in lagging][:6], ensure_ascii=False)
    prompt = P.ATTRIBUTION_TEMPLATE.format(lagging_nodes="、".join(lagging),
                                           mastery_trend_json=json.dumps(trend, ensure_ascii=False),
                                           attempt_snippet=snippet)
    mock_example = {
        "attribution": f"{'、'.join(lagging)} 的连续低分集中在概念辨析环节，疑似对定义边界理解不清。",
        "suggestions": ["安排概念对比表练习", "错题归因讲评（学生互讲）", "一周后同知识点变式复测"],
    }
    data = get_gateway().chat_json(
        [{"role": "system", "content": P.SYSTEM}, {"role": "user", "content": prompt}],
        scene="profile", mock_example=mock_example)
    return _audit("attribute", "done", {"nodes": lagging}, state) | {
        "attribution": data.get("attribution", "")}


def render_report(state: ProfileState) -> Dict[str, Any]:
    """组装学情简报（教师草稿级，复核后方可下发）。"""
    prompt = P.REPORT_TEMPLATE.format(
        mastery_json=json.dumps(state.get("mastery", {}), ensure_ascii=False),
        lagging_nodes="、".join(state.get("lagging_nodes", [])) or "无",
        attribution=state.get("attribution", ""))
    mock_report = (f"现状：掌握度 {json.dumps(state.get('mastery', {}), ensure_ascii=False)}。\n"
                   f"风险：{'、'.join(state.get('lagging_nodes', [])) or '暂无预警'}。\n"
                   "建议：针对预警知识点安排变式练习并于一周后复测。")
    result = get_gateway().chat(
        [{"role": "system", "content": P.SYSTEM}, {"role": "user", "content": prompt}],
        scene="profile", mock_example=mock_report)
    return _audit("render_report", "draft", {"risk": "teacher_draft"}, state) | {"report": result.text}


def build_profile_graph():
    """编译学情子图。"""
    builder = StateGraph(ProfileState)
    builder.add_node("load_attempts", load_attempts)
    builder.add_node("compute_mastery", compute_mastery)
    builder.add_node("lagging_rules", lagging_rules)
    builder.add_node("attribute", attribute)
    builder.add_node("render_report", render_report)
    builder.add_edge(START, "load_attempts")
    builder.add_edge("load_attempts", "compute_mastery")
    builder.add_edge("compute_mastery", "lagging_rules")
    builder.add_edge("lagging_rules", "attribute")
    builder.add_edge("attribute", "render_report")
    builder.add_edge("render_report", END)
    return builder.compile()
