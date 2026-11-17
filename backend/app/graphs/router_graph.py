"""顶层路由图：角色白名单收口 → BERT 意图分类 → 派发闸值 → 子图执行。

原创流程设计（区别于"LLM 自由路由"的常见模板）：
1. 先收口后辨意 —— 按用户角色（教师/学生/家长）先收敛可用 Agent 白名单，
   意图分类只在白名单内进行，越权意图根本不会被派发；
2. 派发闸值 —— 置信度 ≥ 0.92（可配）才自动派发；否则最多两轮澄清话术，
   仍不明确则打包转教师复核队列，绝不"猜着路由"；
3. 全链路审计 —— 每个节点向 audit_trail 追加记录，路由决策可复盘。
"""
from __future__ import annotations

import time
from typing import Any, Dict, List

from langgraph.graph import END, START, StateGraph

from backend.app.config import get_settings
from backend.app.graphs.state import AuditEntry, RouterState

# 角色 -> 可用 Agent 白名单（权限矩阵的唯一事实来源）
ROLE_AGENT_WHITELIST: Dict[str, List[str]] = {
    "teacher": ["lesson", "qa", "grading", "profile"],
    "student": ["qa", "grading"],
    "parent": ["qa", "profile"],
}

_INTENT_LEXICON: Dict[str, List[str]] = {
    "lesson": ["备课", "课件", "教案", "组卷", "大纲", "教学设计", "出题", "练习卷"],
    "grading": ["批改", "打分", "判分", "对答案", "作业", "申诉", "得分", "错题复核"],
    "profile": ["学情", "掌握度", "掌握", "薄弱", "薄弱知识点", "掉队", "学习报告",
                "知识点分析", "学得不好", "偏科", "分析"],
    "qa": ["为什么", "怎么", "什么意思", "解答", "不懂", "请问", "帮我看看",
           "多少", "等于", "吗", "？", "?"],
}


class IntentClassifier:
    """意图分类器：联网模式调用 BERT 分类服务，离线模式用确定性词频评分。

    离线评分同时输出置信度：以"最高分占比 + 领先优势"近似分类器 softmax，
    保证单测可稳定复现高/低置信两种路径。
    """

    def __init__(self) -> None:
        self.settings = get_settings()

    def classify(self, text: str, allowed: List[str]) -> Dict[str, Any]:
        if not self.settings.mock_mode:
            try:
                return self._classify_remote(text, allowed)
            except Exception:
                pass  # 服务异常退回本地词频，路由不中断
        scores: Dict[str, float] = {intent: 0.0 for intent in _INTENT_LEXICON}
        scores["chitchat"] = 0.0
        for intent, words in _INTENT_LEXICON.items():
            for w in words:
                if w in text:
                    scores[intent] += 1.0 + 0.25 * len(w)  # 长关键词证据更强
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        top_intent, top_score = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        if top_score <= 0:
            return {"intent": "chitchat", "confidence": 0.5}
        ratio = top_score / (top_score + second + 1.0)
        margin_bonus = 0.12 if top_score - second >= 1.5 else 0.0
        confidence = min(0.985, 0.50 + 0.42 * ratio + margin_bonus)
        return {"intent": top_intent, "confidence": round(confidence, 4)}

    def _classify_remote(self, text: str, allowed: List[str]) -> Dict[str, Any]:
        """调用校内 BERT 意图服务（OpenAI 兼容网关之外的独立小模型端点）。"""
        import os

        import httpx

        url = os.getenv("INTENT_API_URL", "")
        if not url:
            raise RuntimeError("INTENT_API_URL 未配置")
        with httpx.Client(timeout=5) as client:
            resp = client.post(url, json={"text": text, "labels": allowed})
            resp.raise_for_status()
            data = resp.json()
        return {"intent": data["intent"], "confidence": float(data["confidence"])}


_CLARIFY_TEMPLATES = {
    "lesson": "您是想让我协助备课（生成教学大纲/课件/组卷）吗？",
    "grading": "您是想提交作业让我批改，还是对某道题的判分提出申诉？",
    "profile": "您是想查看某个学生近期的学情与薄弱知识点分析吗？",
    "qa": "您是想问一道具体的题目，还是想了解某个知识点的概念？",
    "chitchat": "我可以协助备课、答疑、批改作业和学情分析，请告诉我您具体想做什么。",
}


def _entry(node: str, state: RouterState, decision: str, detail: Dict[str, Any]) -> AuditEntry:
    return {"node": node, "actor_role": state.get("user_role", ""),
            "decision": decision, "detail": detail, "ts": time.time()}


class RouterGraph:
    """顶层路由图封装：编译一次，多次调用。"""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.classifier = IntentClassifier()
        self._compiled = self._build()

    # ---------- 节点 ----------
    def narrow_by_role(self, state: RouterState) -> Dict[str, Any]:
        """节点一：按角色收敛可用 Agent 白名单。"""
        role = state.get("user_role", "student")
        allowed = ROLE_AGENT_WHITELIST.get(role, ["qa"])
        return {"allowed_agents": allowed,
                "audit_trail": [_entry("narrow_by_role", state, "whitelist",
                                       {"role": role, "allowed": allowed})]}

    def discern_intent(self, state: RouterState) -> Dict[str, Any]:
        """节点二：白名单内意图分类。"""
        allowed = state.get("allowed_agents", [])
        result = self.classifier.classify(state.get("utterance", ""), allowed)
        intent = result["intent"] if result["intent"] in allowed else "out_of_scope"
        trail = list(state.get("audit_trail", []))
        trail.append(_entry("discern_intent", state, "classify",
                            {"intent": intent, "confidence": result["confidence"]}))
        return {"intent": intent, "intent_confidence": result["confidence"],
                "audit_trail": trail}

    def route_after_intent(self, state: RouterState) -> str:
        """条件边：派发闸值判断。"""
        intent = state.get("intent", "chitchat")
        confidence = state.get("intent_confidence", 0.0)
        allowed = state.get("allowed_agents", [])
        if intent in allowed and confidence >= self.settings.intent_threshold:
            return "dispatch"
        return "clarify_or_escalate"

    def dispatch(self, state: RouterState) -> Dict[str, Any]:
        """节点三：派发到子图执行。"""
        intent = state["intent"]
        graph = _get_subgraph(intent)
        payload = _project_state(intent, state)
        final = graph.invoke(payload)
        trail = list(state.get("audit_trail", []))
        trail.append(_entry("dispatch", state, "run_subgraph",
                            {"target": intent, "input_keys": sorted(payload.keys())}))
        return {"dispatch_target": intent, "result": _extract_result(intent, final),
                "audit_trail": trail}

    def clarify_or_escalate(self, state: RouterState) -> Dict[str, Any]:
        """节点四：低置信/越权 -> 澄清话术（≤N 轮）或转教师复核。"""
        rounds = state.get("clarify_rounds", 0) + 1
        intent = state.get("intent", "chitchat")
        trail = list(state.get("audit_trail", []))
        if rounds <= self.settings.max_clarify_rounds:
            question = _CLARIFY_TEMPLATES.get(intent, _CLARIFY_TEMPLATES["chitchat"])
            trail.append(_entry("clarify_or_escalate", state, "clarify",
                                {"round": rounds}))
            return {"clarify_rounds": rounds, "clarify_question": question,
                    "escalate_review": False, "audit_trail": trail}
        # 澄清耗尽：打包上下文转教师复核队列
        from backend.app.tools.academic_affairs import run as academic_run

        task = {"kind": "router_escalation", "utterance": state.get("utterance", ""),
                "session_id": state.get("session_id", ""), "intent_guess": intent}
        pushed = academic_run("push_review_task", task=task)
        trail.append(_entry("clarify_or_escalate", state, "escalate", pushed))
        return {"clarify_rounds": rounds, "escalate_review": True,
                "clarify_question": "这个问题我先记下来，转给老师确认后回复您。",
                "result": {"escalated": True, "review_task_id": pushed.get("task_id", "")},
                "audit_trail": trail}

    # ---------- 构图 ----------
    def _build(self):
        builder = StateGraph(RouterState)
        builder.add_node("narrow_by_role", self.narrow_by_role)
        builder.add_node("discern_intent", self.discern_intent)
        builder.add_node("dispatch", self.dispatch)
        builder.add_node("clarify_or_escalate", self.clarify_or_escalate)
        builder.add_edge(START, "narrow_by_role")
        builder.add_edge("narrow_by_role", "discern_intent")
        builder.add_conditional_edges(
            "discern_intent", self.route_after_intent,
            {"dispatch": "dispatch", "clarify_or_escalate": "clarify_or_escalate"})
        builder.add_edge("dispatch", END)
        builder.add_edge("clarify_or_escalate", END)
        return builder.compile()

    def run(self, state: RouterState) -> RouterState:
        return self._compiled.invoke(state)  # type: ignore[return-value]

    @property
    def compiled(self):
        return self._compiled


_subgraph_cache: Dict[str, Any] = {}


def _get_subgraph(intent: str):
    """懒加载子图，避免循环导入与冷启动开销。"""
    if intent not in _subgraph_cache:
        if intent == "lesson":
            from backend.app.graphs.lesson_prep import build_lesson_graph

            _subgraph_cache[intent] = build_lesson_graph()
        elif intent == "qa":
            from backend.app.graphs.qa import build_qa_graph

            _subgraph_cache[intent] = build_qa_graph()
        elif intent == "grading":
            from backend.app.graphs.grading import build_grading_graph

            _subgraph_cache[intent] = build_grading_graph()
        elif intent == "profile":
            from backend.app.graphs.profile import build_profile_graph

            _subgraph_cache[intent] = build_profile_graph()
        else:
            raise KeyError(f"没有名为 {intent} 的子图")
    return _subgraph_cache[intent]


_STATE_PROJECTION: Dict[str, List[str]] = {
    "lesson": ["subject", "grade", "topic", "teacher_id"],
    "qa": ["question", "grade", "subject", "student_ref", "session_id"],
    "grading": ["items", "student_ref", "grade", "subject", "task_id"],
    "profile": ["student_ref", "grade", "subject", "knowledge_tree"],
}


def _project_state(intent: str, state: RouterState) -> Dict[str, Any]:
    """从路由状态裁剪出子图输入。"""
    payload: Dict[str, Any] = {}
    source = state.get("result_input") or {}
    for key in _STATE_PROJECTION.get(intent, []):
        if key in source:
            payload[key] = source[key]
    if intent == "qa":
        payload.setdefault("question", state.get("utterance", ""))
    return payload


def _extract_result(intent: str, final: Dict[str, Any]) -> Dict[str, Any]:
    """摘取子图终态中的业务结果（不带内部大字段）。"""
    keep = {
        "lesson": ["outline", "slides_content", "quiz_items", "draft_id", "draft_status"],
        "qa": ["answer", "citation_valid", "consistency_ok", "refuse_reason", "cached"],
        "grading": ["objective_results", "subjective_results", "low_confidence_items",
                     "review_status", "comment"],
        "profile": ["mastery", "lagging_nodes", "attribution", "report"],
    }
    return {k: final.get(k) for k in keep.get(intent, []) if k in final}


_router_singleton: RouterGraph | None = None


def get_router_graph() -> RouterGraph:
    global _router_singleton
    if _router_singleton is None:
        _router_singleton = RouterGraph()
    return _router_singleton
