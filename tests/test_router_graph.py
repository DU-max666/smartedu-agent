"""路由图测试：角色白名单收敛、派发闸值、澄清与转复核。"""
from __future__ import annotations

from backend.app.graphs.router_graph import ROLE_AGENT_WHITELIST, IntentClassifier, get_router_graph


def test_whitelist_matrix():
    """权限矩阵：学生无备课/学情，家长无批改。"""
    assert "lesson" not in ROLE_AGENT_WHITELIST["student"]
    assert "profile" not in ROLE_AGENT_WHITELIST["student"]
    assert "grading" not in ROLE_AGENT_WHITELIST["parent"]
    assert set(ROLE_AGENT_WHITELIST["teacher"]) == {"lesson", "qa", "grading", "profile"}


def test_intent_classifier_high_confidence():
    clf = IntentClassifier()
    out = clf.classify("帮我批改作业打分", ["grading", "qa"])
    assert out["intent"] == "grading"
    assert out["confidence"] >= 0.92


def test_intent_classifier_low_confidence_for_vague():
    clf = IntentClassifier()
    out = clf.classify("你好呀", ["qa", "grading"])
    assert out["confidence"] < 0.92


def test_dispatch_when_confident():
    g = get_router_graph()
    final = g.run({"session_id": "t1", "user_role": "teacher",
                   "utterance": "帮我批改作业打分",
                   "result_input": {"items": []}})
    assert final.get("dispatch_target") == "grading"
    assert final.get("escalate_review") is not True


def test_student_cannot_reach_lesson():
    """越权意图：白名单外不派发，走澄清/复核。"""
    g = get_router_graph()
    final = g.run({"session_id": "t2", "user_role": "student",
                   "utterance": "帮我组一份单元测试卷并出题"})
    assert final.get("dispatch_target") in (None, "")
    assert final.get("clarify_question")


def test_clarify_then_escalate():
    """澄清轮数耗尽后必须转教师复核，绝不无限循环。"""
    g = get_router_graph()
    final = g.run({"session_id": "t3", "user_role": "student",
                   "utterance": "你好", "clarify_rounds": 2})
    assert final.get("escalate_review") is True
    assert final.get("result", {}).get("review_task_id")


def test_audit_trail_recorded():
    g = get_router_graph()
    final = g.run({"session_id": "t4", "user_role": "teacher",
                   "utterance": "帮我批改作业打分", "result_input": {"items": []}})
    nodes = [a["node"] for a in final.get("audit_trail", [])]
    assert "narrow_by_role" in nodes and "discern_intent" in nodes
