"""备课子图：课标检索 → 教学大纲 → 课件内容 → 组卷 → 教师草稿打包。

产物定位永远是"教师草稿"（risk_level=draft）：系统只加速备课初稿，
教学决策权在教师；打包节点生成 ArtifactVersion 记录供复核签发。
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Dict, List

from langgraph.graph import END, START, StateGraph

from backend.app.config import get_settings
from backend.app.gateway.model_gateway import get_gateway
from backend.app.graphs.state import LessonState
from backend.app.tools import curriculum_tool, question_bank
from backend.prompts import lesson_prep as P


def _audit(node: str, decision: str, detail: Dict[str, Any], state: LessonState) -> Dict[str, Any]:
    trail = list(state.get("audit_trail", []))
    trail.append({"node": node, "actor_role": "teacher", "decision": decision,
                  "detail": detail, "ts": time.time()})
    return {"audit_trail": trail}


def fetch_curriculum(state: LessonState) -> Dict[str, Any]:
    """课标检索：优先语料库，回退内置要点。"""
    result = curriculum_tool.run(grade=state.get("grade", ""),
                                 subject=state.get("subject", ""),
                                 strand=state.get("topic", ""))
    points = result.get("points", [])
    return _audit("fetch_curriculum", "done", {"count": len(points),
                                               "source": result.get("source", "")},
                  state) | {"curriculum_points": points}


def draft_outline(state: LessonState) -> Dict[str, Any]:
    """教学大纲生成（JSON 输出 + schema 校验）。"""
    points_block = "\n".join(f"- {p}" for p in state.get("curriculum_points", []))
    prompt = P.OUTLINE_TEMPLATE.format(topic=state.get("topic", ""),
                                       grade=state.get("grade", ""),
                                       subject=state.get("subject", ""),
                                       curriculum_points=points_block or "（未检索到课标要点）",
                                       excerpts="（备课初稿阶段略）")
    mock_outline = [
        {"stage": "情境导入", "minutes": 5, "goal": f"以生活实例引出《{state.get('topic', '')}》",
         "evidence_ref": "课标要点1"},
        {"stage": "概念新授", "minutes": 15, "goal": "建立核心概念与进率关系",
         "evidence_ref": "课标要点2"},
        {"stage": "例题精讲", "minutes": 10, "goal": "示范规范解题步骤",
         "evidence_ref": "课标要点2"},
        {"stage": "随堂练习", "minutes": 8, "goal": "即时反馈纠偏", "evidence_ref": "题库取样"},
        {"stage": "小结与作业", "minutes": 2, "goal": "结构化回顾", "evidence_ref": "课标要点3"},
    ]
    data = get_gateway().chat_json(
        [{"role": "system", "content": P.SYSTEM}, {"role": "user", "content": prompt}],
        scene="lesson", mock_example={"outline": mock_outline})
    outline = data.get("outline", mock_outline)
    return _audit("draft_outline", "done", {"stages": len(outline)}, state) | {"outline": outline}


def draft_slides(state: LessonState) -> Dict[str, Any]:
    """逐页课件内容。"""
    prompt = P.SLIDES_TEMPLATE.format(outline_json=json.dumps(state.get("outline", []),
                                                              ensure_ascii=False))
    mock_slides = [{"page": i + 1, "title": stage.get("stage", f"第{i + 1}页"),
                    "bullets": [stage.get("goal", ""), f"围绕《{state.get('topic', '')}》展开"]}
                   for i, stage in enumerate(state.get("outline", []))]
    data = get_gateway().chat_json(
        [{"role": "system", "content": P.SYSTEM}, {"role": "user", "content": prompt}],
        scene="lesson", mock_example={"slides": mock_slides})
    return _audit("draft_slides", "done", {"pages": len(data.get("slides", []))},
                  state) | {"slides_content": data.get("slides", [])}


def compose_quiz(state: LessonState) -> Dict[str, Any]:
    """组卷：题库优先复用原题，数量不足时模型自创并强制附评分要点。"""
    bank = question_bank.run(subject=state.get("subject", ""),
                             grade=state.get("grade", ""),
                             topic=state.get("topic", ""), limit=5)
    quiz: List[Dict[str, Any]] = []
    for q in bank.get("questions", []):
        quiz.append({**q, "source": q["qid"]})
    if len(quiz) < 3:
        prompt = P.QUIZ_TEMPLATE.format(topic=state.get("topic", ""),
                                       bank_items=json.dumps(bank.get("questions", []),
                                                             ensure_ascii=False))
        data = get_gateway().chat_json(
            [{"role": "system", "content": P.SYSTEM}, {"role": "user", "content": prompt}],
            scene="lesson", mock_example={"quiz": []})
        for item in data.get("quiz", []):
            if item.get("points"):  # 无评分要点的自创题不予采纳（批改链路依赖要点）
                quiz.append({**item, "source": "self"})
    return _audit("compose_quiz", "done", {"count": len(quiz)}, state) | {"quiz_items": quiz}


def package_draft(state: LessonState) -> Dict[str, Any]:
    """打包教师草稿并入库版本链（draft 状态，等待教师复核）。"""
    payload = {
        "topic": state.get("topic", ""),
        "outline": state.get("outline", []),
        "slides": state.get("slides_content", []),
        "quiz": state.get("quiz_items", []),
    }
    content = json.dumps(payload, ensure_ascii=False)
    draft_id = hashlib.sha1(content.encode("utf-8")).hexdigest()[:16]
    _persist_artifact(state, draft_id, content)
    return _audit("package_draft", "drafted", {"draft_id": draft_id}, state) | {
        "draft_id": draft_id, "draft_status": "draft"}


def _persist_artifact(state: LessonState, draft_id: str, content: str) -> None:
    """有库写库、无库跳过：备课产物落 ArtifactVersion。"""
    import os

    if os.getenv("SKIP_DB", "0") == "1":
        return
    try:
        from sqlalchemy.orm import Session as SASession

        from backend.app.db.models import ArtifactVersion, get_engine, init_db
        from backend.app.config import get_settings as gs

        engine = get_engine(gs().database_url)
        init_db(engine)
        with SASession(engine) as db:
            db.add(ArtifactVersion(task_id=0, version=1, content_json=content,
                                   review_state="draft"))
            db.commit()
    except Exception:
        # 演示/离线环境允许无库运行，产物仍在响应体内返回
        pass


def build_lesson_graph():
    """编译备课子图。"""
    builder = StateGraph(LessonState)
    builder.add_node("fetch_curriculum", fetch_curriculum)
    builder.add_node("draft_outline", draft_outline)
    builder.add_node("draft_slides", draft_slides)
    builder.add_node("compose_quiz", compose_quiz)
    builder.add_node("package_draft", package_draft)
    builder.add_edge(START, "fetch_curriculum")
    builder.add_edge("fetch_curriculum", "draft_outline")
    builder.add_edge("draft_outline", "draft_slides")
    builder.add_edge("draft_slides", "compose_quiz")
    builder.add_edge("compose_quiz", "package_draft")
    builder.add_edge("package_draft", END)
    return builder.compile()
