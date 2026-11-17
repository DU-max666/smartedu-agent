"""答疑子图：镜像缓存查询 → 问题改写 → 混合检索 → 带引用生成 → 一致性校验。

质量闸门（两道）：
1. 引用回查：答案中的每个【来源】必须能对上本轮证据坐标，对不上即判幻觉；
2. 一致性校验：mock 模式用"答案词对证据的词面覆盖度"做确定性校验，
   联网模式由模型复核"结论是否可由证据推出"。
任一闸门失败：先做一次"收紧重生成"，仍失败则拒答并转教师复核，
绝不下发无据答案。
"""
from __future__ import annotations

import re
import time
from typing import Any, Dict, List

from langgraph.graph import END, START, StateGraph

from backend.app.config import get_settings
from backend.app.graphs.state import QAState
from backend.app.memory.session import get_session_memory
from backend.app.gateway.model_gateway import get_gateway
from backend.app.retrieval.hybrid_search import get_retriever
from backend.app.safety.citation_checker import check_answer

from backend.prompts import qa as P


def _audit(node: str, decision: str, detail: Dict[str, Any], state: QAState) -> Dict[str, Any]:
    trail = list(state.get("audit_trail", []))
    trail.append({"node": node, "actor_role": "system", "decision": decision,
                  "detail": detail, "ts": time.time()})
    return {"audit_trail": trail}


def _evidence_block(evidence: List[Dict[str, Any]]) -> str:
    """把证据列表渲染为带引用坐标的提示词块。"""
    lines = []
    for i, item in enumerate(evidence, 1):
        body = item.get("parent_text") or item.get("text", "")
        coord = f"【来源:{item.get('doc_id', '')} | {item.get('page', '')} | {item.get('question_no', '')}】".replace(" | 】", "】")
        lines.append(f"证据{i} {coord}\n{body}")
    return "\n---\n".join(lines)


def cache_lookup(state: QAState) -> Dict[str, Any]:
    """镜像缓存查询：只可能命中"教师定稿或高相似再验证"的答案。"""
    from backend.app.db.cache import get_mirror_cache

    hit = get_mirror_cache().lookup(state.get("question", ""),
                                    grade=state.get("grade", ""),
                                    subject=state.get("subject", ""))
    if hit:
        entry, sim = hit
        out = _audit("cache_lookup", "hit", {"sim": round(sim, 4)}, state)
        out.update({"answer": entry.answer, "cached": True,
                    "citation_valid": True, "consistency_ok": True})
        return out
    return _audit("cache_lookup", "miss", {}, state)


def route_after_cache(state: QAState) -> str:
    return "finalize" if state.get("cached") else "rewrite"


def rewrite_query(state: QAState) -> Dict[str, Any]:
    """口语问题 -> 规范化检索查询（携带会话历史做指代消解）。"""
    memory = get_session_memory()
    history = memory.context_block(state.get("session_id", "")) if state.get("session_id") else ""
    prompt = P.REWRITE_TEMPLATE.format(history=history or "（无历史）",
                                       question=state.get("question", ""))
    gateway = get_gateway()
    # mock 模式下用确定性去口语化结果作为注入示例
    rewritten = re.sub(r"^(老师|请问|帮我看看)[，,：:]?\s*", "", state.get("question", "")).strip()
    result = gateway.chat([{"role": "system", "content": P.SYSTEM},
                           {"role": "user", "content": prompt}], scene="qa",
                          mock_example=rewritten)
    return _audit("rewrite_query", "done", {"rewritten": result.text[:60]},
                  state) | {"rewritten_query": result.text.strip(), "cached": False}


def retrieve(state: QAState) -> Dict[str, Any]:
    """混合检索：学段×学科元数据过滤。"""
    filters = {k: v for k, v in {"grade": state.get("grade", ""),
                                 "subject": state.get("subject", "")}.items() if v}
    evidence = get_retriever().search(state.get("rewritten_query") or state.get("question", ""),
                                      top_k=5, filters=filters or None)
    return _audit("retrieve", "done", {"hits": len(evidence)}, state) | {"evidence": evidence}


def generate(state: QAState) -> Dict[str, Any]:
    """带教材页码/题号引用的答案生成。"""
    evidence = state.get("evidence", [])
    if not evidence:
        out = _audit("generate", "refuse_no_evidence", {}, state)
        out.update({"refuse_reason": "知识库中没有找到相关教材内容",
                    "answer": "这个问题老师还需要查一下，已转给你的任课老师。"})
        return out
    prompt = P.ANSWER_TEMPLATE.format(
        rewritten_query=state.get("rewritten_query", ""),
        grade=state.get("grade", "") or "未指定", subject=state.get("subject", "") or "未指定",
        evidence_block=_evidence_block(evidence),
    )
    gateway = get_gateway()
    messages = [{"role": "system", "content": P.SYSTEM}, {"role": "user", "content": prompt}]
    if get_settings().mock_mode:
        first = evidence[0]
        mock = (f"依据教材内容：{first.get('text', '')[:80]}……"
                f"【来源:{first.get('doc_id', '')} | {first.get('page', '')}】")
        result = gateway.chat(messages, scene="qa", mock_example=mock)
    else:
        result = gateway.chat(messages, scene="qa")
    return _audit("generate", "done", {"len": len(result.text)}, state) | {"answer": result.text}


def _lexical_consistency(answer: str, evidence: List[Dict[str, Any]]) -> bool:
    """离线确定性一致性校验：答案中的数字/单位结论词需有证据支撑。"""
    evidence_text = "".join(e.get("parent_text") or e.get("text", "") for e in evidence)
    claims = re.findall(r"\d+(?:\.\d+)?(?:千米|米|分米|厘米|毫米|克|千克|吨|平方|cm|km|kg|g|m)", answer)
    if not claims:
        return True
    return all(c in evidence_text for c in claims)


def verify(state: QAState) -> Dict[str, Any]:
    """引用回查 + 一致性校验双闸门。"""
    answer = state.get("answer", "")
    evidence = state.get("evidence", [])
    citation = check_answer(answer, evidence, require_citation=True)
    if get_settings().mock_mode:
        consistent = _lexical_consistency(answer, evidence)
    else:
        prompt = P.CONSISTENCY_TEMPLATE.format(evidence_block=_evidence_block(evidence),
                                               answer=answer)
        data = get_gateway().chat_json(
            [{"role": "system", "content": "你是严格的答案审核员。"},
             {"role": "user", "content": prompt}], scene="qa")
        consistent = bool(data.get("consistent", False))
    return _audit("verify", "pass" if (citation.ok and consistent) else "fail",
                  {"citation_issues": [i.kind for i in citation.issues],
                   "consistent": consistent}, state) | {
        "citation_valid": citation.ok, "consistency_ok": consistent}


def route_after_verify(state: QAState) -> str:
    if state.get("citation_valid") and state.get("consistency_ok"):
        return "finalize"
    return "revise_or_refuse"


def _refuse(state: QAState, attempts: int) -> Dict[str, Any]:
    out = _audit("revise_or_refuse", "refuse", {"attempts": attempts}, state)
    out.update({"refuse_reason": "答案未通过引用/一致性双闸门",
                "answer": "这个问题老师还需要查一下，已转给你的任课老师。",
                "citation_valid": False, "consistency_ok": False,
                "revise_attempts": attempts})
    return out


def revise_or_refuse(state: QAState) -> Dict[str, Any]:
    """一次收紧重生成；仍不合格则拒答转人工。"""
    attempts = state.get("revise_attempts", 0) + 1
    if attempts >= 2:
        return _refuse(state, attempts)
    evidence = state.get("evidence", [])
    if not evidence:
        return _refuse(state, attempts)
    strict_prompt = (P.ANSWER_TEMPLATE.format(
        rewritten_query=state.get("rewritten_query", ""),
        grade=state.get("grade", "") or "未指定",
        subject=state.get("subject", "") or "未指定",
        evidence_block=_evidence_block(evidence))
        + "\n注意：上一版答案存在无出处结论。只复述证据中的事实，每个结论句必须带引用。")
    first = evidence[0]
    mock_answer = (f"依据证据复述：{first.get('text', '')[:60]}"
                   f"【来源:{first.get('doc_id', '')} | {first.get('page', '')}】")
    result = get_gateway().chat(
        [{"role": "system", "content": P.SYSTEM}, {"role": "user", "content": strict_prompt}],
        scene="qa", mock_example=mock_answer)
    out = _audit("revise_or_refuse", "revise", {"attempts": attempts}, state)
    out.update({"answer": result.text, "revise_attempts": attempts})
    return out


def finalize(state: QAState) -> Dict[str, Any]:
    """出口节点：写会话记忆；定稿答案按镜像缓存纪律尝试写回。"""
    session_id = state.get("session_id", "")
    if session_id:
        get_session_memory().append(session_id, "assistant", state.get("answer", ""))
    return _audit("finalize", "done", {"refused": bool(state.get("refuse_reason"))}, state)


def build_qa_graph():
    """编译答疑子图。"""
    builder = StateGraph(QAState)
    builder.add_node("cache_lookup", cache_lookup)
    builder.add_node("rewrite_query", rewrite_query)
    builder.add_node("retrieve", retrieve)
    builder.add_node("generate", generate)
    builder.add_node("verify", verify)
    builder.add_node("revise_or_refuse", revise_or_refuse)
    builder.add_node("finalize", finalize)
    builder.add_edge(START, "cache_lookup")
    builder.add_conditional_edges("cache_lookup", route_after_cache,
                                  {"rewrite": "rewrite_query", "finalize": "finalize"})
    builder.add_edge("rewrite_query", "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", "verify")
    builder.add_conditional_edges("verify", route_after_verify,
                                  {"finalize": "finalize", "revise_or_refuse": "revise_or_refuse"})
    builder.add_edge("revise_or_refuse", "verify")
    builder.add_edge("finalize", END)
    return builder.compile()
