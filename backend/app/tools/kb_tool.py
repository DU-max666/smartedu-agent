"""知识库检索工具：把混合检索封装为 Function Calling 工具。"""
from __future__ import annotations

from typing import Any, Dict, List

TOOL_SPEC: Dict[str, Any] = {
    "name": "search_knowledge_base",
    "description": "在慧学语料库（教材/课标/题库/对话）中做混合检索，"
                   "返回带页码与题号引用的证据片段。",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "检索查询语句"},
            "grade": {"type": "string", "description": "学段年级过滤，如 小学三年级"},
            "subject": {"type": "string", "description": "学科过滤，如 数学"},
            "source_type": {
                "type": "string", "enum": ["textbook", "curriculum", "qa_bank", "dialogue", ""],
                "description": "语料来源类型，空串表示不限",
            },
            "top_k": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5},
        },
        "required": ["query"],
        "additionalProperties": False,
    },
}


def run(query: str, grade: str = "", subject: str = "",
        source_type: str = "", top_k: int = 5) -> Dict[str, Any]:
    """执行检索并序列化为工具返回结构。"""
    from backend.app.retrieval.hybrid_search import get_retriever

    filters = {k: v for k, v in
               {"grade": grade, "subject": subject, "source_type": source_type}.items() if v}
    evidence = get_retriever().search(query, top_k=top_k, filters=filters or None)
    items: List[Dict[str, Any]] = []
    for item in evidence:
        entry = {
            "doc_id": item.get("doc_id", ""),
            "chunk_id": item.get("chunk_id", ""),
            "text": item.get("parent_text") or item.get("text", ""),
            "page": item.get("page", ""),
            "question_no": item.get("question_no", ""),
            "source_type": item.get("source_type", "textbook"),
            "score": round(float(item.get("score", 0.0)), 4),
        }
        items.append(entry)
    return {"hits": items, "count": len(items)}
