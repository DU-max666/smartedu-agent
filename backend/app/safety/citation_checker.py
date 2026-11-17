"""引用回查：校验生成文本中的引用坐标是否真实存在于检索证据。

引用格式约定（与提示词一致）：【来源:文档ID | 第12页 | 第3题】，
页码/题号段可省略。校验规则：
1. 文档 ID 必须出现在本轮证据集中；
2. 若标注了页码，需与该文档证据的 page 字段能对上（容忍"第12页"/"12"两种写法）；
3. 若标注了题号，需能对上证据的 question_no 字段；
4. 无任何引用的"事实性长答案"视为不合格（拒答或补检索由上层决定）。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from backend.app.graphs.state import EvidenceItem

_CITATION_RE = re.compile(r"【来源[:：]\s*([^】|]+?)((?:\s*\|\s*[^】]+?)*)】")


@dataclass
class CitationIssue:
    """一条引用问题。"""

    raw: str
    kind: str  # unknown_doc / page_mismatch / no_citation
    detail: str


@dataclass
class CitationReport:
    """回查报告。"""

    citations: List[str] = field(default_factory=list)
    issues: List[CitationIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues


def extract_citations(text: str) -> List[Dict[str, Optional[str]]]:
    """解析引用 -> [{doc_id, page, question_no}]。"""
    out: List[Dict[str, Optional[str]]] = []
    for match in _CITATION_RE.finditer(text):
        doc_id = match.group(1).strip()
        parts = [p.strip() for p in (match.group(2) or "").split("|") if p.strip()]
        page = question_no = None
        for part in parts:
            if re.search(r"页", part):
                page = part
            elif re.search(r"题", part):
                question_no = part
        out.append({"doc_id": doc_id, "page": page, "question_no": question_no})
    return out


def _norm_page(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    digits = re.sub(r"\D", "", raw)
    return digits or raw


def check_answer(text: str, evidence: List[EvidenceItem],
                 require_citation: bool = True) -> CitationReport:
    """对生成答案做引用回查。"""
    report = CitationReport()
    parsed = extract_citations(text)
    report.citations = [f"{c['doc_id']}" for c in parsed]

    # 同一文档可能命中多页/多题：坐标收集为集合，引用命中任一坐标即有效
    doc_index: Dict[str, Dict[str, set]] = {}
    for item in evidence:
        entry = doc_index.setdefault(item.get("doc_id", ""), {"page": set(), "question_no": set()})
        if item.get("page"):
            entry["page"].add(_norm_page(item["page"]))
        if item.get("question_no"):
            entry["question_no"].add(_norm_page(item["question_no"]))

    if not parsed and require_citation and len(text) > 60:
        report.issues.append(CitationIssue(raw="", kind="no_citation",
                                           detail="长答案未携带任何来源引用"))
    for cite in parsed:
        raw = f"【来源:{cite['doc_id']}】"
        if cite["doc_id"] not in doc_index:
            report.issues.append(CitationIssue(raw=raw, kind="unknown_doc",
                                               detail="引用的文档不在本轮证据集中"))
            continue
        entry = doc_index[cite["doc_id"]]
        if cite["page"] and entry["page"] and _norm_page(cite["page"]) not in entry["page"]:
            report.issues.append(CitationIssue(raw=raw, kind="page_mismatch",
                                               detail=f"页码 {cite['page']} 与证据 {sorted(entry['page'])} 均不符"))
        if cite["question_no"] and entry["question_no"] and \
                _norm_page(cite["question_no"]) not in entry["question_no"]:
            report.issues.append(CitationIssue(raw=raw, kind="page_mismatch",
                                               detail=f"题号 {cite['question_no']} 与证据不符"))
    return report
