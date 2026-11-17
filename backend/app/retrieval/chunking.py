"""语料切片器：三种切分策略，统一产出 ChunkRecord。

与常见"固定长度滑窗"模板不同，本项目按语料形态分治：
1. ParentChildChunker —— 教材/课标 Markdown：按标题层级切"父块"，
   父块内再按句子窗口切"子块"。检索命中子块（语义聚焦），
   生成时回填父块（上下文完整），parent_id 负责两级联动。
2. TableStructuralChunker —— 结构化表格（成绩表/知识点对照表）：
   表头与每一行绑定成独立记录，避免表格被滑窗切碎后语义丢失。
3. DialogueTurnChunker —— 课堂问答/答疑对话记录：以学生问+教师答
   为一轮，整轮入库，保留"问法多样性"供答疑改写节点做少样本参考。
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ChunkRecord:
    """切片统一产物。"""

    chunk_id: str
    doc_id: str
    text: str
    parent_id: str = ""
    meta: Dict[str, str] = field(default_factory=dict)


def _make_id(doc_id: str, tag: str, text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    return f"{doc_id}#{tag}#{digest}"


_SENT_SPLIT = re.compile(r"(?<=[。！？；\n])")


def _split_sentences(text: str) -> List[str]:
    parts = [p for p in _SENT_SPLIT.split(text) if p and p.strip()]
    return parts


class ParentChildChunker:
    """父子两级切分器。"""

    def __init__(self, child_max_chars: int = 240, overlap_sentences: int = 1) -> None:
        self.child_max_chars = child_max_chars
        self.overlap_sentences = overlap_sentences

    def chunk(self, doc_id: str, markdown: str, base_meta: Dict[str, str] | None = None) -> List[ChunkRecord]:
        base_meta = base_meta or {}
        records: List[ChunkRecord] = []
        parents = self._split_by_heading(markdown)
        for heading, body in parents:
            parent_id = _make_id(doc_id, "P", heading + body)
            records.append(ChunkRecord(
                chunk_id=parent_id, doc_id=doc_id, text=(body or heading).strip(),
                parent_id="", meta={**base_meta, "kind": "parent", "heading": heading},
            ))
            sentences = _split_sentences(body)
            window: List[str] = []
            for sent in sentences:
                if window and sum(len(s) for s in window) + len(sent) > self.child_max_chars:
                    records.append(self._child(doc_id, parent_id, window, base_meta, heading))
                    window = window[-self.overlap_sentences:] if self.overlap_sentences else []
                window.append(sent)
            if window:
                records.append(self._child(doc_id, parent_id, window, base_meta, heading))
        return records

    @staticmethod
    def _child(doc_id: str, parent_id: str, sentences: List[str],
               base_meta: Dict[str, str], heading: str) -> ChunkRecord:
        text = "".join(sentences).strip()
        return ChunkRecord(
            chunk_id=_make_id(doc_id, "C", text), doc_id=doc_id, text=text,
            parent_id=parent_id, meta={**base_meta, "kind": "child", "heading": heading},
        )

    @staticmethod
    def _split_by_heading(markdown: str) -> List[tuple]:
        """按 #/##/### 标题切段；无标题内容归入"正文"段。"""
        sections: List[tuple] = []
        current_heading, buf = "", []
        for line in markdown.splitlines():
            match = re.match(r"^(#{1,3})\s+(.*)$", line)
            if match:
                if "".join(buf).strip() or current_heading:
                    sections.append((current_heading, "\n".join(buf)))
                current_heading, buf = match.group(2).strip(), []
            else:
                buf.append(line)
        if "".join(buf).strip() or current_heading:
            sections.append((current_heading, "\n".join(buf)))
        return sections


class TableStructuralChunker:
    """Markdown 表格结构化切分：表头绑定每行。"""

    def chunk(self, doc_id: str, markdown: str, base_meta: Dict[str, str] | None = None) -> List[ChunkRecord]:
        base_meta = base_meta or {}
        records: List[ChunkRecord] = []
        for table_lines in self._extract_tables(markdown):
            header = [c.strip() for c in table_lines[0].strip("|").split("|")]
            for row in table_lines[2:]:  # 跳过分隔行
                cells = [c.strip() for c in row.strip("|").split("|")]
                pairs = "; ".join(f"{h}:{v}" for h, v in zip(header, cells) if v)
                if not pairs:
                    continue
                text = f"表格行（{header[0]}={cells[0]}）：{pairs}"
                records.append(ChunkRecord(
                    chunk_id=_make_id(doc_id, "T", text), doc_id=doc_id, text=text,
                    meta={**base_meta, "kind": "table_row", "columns": ",".join(header)},
                ))
        return records

    @staticmethod
    def _extract_tables(markdown: str) -> List[List[str]]:
        tables, buf = [], []
        for line in markdown.splitlines():
            if line.strip().startswith("|") and line.strip().endswith("|"):
                buf.append(line)
            else:
                if len(buf) >= 2:
                    tables.append(buf)
                buf = []
        if len(buf) >= 2:
            tables.append(buf)
        return tables


class DialogueTurnChunker:
    """对话轮次切分：一问一答成对入库。"""

    _SPEAKER = re.compile(r"^(学生|教师|家长|老师)[：:]\s*")

    def chunk(self, doc_id: str, dialogue_text: str, base_meta: Dict[str, str] | None = None) -> List[ChunkRecord]:
        base_meta = base_meta or {}
        # 第一步：按说话人前缀把文本拆成 (speaker, content) 轮次序列
        turns: List[tuple] = []
        current_speaker, buf = "", []
        for line in dialogue_text.splitlines():
            line = line.strip()
            if not line:
                continue
            match = self._SPEAKER.match(line)
            if match:
                if buf and current_speaker:
                    turns.append((current_speaker, "".join(buf).strip()))
                current_speaker = match.group(1)
                buf = [line[match.end():]]
            elif current_speaker:
                buf.append(line)
        if buf and current_speaker:
            turns.append((current_speaker, "".join(buf).strip()))
        # 第二步：学生/家长的问 与 紧随其后的教师/老师的答 配成一对
        records: List[ChunkRecord] = []
        pending_question = ""
        for speaker, content in turns:
            if not content:
                continue
            if speaker in ("学生", "家长"):
                if pending_question:  # 连续两问：前一无应答，单独入库
                    self._emit(doc_id, pending_question, "", base_meta, records)
                pending_question = content
            elif speaker in ("教师", "老师"):
                self._emit(doc_id, pending_question, content, base_meta, records)
                pending_question = ""
        if pending_question:
            self._emit(doc_id, pending_question, "", base_meta, records)
        return records

    @staticmethod
    def _emit(doc_id: str, question: str, answer: str,
              base_meta: Dict[str, str], records: List[ChunkRecord]) -> None:
        text = f"问：{question}\n答：{answer}".strip()
        records.append(ChunkRecord(
            chunk_id=_make_id(doc_id, "D", text), doc_id=doc_id, text=text,
            meta={**base_meta, "kind": "dialogue_turn", "question": question[:60]},
        ))
