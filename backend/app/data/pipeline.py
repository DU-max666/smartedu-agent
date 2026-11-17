"""语料接入管道：多源数据 → 元数据抽取 → 清洗去重 → 切片 → 入库。

流程对应 docs/data_governance.md 的"入库五步"：
1. 接入：Markdown 直读；扫描件走 OCR 适配器；课堂录音走 ASR 适配器
   （PaddleOCR / Whisper 均为可选依赖，缺库时给出明确安装指引）；
2. 元数据抽取：文件头部 front-matter（doc_id/学段/学科/来源类型/起始页）；
3. 清洗去重：控制字符归一 + 内容哈希去重（重复样本直接拒收并计数）；
4. 切片：按来源类型路由到三种切片器（父子块/表格结构化/对话轮次）；
5. 入库：子块进检索索引，父块只进向量库供上下文回填——
   父块不参与召回，避免长文本稀释匹配精度。
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

from backend.app.retrieval.chunking import (
    ChunkRecord, DialogueTurnChunker, ParentChildChunker, TableStructuralChunker,
)
from backend.app.retrieval.hybrid_search import HybridRetriever, get_retriever
from backend.app.retrieval.vector_store import VectorRecord

_FRONT_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.S)


@dataclass
class RawDoc:
    """一篇待入库原始文档。"""

    doc_id: str
    body: str
    meta: Dict[str, str] = field(default_factory=dict)


class OcrAdapter:
    """扫描件 OCR 适配器（PaddleOCR 可选依赖）。"""

    def read_image(self, path: str) -> str:
        try:
            from paddleocr import PaddleOCR  # type: ignore
        except ImportError as exc:
            raise RuntimeError("OCR 依赖未安装：pip install paddleocr paddlepaddle") from exc
        engine = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
        lines: List[str] = []
        for line in engine.ocr(path, cls=True)[0] or []:
            lines.append(str(line[1][0]))
        return "\n".join(lines)


class AsrAdapter:
    """课堂录音转写适配器（Whisper 可选依赖）。"""

    def transcribe(self, path: str) -> str:
        try:
            import whisper  # type: ignore
        except ImportError as exc:
            raise RuntimeError("ASR 依赖未安装：pip install openai-whisper") from exc
        model = whisper.load_model("small")
        return str(model.transcribe(path, language="zh")["text"])


def extract_front_matter(text: str) -> Tuple[Dict[str, str], str]:
    """解析头部 front-matter，返回 (元数据, 正文)。"""
    match = _FRONT_RE.match(text)
    if not match:
        return {}, text
    meta: Dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip()
    return meta, text[match.end():]


def clean_text(text: str) -> str:
    """清洗：去控制字符、统一空白、压缩连续空行。"""
    text = re.sub(r"[\x00-\x08\x0b-\x1f]", "", text)
    text = re.sub(r"[ \t\u3000]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class CorpusPipeline:
    """语料入库管道。"""

    def __init__(self, retriever: HybridRetriever | None = None) -> None:
        self.retriever = retriever or get_retriever()
        self.parent_child = ParentChildChunker()
        self.table = TableStructuralChunker()
        self.dialogue = DialogueTurnChunker()
        self._seen_hashes: set = set()

    # ---------- 接入与预处理 ----------
    def load_dir(self, directory: str) -> List[RawDoc]:
        docs: List[RawDoc] = []
        for path in sorted(Path(directory).glob("*.md")):
            raw = path.read_text(encoding="utf-8")
            meta, body = extract_front_matter(raw)
            meta.setdefault("doc_id", path.stem)
            meta["file_name"] = path.name
            docs.append(RawDoc(doc_id=meta["doc_id"], body=clean_text(body), meta=meta))
        return docs

    def dedup(self, docs: List[RawDoc]) -> Tuple[List[RawDoc], int]:
        kept: List[RawDoc] = []
        dropped = 0
        for doc in docs:
            digest = hashlib.sha1(doc.body.encode("utf-8")).hexdigest()
            if digest in self._seen_hashes:
                dropped += 1
                continue
            self._seen_hashes.add(digest)
            kept.append(doc)
        return kept, dropped

    # ---------- 切片路由 ----------
    def chunk(self, doc: RawDoc) -> List[ChunkRecord]:
        source_type = doc.meta.get("source_type", "textbook")
        if source_type == "dialogue":
            return self.dialogue.chunk(doc.doc_id, doc.body, doc.meta)
        if source_type == "table":
            return self.table.chunk(doc.doc_id, doc.body, doc.meta)
        return self.parent_child.chunk(doc.doc_id, doc.body, doc.meta)

    # ---------- 入库 ----------
    def to_vector_records(self, doc: RawDoc, chunks: List[ChunkRecord]) -> List[VectorRecord]:
        page_start = int(re.sub(r"\D", "", doc.meta.get("page_start", "0")) or 0)
        records: List[VectorRecord] = []
        for i, chunk in enumerate(chunks):
            page = chunk.meta.get("page") or (f"第{page_start + i // 2}页" if page_start else "")
            records.append(VectorRecord(
                chunk_id=chunk.chunk_id, doc_id=doc.doc_id, text=chunk.text,
                embedding=[], parent_id=chunk.parent_id, page=page,
                question_no=chunk.meta.get("question_no", ""),
                grade=doc.meta.get("grade", ""), subject=doc.meta.get("subject", ""),
                source_type=doc.meta.get("source_type", "textbook"),
                extra={"kind": chunk.meta.get("kind", "child")},
            ))
        return records

    def ingest_dir(self, directory: str) -> Dict[str, Any]:
        """整目录入库，返回统计信息。"""
        docs = self.load_dir(directory)
        docs, dropped_dupes = self.dedup(docs)
        total_chunks = total_indexed = 0
        for doc in docs:
            chunks = self.chunk(doc)
            records = self.to_vector_records(doc, chunks)
            self.retriever.index(records)
            total_chunks += len(records)
            total_indexed += sum(1 for r in records if r.extra.get("kind") != "parent")
        return {"docs": len(docs), "chunks": total_chunks,
                "indexed_children": total_indexed, "dropped_duplicates": dropped_dupes}

    def retire_doc(self, doc_id: str) -> None:
        """坏样本下线：从全部索引中移除。"""
        self.retriever.drop_doc(doc_id)
