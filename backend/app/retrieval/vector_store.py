"""向量库封装：Milvus 优先，自动降级到进程内 numpy 余弦索引。

对外只暴露统一契约（upsert / search / delete_by_doc），上层检索代码
不感知后端差异；元数据过滤（学段×学科）在两种后端中语义一致。
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from backend.app.config import Settings, get_settings


@dataclass
class VectorRecord:
    """一条可检索的向量化切片。"""

    chunk_id: str
    doc_id: str
    text: str
    embedding: List[float]
    parent_id: str = ""
    page: str = ""
    question_no: str = ""
    grade: str = ""
    subject: str = ""
    source_type: str = "textbook"
    extra: Dict[str, str] = field(default_factory=dict)


class BaseVectorStore:
    """向量库统一接口。"""

    def upsert(self, records: List[VectorRecord]) -> None:
        raise NotImplementedError

    def search(self, query_vec: List[float], top_k: int = 10,
               filters: Optional[Dict[str, str]] = None) -> List[Tuple[VectorRecord, float]]:
        raise NotImplementedError

    def delete_by_doc(self, doc_id: str) -> None:
        raise NotImplementedError

    def get_by_ids(self, chunk_ids: List[str]) -> Dict[str, VectorRecord]:
        raise NotImplementedError


class InMemoryVectorStore(BaseVectorStore):
    """离线降级实现：numpy 矩阵 + 余弦相似度，线程安全。"""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._records: Dict[str, VectorRecord] = {}
        self._matrix: Optional[np.ndarray] = None
        self._ids: List[str] = []

    def upsert(self, records: List[VectorRecord]) -> None:
        with self._lock:
            for r in records:
                self._records[r.chunk_id] = r
            self._ids = list(self._records.keys())
            if self._ids:
                self._matrix = np.asarray([self._records[i].embedding for i in self._ids], dtype=np.float32)
                norms = np.linalg.norm(self._matrix, axis=1, keepdims=True)
                self._matrix = self._matrix / np.clip(norms, 1e-9, None)

    def search(self, query_vec: List[float], top_k: int = 10,
               filters: Optional[Dict[str, str]] = None) -> List[Tuple[VectorRecord, float]]:
        with self._lock:
            if self._matrix is None or not self._ids:
                return []
            q = np.asarray(query_vec, dtype=np.float32)
            q = q / max(float(np.linalg.norm(q)), 1e-9)
            sims = self._matrix @ q
            order = np.argsort(-sims)
            out: List[Tuple[VectorRecord, float]] = []
            for idx in order:
                rec = self._records[self._ids[int(idx)]]
                if filters and not _match(rec, filters):
                    continue
                out.append((rec, float(sims[int(idx)])))
                if len(out) >= top_k:
                    break
            return out

    def delete_by_doc(self, doc_id: str) -> None:
        with self._lock:
            for cid in [c for c, r in self._records.items() if r.doc_id == doc_id]:
                del self._records[cid]
            self.upsert([])  # 触发矩阵重建

    def get_by_ids(self, chunk_ids: List[str]) -> Dict[str, VectorRecord]:
        with self._lock:
            return {cid: self._records[cid] for cid in chunk_ids if cid in self._records}


class MilvusVectorStore(BaseVectorStore):
    """Milvus 后端：schema 含引用坐标与学段/学科标量字段，支持过滤检索。"""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        from pymilvus import (  # 延迟导入：离线环境不装 pymilvus 也能跑
            Collection, DataType, FieldSchema, CollectionSchema, connections, utility,
        )
        self._m = {"Collection": Collection, "DataType": DataType,
                   "FieldSchema": FieldSchema, "CollectionSchema": CollectionSchema,
                   "connections": connections, "utility": utility}
        host = self.settings.milvus_uri.replace("http://", "").split(":")
        connections.connect(alias="default", host=host[0], port=host[1] if len(host) > 1 else "19530")
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        m, name = self._m, self.settings.milvus_collection
        if m["utility"].has_collection(name):
            self.collection = m["Collection"](name)
            self.collection.load()
            return
        dt = m["DataType"]
        schema = m["CollectionSchema"](fields=[
            m["FieldSchema"]("pk", dt.INT64, is_primary=True, auto_id=True),
            m["FieldSchema"]("chunk_id", dt.VARCHAR, max_length=128),
            m["FieldSchema"]("doc_id", dt.VARCHAR, max_length=64),
            m["FieldSchema"]("parent_id", dt.VARCHAR, max_length=128),
            m["FieldSchema"]("text", dt.VARCHAR, max_length=8192),
            m["FieldSchema"]("page", dt.VARCHAR, max_length=32),
            m["FieldSchema"]("question_no", dt.VARCHAR, max_length=32),
            m["FieldSchema"]("grade", dt.VARCHAR, max_length=32),
            m["FieldSchema"]("subject", dt.VARCHAR, max_length=32),
            m["FieldSchema"]("source_type", dt.VARCHAR, max_length=32),
            m["FieldSchema"]("embedding", dt.FLOAT_VECTOR, dim=self.settings.embed_dim),
        ])
        self.collection = m["Collection"].create(name, schema)
        self.collection.create_index("embedding", {"index_type": "HNSW", "metric_type": "IP",
                                                   "params": {"M": 16, "efConstruction": 200}})
        self.collection.load()

    def upsert(self, records: List[VectorRecord]) -> None:
        rows = [{
            "chunk_id": r.chunk_id, "doc_id": r.doc_id, "parent_id": r.parent_id,
            "text": r.text[:8000], "page": r.page, "question_no": r.question_no,
            "grade": r.grade, "subject": r.subject, "source_type": r.source_type,
            "embedding": r.embedding,
        } for r in records]
        self.collection.upsert(rows)

    def search(self, query_vec: List[float], top_k: int = 10,
               filters: Optional[Dict[str, str]] = None) -> List[Tuple[VectorRecord, float]]:
        expr = " and ".join(f'{k} == "{v}"' for k, v in (filters or {}).items()
                            if k in ("grade", "subject", "source_type", "doc_id"))
        res = self.collection.search(
            data=[query_vec], anns_field="embedding",
            param={"metric_type": "IP", "params": {"ef": 64}},
            limit=top_k * 2, expr=expr or None,
            output_fields=["chunk_id", "doc_id", "parent_id", "text", "page",
                           "question_no", "grade", "subject", "source_type"],
        )
        out: List[Tuple[VectorRecord, float]] = []
        for hit in res[0]:
            e = hit.entity
            rec = VectorRecord(chunk_id=e.get("chunk_id"), doc_id=e.get("doc_id"),
                               parent_id=e.get("parent_id") or "", text=e.get("text"),
                               page=e.get("page") or "", question_no=e.get("question_no") or "",
                               grade=e.get("grade") or "", subject=e.get("subject") or "",
                               source_type=e.get("source_type") or "textbook", embedding=[])
            out.append((rec, float(hit.distance)))
        return out[:top_k]

    def delete_by_doc(self, doc_id: str) -> None:
        self.collection.delete(f'doc_id == "{doc_id}"')

    def get_by_ids(self, chunk_ids: List[str]) -> Dict[str, VectorRecord]:
        if not chunk_ids:
            return {}
        expr = " or ".join(f'chunk_id == "{cid}"' for cid in chunk_ids)
        rows = self.collection.query(expr=expr, output_fields=[
            "chunk_id", "doc_id", "parent_id", "text", "page", "question_no",
            "grade", "subject", "source_type"])
        out: Dict[str, VectorRecord] = {}
        for row in rows:
            out[row["chunk_id"]] = VectorRecord(
                chunk_id=row["chunk_id"], doc_id=row["doc_id"], text=row["text"],
                parent_id=row.get("parent_id") or "", page=row.get("page") or "",
                question_no=row.get("question_no") or "", grade=row.get("grade") or "",
                subject=row.get("subject") or "", source_type=row.get("source_type") or "textbook",
                embedding=[])
        return out


def _match(rec: VectorRecord, filters: Dict[str, str]) -> bool:
    """内存后端的元数据过滤，与 Milvus 表达式语义对齐。"""
    for key, want in filters.items():
        if not want:
            continue
        if getattr(rec, key, "") != want:
            return False
    return True


_store_singleton: Optional[BaseVectorStore] = None


def get_vector_store() -> BaseVectorStore:
    """工厂：显式开启且 pymilvus 可用时走 Milvus，否则内存降级。"""
    global _store_singleton
    if _store_singleton is not None:
        return _store_singleton
    settings = get_settings()
    import os
    if os.getenv("USE_MILVUS", "0") == "1":
        try:
            _store_singleton = MilvusVectorStore(settings)
            return _store_singleton
        except Exception:  # 连接失败自动降级，不让检索阻塞服务启动
            pass
    _store_singleton = InMemoryVectorStore()
    return _store_singleton
