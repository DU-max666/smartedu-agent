"""混合检索：向量 + BM25 双路召回 → RRF 融合 → 重排 → 父子块回填。

原创设计点：
1. 元数据过滤贯穿全链路（学段×学科×来源），BM25 路在候选阶段同样过滤，
   避免"向量过滤了、词面没过滤"导致跨学段证据混入；
2. RRF 融合后再重排（先保召回多样性、再精排），重排失败自动跳过不阻塞；
3. 父子块回填：命中子块后取父块全文写入 parent_text，生成节点用父块
   上下文、引用标注用子块坐标，兼顾语义聚焦与上下文完整。
"""
from __future__ import annotations

from typing import Dict, List, Optional

from backend.app.graphs.state import EvidenceItem
from backend.app.retrieval.bm25 import BM25Index
from backend.app.retrieval.embeddings import EmbeddingClient, get_embedder
from backend.app.retrieval.reranker import Reranker
from backend.app.retrieval.vector_store import BaseVectorStore, VectorRecord, get_vector_store


def rrf_fuse(rank_lists: List[List[str]], k: int = 60) -> Dict[str, float]:
    """Reciprocal Rank Fusion：score(d) = Σ 1/(k + rank)。"""
    fused: Dict[str, float] = {}
    for ranked in rank_lists:
        for rank, item_id in enumerate(ranked, start=1):
            fused[item_id] = fused.get(item_id, 0.0) + 1.0 / (k + rank)
    return fused


class HybridRetriever:
    """慧学统一检索入口。"""

    def __init__(self, store: Optional[BaseVectorStore] = None,
                 embedder: Optional[EmbeddingClient] = None,
                 reranker: Optional[Reranker] = None,
                 rrf_k: int = 60) -> None:
        self.store = store or get_vector_store()
        self.embedder = embedder or get_embedder()
        self.reranker = reranker or Reranker()
        self.rrf_k = rrf_k
        self.bm25 = BM25Index()
        self._corpus: Dict[str, VectorRecord] = {}

    # ---------- 索引 ----------
    def index(self, records: List[VectorRecord]) -> None:
        if not records:
            return
        vectors = self.embedder.embed([r.text for r in records])
        enriched: List[VectorRecord] = []
        for rec, vec in zip(records, vectors):
            rec.embedding = vec
            enriched.append(rec)
        self.store.upsert(enriched)
        # 子块（及表格行/对话轮）参与召回；父块只入向量库供上下文回填
        recallable = [r for r in enriched if r.extra.get("kind") != "parent"]
        self.bm25.add_documents([(r.chunk_id, r.text) for r in recallable])
        for r in recallable:
            self._corpus[r.chunk_id] = r

    def drop_doc(self, doc_id: str) -> None:
        """坏样本下线：同步清理向量、BM25 与语料映射。"""
        self.store.delete_by_doc(doc_id)
        for cid in [c for c, r in self._corpus.items() if r.doc_id == doc_id]:
            self.bm25.invalidate(cid)
            self._corpus.pop(cid, None)

    # ---------- 查询 ----------
    def search(self, query: str, top_k: int = 5,
               filters: Optional[Dict[str, str]] = None,
               expand_parent: bool = True) -> List[EvidenceItem]:
        filters = filters or {}
        query_vec = self.embedder.embed_one(query)
        pool = max(top_k * 4, 12)

        vector_hits = self.store.search(query_vec, top_k=pool, filters=filters)
        vector_ids = [rec.chunk_id for rec, _ in vector_hits if rec.chunk_id in self._corpus]

        bm25_ids = [cid for cid, _ in self.bm25.search(query, top_k=pool)
                    if self._pass_filter(cid, filters)]

        fused = rrf_fuse([vector_ids, bm25_ids], k=self.rrf_k)
        candidate_ids = sorted(fused, key=fused.get, reverse=True)[:pool]
        if not candidate_ids:
            return []

        # 重排（失败不阻塞：直接沿用 RRF 顺序）
        try:
            texts = [(cid, self._corpus[cid].text) for cid in candidate_ids if cid in self._corpus]
            rerank_scores = dict(self.reranker.rerank(query, texts))
        except Exception:
            rerank_scores = {cid: fused[cid] for cid in candidate_ids}

        ordered = sorted(candidate_ids,
                         key=lambda cid: (rerank_scores.get(cid, 0.0), fused.get(cid, 0.0)),
                         reverse=True)[:top_k]

        parent_ids = [self._corpus[cid].parent_id for cid in ordered
                      if cid in self._corpus and self._corpus[cid].parent_id]
        parents = self.store.get_by_ids(parent_ids) if expand_parent and parent_ids else {}

        evidence: List[EvidenceItem] = []
        for cid in ordered:
            rec = self._corpus.get(cid)
            if rec is None:
                continue
            item: EvidenceItem = {
                "doc_id": rec.doc_id, "chunk_id": rec.chunk_id, "parent_id": rec.parent_id,
                "text": rec.text, "page": rec.page, "question_no": rec.question_no,
                "score": float(rerank_scores.get(cid, 0.0)), "source_type": rec.source_type,
            }
            parent = parents.get(rec.parent_id)
            if parent is not None:
                item["parent_text"] = parent.text  # type: ignore[typeddict-item]
            evidence.append(item)
        return evidence

    # ---------- 辅助 ----------
    def _pass_filter(self, chunk_id: str, filters: Dict[str, str]) -> bool:
        rec = self._corpus.get(chunk_id)
        if rec is None:
            return False
        return all(not want or getattr(rec, key, "") == want
                   for key, want in filters.items())


_retriever_singleton: Optional[HybridRetriever] = None


def get_retriever() -> HybridRetriever:
    """进程级检索单例（graphs 与 tools 共用同一份语料索引）。"""
    global _retriever_singleton
    if _retriever_singleton is None:
        _retriever_singleton = HybridRetriever()
    return _retriever_singleton
