"""混合检索测试：RRF 融合、元数据过滤、父子块回填、坏样本下线。"""
from __future__ import annotations

from backend.app.retrieval.embeddings import EmbeddingClient
from backend.app.retrieval.hybrid_search import HybridRetriever, rrf_fuse
from backend.app.retrieval.vector_store import InMemoryVectorStore, VectorRecord


def _build_retriever() -> HybridRetriever:
    emb = EmbeddingClient()
    store = InMemoryVectorStore()
    retriever = HybridRetriever(store=store, embedder=emb)
    parent = VectorRecord(chunk_id="d1#P#1", doc_id="d1", text="长度单位总述：千米米分米厘米毫米。",
                          embedding=[], extra={"kind": "parent"},
                          grade="小学三年级", subject="数学", page="第10页")
    child = VectorRecord(chunk_id="d1#C#1", doc_id="d1", parent_id=parent.chunk_id,
                         text="1千米等于1000米，千米与米之间进率是1000。",
                         embedding=[], extra={"kind": "child"},
                         grade="小学三年级", subject="数学", page="第10页")
    other = VectorRecord(chunk_id="d2#C#1", doc_id="d2",
                         text="水蒸气遇冷液化成小水珠。",
                         embedding=[], extra={"kind": "child"},
                         grade="小学四年级", subject="科学", page="第22页")
    retriever.index([parent, child, other])
    return retriever


def test_rrf_fuse_basic():
    fused = rrf_fuse([["a", "b", "c"], ["b", "a", "d"]], k=60)
    assert fused["b"] > fused["d"]
    assert abs(fused["a"] - (1 / 61 + 1 / 62)) < 1e-9


def test_hybrid_search_relevance():
    r = _build_retriever()
    hits = r.search("1千米等于多少米", top_k=2)
    assert hits and hits[0]["chunk_id"] == "d1#C#1"


def test_metadata_filter_blocks_cross_grade():
    r = _build_retriever()
    hits = r.search("水珠", top_k=3, filters={"grade": "小学三年级", "subject": "数学"})
    # 四年级科学文档被学段×学科过滤挡在召回之外
    assert not any(h["doc_id"] == "d2" for h in hits)


def test_parent_expansion():
    r = _build_retriever()
    hits = r.search("1千米等于多少米", top_k=1, expand_parent=True)
    assert hits and hits[0].get("parent_text"), "命中子块应回填父块上下文"


def test_parent_not_recallable():
    """父块不参与召回，避免长文本抢占子块位置。"""
    r = _build_retriever()
    hits = r.search("长度单位 千米 米", top_k=5)
    assert all(h["chunk_id"] != "d1#P#1" for h in hits)


def test_drop_doc_removes_from_index():
    r = _build_retriever()
    r.drop_doc("d1")
    hits = r.search("1千米等于多少米", top_k=3)
    assert not any(h["doc_id"] == "d1" for h in hits)
