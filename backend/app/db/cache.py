"""镜像双缓存：热层（Redis，可选）+ 暖层（进程内/数据库）。

写回纪律（与简历口径一致的原创设计）：
只有两种内容允许进入缓存——
1. 教师复核定稿（reviewed=True）的答案；
2. 与既有缓存问题语义相似度 ≥ 0.92 的刷新（视为同一问题的再验证）。
未达标的机器答案一律不缓存，从源头避免"错误答案被放大复用"。

失效纪律：缓存条目绑定语料版本号（corpus_version），课程修订后
版本号变更即整体失效，不会出现"新教材配旧答案"。
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from backend.app.config import get_settings
from backend.app.retrieval.embeddings import get_embedder


@dataclass
class CacheEntry:
    """一条缓存的问答对。"""

    query: str
    answer: str
    grade: str
    subject: str
    query_vec: List[float]
    reviewed: bool = False
    hit_count: int = 0
    corpus_version: str = ""
    updated_at: float = field(default_factory=lambda: time.time())


class MirrorCache:
    """镜像双缓存实现。"""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._entries: List[CacheEntry] = []
        self._redis = self._connect_redis()
        self.settings = get_settings()

    # ---------- 查询 ----------
    def lookup(self, query: str, grade: str = "", subject: str = ""
               ) -> Optional[Tuple[CacheEntry, float]]:
        """语义命中：相似度 ≥ 阈值 且 学段/学科/语料版本一致。"""
        vec = np.asarray(get_embedder().embed_one(query), dtype=np.float32)
        threshold = self.settings.cache_similarity_threshold
        best: Optional[Tuple[CacheEntry, float]] = None
        with self._lock:
            for entry in self._entries:
                if entry.corpus_version != self.settings.corpus_version:
                    continue
                if grade and entry.grade != grade:
                    continue
                if subject and entry.subject != subject:
                    continue
                ev = np.asarray(entry.query_vec, dtype=np.float32)
                sim = float(np.dot(vec, ev) / (np.linalg.norm(vec) * np.linalg.norm(ev) + 1e-9))
                if sim >= threshold and (best is None or sim > best[1]):
                    best = (entry, sim)
            if best:
                best[0].hit_count += 1
                best[0].updated_at = time.time()
        return best

    # ---------- 写回 ----------
    def put(self, query: str, answer: str, grade: str, subject: str,
            reviewed: bool = False) -> bool:
        """按写回纪律尝试入缓存；返回是否真正写入。"""
        threshold = self.settings.cache_similarity_threshold
        existing = self.lookup(query, grade, subject)
        with self._lock:
            if existing:
                entry, _ = existing
                # 相似刷新：保留已定稿答案，未定稿则更新内容与时间
                if not entry.reviewed:
                    entry.query, entry.answer = query, answer
                    entry.query_vec = get_embedder().embed_one(query)
                entry.corpus_version = self.settings.corpus_version
                entry.updated_at = time.time()
                self._write_hot(entry)
                return True
            if not reviewed:
                return False  # 未定稿且无相似定稿支撑 -> 拒绝写回
            entry = CacheEntry(
                query=query, answer=answer, grade=grade, subject=subject,
                query_vec=get_embedder().embed_one(query), reviewed=True,
                corpus_version=self.settings.corpus_version,
            )
            self._entries.append(entry)
            self._write_hot(entry)
            return True

    # ---------- 维护 ----------
    def invalidate_corpus(self) -> None:
        """课程修订：清空全部条目并同步清热层。"""
        with self._lock:
            self._entries.clear()
        if self._redis is not None:
            try:
                self._redis.delete(*self._redis.keys("huixue:cache:*") or "huixue:cache:__none__")
            except Exception:
                pass

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "entries": len(self._entries),
                "reviewed": sum(1 for e in self._entries if e.reviewed),
                "total_hits": sum(e.hit_count for e in self._entries),
            }

    # ---------- 热层 ----------
    @staticmethod
    def _connect_redis():
        if get_settings().mock_mode:
            return None
        try:
            import redis  # 可选依赖

            client = redis.Redis.from_url(get_settings().redis_url, decode_responses=True)
            client.ping()
            return client
        except Exception:
            return None

    def _write_hot(self, entry: CacheEntry) -> None:
        """暖层写入成功后同步一条精简副本到热层（失败不影响主流程）。"""
        if self._redis is None:
            return
        import json

        key = f"huixue:cache:{entry.corpus_version}:{entry.grade}:{entry.subject}"
        try:
            self._redis.hset(key, mapping={entry.query[:64]: json.dumps(
                {"answer": entry.answer, "reviewed": entry.reviewed}, ensure_ascii=False)})
            self._redis.expire(key, 86400)
        except Exception:
            pass


_cache_singleton: Optional[MirrorCache] = None


def get_mirror_cache() -> MirrorCache:
    global _cache_singleton
    if _cache_singleton is None:
        _cache_singleton = MirrorCache()
    return _cache_singleton
