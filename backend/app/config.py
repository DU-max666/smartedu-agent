"""全局配置。

从环境变量（或项目根目录 .env 文件）读取配置；未提供模型密钥时
自动进入 mock 模式，保证无密钥环境下全链路与单测可离线跑通。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

# 项目根目录：backend/app/config.py -> 上溯两级到仓库根
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv(path: Path) -> None:
    """极简 .env 解析：KEY=VALUE，# 开头为注释，已存在的环境变量不覆盖。"""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


@dataclass(frozen=True)
class Settings:
    """慧学平台运行期配置（全部可由环境变量覆盖，默认值面向本地开发）。"""

    app_name: str = "hui-xue-smartedu-agent"
    env: str = field(default_factory=lambda: os.getenv("APP_ENV", "dev"))
    # ---- OpenAI 兼容模型网关 ----
    openai_base_url: str = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL", ""))
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    chat_model: str = field(default_factory=lambda: os.getenv("CHAT_MODEL", "gpt-4o-mini"))
    temperature: float = field(default_factory=lambda: float(os.getenv("CHAT_TEMPERATURE", "0.2")))
    max_retries: int = field(default_factory=lambda: int(os.getenv("GATEWAY_MAX_RETRIES", "3")))
    rate_limit_qps: float = field(default_factory=lambda: float(os.getenv("GATEWAY_RATE_QPS", "8")))
    # ---- 嵌入 / 重排 ----
    embed_model: str = field(default_factory=lambda: os.getenv("EMBED_MODEL", "BGE-M3"))
    embed_dim: int = field(default_factory=lambda: int(os.getenv("EMBED_DIM", "512")))
    rerank_model: str = field(default_factory=lambda: os.getenv("RERANK_MODEL", "BGE-Reranker-V2-M3"))
    # ---- 存储 ----
    milvus_uri: str = field(default_factory=lambda: os.getenv("MILVUS_URI", "http://localhost:19530"))
    milvus_collection: str = field(default_factory=lambda: os.getenv("MILVUS_COLLECTION", "huixue_corpus"))
    redis_url: str = field(default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    database_url: str = field(default_factory=lambda: os.getenv("DATABASE_URL", "sqlite:///./huixue.db"))
    # ---- 智能体调度阈值（与简历口径一致）----
    intent_threshold: float = field(default_factory=lambda: float(os.getenv("INTENT_THRESHOLD", "0.92")))
    cache_similarity_threshold: float = field(default_factory=lambda: float(os.getenv("CACHE_SIM_THRESHOLD", "0.92")))
    max_clarify_rounds: int = field(default_factory=lambda: int(os.getenv("MAX_CLARIFY_ROUNDS", "2")))
    recent_turns: int = field(default_factory=lambda: int(os.getenv("RECENT_TURNS", "6")))
    # ---- 语料版本：课程修订后用于整体失效镜像缓存 ----
    corpus_version: str = field(default_factory=lambda: os.getenv("CORPUS_VERSION", "v2026.03"))

    @property
    def mock_mode(self) -> bool:
        """无密钥或显式声明离线时，走本地确定性 mock 实现。"""
        return (not self.openai_api_key) or self.env in ("test", "offline")


def _build_settings() -> Settings:
    _load_dotenv(_REPO_ROOT / ".env")
    return Settings()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """进程级单例配置。测试中可通过 cache_clear() 重载 .env。"""
    return _build_settings()
