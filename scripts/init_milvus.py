"""初始化 Milvus collection（独立脚本，可离线试跑给出降级提示）。

用法：
    python scripts/init_milvus.py            # 建 collection + HNSW 索引
    python scripts/init_milvus.py --check    # 仅检查存在性
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.config import get_settings  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="初始化慧学 Milvus collection")
    parser.add_argument("--check", action="store_true", help="仅检查 collection 是否存在")
    args = parser.parse_args()

    settings = get_settings()
    try:
        from pymilvus import connections, utility  # noqa: F401
    except ImportError:
        print("[降级] 未安装 pymilvus。离线环境无需本脚本："
              "服务启动时自动使用内存向量索引（USE_MILVUS=0）。")
        return 0

    host = settings.milvus_uri.replace("http://", "").split(":")
    try:
        connections.connect(alias="default", host=host[0],
                            port=host[1] if len(host) > 1 else "19530")
    except Exception as exc:
        print(f"[错误] 无法连接 Milvus {settings.milvus_uri}: {exc}")
        return 1

    exists = utility.has_collection(settings.milvus_collection)
    if args.check:
        print(f"collection {settings.milvus_collection} 存在: {exists}")
        return 0

    if exists:
        print(f"collection {settings.milvus_collection} 已存在，跳过创建。")
        return 0

    from backend.app.retrieval.vector_store import MilvusVectorStore

    MilvusVectorStore(settings)
    print(f"已创建 collection {settings.milvus_collection}（HNSW / 内积）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
