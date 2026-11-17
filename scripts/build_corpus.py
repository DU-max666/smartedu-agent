"""语料初始化脚本：把目录中的 Markdown 语料灌入检索索引。

用法：
    python scripts/build_corpus.py                       # 灌入默认 seed_corpus
    python scripts/build_corpus.py --dir path/to/corpus  # 灌入自定义语料目录
    python scripts/build_corpus.py --retire demo-math-3a # 坏样本下线
    python scripts/build_corpus.py --verify "3千米等于多少米"  # 入库后抽样验证召回
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.data.pipeline import CorpusPipeline  # noqa: E402

DEFAULT_DIR = ROOT / "backend" / "app" / "data" / "seed_corpus"


def main() -> int:
    parser = argparse.ArgumentParser(description="慧学语料入库")
    parser.add_argument("--dir", default=str(DEFAULT_DIR), help="语料目录（*.md，含 front-matter）")
    parser.add_argument("--retire", default="", help="按 doc_id 下线坏样本")
    parser.add_argument("--verify", default="", help="入库后用该查询抽样验证召回")
    args = parser.parse_args()

    pipeline = CorpusPipeline()
    if args.retire:
        pipeline.retire_doc(args.retire)
        print(f"已下线文档: {args.retire}")

    stats = pipeline.ingest_dir(args.dir)
    print("入库统计:", json.dumps(stats, ensure_ascii=False))

    if args.verify:
        from backend.app.retrieval.hybrid_search import get_retriever

        hits = get_retriever().search(args.verify, top_k=3)
        print(f"验证查询「{args.verify}」召回 {len(hits)} 条：")
        for h in hits:
            print(f"  - [{h.get('doc_id')} | {h.get('page')}] {h.get('text', '')[:50]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
