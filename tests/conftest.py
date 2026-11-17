"""pytest 公共夹具：强制离线 mock 模式并打通 import 路径。"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ["APP_ENV"] = "test"   # 触发 mock 模式（无密钥可跑）
os.environ["SKIP_DB"] = "1"      # 单测不落数据库
sys.path.insert(0, str(ROOT))
