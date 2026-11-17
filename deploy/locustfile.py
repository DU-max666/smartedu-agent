"""Locust 压测脚本：模拟答疑高峰场景（教师/学生混合流量）。

用法：
    locust -f deploy/locustfile.py --host http://127.0.0.1:8000
    locust -f deploy/locustfile.py --headless -u 4000 -r 200 -t 5m --host http://127.0.0.1:8000

注意：4000 并发为区域平台峰值口径；本地压测请按机器容量调低 -u。
"""
from __future__ import annotations

import random
import string

from locust import HttpUser, between, task

_QUESTIONS = [
    "3千米等于多少米",
    "5米40厘米和54分米哪个大",
    "为什么冬天窗户里面会有小水珠",
    "1厘米等于多少毫米",
    "估测长度可以用什么参照物",
]


class SmartEduUser(HttpUser):
    """单虚拟用户：答疑为主，偶发健康检查。"""

    wait_time = between(0.5, 2.0)

    def _headers(self) -> dict:
        role = random.choice(["student", "student", "student", "teacher"])
        ref = "S-" + "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
        return {
            "X-Huixue-Key": "dev-local-key",
            "X-User-Role": role,
            "X-User-Ref": ref if role == "student" else "",
        }

    @task(8)
    def ask_question(self) -> None:
        self.client.post(
            "/api/v1/qa/ask",
            json={
                "question": random.choice(_QUESTIONS),
                "grade": "小学三年级",
                "subject": "数学",
                "session_id": f"load-{random.randint(1, 5000)}",
            },
            headers=self._headers(),
        )

    @task(1)
    def health(self) -> None:
        self.client.get("/healthz")
