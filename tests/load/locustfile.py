"""Load-test profile for the `app` service.

Run against a stack with an ECHO or a fast self-hosted model (do NOT point it
at a paid public API):

    locust -f tests/load/locustfile.py --host http://localhost:8000

Each simulated user runs a full interview: create session -> answer every
question -> generate the report. This exercises the DB write path, the RAG
call path, and the LLM call path together, which is what production concurrency
stresses.
"""

from __future__ import annotations

import uuid

from locust import HttpUser, between, task


class DueDiligenceUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def full_interview(self) -> None:
        r = self.client.post(
            "/sessions",
            json={
                "company_id": f"load-{uuid.uuid4().hex[:8]}",
                "startup_info": "B2B SaaS, 15 staff, ~2 crore INR ARR, 70% revenue from top 5.",
                "startup_stage": "Seed",
            },
            headers={"Idempotency-Key": uuid.uuid4().hex},
            name="POST /sessions",
        )
        if r.status_code != 201:
            return
        session_id = r.json()["session_id"]

        q = self.client.get(
            f"/sessions/{session_id}/questions", name="GET /sessions/:id/questions"
        )
        current = q.json().get("current_question")

        guard = 0
        while current and guard < 60:
            guard += 1
            a = self.client.post(
                f"/questions/{current['question_id']}/answer",
                json={"answer": "We track this monthly; the figure is stable and documented."},
                name="POST /questions/:id/answer",
            )
            if a.status_code != 200:
                break
            data = a.json()
            current = None if data["type"] == "complete" else data.get("question")

        self.client.post(
            f"/sessions/{session_id}/complete", name="POST /sessions/:id/complete"
        )
