import unittest

from fastapi.testclient import TestClient

from scripts.mock_dres_server import app, reset_mock_state
from services.dres_submission import build_dres_submission_payload


class MockDresServerTests(unittest.TestCase):
    def setUp(self):
        reset_mock_state()
        self.client = TestClient(app)
        login = self.client.post(
            "/api/v2/login",
            json={"username": "team_197", "password": "mock-password"},
        )
        self.assertEqual(login.status_code, 200)
        self.session = login.json()["sessionId"]

    def test_full_login_evaluation_and_three_submission_flows(self):
        evaluations = self.client.get(
            "/api/v2/client/evaluation/list",
            params={"session": self.session},
        )
        self.assertEqual(evaluations.status_code, 200)
        self.assertEqual(len(evaluations.json()), 1)
        self.assertEqual(evaluations.json()[0]["id"], "mock-final")

        task = self.client.get(
            "/api/v2/client/evaluation/currentTask/mock-final",
            params={"session": self.session},
        )
        self.assertEqual(task.status_code, 200)
        self.assertEqual(task.json()["taskType"], "MANUAL")

        cases = [
            (
                "mock-final",
                build_dres_submission_payload(
                    mode="kis", video_id="L24_V044.mp4", time_ms=7520
                ),
            ),
            (
                "mock-final",
                build_dres_submission_payload(
                    mode="qa",
                    video_id="L24_V044",
                    time_ms=1250,
                    answer="màu đỏ",
                ),
            ),
            (
                "mock-final",
                build_dres_submission_payload(
                    mode="trake",
                    video_id="L24_V044",
                    frame_ids=[12, 48, 103],
                ),
            ),
        ]

        for evaluation_id, payload in cases:
            with self.subTest(evaluation_id=evaluation_id):
                submitted = self.client.post(
                    f"/api/v2/submit/{evaluation_id}",
                    params={"session": self.session},
                    json=payload,
                )
                self.assertEqual(submitted.status_code, 200)
                self.assertEqual(submitted.json()["captured"]["payload"], payload)

        captured = self.client.get("/debug/submissions").json()
        self.assertEqual(captured["count"], 3)
        self.assertEqual(
            [item["mode"] for item in captured["submissions"]],
            ["kis", "qa", "trake"],
        )

    def test_wrong_payload_for_evaluation_is_rejected(self):
        response = self.client.post(
            "/api/v2/submit/mock-final",
            params={"session": self.session},
            json={"answerSets": [{"answers": [{"text": "NOT-A-DRES-PAYLOAD"}]}]},
        )
        self.assertEqual(response.status_code, 412)

    def test_unknown_session_is_rejected(self):
        response = self.client.get(
            "/api/v2/client/evaluation/list",
            params={"session": "unknown"},
        )
        self.assertEqual(response.status_code, 401)


if __name__ == "__main__":
    unittest.main()
