import unittest

import httpx

import app as application
from scripts.mock_dres_server import (
    EVALUATIONS,
    app as mock_dres_app,
    reset_mock_state,
    set_next_mock_response,
)
from services.dres_client import DresClient
from services.dres_session import DresSessionStore


class DresApiIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        reset_mock_state()
        application.DRES_DEFAULT_USERNAME = "team_197"
        application.DRES_DEFAULT_PASSWORD = "mock-password"
        application.DRES_COOKIE_SECURE = False
        application.dres_sessions = DresSessionStore()
        application.dres_client = DresClient(
            "http://127.0.0.1:19100",
            transport=httpx.ASGITransport(app=mock_dres_app),
        )
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=application.app),
            base_url="http://application.local",
        )

        login = await self.client.post("/api/dres/login/default", json={})
        self.assertEqual(login.status_code, 200)
        selected = await self.client.post(
            "/api/dres/evaluation",
            json={"evaluationId": "mock-final"},
        )
        self.assertEqual(selected.status_code, 200)

    async def asyncTearDown(self):
        await self.client.aclose()

    async def _submit(self, time_ms: int):
        return await self.client.post(
            "/api/dres/submit",
            json={
                "evaluationId": "mock-final",
                "mode": "kis",
                "videoId": "L24_V044",
                "timeMs": time_ms,
            },
        )

    async def test_submission_result_controls_are_served(self):
        response = await self.client.get("/")
        self.assertEqual(response.status_code, 200)
        html = response.text
        for element_id in (
            "submission-result",
            "trake-result",
            "copy-submission-payload-btn",
            "copy-trake-payload-btn",
        ):
            self.assertIn(f'id="{element_id}"', html)
        self.assertIn('/static/js/main.js?v=13', html)
        self.assertIn('id="refresh-evaluations-btn"', html)

    async def test_refresh_evaluations_fetches_new_active_evaluations(self):
        added_evaluation = {
            "id": "mock-new",
            "name": "MOCK • Evaluation mới",
            "type": "SYNCHRONOUS",
            "status": "ACTIVE",
        }
        EVALUATIONS.append(added_evaluation)
        try:
            response = await self.client.get("/api/dres/evaluations")
        finally:
            EVALUATIONS.remove(added_evaluation)

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(
            [evaluation["id"] for evaluation in body["evaluations"]],
            ["mock-final", "mock-new"],
        )
        self.assertEqual(body["selectedEvaluationId"], "mock-final")

    async def test_wrong_verdict_is_received_submission_not_transport_error(self):
        set_next_mock_response("WRONG")
        response = await self._submit(1000)

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["accepted"])
        self.assertFalse(body["pending"])
        self.assertEqual(body["verdict"], "WRONG")
        self.assertEqual(body["rawResult"]["status"], True)
        self.assertIn("SESSION_ID_ẨN", body["destination"])
        self.assertNotIn("?session=mock-", body["destination"])

    async def test_pending_is_returned_without_guessing_verdict(self):
        set_next_mock_response("PENDING")
        response = await self._submit(2000)

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["accepted"])
        self.assertTrue(body["pending"])
        self.assertIsNone(body["verdict"])
        self.assertEqual(body["dresHttpStatus"], 202)

    async def test_rejected_submission_is_not_recorded_as_success(self):
        set_next_mock_response("REJECTED")
        rejected = await self._submit(3000)
        self.assertEqual(rejected.status_code, 412)
        self.assertIn("từ chối", rejected.json()["detail"])

        set_next_mock_response("CORRECT")
        retry = await self._submit(3000)
        self.assertEqual(retry.status_code, 200)
        self.assertEqual(retry.json()["verdict"], "CORRECT")

    async def test_immediate_identical_resubmission_is_temporarily_blocked(self):
        first = await self._submit(4000)
        second = await self._submit(4000)

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 409)
        self.assertIn("3 giây", second.json()["detail"])
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=mock_dres_app),
            base_url="http://mock.local",
        ) as mock_client:
            captured = await mock_client.get("/debug/submissions")
        self.assertEqual(captured.json()["count"], 1)


if __name__ == "__main__":
    unittest.main()
