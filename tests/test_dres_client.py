import asyncio
import unittest

import httpx

from services.dres_client import DresApiError, DresClient


class DresClientTests(unittest.TestCase):
    def test_login_evaluations_current_task_and_submit(self):
        seen = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            if request.url.path == "/api/v2/login":
                return httpx.Response(
                    200,
                    json={"sessionId": "secret-session", "username": "team_197"},
                )
            if request.url.path == "/api/v2/client/evaluation/list":
                return httpx.Response(200, json=[{"id": "eval-1", "status": "ACTIVE"}])
            if request.url.path == "/api/v2/client/evaluation/currentTask/eval-1":
                return httpx.Response(200, json={"name": "KIS task", "taskType": "KIS"})
            if request.url.path == "/api/v2/submit/eval-1":
                return httpx.Response(202, json={"submission": "received"})
            raise AssertionError(f"Unexpected request: {request.method} {request.url}")

        client = DresClient(
            "https://eventretrieval.one",
            transport=httpx.MockTransport(handler),
        )
        payload = {
            "answerSets": [
                {"answers": [{"mediaItemName": "L24_V044", "start": 1000, "end": 1000}]}
            ]
        }

        async def exercise():
            user = await client.login("team_197", "test-password")
            evaluations = await client.list_evaluations(user["sessionId"])
            task = await client.current_task(user["sessionId"], "eval-1")
            status, result = await client.submit(
                user["sessionId"],
                "eval-1",
                payload,
            )
            return user, evaluations, task, status, result

        user, evaluations, task, status, result = asyncio.run(exercise())

        self.assertEqual(user["sessionId"], "secret-session")
        self.assertEqual(evaluations[0]["id"], "eval-1")
        self.assertEqual(task["taskType"], "KIS")
        self.assertEqual(status, 202)
        self.assertEqual(result["submission"], "received")
        self.assertEqual(seen[0].method, "POST")
        self.assertEqual(seen[0].url.path, "/api/v2/login")
        self.assertEqual(seen[1].url.params["session"], "secret-session")
        self.assertEqual(seen[3].url.params["session"], "secret-session")

    def test_dres_error_is_exposed_without_losing_status(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(412, json={"description": "Submission rejected"})

        client = DresClient(
            "https://eventretrieval.one",
            transport=httpx.MockTransport(handler),
        )

        async def exercise():
            await client.submit("session", "eval", {"answerSets": []})

        with self.assertRaises(DresApiError) as raised:
            asyncio.run(exercise())

        self.assertEqual(raised.exception.status_code, 412)
        self.assertEqual(str(raised.exception), "Submission rejected")

    def test_non_local_http_url_is_rejected(self):
        with self.assertRaises(ValueError):
            DresClient("http://example.com")


if __name__ == "__main__":
    unittest.main()
