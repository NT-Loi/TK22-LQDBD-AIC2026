import asyncio
import unittest

import httpx

from services.dres_client import DresApiError, DresClient, normalize_submission_response


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
                return httpx.Response(
                    202,
                    json={
                        "status": True,
                        "submission": "INDETERMINATE",
                        "description": "Accepted, verdict pending",
                    },
                )
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
            receipt = await client.submit(
                user["sessionId"],
                "eval-1",
                payload,
            )
            return user, evaluations, task, receipt

        user, evaluations, task, receipt = asyncio.run(exercise())

        self.assertEqual(user["sessionId"], "secret-session")
        self.assertEqual(evaluations[0]["id"], "eval-1")
        self.assertEqual(task["taskType"], "KIS")
        self.assertEqual(receipt["dresHttpStatus"], 202)
        self.assertTrue(receipt["accepted"])
        self.assertTrue(receipt["pending"])
        self.assertIsNone(receipt["verdict"])
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

    def test_all_documented_submit_errors_preserve_status_and_description(self):
        for status_code in (400, 401, 404, 412):
            with self.subTest(status_code=status_code):
                def handler(request: httpx.Request) -> httpx.Response:
                    return httpx.Response(
                        status_code,
                        json={"status": False, "description": f"DRES error {status_code}"},
                    )

                client = DresClient(
                    "https://eventretrieval.one",
                    transport=httpx.MockTransport(handler),
                )

                async def exercise():
                    await client.submit("session", "eval", {"answerSets": []})

                with self.assertRaises(DresApiError) as raised:
                    asyncio.run(exercise())
                self.assertEqual(raised.exception.status_code, status_code)
                self.assertEqual(str(raised.exception), f"DRES error {status_code}")

    def test_login_without_session_id_is_rejected(self):
        client = DresClient(
            "https://eventretrieval.one",
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json={"username": "team_197"})
            ),
        )

        with self.assertRaises(DresApiError):
            asyncio.run(client.login("team_197", "password"))

    def test_non_local_http_url_is_rejected(self):
        with self.assertRaises(ValueError):
            DresClient("http://example.com")

    def test_all_official_verdicts_are_normalized(self):
        for verdict in ("CORRECT", "WRONG", "INDETERMINATE", "UNDECIDABLE"):
            with self.subTest(verdict=verdict):
                result = normalize_submission_response(
                    200,
                    {
                        "status": True,
                        "submission": verdict,
                        "description": f"Verdict: {verdict}",
                        "sessionId": "must-not-reach-the-browser",
                    },
                )
                self.assertTrue(result["accepted"])
                self.assertFalse(result["pending"])
                self.assertEqual(result["verdict"], verdict)
                self.assertNotIn("sessionId", result["rawResult"])

    def test_false_status_or_malformed_success_is_rejected(self):
        invalid_responses = [
            {"status": False, "submission": "CORRECT", "description": "Rejected"},
            {"status": True, "submission": "UNKNOWN", "description": "Bad verdict"},
            {"status": True, "submission": "CORRECT"},
            "not-an-object",
        ]
        for response in invalid_responses:
            with self.subTest(response=response):
                with self.assertRaises(DresApiError):
                    normalize_submission_response(200, response)

    def test_masked_submission_url_never_contains_session(self):
        client = DresClient("https://eventretrieval.one")
        destination = client.masked_submission_url("eval id")
        self.assertEqual(
            destination,
            "https://eventretrieval.one/api/v2/submit/eval%20id?session=<SESSION_ID_ẨN>",
        )


if __name__ == "__main__":
    unittest.main()
