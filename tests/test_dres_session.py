import unittest

from services.dres_session import DresSessionStore


class DresSessionStoreTests(unittest.TestCase):
    def test_session_selection_and_short_duplicate_window(self):
        store = DresSessionStore()
        created = store.create(
            "dres-secret",
            {"username": "team_197"},
            [{"id": "eval-1", "status": "ACTIVE"}],
        )

        self.assertEqual(store.get(created.local_id).dres_session_id, "dres-secret")
        store.select_evaluation(created.local_id, "eval-1")
        self.assertEqual(store.get(created.local_id).selected_evaluation_id, "eval-1")
        self.assertTrue(store.begin_submission(created.local_id, "fingerprint"))
        self.assertFalse(store.begin_submission(created.local_id, "fingerprint"))
        store.finish_submission(created.local_id, "fingerprint", accepted=False)
        self.assertTrue(store.begin_submission(created.local_id, "fingerprint"))
        store.finish_submission(created.local_id, "fingerprint", accepted=True)
        self.assertFalse(store.begin_submission(created.local_id, "fingerprint"))
        self.assertTrue(
            store.begin_submission(
                created.local_id,
                "fingerprint",
                duplicate_window_seconds=0,
            )
        )

        deleted = store.delete(created.local_id)
        self.assertEqual(deleted.dres_session_id, "dres-secret")
        self.assertIsNone(store.get(created.local_id))


if __name__ == "__main__":
    unittest.main()
