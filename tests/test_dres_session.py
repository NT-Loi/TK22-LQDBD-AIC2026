import unittest

from services.dres_session import DresSessionStore


class DresSessionStoreTests(unittest.TestCase):
    def test_session_selection_and_duplicate_fingerprint(self):
        store = DresSessionStore()
        created = store.create(
            "dres-secret",
            {"username": "team_197"},
            [{"id": "eval-1", "status": "ACTIVE"}],
        )

        self.assertEqual(store.get(created.local_id).dres_session_id, "dres-secret")
        store.select_evaluation(created.local_id, "eval-1")
        self.assertEqual(store.get(created.local_id).selected_evaluation_id, "eval-1")
        self.assertFalse(store.has_fingerprint(created.local_id, "fingerprint"))
        store.record_fingerprint(created.local_id, "fingerprint")
        self.assertTrue(store.has_fingerprint(created.local_id, "fingerprint"))

        deleted = store.delete(created.local_id)
        self.assertEqual(deleted.dres_session_id, "dres-secret")
        self.assertIsNone(store.get(created.local_id))


if __name__ == "__main__":
    unittest.main()
