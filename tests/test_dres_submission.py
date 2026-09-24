import unittest

from services.dres_submission import (
    build_dres_submission_payload,
    normalize_video_id,
)


class DresSubmissionPayloadTests(unittest.TestCase):
    def test_kis_payload_uses_video_basename_and_milliseconds(self):
        payload = build_dres_submission_payload(
            mode="kis",
            video_id="L24_V044.mp4",
            time_ms=7_520,
        )

        self.assertEqual(
            payload,
            {
                "answerSets": [
                    {
                        "answers": [
                            {
                                "mediaItemName": "L24_V044",
                                "start": 7_520,
                                "end": 7_520,
                            }
                        ]
                    }
                ]
            },
        )

    def test_qa_payload_preserves_vietnamese_answer(self):
        payload = build_dres_submission_payload(
            mode="qa",
            video_id="L24_V044",
            time_ms=1_250,
            answer="  màu đỏ  ",
        )

        self.assertEqual(
            payload,
            {"answerSets": [{"answers": [{"text": "QA-màu đỏ-L24_V044-1250"}]}]},
        )

    def test_trake_payload_keeps_semantic_order(self):
        payload = build_dres_submission_payload(
            mode="trake",
            video_id="L24_V044.mov",
            frame_ids=[12, 48, 103],
        )

        self.assertEqual(
            payload,
            {"answerSets": [{"answers": [{"text": "TR-L24_V044-12,48,103"}]}]},
        )

    def test_trake_rejects_duplicate_or_non_increasing_frames(self):
        for frames in ([12, 12], [48, 12]):
            with self.subTest(frames=frames):
                with self.assertRaisesRegex(ValueError, "TRAKE"):
                    build_dres_submission_payload(
                        mode="trake",
                        video_id="L24_V044",
                        frame_ids=frames,
                    )

    def test_invalid_video_id_and_time_are_rejected(self):
        with self.assertRaises(ValueError):
            normalize_video_id("../L24_V044.mp4")
        with self.assertRaises(ValueError):
            build_dres_submission_payload(
                mode="kis",
                video_id="L24_V044",
                time_ms=-1,
            )


if __name__ == "__main__":
    unittest.main()
