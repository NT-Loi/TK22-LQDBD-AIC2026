import unittest
from utils.pts_mapper import PTSMapper, get_video_pts_time, get_video_time_ms, annotate_result_item


class TestPTSMapper(unittest.TestCase):
    def setUp(self):
        self.mapper = PTSMapper()

    def test_cfr_fallback(self):
        # A video that has no map file should fallback to frame_idx / fps
        pts = self.mapper.get_pts_time("UNKNOWN_VID", 50, default_fps=25.0)
        self.assertEqual(pts, 2.0)
        time_ms = self.mapper.get_time_ms("UNKNOWN_VID", 50, default_fps=25.0)
        self.assertEqual(time_ms, 2000)

    def test_map_anchor_points(self):
        # Test N001-V001 frame 0 and 50
        pts0 = self.mapper.get_pts_time("N001-V001", 0)
        self.assertAlmostEqual(pts0, 0.0, places=3)
        pts50 = self.mapper.get_pts_time("N001-V001", 50)
        self.assertAlmostEqual(pts50, 2.0, places=3)

    def test_map_interpolation(self):
        # Test interpolation between frame 0 and 50 in N001-V001 (frame 25 should be 1.0s)
        pts25 = self.mapper.get_pts_time("N001-V001", 25)
        self.assertAlmostEqual(pts25, 1.0, places=2)
        ms25 = self.mapper.get_time_ms("N001-V001", 25)
        self.assertEqual(ms25, 1000)

    def test_video_id_variants(self):
        # Hyphen vs underscore
        ms1 = self.mapper.get_time_ms("N001-V001", 50)
        ms2 = self.mapper.get_time_ms("N001_V001", 50)
        ms3 = self.mapper.get_time_ms("N001-V001.mp4", 50)
        self.assertEqual(ms1, ms2)
        self.assertEqual(ms1, ms3)

    def test_annotate_result_item(self):
        item = {
            "video_id": "N001-V001",
            "keyframe_index": 50,
            "score": 0.9,
            "frames": [{"keyframe_index": 25}],
        }
        res = annotate_result_item(item)
        self.assertIn("fps", res)
        self.assertIn("timeMs", res)
        self.assertEqual(res["timeMs"], 2000)
        self.assertEqual(res["frames"][0]["timeMs"], 1000)


if __name__ == "__main__":
    unittest.main()
