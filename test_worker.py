import unittest
from worker import build_processing_plan, create_report

class TestWorkerPipeline(unittest.TestCase):
    def test_build_processing_plan_high_percussive(self):
        analysis = {
            "percussive_ratio": 0.70,
            "harmonic_ratio": 0.30,
            "tempo_bpm": 120.0,
            "peak_dbfs": -1.0,
            "rms_dbfs": -12.0,
        }
        plan = build_processing_plan(analysis, ["drums", "bass"])
        self.assertEqual(plan["demucs"]["overlap"], 0.40)
        self.assertEqual(plan["demucs"]["shifts"], 1)
        self.assertIn("drums", plan["midi"])
        self.assertIn("bass", plan["midi"])
        self.assertAlmostEqual(plan["midi"]["drums"]["onset_threshold"], 0.35 + 0.70 * 0.15)

    def test_build_processing_plan_high_harmonic_low_dynamic(self):
        analysis = {
            "percussive_ratio": 0.20,
            "harmonic_ratio": 0.80,
            "tempo_bpm": 95.0,
            "peak_dbfs": -2.0,
            "rms_dbfs": -6.0, # dynamic range is 4 (which is < 8)
        }
        plan = build_processing_plan(analysis, ["piano", "guitar"])
        self.assertEqual(plan["demucs"]["overlap"], 0.30)
        self.assertEqual(plan["demucs"]["shifts"], 2)
        self.assertIn("piano", plan["midi"])
        self.assertIn("guitar", plan["midi"])

    def test_create_report(self):
        analysis = {
            "duration_seconds": 120.5,
            "sample_rate": 44100,
            "peak_dbfs": -0.5,
            "rms_dbfs": -14.2,
            "tempo_bpm": 115.0,
            "harmonic_ratio": 0.60,
            "percussive_ratio": 0.40,
        }
        plan = {
            "demucs": {
                "model": "htdemucs_6s",
                "overlap": 0.35,
                "shifts": 1,
            }
        }
        report = create_report(analysis, plan)
        self.assertIn("StemForge Processing Report", report)
        self.assertIn("120.5 seconds", report)
        self.assertIn("htdemucs_6s", report)

if __name__ == "__main__":
    unittest.main()
