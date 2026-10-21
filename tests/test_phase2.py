import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from src import data_manager as dm, analysis as an


class AnalysisTests(unittest.TestCase):
    def test_cpp_matches_python(self):
        x = np.random.default_rng(1).normal(0, 1, 500)
        np.testing.assert_allclose(an.rolling_z(x), an.rolling_z(x, force_python=True), atol=1e-6)

    def test_finds_synthetic_events(self):
        df, ev = dm.generate_sample()
        e = an.group_events(an.detect_anomalies(df))
        self.assertGreaterEqual(len(e), 3)
        for known in ev:
            s = np.datetime64(known["start"])
            self.assertTrue(((e["start"].values <= s + np.timedelta64(6, "h")) & (e["end"].values >= s)).any())

    def test_isolation_forest_runs(self):
        df, _ = dm.generate_sample()
        self.assertTrue(an.detect_anomalies(df, method="iforest")["is_anomaly"].any())

    def test_forecast(self):
        df, _ = dm.generate_sample()
        h, fc = an.forecast_daily(dm.daily_totals(df), 30)
        self.assertEqual(len(fc), 30); self.assertTrue((fc.kwh > 0).all())
        with self.assertRaises(ValueError):
            an.forecast_daily(dm.daily_totals(dm.generate_sample(days=5)[0]))

if __name__ == "__main__":
    unittest.main()
