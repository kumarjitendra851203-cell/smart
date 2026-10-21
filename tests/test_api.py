import io, os, sys, json, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import app

HDR = "timestamp,property_id,property_type,consumption_kwh\n"


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.c = app.test_client()

    def post(self, settings=None, csv=None):
        data = {"settings": json.dumps(settings or {})}
        if csv is not None:
            data["file"] = (io.BytesIO(csv.encode()), "x.csv")
        return self.c.post("/api/analyze", data=data, content_type="multipart/form-data")

    def test_index_and_sample(self):
        self.assertEqual(self.c.get("/").status_code, 200)
        self.assertEqual(self.c.get("/api/sample.csv").status_code, 200)

    def test_demo_analysis(self):
        j = self.post().get_json()
        self.assertEqual(j["meta"]["property"], "HOSTEL-DEMO-1")
        self.assertGreaterEqual(j["anomalies"]["count"], 3)
        self.assertEqual(len(j["forecast"]["fc"]["kwh"]), 30)
        self.assertEqual(len(j["overview"]["hourly"]), 24)

    def test_tiered_and_iforest(self):
        t = {"model": "tiered", "tiers": [{"upto": 1000, "rate": 4}, {"upto": 3000, "rate": 5.5}, {"upto": None, "rate": 7}]}
        j = self.post({"tariff": t, "method": "iforest"}).get_json()
        self.assertGreater(j["forecast"]["pred_bill"], 0)

    def test_bad_inputs(self):
        self.assertEqual(self.post(csv="a,b\n1,2\n").status_code, 400)
        self.assertEqual(self.post({"tariff": {"model": "tiered", "tiers": [{"upto": 5, "rate": 1}]}}).status_code, 400)

    def test_short_csv_reports_invalid_and_forecast_error(self):
        j = self.post(csv=HDR + "2025-01-01 00:00,A,home,1\nbad,A,home,1\n").get_json()
        self.assertEqual(j["meta"]["invalid_count"], 1)
        self.assertIn("error", j["forecast"])

if __name__ == "__main__":
    unittest.main()
