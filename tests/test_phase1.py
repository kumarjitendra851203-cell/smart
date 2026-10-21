import os, sys, tempfile, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from src import data_manager as dm
from src.tariff_calculator import Tariff, estimate_bill


def write(text):
    f = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False)
    f.write(text); f.close(); return f.name

HDR = "timestamp,property_id,property_type,consumption_kwh\n"


class DataTests(unittest.TestCase):
    def test_load_valid(self):
        v, bad = dm.load_csv(write(HDR + "2025-01-01 00:00,A,home,1.5\n2025-01-01 01:00,A,home,2.5\n"))
        self.assertEqual(len(v), 2); self.assertEqual(bad, [])
        self.assertAlmostEqual(v.consumption_kwh.sum(), 4.0)

    def test_missing_columns(self):
        with self.assertRaises(dm.DataError):
            dm.load_csv(write("timestamp,consumption_kwh\n2025-01-01,1\n"))

    def test_invalid_rows_reported_with_row_numbers(self):
        v, bad = dm.load_csv(write(HDR + "2025-01-01 00:00,A,home,1\nnot-a-date,A,home,1\n"
                                   "2025-01-01 02:00,A,home,-3\n2025-01-01 03:00,A,home,abc\n"))
        self.assertEqual(len(v), 1)
        self.assertEqual([b["row"] for b in bad], [3, 4, 5])

    def test_duplicates_flagged_first_kept(self):
        v, bad = dm.load_csv(write(HDR + "2025-01-01 00:00,A,home,1\n2025-01-01 00:00,A,home,9\n"))
        self.assertEqual(len(v), 1); self.assertEqual(v.consumption_kwh[0], 1)
        self.assertIn("duplicate", bad[0]["reason"])

    def test_daily_aggregation_and_no_zero_fill(self):
        v, _ = dm.load_csv(write(HDR + "2025-01-01 00:00,A,home,1\n2025-01-01 12:00,A,home,2\n2025-01-03 00:00,A,home,4\n"))
        d = dm.daily_totals(v)
        self.assertEqual(list(d.kwh), [3, 4]); self.assertEqual(len(d), 2)

    def test_empty(self):
        with self.assertRaises(dm.DataError):
            dm.load_csv(write(HDR))
        self.assertTrue(dm.daily_totals(pd.DataFrame()).empty)

    def test_sample_reproducible_and_valid(self):
        a, ev = dm.generate_sample(); b, _ = dm.generate_sample()
        self.assertEqual(len(a), 90 * 24); self.assertTrue(a.equals(b)); self.assertEqual(len(ev), 3)
        v, bad = dm.validate_dataframe(a.astype({"timestamp": str}))
        self.assertEqual(bad, []); self.assertEqual(len(v), len(a))

    def test_save_creates_backup(self):
        d = tempfile.mkdtemp(); p = os.path.join(d, "x.csv")
        df, _ = dm.generate_sample(days=1)
        dm.save_csv(df, p); dm.save_csv(df, p)
        self.assertTrue(any(f.endswith(".bak") for f in os.listdir(d)))


class TariffTests(unittest.TestCase):
    def test_flat(self):
        self.assertAlmostEqual(estimate_bill(150, Tariff(price_per_kwh=6.0))["total"], 900.0)

    def test_tiered(self):  # 100*3 + 100*4.5 + 50*6 = 1050
        t = Tariff(tiers=[{"upto": 100, "rate": 3}, {"upto": 200, "rate": 4.5}, {"upto": None, "rate": 6}])
        self.assertAlmostEqual(estimate_bill(250, t)["total"], 1050.0)
        self.assertAlmostEqual(estimate_bill(80, t)["total"], 240.0)

    def test_fixed_and_percent(self):  # (100*5 + 50) * 1.10 = 605
        t = Tariff(price_per_kwh=5, fixed_monthly_charge=50, additional_percent=10)
        self.assertAlmostEqual(estimate_bill(100, t)["total"], 605.0)

    def test_invalid(self):
        with self.assertRaises(ValueError): estimate_bill(10, Tariff(price_per_kwh=-1))
        with self.assertRaises(ValueError): estimate_bill(-5, Tariff())
        with self.assertRaises(ValueError):
            estimate_bill(10, Tariff(tiers=[{"upto": 100, "rate": 3}]))

if __name__ == "__main__":
    unittest.main()
