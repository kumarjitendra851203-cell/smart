"""CSV loading, validation, saving, aggregation and synthetic sample data.
Consumption must be ENERGY in kWh per reading interval - not instantaneous power in kW.
"""
import os, shutil, time
import numpy as np
import pandas as pd

REQUIRED = ["timestamp", "property_id", "property_type", "consumption_kwh"]
OPTIONAL = ["occupancy", "temperature_c", "appliance_category"]


class DataError(Exception):
    pass


def validate_dataframe(raw: pd.DataFrame):
    """Return (valid_df, invalid_rows). invalid_rows: list of {row, reason}.
    Row numbers are CSV line numbers (header = line 1). Nothing is silently fixed."""
    raw = raw.copy()
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    missing = [c for c in REQUIRED if c not in raw.columns]
    if missing:
        raise DataError("Missing required column(s): " + ", ".join(missing))
    if raw.empty:
        raise DataError("The file contains no data rows.")
    raw["_row"] = np.arange(2, len(raw) + 2)
    reasons = {i: [] for i in raw.index}

    ts = pd.to_datetime(raw["timestamp"], errors="coerce")
    kwh = pd.to_numeric(raw["consumption_kwh"], errors="coerce")
    for i in raw.index:
        if pd.isna(raw.at[i, "timestamp"]) or pd.isna(ts[i]):
            reasons[i].append("unparseable or missing timestamp")
        if pd.isna(raw.at[i, "consumption_kwh"]) or pd.isna(kwh[i]):
            reasons[i].append("missing or non-numeric consumption_kwh")
        elif kwh[i] < 0:
            reasons[i].append("negative consumption_kwh")
        for c in ("property_id", "property_type"):
            if pd.isna(raw.at[i, c]) or str(raw.at[i, c]).strip() == "":
                reasons[i].append(f"missing {c}")
    raw["timestamp"], raw["consumption_kwh"] = ts, kwh
    ok_mask = pd.Series([not reasons[i] for i in raw.index], index=raw.index)
    dup = raw[ok_mask].duplicated(subset=["property_id", "timestamp"], keep="first")
    for i in dup[dup].index:
        reasons[i].append("duplicate timestamp for this property")
    ok_mask = pd.Series([not reasons[i] for i in raw.index], index=raw.index)
    invalid = [{"row": int(raw.at[i, "_row"]), "reason": "; ".join(reasons[i])}
               for i in raw.index if reasons[i]]
    valid = raw[ok_mask].drop(columns="_row").sort_values(["property_id", "timestamp"]).reset_index(drop=True)
    return valid, invalid


def load_csv(path: str):
    try:
        raw = pd.read_csv(path, comment="#")
    except FileNotFoundError:
        raise DataError(f"File not found: {path}")
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as e:
        raise DataError(f"Could not read CSV: {e}")
    return validate_dataframe(raw)


def save_csv(df: pd.DataFrame, path: str, backup: bool = True):
    if backup and os.path.exists(path):
        shutil.copy2(path, f"{path}.{time.strftime('%Y%m%d_%H%M%S')}.bak")
    df.to_csv(path, index=False)


def daily_totals(df: pd.DataFrame) -> pd.DataFrame:
    """Daily kWh per property. 'readings' shows how many readings made each day, so
    incomplete days are visible; missing days are NOT filled with zero."""
    if df.empty:
        return pd.DataFrame(columns=["property_id", "date", "kwh", "readings"])
    d = df.assign(date=df["timestamp"].dt.normalize())
    return (d.groupby(["property_id", "date"])["consumption_kwh"]
              .agg(kwh="sum", readings="count").reset_index())


def generate_sample(days: int = 90, seed: int = 42, start: str = "2025-01-01"):
    """SYNTHETIC demo hostel data (not real meter data). Returns (df, events)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=days * 24, freq="h")
    hour, dow = idx.hour.values, idx.dayofweek.values
    occ = np.where((hour >= 22) | (hour < 6), 120, np.where((hour >= 9) & (hour < 17), 50, 100))
    occ = np.where(dow >= 5, occ * 0.8, occ) + rng.normal(0, 4, len(idx))
    occ = np.clip(occ, 0, 150).round()
    temp = 27 + 6 * np.sin((hour - 9) / 24 * 2 * np.pi) + rng.normal(0, 1, len(idx))
    kwh = 3.0 + 0.045 * occ + 0.08 * np.clip(temp - 28, 0, None) + rng.normal(0, 0.25, len(idx))
    events = []
    for day in (20, 45, 70):  # documented synthetic high-consumption events
        s = day * 24 + 1
        if s + 5 > len(idx):  # skip events outside a short dataset
            continue
        kwh[s:s + 5] += 6.0
        events.append({"start": str(idx[s]), "hours": 5, "extra_kwh_per_hour": 6.0,
                       "note": "synthetic overnight equipment left running"})
    df = pd.DataFrame({"timestamp": idx, "property_id": "HOSTEL-DEMO-1", "property_type": "hostel",
                       "consumption_kwh": kwh.clip(0).round(3), "occupancy": occ,
                       "temperature_c": temp.round(1)})
    return df, events


if __name__ == "__main__":
    here = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    df, ev = generate_sample()
    save_csv(df, os.path.join(here, "sample_electricity.csv"), backup=False)
    pd.DataFrame(ev).to_csv(os.path.join(here, "sample_events_SYNTHETIC.csv"), index=False)
    print(f"Wrote {len(df)} synthetic hourly rows and {len(ev)} documented events to {here}")
