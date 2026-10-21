"""Smart Energy Analyzer - Flask backend. Exports `app` (Vercel / Render / gunicorn compatible).
Local run:  python app.py   ->  http://127.0.0.1:5000
"""
import functools, io, json, os, sys
import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, request, send_from_directory

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from src import data_manager as dm, analysis as an
from src.tariff_calculator import Tariff, estimate_bill, DISCLAIMER

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024  # serverless request-body limit is ~4.5 MB


def _json_default(o):
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, pd.Timestamp): return o.isoformat()
    raise TypeError(type(o))


app.json.default = _json_default


@functools.lru_cache(maxsize=1)
def sample():
    return dm.generate_sample()


def make_tariff(t):
    t = t or {}
    common = dict(currency=str(t.get("currency") or "\u20b9"), fixed_monthly_charge=float(t.get("fixed") or 0),
                  additional_percent=float(t.get("extra") or 0))
    if t.get("model") == "tiered":
        tr = [{"upto": float(x["upto"]) if x.get("upto") is not None else None, "rate": float(x["rate"])}
              for x in t.get("tiers", [])]
        tariff = Tariff(tiers=tr, **common)
    else:
        tariff = Tariff(price_per_kwh=float(t.get("price", 6.0)), **common)
    tariff.validate()
    return tariff


def r(x, n=2):
    return None if x is None or pd.isna(x) else round(float(x), n)


def build(df, invalid, s, tariff, known_events):
    props = sorted(df["property_id"].unique())
    prop = s.get("property") if s.get("property") in props else props[0]
    d = df[df["property_id"] == prop]
    daily = dm.daily_totals(d)
    last30 = float(daily[daily["date"] > daily["date"].max() - pd.Timedelta(days=30)]["kwh"].sum())
    hourly = d.groupby(d["timestamp"].dt.hour)["consumption_kwh"].mean().reindex(range(24)).fillna(0)
    heat = (d.assign(h=d["timestamp"].dt.hour, w=d["timestamp"].dt.dayofweek)
              .pivot_table(index="w", columns="h", values="consumption_kwh")
              .reindex(index=range(7), columns=range(24)))
    overview = {"total": r(d["consumption_kwh"].sum()), "avg_daily": r(daily["kwh"].mean(), 1),
                "peak_hour": int(hourly.idxmax()), "last30_kwh": r(last30), "last30_bill": r(estimate_bill(last30, tariff)["total"], 0),
                "incomplete_days": int((daily["readings"] < daily["readings"].max()).sum()),
                "daily": {"d": daily["date"].dt.strftime("%Y-%m-%d").tolist(), "kwh": daily["kwh"].round(2).tolist()},
                "hourly": hourly.round(3).tolist(),
                "heat": [[r(v, 3) for v in row] for row in heat.values]}

    method = "iforest" if s.get("method") == "iforest" else "zscore"
    res = an.detect_anomalies(d, method, threshold=float(s.get("threshold", 4.0)),
                              contamination=min(max(float(s.get("contamination", 0.005)), 0.001), 0.05))
    ev = an.group_events(res)
    rate = tariff.tiers[-1]["rate"] if tariff.tiers else tariff.price_per_kwh
    extra_kwh = float(ev.loc[ev["type"] == "High usage", "extra_kwh"].sum()) if len(ev) else 0.0
    tail = res.tail(5000).reset_index(drop=True)
    evo = ev.assign(start=ev["start"].dt.strftime("%Y-%m-%d %H:%M"), end=ev["end"].dt.strftime("%Y-%m-%d %H:%M")) \
        if len(ev) else ev
    anomalies = {"events": evo.to_dict("records"), "count": len(ev), "extra_kwh": r(extra_kwh, 1),
                 "extra_cost": r(extra_kwh * rate, 0),
                 "series": {"t": tail["timestamp"].dt.strftime("%Y-%m-%d %H:%M").tolist(),
                            "kwh": tail["consumption_kwh"].round(3).tolist(),
                            "anom": [int(i) for i in tail.index[tail["is_anomaly"]]]}}

    try:
        hist, fc = an.forecast_daily(daily, int(min(max(s.get("horizon", 30), 30), 90)))
        month = fc.head(30)
        pb, ob = estimate_bill(month["kwh"].sum(), tariff), estimate_bill(last30, tariff)
        forecast = {"hist": {"d": hist["date"].dt.strftime("%Y-%m-%d").tolist(), "kwh": hist["kwh"].round(2).tolist()},
                    "fc": {"d": fc["date"].dt.strftime("%Y-%m-%d").tolist(), "kwh": fc["kwh"].round(2).tolist(),
                           "low": fc["low"].round(2).tolist(), "high": fc["high"].round(2).tolist()},
                    "pred_kwh": r(month["kwh"].sum(), 0), "pred_bill": r(pb["total"], 0), "prev_bill": r(ob["total"], 0),
                    "low_bill": r(estimate_bill(month["low"].sum(), tariff)["total"], 0),
                    "high_bill": r(estimate_bill(month["high"].sum(), tariff)["total"], 0),
                    "breakdown": {"Energy charge": r(pb["energy_charge"]), "Fixed charge": r(pb["fixed_charge"]),
                                  "Taxes / surcharge": r(pb["additional_charge"]), "Total": r(pb["total"])}}
    except ValueError as e:
        forecast = {"error": str(e)}

    meta = {"engine": an.ENGINE, "ml": "scikit-learn" if an.IsolationForest else "NumPy fallback",
            "property": prop, "properties": props, "rows": len(df), "invalid": invalid[:200], "invalid_count": len(invalid),
            "known_events": known_events, "disclaimer": DISCLAIMER}
    return {"currency": tariff.currency, "meta": meta, "overview": overview, "anomalies": anomalies, "forecast": forecast}


@app.get("/")
def index():
    return send_from_directory(os.path.join(ROOT, "static"), "index.html")


@app.get("/api/sample.csv")
def sample_csv():
    return Response(sample()[0].to_csv(index=False), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=sample_electricity.csv"})


@app.post("/api/analyze")
def analyze():
    try:
        s = json.loads(request.form.get("settings") or "{}")
        tariff = make_tariff(s.get("tariff"))
    except (ValueError, KeyError, TypeError) as e:
        return jsonify(error=f"Invalid settings / tariff: {e}"), 400
    up = request.files.get("file")
    try:
        if up and up.filename:
            df, invalid = dm.validate_dataframe(pd.read_csv(io.BytesIO(up.read()), comment="#"))
            known = []
        else:
            df, known = sample()
            invalid = []
        if df.empty:
            raise dm.DataError("No valid rows found in the file.")
    except dm.DataError as e:
        return jsonify(error=str(e)), 400
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as e:
        return jsonify(error=f"Could not read CSV: {e}"), 400
    out = build(df, invalid, s, tariff, known)
    out["meta"]["demo"] = not (up and up.filename)
    return jsonify(out)


@app.errorhandler(413)
def too_big(_):
    return jsonify(error="File too large (max 4 MB for this deployment)."), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)))
