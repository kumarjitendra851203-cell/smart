"""Anomaly detection + forecasting. Heavy loop runs in C++ (ctypes); falls back to NumPy/pandas."""
import ctypes, os, subprocess, sys
import numpy as np
import pandas as pd
try:  # scikit-learn is optional: keeps serverless bundles small
    from sklearn.ensemble import IsolationForest
except ImportError:
    IsolationForest = None

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CPP = os.path.join(_ROOT, "cpp", "fast_analysis.cpp")
_EXT = ".dll" if sys.platform == "win32" else ".dylib" if sys.platform == "darwin" else ".so"
_LIB = os.path.join(_ROOT, "cpp", "libfast_analysis" + _EXT)
_P = ctypes.POINTER(ctypes.c_double)
_lib, ENGINE = None, "Python (NumPy fallback)"


def _load():
    global _lib, ENGINE
    try:
        if not os.path.exists(_LIB) or os.path.getmtime(_LIB) < os.path.getmtime(_CPP):
            subprocess.run(["g++", "-O2", "-shared", "-fPIC", "-o", _LIB, _CPP],
                           check=True, capture_output=True)
        lib = ctypes.CDLL(_LIB)
        lib.rolling_zscore.argtypes = [_P, ctypes.c_int, ctypes.c_int, _P]
        lib.rolling_zscore.restype = None
        _lib, ENGINE = lib, "C++ (fast engine)"
    except Exception:
        _lib = None


_load()


def rolling_z(x, window=168, force_python=False):
    x = np.ascontiguousarray(x, dtype=np.float64)
    if _lib is not None and not force_python:
        out = np.zeros(len(x))
        _lib.rolling_zscore(x.ctypes.data_as(_P), len(x), int(window), out.ctypes.data_as(_P))
        return out
    s = pd.Series(x).shift(1)
    m = s.rolling(window, min_periods=8).mean()
    sd = s.rolling(window, min_periods=8).std(ddof=0).clip(lower=1e-3)
    return ((pd.Series(x) - m) / sd).fillna(0).to_numpy()


def detect_anomalies(df, method="zscore", threshold=4.0, window=168, contamination=0.005):
    """df: one property. Expected = median for that hour of day. Returns per-hour frame."""
    d = df.sort_values("timestamp").reset_index(drop=True)[["timestamp", "consumption_kwh"]].copy()
    hour = d["timestamp"].dt.hour
    d["expected"] = d.groupby(hour)["consumption_kwh"].transform("median")
    d["resid"] = d["consumption_kwh"] - d["expected"]
    d["z"] = rolling_z(d["resid"].values, window)
    if method == "zscore":
        d["is_anomaly"] = d["z"].abs() > threshold
    elif len(d) < 50:
        d["is_anomaly"] = False
    elif IsolationForest is not None:
        X = np.c_[d["resid"].values, hour.values]
        d["is_anomaly"] = IsolationForest(contamination=contamination, random_state=0).fit_predict(X) == -1
    else:  # fallback when scikit-learn is not installed: top share of largest deviations
        a = d["resid"].abs()
        d["is_anomaly"] = a >= a.quantile(1 - contamination)
    return d


def group_events(d):
    cols = ["start", "end", "hours", "type", "extra_kwh", "peak_kwh"]
    f = d[d["is_anomaly"]]
    if f.empty:
        return pd.DataFrame(columns=cols)
    gid = (f.index.to_series().diff() != 1).cumsum().values
    ev = f.groupby(gid).agg(start=("timestamp", "min"), end=("timestamp", "max"),
                            hours=("timestamp", "count"), extra_kwh=("resid", "sum"),
                            peak_kwh=("consumption_kwh", "max")).reset_index(drop=True)
    ev["type"] = np.where(ev["extra_kwh"] >= 0, "High usage", "Unusually low")
    ev[["extra_kwh", "peak_kwh"]] = ev[["extra_kwh", "peak_kwh"]].round(2)
    return ev[cols]


def _ridge(X, y, alpha=1.0):
    Xb = np.c_[np.ones(len(X)), X]
    P = np.eye(Xb.shape[1]) * alpha
    P[0, 0] = 0  # intercept not penalised
    w = np.linalg.solve(Xb.T @ Xb + P, Xb.T @ y)
    return lambda Z: np.c_[np.ones(len(Z)), Z] @ w


def forecast_daily(daily, horizon=30):
    """Ridge regression on trend + day-of-week. Incomplete days are ignored. Returns (history, forecast)."""
    d = daily.sort_values("date").copy()
    d = d[d["readings"] >= 0.8 * d["readings"].max()]
    if len(d) < 14:
        raise ValueError("Forecast needs at least 14 complete days of data.")
    t0 = d["date"].min()

    def feats(dates):
        dates = pd.Series(pd.to_datetime(dates)).reset_index(drop=True)
        t = (dates - t0).dt.days.to_numpy().reshape(-1, 1)
        dow = pd.get_dummies(dates.dt.dayofweek).reindex(columns=range(7), fill_value=0).astype(float).to_numpy()
        return np.hstack([t, dow])

    predict = _ridge(feats(d["date"]), d["kwh"].values)
    sd = float(np.std(d["kwh"].values - predict(feats(d["date"]))))
    fd = pd.date_range(d["date"].max() + pd.Timedelta(days=1), periods=horizon)
    pred = np.clip(predict(feats(fd)), 0, None)
    fc = pd.DataFrame({"date": fd, "kwh": pred, "low": np.clip(pred - 1.96 * sd, 0, None), "high": pred + 1.96 * sd})
    return d.reset_index(drop=True), fc
