# Smart Energy Analyzer

Python (Flask backend, analysis, ML) + C++ (fast rolling z-score engine) + thin HTML/JS frontend (Chart.js).

## Run locally
    pip install -r requirements.txt
    python app.py            # http://127.0.0.1:5000

## Deploy
- **Vercel**: push repo, import it. `app.py` exports `app` (Flask is auto-detected).
  C++ is not compiled there, so the NumPy fallback runs automatically (same results).
  If the build fails on bundle size (250 MB), delete `scikit-learn` from requirements.txt -
  Isolation Forest then falls back to a simple outlier rule; everything else is unchanged.
  Upload limit: ~4 MB CSV.
- **Render / Railway / Heroku-style**: uses `Procfile` (`gunicorn app:app`). g++ available -> C++ engine used.

## API
`POST /api/analyze` (multipart: optional `file` CSV, `settings` JSON) -> overview, anomalies, forecast, data-quality JSON.
`GET /api/sample.csv` -> demo CSV.

## Tests
    python -m unittest discover -s tests

Bills are estimates only. CSV consumption must be kWh per interval (not kW).
