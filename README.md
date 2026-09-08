# Disease Outbreak Early Warning System — backend

A small Flask API backing the project's frontend. It stores case reports in
SQLite, and uses pandas + scikit-learn to turn those reports into geographic
clusters and anomaly alerts — the same two ideas the project's "Proposed
Methodology" describes (outbreak detection → automated alerts).

## What it actually does

- Stores case reports (location, zone, symptom, source) in a local SQLite
  file (`outbreak.db`), auto-created and seeded with 22 days of realistic
  mock data across five Navi Mumbai zones on first run.
- **Clustering** (`/api/clusters`): runs DBSCAN on report coordinates to find
  geographic hotspots.
- **Anomaly detection** (`/api/anomalies`): for each zone, compares the most
  recent day's report count to a 14-day baseline using a z-score, and flags
  the zone if it's more than 2 standard deviations above normal.
- **Alerts** (`/api/alerts`): plain-language alerts generated from flagged
  anomalies.
- The seed data gives the "Kamothe" zone a rising case count over the last
  week, so out of the box you'll see it flagged as an anomaly and its
  cluster marked `hot`.

## Run it

```bash
cd outbreak-backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

The API serves on `http://127.0.0.1:5000`. Leave it running, then open
`index.html` (the frontend) in your browser — it will detect the backend
automatically and switch from demo data to live data.

## Endpoints

| Method | Path              | Description                                   |
|--------|-------------------|------------------------------------------------|
| GET    | `/api/health`     | Liveness check                                 |
| GET    | `/api/reports`    | List reports (`?days=N`, default 21)           |
| POST   | `/api/reports`    | Add a report: `{lat, lng, zone, symptom}`      |
| GET    | `/api/clusters`   | DBSCAN clusters (`?days=N`, default 7)         |
| GET    | `/api/anomalies`  | Per-zone daily counts + z-score flags          |
| GET    | `/api/alerts`     | Alerts generated from flagged anomalies        |
| GET    | `/api/summary`    | Dashboard counters                             |

Example — submit a report:

```bash
curl -X POST http://127.0.0.1:5000/api/reports \
  -H "Content-Type: application/json" \
  -d '{"lat": 19.02, "lng": 73.09, "zone": "Kamothe", "symptom": "fever"}'
```

## Notes on scope

This is a working prototype of the detection logic, not a production
system — it's meant to demonstrate the pipeline described in the project
(collection → validation → detection → alerts → dashboard) end to end. To
harden it: add authentication/role-based access, input validation, a
production database (the project's slides propose MySQL/MongoDB) instead of
SQLite, and a real message/notification channel for `/api/alerts` instead of
returning JSON.
