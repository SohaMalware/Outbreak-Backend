"""
Disease Outbreak Early Warning System — backend API.

Endpoints:
  GET  /api/health      -> liveness check
  GET  /api/reports      -> list case reports (?days=N)
  POST /api/reports      -> submit a new case report
  GET  /api/clusters     -> DBSCAN geographic clustering of recent reports (?days=N)
  GET  /api/anomalies    -> per-zone daily counts + z-score anomaly flags
  GET  /api/alerts       -> alerts generated from flagged anomalies
  GET  /api/summary      -> counters for the dashboard header

Run:
  pip install -r requirements.txt
  python app.py
  -> serves on http://127.0.0.1:5000
"""

import os
import random
import sqlite3
from datetime import datetime, timedelta

import pandas as pd
from flask import Flask, g, jsonify, request
from sklearn.cluster import DBSCAN

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outbreak.db")

app = Flask(__name__)

# ---------------------------------------------------------------------------
# CORS (manual — avoids requiring the flask-cors package). The frontend is a
# static HTML file opened directly in the browser, so it needs permissive
# CORS to call this API from a different origin.
# ---------------------------------------------------------------------------
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


@app.route("/api/<path:_any>", methods=["OPTIONS"])
def cors_preflight(_any):
    return "", 204


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(_exc):
    db = g.pop("db", None)
    if db is not None:
        db.close()


ZONES = {
    "Kamothe": (19.0176, 73.0961),
    "Kharghar": (19.0474, 73.0662),
    "Panvel": (18.9894, 73.1175),
    "Belapur": (19.0234, 73.0356),
    "Vashi": (19.0771, 73.0000),
}
SYMPTOMS = ["fever", "cough", "diarrhea", "rash", "respiratory distress"]


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            lat REAL NOT NULL,
            lng REAL NOT NULL,
            zone TEXT NOT NULL,
            symptom TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT 'citizen',
            reported_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    count = conn.execute("SELECT COUNT(*) AS c FROM reports").fetchone()[0]
    if count == 0:
        seed(conn)
    conn.close()


def seed(conn):
    """Populate 22 days of mock reports across five zones, with Kamothe
    developing a clear outbreak signal over the last week — this is what
    lets /api/clusters and /api/anomalies have something real to detect."""
    random.seed(42)
    today = datetime.utcnow().date()
    rows = []
    for day_offset in range(21, -1, -1):
        date = today - timedelta(days=day_offset)
        for zone, (lat, lng) in ZONES.items():
            base = random.randint(2, 6)
            if zone == "Kamothe" and day_offset <= 6:
                base += (7 - day_offset) * 3
            for _ in range(base):
                jlat = lat + random.uniform(-0.01, 0.01)
                jlng = lng + random.uniform(-0.01, 0.01)
                rows.append(
                    (
                        jlat,
                        jlng,
                        zone,
                        random.choice(SYMPTOMS),
                        random.choice(["citizen", "citizen", "health_worker"]),
                        date.isoformat(),
                    )
                )
    conn.executemany(
        "INSERT INTO reports (lat, lng, zone, symptom, source, reported_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Analytics helpers (plain Python return values — routes wrap these in jsonify)
# ---------------------------------------------------------------------------
def fetch_reports_df(days=None):
    conn = get_db()
    if days is not None:
        cutoff = (datetime.utcnow().date() - timedelta(days=days)).isoformat()
        rows = conn.execute(
            "SELECT * FROM reports WHERE reported_at >= ? ORDER BY reported_at", (cutoff,)
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM reports ORDER BY reported_at").fetchall()
    return pd.DataFrame([dict(r) for r in rows])


def compute_clusters(days=7):
    df = fetch_reports_df(days)
    if df.empty:
        return []
    coords = df[["lat", "lng"]].to_numpy()
    # eps ~0.015 degrees ≈ 1.5km — tuned for city-district-scale clustering
    labels = DBSCAN(eps=0.015, min_samples=6).fit(coords).labels_
    df["cluster"] = labels

    # A cluster's geography alone doesn't say whether it's *unusual* — that
    # comes from the anomaly detector. A cluster is "hot" if its zone is
    # currently reporting well above its own baseline, not just because it
    # has a lot of raw reports (dense zones would always trip a flat count).
    flagged_zones = {a["zone"] for a in compute_anomalies() if a["flagged"]}

    results = []
    for label, group in df[df["cluster"] != -1].groupby("cluster"):
        zone = group["zone"].mode().iat[0]
        results.append(
            {
                "cluster_id": int(label),
                "zone": zone,
                "center_lat": round(float(group["lat"].mean()), 5),
                "center_lng": round(float(group["lng"].mean()), 5),
                "report_count": int(len(group)),
                "hot": zone in flagged_zones,
            }
        )
    results.sort(key=lambda c: -c["report_count"])
    return results


def compute_anomalies():
    df = fetch_reports_df()
    if df.empty:
        return []
    df["reported_at"] = pd.to_datetime(df["reported_at"])
    daily = df.groupby(["zone", "reported_at"]).size().reset_index(name="count")
    out = []
    for zone, g_ in daily.groupby("zone"):
        g_ = g_.sort_values("reported_at")
        baseline = g_["count"].iloc[:14]
        mean = float(baseline.mean())
        std = float(baseline.std()) or 1.0
        latest = int(g_["count"].iloc[-1])
        z = round((latest - mean) / std, 2)
        series = [
            {"date": d.strftime("%Y-%m-%d"), "count": int(c)}
            for d, c in zip(g_["reported_at"], g_["count"])
        ]
        out.append(
            {
                "zone": zone,
                "baseline_mean": round(mean, 2),
                "threshold": round(mean + 2 * std, 2),
                "latest_count": latest,
                "z_score": z,
                "flagged": bool(z > 2),
                "series": series,
            }
        )
    out.sort(key=lambda a: -a["z_score"])
    return out


def compute_alerts():
    alerts = []
    for a in compute_anomalies():
        if a["flagged"]:
            alerts.append(
                {
                    "zone": a["zone"],
                    "message": (
                        f"Case reports in {a['zone']} are {a['z_score']}\u03c3 above "
                        f"baseline ({a['latest_count']} vs threshold {a['threshold']})."
                    ),
                    "severity": "high" if a["z_score"] > 4 else "moderate",
                    "triggered_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                }
            )
    return alerts


def compute_summary():
    conn = get_db()
    today = datetime.utcnow().date().isoformat()
    reports_today = conn.execute(
        "SELECT COUNT(*) AS c FROM reports WHERE reported_at = ?", (today,)
    ).fetchone()["c"]
    hot_clusters = [c for c in compute_clusters(days=7) if c["hot"]]
    return {
        "reports_today": reports_today,
        "active_clusters": len(hot_clusters),
        "alerts_sent": len(compute_alerts()),
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/api/reports", methods=["GET"])
def get_reports():
    days = request.args.get("days", default=21, type=int)
    df = fetch_reports_df(days)
    return jsonify(df.to_dict(orient="records"))


@app.route("/api/reports", methods=["POST"])
def add_report():
    data = request.get_json(force=True, silent=True) or {}
    required = ["lat", "lng", "zone", "symptom"]
    missing = [k for k in required if k not in data]
    if missing:
        return jsonify({"error": f"missing fields: {missing}"}), 400

    conn = get_db()
    conn.execute(
        "INSERT INTO reports (lat, lng, zone, symptom, source, reported_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            data["lat"],
            data["lng"],
            data["zone"],
            data["symptom"],
            data.get("source", "citizen"),
            data.get("reported_at", datetime.utcnow().date().isoformat()),
        ),
    )
    conn.commit()
    return jsonify({"status": "created"}), 201


@app.route("/api/clusters")
def clusters_route():
    days = request.args.get("days", default=7, type=int)
    return jsonify(compute_clusters(days))


@app.route("/api/anomalies")
def anomalies_route():
    return jsonify(compute_anomalies())


@app.route("/api/alerts")
def alerts_route():
    return jsonify(compute_alerts())


@app.route("/api/summary")
def summary_route():
    return jsonify(compute_summary())


# Initialize/seed the database at import time (not just under __main__) so a
# production WSGI server like gunicorn — which imports this module and never
# runs the block below — still gets a ready database.
init_db()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
