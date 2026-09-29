"""
Disease Outbreak Early Warning System — backend API (v3).

Major change from v2: NO accounts, NO login, NO admin panel. Everyone who
opens the site sees the same thing: predicted outbreak clusters, case
counts per hospital/locality, and related news. There is no raw personal
data to protect because there are no personal accounts — hospital reports
are aggregate daily counts (not individual patient records), and citizen
reports are anonymous (no identity attached at all).

Data sources:
- Hospitals: a small fixed set of nearby hospitals (seeded in the database
  as a directory — see HOSPITALS below). In a real deployment these would
  push data automatically from each hospital's own system; here, any
  client can POST a daily case count for a hospital + disease. This is a
  simulation of a real-time feed, not a real hospital integration — there
  is deliberately no authentication on this endpoint, per the project's
  new requirement to drop all logins. That is a real, known weakness
  (anyone could submit fake hospital numbers) — worth flagging to
  reviewers, and worth fixing with an API key per hospital before any real
  use.
- Citizens: anonymous reports of a disease/symptom + GPS location. No
  account, no identity stored — just a coordinate, a disease label, and a
  timestamp.

Prediction, not just detection: for each of a fixed list of diseases, at
each hospital's locality, the last 14 days of combined case data (hospital
counts at full weight, citizen reports at partial weight — see
CITIZEN_WEIGHT) are turned into a daily count series. The last 3 days are
compared against the preceding week's average. A rising trend over a
minimum case floor is flagged as a predicted outbreak, with a risk level
based on how steep the rise is. Because this runs independently per
disease per locality, several different outbreaks can be flagged at once
across different areas — that's the "numerous outbreaks possible" the
project asked for.

Endpoints:
  GET  /api/health              -> liveness check
  GET  /api/hospitals           -> directory of hospitals (id, name, lat, lng)
  POST /api/hospital-reports    -> {hospital_id, disease, case_count, report_date?}
  GET  /api/hospital-reports    -> raw daily hospital counts (?days=N)
  POST /api/citizen-reports     -> {lat, lng, disease} — anonymous
  GET  /api/predictions         -> the core output: predicted outbreaks
  GET  /api/case-counts         -> per-hospital case totals, hospital vs citizen
  GET  /api/news                -> ?disease=X&locality=Y -> related headlines
  GET  /api/summary             -> dashboard counters

Run:
  pip install -r requirements.txt
  python app.py
  -> serves on http://127.0.0.1:5000
"""

import math
import os
import sqlite3
from datetime import date, datetime, timedelta, timezone
from urllib.parse import quote
from xml.etree import ElementTree

import pandas as pd
import requests
from flask import Flask, g, jsonify, request

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outbreak.db")

app = Flask(__name__)

DISEASES = ["Dengue", "Malaria", "Cholera", "Typhoid", "Influenza", "COVID-19", "Chikungunya"]

# A small fixed set of nearby hospitals — seeded once, never duplicated.
HOSPITALS = [
    {"name": "Kamothe Community Hospital", "lat": 19.0176, "lng": 73.0961},
    {"name": "Kharghar Multispecialty Hospital", "lat": 19.0474, "lng": 73.0662},
    {"name": "Panvel Civic Hospital", "lat": 18.9894, "lng": 73.1175},
    {"name": "Belapur General Hospital", "lat": 19.0234, "lng": 73.0356},
    {"name": "Vashi Health Centre", "lat": 19.0771, "lng": 73.0000},
]

# How much one anonymous citizen report counts toward the case-equivalent
# total used for trend detection, relative to one hospital-confirmed case
# (weight 1.0). Self-reported symptoms are weaker evidence than a hospital
# diagnosis, so they count for less but still contribute.
CITIZEN_WEIGHT = 0.3

MIN_CASES_FOR_PREDICTION = 4   # floor below which a "trend" is just noise
GROWTH_RATIO_MODERATE = 1.5
GROWTH_RATIO_HIGH = 2.5
SUSTAINED_HIGH_ABS = 15        # flagged even without growth, if just persistently high


# ---------------------------------------------------------------------------
# CORS (manual — the frontend is a static file on a different origin)
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


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS hospitals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            lat REAL NOT NULL,
            lng REAL NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS hospital_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hospital_id INTEGER NOT NULL,
            disease TEXT NOT NULL,
            case_count INTEGER NOT NULL,
            report_date TEXT NOT NULL,
            UNIQUE(hospital_id, disease, report_date),
            FOREIGN KEY (hospital_id) REFERENCES hospitals (id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS citizen_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            lat REAL NOT NULL,
            lng REAL NOT NULL,
            disease TEXT NOT NULL,
            reported_at TEXT NOT NULL
        )
        """
    )
    conn.commit()

    count = conn.execute("SELECT COUNT(*) AS c FROM hospitals").fetchone()[0]
    if count == 0:
        conn.executemany(
            "INSERT INTO hospitals (name, lat, lng) VALUES (:name, :lat, :lng)", HOSPITALS
        )
        conn.commit()
    conn.close()


def get_hospitals_df():
    conn = get_db()
    rows = conn.execute("SELECT * FROM hospitals").fetchall()
    return pd.DataFrame([dict(r) for r in rows])


def haversine_km(lat1, lng1, lat2, lng2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def nearest_hospital_id(lat, lng, hospitals_df):
    best_id, best_dist = None, None
    for _, h in hospitals_df.iterrows():
        d = haversine_km(lat, lng, h["lat"], h["lng"])
        if best_dist is None or d < best_dist:
            best_dist, best_id = d, int(h["id"])
    return best_id


# ---------------------------------------------------------------------------
# Public routes
# ---------------------------------------------------------------------------
@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/api/hospitals")
def list_hospitals():
    df = get_hospitals_df()
    return jsonify(df.to_dict(orient="records"))


@app.route("/api/hospital-reports", methods=["GET"])
def get_hospital_reports():
    days = request.args.get("days", default=21, type=int)
    conn = get_db()
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    rows = conn.execute(
        """
        SELECT hospital_reports.id, hospitals.name AS hospital, hospital_reports.disease,
               hospital_reports.case_count, hospital_reports.report_date
        FROM hospital_reports JOIN hospitals ON hospital_reports.hospital_id = hospitals.id
        WHERE hospital_reports.report_date >= ?
        ORDER BY hospital_reports.report_date DESC
        """,
        (cutoff,),
    ).fetchall()
    return jsonify([dict(r) for r in rows])


@app.route("/api/hospital-reports", methods=["POST"])
def add_hospital_report():
    data = request.get_json(force=True, silent=True) or {}
    required = ["hospital_id", "disease", "case_count"]
    missing = [k for k in required if k not in data]
    if missing:
        return jsonify({"error": f"missing fields: {missing}"}), 400
    if data["disease"] not in DISEASES:
        return jsonify({"error": f"disease must be one of {DISEASES}"}), 400
    try:
        hospital_id = int(data["hospital_id"])
        case_count = int(data["case_count"])
    except (TypeError, ValueError):
        return jsonify({"error": "hospital_id and case_count must be integers"}), 400
    if case_count < 0:
        return jsonify({"error": "case_count cannot be negative"}), 400
    report_date = data.get("report_date") or date.today().isoformat()

    conn = get_db()
    exists = conn.execute("SELECT id FROM hospitals WHERE id = ?", (hospital_id,)).fetchone()
    if not exists:
        return jsonify({"error": "unknown hospital_id"}), 400

    conn.execute(
        """
        INSERT INTO hospital_reports (hospital_id, disease, case_count, report_date)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(hospital_id, disease, report_date)
        DO UPDATE SET case_count = excluded.case_count
        """,
        (hospital_id, data["disease"], case_count, report_date),
    )
    conn.commit()
    return jsonify({"status": "recorded"}), 201


@app.route("/api/citizen-reports", methods=["POST"])
def add_citizen_report():
    data = request.get_json(force=True, silent=True) or {}
    required = ["lat", "lng", "disease"]
    missing = [k for k in required if k not in data]
    if missing:
        return jsonify({"error": f"missing fields: {missing}"}), 400
    if data["disease"] not in DISEASES:
        return jsonify({"error": f"disease must be one of {DISEASES}"}), 400
    try:
        lat, lng = float(data["lat"]), float(data["lng"])
    except (TypeError, ValueError):
        return jsonify({"error": "lat/lng must be numbers"}), 400
    if not (-90 <= lat <= 90 and -180 <= lng <= 180):
        return jsonify({"error": "lat/lng out of range"}), 400

    conn = get_db()
    conn.execute(
        "INSERT INTO citizen_reports (lat, lng, disease, reported_at) VALUES (?, ?, ?, ?)",
        (lat, lng, data["disease"], datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    return jsonify({"status": "recorded"}), 201


# ---------------------------------------------------------------------------
# Prediction engine
# ---------------------------------------------------------------------------
def build_daily_series(days=14):
    """Returns a dict keyed by (hospital_id, disease) -> {date_str: count},
    combining hospital-confirmed counts (full weight) with anonymous citizen
    reports assigned to their nearest hospital locality (partial weight)."""
    hospitals_df = get_hospitals_df()
    conn = get_db()
    cutoff = (date.today() - timedelta(days=days)).isoformat()

    series = {}

    hosp_rows = conn.execute(
        "SELECT hospital_id, disease, case_count, report_date FROM hospital_reports WHERE report_date >= ?",
        (cutoff,),
    ).fetchall()
    for r in hosp_rows:
        key = (r["hospital_id"], r["disease"])
        series.setdefault(key, {})
        series[key][r["report_date"]] = series[key].get(r["report_date"], 0) + r["case_count"]

    cutoff_ts = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    citizen_rows = conn.execute(
        "SELECT lat, lng, disease, reported_at FROM citizen_reports WHERE reported_at >= ?",
        (cutoff_ts,),
    ).fetchall()
    for r in citizen_rows:
        if hospitals_df.empty:
            continue
        hid = nearest_hospital_id(r["lat"], r["lng"], hospitals_df)
        d = r["reported_at"][:10]  # date portion of the ISO timestamp
        key = (hid, r["disease"])
        series.setdefault(key, {})
        series[key][d] = series[key].get(d, 0) + CITIZEN_WEIGHT

    return series, hospitals_df


def compute_predictions(days=14):
    series, hospitals_df = build_daily_series(days=days)
    if hospitals_df.empty:
        return []

    hospitals_by_id = {int(h["id"]): h for _, h in hospitals_df.iterrows()}
    today = date.today()
    predictions = []

    for (hospital_id, disease), counts_by_date in series.items():
        hospital = hospitals_by_id.get(hospital_id)
        if hospital is None:
            continue

        # Build a full daily series for the window, filling gaps with 0.
        daily = []
        for i in range(days - 1, -1, -1):
            d = (today - timedelta(days=i)).isoformat()
            daily.append(counts_by_date.get(d, 0))

        recent = daily[-3:]
        baseline = daily[-10:-3] if len(daily) >= 10 else daily[:-3]
        recent_avg = sum(recent) / len(recent) if recent else 0
        baseline_avg = (sum(baseline) / len(baseline)) if baseline else 0
        ratio = recent_avg / max(baseline_avg, 0.5)
        recent_total = round(sum(recent), 1)

        predicted, risk = False, None
        if recent_total >= MIN_CASES_FOR_PREDICTION and ratio >= GROWTH_RATIO_MODERATE:
            predicted = True
            risk = "high" if ratio >= GROWTH_RATIO_HIGH else "moderate"
        elif recent_total >= SUSTAINED_HIGH_ABS:
            predicted = True
            risk = "moderate"

        if predicted:
            predictions.append(
                {
                    "disease": disease,
                    "locality": hospital["name"],
                    "hospital_id": hospital_id,
                    "lat": float(hospital["lat"]),
                    "lng": float(hospital["lng"]),
                    "recent_case_equivalent": recent_total,
                    "growth_ratio": round(ratio, 2),
                    "risk": risk,
                }
            )

    order = {"high": 0, "moderate": 1}
    predictions.sort(key=lambda p: (order.get(p["risk"], 2), -p["growth_ratio"]))
    return predictions


@app.route("/api/predictions")
def predictions_route():
    return jsonify(compute_predictions())


# ---------------------------------------------------------------------------
# Case counts (public — these are aggregate numbers, not individual records)
# ---------------------------------------------------------------------------
@app.route("/api/case-counts")
def case_counts():
    days = request.args.get("days", default=14, type=int)
    hospitals_df = get_hospitals_df()
    conn = get_db()
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    cutoff_ts = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

    hosp_totals = conn.execute(
        """
        SELECT hospital_id, disease, SUM(case_count) AS total
        FROM hospital_reports WHERE report_date >= ? GROUP BY hospital_id, disease
        """,
        (cutoff,),
    ).fetchall()

    citizen_rows = conn.execute(
        "SELECT lat, lng, disease FROM citizen_reports WHERE reported_at >= ?",
        (cutoff_ts,),
    ).fetchall()
    citizen_totals = {}
    for r in citizen_rows:
        if hospitals_df.empty:
            continue
        hid = nearest_hospital_id(r["lat"], r["lng"], hospitals_df)
        key = (hid, r["disease"])
        citizen_totals[key] = citizen_totals.get(key, 0) + 1

    result = []
    for _, h in hospitals_df.iterrows():
        hid = int(h["id"])
        by_disease = {}
        for row in hosp_totals:
            if row["hospital_id"] == hid:
                by_disease.setdefault(row["disease"], {"confirmed": 0, "citizen_reported": 0})
                by_disease[row["disease"]]["confirmed"] = row["total"]
        for (chid, disease), c in citizen_totals.items():
            if chid == hid:
                by_disease.setdefault(disease, {"confirmed": 0, "citizen_reported": 0})
                by_disease[disease]["citizen_reported"] = c
        if by_disease:
            result.append(
                {
                    "hospital": h["name"],
                    "lat": float(h["lat"]),
                    "lng": float(h["lng"]),
                    "by_disease": by_disease,
                }
            )
    return jsonify(result)


@app.route("/api/summary")
def summary():
    conn = get_db()
    today_str = date.today().isoformat()
    today_start_ts = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).isoformat()

    hospital_cases_today = conn.execute(
        "SELECT COALESCE(SUM(case_count), 0) AS c FROM hospital_reports WHERE report_date = ?",
        (today_str,),
    ).fetchone()["c"]
    citizen_reports_today = conn.execute(
        "SELECT COUNT(*) AS c FROM citizen_reports WHERE reported_at >= ?", (today_start_ts,)
    ).fetchone()["c"]
    hospitals_reporting = conn.execute(
        "SELECT COUNT(DISTINCT hospital_id) AS c FROM hospital_reports WHERE report_date >= ?",
        ((date.today() - timedelta(days=7)).isoformat(),),
    ).fetchone()["c"]
    predictions = compute_predictions()

    return jsonify(
        {
            "predicted_outbreaks": len(predictions),
            "hospitals_reporting": hospitals_reporting,
            "cases_today": hospital_cases_today,
            "citizen_reports_today": citizen_reports_today,
        }
    )


# ---------------------------------------------------------------------------
# News — server-side fetch of Google News RSS (no API key required).
# Best-effort: any failure returns an empty list rather than an error, so a
# flaky or blocked outbound request never breaks the rest of the page.
# ---------------------------------------------------------------------------
@app.route("/api/news")
def news():
    disease = request.args.get("disease", "").strip()
    locality = request.args.get("locality", "").strip()
    if not disease:
        return jsonify({"error": "disease query param required"}), 400

    query = f"{disease} outbreak {locality} India".strip()
    url = f"https://news.google.com/rss/search?q={quote(query)}&hl=en-IN&gl=IN&ceid=IN:en"

    try:
        resp = requests.get(url, timeout=5, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        root = ElementTree.fromstring(resp.content)
        items = []
        for item in root.findall(".//item")[:6]:
            title = item.findtext("title") or ""
            link = item.findtext("link") or ""
            pub_date = item.findtext("pubDate") or ""
            source_el = item.find("source")
            source = source_el.text if source_el is not None else ""
            items.append({"title": title, "link": link, "source": source, "published": pub_date})
        return jsonify(items)
    except Exception:
        return jsonify([])


# Initialize the database at import time (not just under __main__) so a
# production WSGI server like gunicorn — which imports this module and never
# runs the block below — still gets a ready database.
init_db()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
