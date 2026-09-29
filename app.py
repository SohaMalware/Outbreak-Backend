"""
Disease Outbreak Early Warning System — backend API (v4).

Major change from v3: hospital data is no longer manually entered by
anyone. Instead, the backend periodically pulls local news coverage for
each monitored locality + disease combination and turns that into a
case-equivalent signal. This is a real, recognized surveillance technique
called "event-based surveillance" (as opposed to "indicator-based
surveillance", which is what the citizen-report and hospital-report paths
are) — India's own disease surveillance platform (IHIP) explicitly uses
media reports the same way, as a complement to formal reporting. See
fetch_and_store_news_signal() for exactly how a headline becomes a number.

Why this replaces the old hospital-report form: there is no free, open,
real-time, hospital-level case-count API available to pull from (checked —
India's real system, IHIP, is internal/password-protected for health
officials). A manual entry form was the only alternative, and it had an
honest problem: nothing stopped anyone from typing in fake numbers. Pulling
from public news removes that specific risk, at the cost of the data being
approximate rather than authoritative — headlines rarely state an exact
daily case count, and coverage of small neighborhoods is sparse. Both
tradeoffs are real; they're documented here rather than hidden.

Important honesty note: the five "localities" below are real place names
(they're neighborhoods in Navi Mumbai), but this project has no
affiliation with any actual hospital — earlier versions used invented
hospital names, which would never appear in real news coverage, so this
version anchors to plain area names instead.

Two data sources feed the prediction engine:
- News signal (event-based): fetch_and_store_news_signal() / refresh_news_signals()
- Citizen reports (indicator-based): anonymous, GPS + disease label, no account

Seasonal awareness: some diseases have a well-known seasonal pattern in
India (Dengue/Malaria/Cholera/Chikungunya/Typhoid rise with the monsoon,
Influenza rises in winter). SEASONAL_PROFILE encodes that as a per-disease,
per-month multiplier. A locality's case count has to clear a seasonally
*raised* bar during a disease's expected peak months to be flagged — and a
seasonally *lower* bar outside them, so an off-season case rise (e.g.
Dengue in January) is caught more sensitively rather than needing to hit
the same absolute numbers as a monsoon rise. COVID-19 has no reliable
seasonal pattern in the data available, so it isn't given one here — flagged
in SEASONAL_PROFILE rather than silently assumed.

Endpoints:
  GET  /api/health              -> liveness check
  GET  /api/localities          -> monitored areas (id, name, lat, lng)
  POST /api/citizen-reports     -> {lat, lng, disease} — anonymous
  POST /api/refresh-news        -> triggers a bounded batch of news-signal refreshes
  GET  /api/predictions         -> the core output: predicted outbreaks (reads cache only, fast)
  GET  /api/case-counts         -> per-locality case-equivalent totals, news vs citizen
  GET  /api/news                -> ?disease=X&locality=Y -> live headlines for display
  GET  /api/summary             -> dashboard counters

Run:
  pip install -r requirements.txt
  python app.py
  -> serves on http://127.0.0.1:5000
"""

import math
import os
import re
import sqlite3
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote
from xml.etree import ElementTree

import pandas as pd
import requests
from flask import Flask, g, jsonify, request

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outbreak.db")

app = Flask(__name__)

DISEASES = ["Dengue", "Malaria", "Cholera", "Typhoid", "Influenza", "COVID-19", "Chikungunya"]

# Real neighborhoods in Navi Mumbai — used only as named geographic anchors
# for clustering citizen reports and scoping news searches. Not hospitals.
LOCALITIES = [
    {"name": "Kamothe", "lat": 19.0176, "lng": 73.0961},
    {"name": "Kharghar", "lat": 19.0474, "lng": 73.0662},
    {"name": "Panvel", "lat": 18.9894, "lng": 73.1175},
    {"name": "Belapur", "lat": 19.0234, "lng": 73.0356},
    {"name": "Vashi", "lat": 19.0771, "lng": 73.0000},
]

# How much weight each source contributes to the case-equivalent series
# used for trend detection. Hospital-confirmed data would be 1.0 (full
# trust) if it existed here; news and citizen signals are both weaker
# evidence, so they're discounted relative to that baseline.
NEWS_SIGNAL_WEIGHT = 0.6
CITIZEN_WEIGHT = 0.3
NEWS_MENTION_WEIGHT = 3  # case-equivalent credited to a relevant headline with no extractable number

MIN_CASES_FOR_PREDICTION = 4
GROWTH_RATIO_MODERATE = 1.5
GROWTH_RATIO_HIGH = 2.5
SUSTAINED_HIGH_ABS = 15

REFRESH_MAX_AGE_HOURS = 6   # don't re-fetch a (locality, disease) pair more often than this
REFRESH_BATCH_LIMIT = 8     # max combos refreshed per /api/refresh-news call, to keep it fast

# Peak months (1=Jan..12=Dec) and how much higher the "normal" baseline is
# expected to run during them. Outside peak months, multiplier is 1.0.
SEASONAL_PROFILE = {
    "Dengue":       {"peak_months": {6, 7, 8, 9, 10}, "peak_multiplier": 1.8},
    "Cholera":      {"peak_months": {6, 7, 8, 9},      "peak_multiplier": 1.6},
    "Malaria":      {"peak_months": {6, 7, 8, 9, 10},  "peak_multiplier": 1.5},
    "Chikungunya":  {"peak_months": {6, 7, 8, 9, 10},  "peak_multiplier": 1.5},
    "Typhoid":      {"peak_months": {6, 7, 8, 9},      "peak_multiplier": 1.3},
    "Influenza":    {"peak_months": {11, 12, 1, 2},    "peak_multiplier": 1.7},
    "COVID-19":     {"peak_months": set(),             "peak_multiplier": 1.0},  # no reliable pattern modeled
}

NUM_CASE_RE = re.compile(
    r"\b(\d{1,5})\b(?:\s+\S+){0,3}?\s+(cases|infections|patients|infected|hospitalisations|hospitalizations)\b",
    re.IGNORECASE,
)


def seasonal_multiplier(disease, month):
    profile = SEASONAL_PROFILE.get(disease)
    if not profile:
        return 1.0
    return profile["peak_multiplier"] if month in profile["peak_months"] else 1.0


def extract_case_count(text):
    matches = NUM_CASE_RE.findall(text or "")
    if not matches:
        return None
    return max(int(m[0]) for m in matches)


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
        CREATE TABLE IF NOT EXISTS localities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            lat REAL NOT NULL,
            lng REAL NOT NULL
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
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS news_signal_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            locality_id INTEGER NOT NULL,
            disease TEXT NOT NULL,
            signal_date TEXT NOT NULL,
            case_equivalent REAL NOT NULL,
            fetched_at TEXT NOT NULL,
            UNIQUE(locality_id, disease, signal_date),
            FOREIGN KEY (locality_id) REFERENCES localities (id)
        )
        """
    )
    conn.commit()

    count = conn.execute("SELECT COUNT(*) AS c FROM localities").fetchone()[0]
    if count == 0:
        conn.executemany(
            "INSERT INTO localities (name, lat, lng) VALUES (:name, :lat, :lng)", LOCALITIES
        )
        conn.commit()
    conn.close()


def get_localities_df():
    conn = get_db()
    rows = conn.execute("SELECT * FROM localities").fetchall()
    return pd.DataFrame([dict(r) for r in rows])


def haversine_km(lat1, lng1, lat2, lng2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def nearest_locality_id(lat, lng, localities_df):
    best_id, best_dist = None, None
    for _, loc in localities_df.iterrows():
        d = haversine_km(lat, lng, loc["lat"], loc["lng"])
        if best_dist is None or d < best_dist:
            best_dist, best_id = d, int(loc["id"])
    return best_id


# ---------------------------------------------------------------------------
# Public routes
# ---------------------------------------------------------------------------
@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/api/localities")
def list_localities():
    df = get_localities_df()
    return jsonify(df.to_dict(orient="records"))


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
# News signal — event-based surveillance via Google News RSS (no API key)
# ---------------------------------------------------------------------------
def fetch_and_store_news_signal(conn, locality_id, locality_name, disease):
    """Best-effort: fetches recent headlines for this locality+disease, buckets
    any extractable case-equivalent by the headline's publish date, and
    upserts into news_signal_cache. Any failure is swallowed — a stale or
    empty cache entry is fine; it just means fewer/no results this cycle."""
    query = f"{disease} cases {locality_name} India"
    url = f"https://news.google.com/rss/search?q={quote(query)}&hl=en-IN&gl=IN&ceid=IN:en"
    now_iso = datetime.now(timezone.utc).isoformat()

    counts_by_date = {}
    try:
        resp = requests.get(url, timeout=5, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        root = ElementTree.fromstring(resp.content)
        cutoff = datetime.now(timezone.utc) - timedelta(days=5)
        for item in root.findall(".//item")[:8]:
            title = item.findtext("title") or ""
            pub_date_raw = item.findtext("pubDate") or ""
            try:
                pub_dt = parsedate_to_datetime(pub_date_raw)
                if pub_dt.tzinfo is None:
                    pub_dt = pub_dt.replace(tzinfo=timezone.utc)
            except Exception:
                pub_dt = datetime.now(timezone.utc)
            if pub_dt < cutoff:
                continue
            d = pub_dt.date().isoformat()
            n = extract_case_count(title)
            counts_by_date[d] = counts_by_date.get(d, 0) + (n if n is not None else NEWS_MENTION_WEIGHT)
    except Exception:
        pass  # network hiccup, blocked, or malformed feed — leave cache as-is below

    if counts_by_date:
        for d, c in counts_by_date.items():
            conn.execute(
                """
                INSERT INTO news_signal_cache (locality_id, disease, signal_date, case_equivalent, fetched_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(locality_id, disease, signal_date)
                DO UPDATE SET case_equivalent = excluded.case_equivalent, fetched_at = excluded.fetched_at
                """,
                (locality_id, disease, d, c, now_iso),
            )
    else:
        # Record that we checked today and found nothing, so we don't
        # hammer the same combo again within REFRESH_MAX_AGE_HOURS.
        today_str = date.today().isoformat()
        conn.execute(
            """
            INSERT INTO news_signal_cache (locality_id, disease, signal_date, case_equivalent, fetched_at)
            VALUES (?, ?, ?, 0, ?)
            ON CONFLICT(locality_id, disease, signal_date)
            DO UPDATE SET fetched_at = excluded.fetched_at
            """,
            (locality_id, disease, today_str, now_iso),
        )
    conn.commit()


@app.route("/api/refresh-news", methods=["POST"])
def refresh_news():
    conn = get_db()
    localities_df = get_localities_df()
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=REFRESH_MAX_AGE_HOURS)).isoformat()

    refreshed = 0
    for _, loc in localities_df.iterrows():
        if refreshed >= REFRESH_BATCH_LIMIT:
            break
        for disease in DISEASES:
            if refreshed >= REFRESH_BATCH_LIMIT:
                break
            row = conn.execute(
                """
                SELECT fetched_at FROM news_signal_cache
                WHERE locality_id = ? AND disease = ? ORDER BY fetched_at DESC LIMIT 1
                """,
                (int(loc["id"]), disease),
            ).fetchone()
            if row and row["fetched_at"] >= cutoff:
                continue
            fetch_and_store_news_signal(conn, int(loc["id"]), loc["name"], disease)
            refreshed += 1

    return jsonify({"refreshed": refreshed})


# ---------------------------------------------------------------------------
# Prediction engine
# ---------------------------------------------------------------------------
def build_daily_series(days=14):
    """(locality_id, disease) -> {date_str: case_equivalent}, combining the
    cached news signal (full weight of what was stored, already
    NEWS_SIGNAL_WEIGHT-able below) with citizen reports at CITIZEN_WEIGHT."""
    localities_df = get_localities_df()
    conn = get_db()
    cutoff = (date.today() - timedelta(days=days)).isoformat()

    series = {}

    news_rows = conn.execute(
        "SELECT locality_id, disease, case_equivalent, signal_date FROM news_signal_cache WHERE signal_date >= ?",
        (cutoff,),
    ).fetchall()
    for r in news_rows:
        key = (r["locality_id"], r["disease"])
        series.setdefault(key, {})
        series[key][r["signal_date"]] = series[key].get(r["signal_date"], 0) + r["case_equivalent"] * NEWS_SIGNAL_WEIGHT

    cutoff_ts = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    citizen_rows = conn.execute(
        "SELECT lat, lng, disease, reported_at FROM citizen_reports WHERE reported_at >= ?",
        (cutoff_ts,),
    ).fetchall()
    for r in citizen_rows:
        if localities_df.empty:
            continue
        lid = nearest_locality_id(r["lat"], r["lng"], localities_df)
        d = r["reported_at"][:10]
        key = (lid, r["disease"])
        series.setdefault(key, {})
        series[key][d] = series[key].get(d, 0) + CITIZEN_WEIGHT

    return series, localities_df


def compute_predictions(days=14):
    series, localities_df = build_daily_series(days=days)
    if localities_df.empty:
        return []

    localities_by_id = {int(l["id"]): l for _, l in localities_df.iterrows()}
    today = date.today()
    predictions = []

    for (locality_id, disease), counts_by_date in series.items():
        locality = localities_by_id.get(locality_id)
        if locality is None:
            continue

        daily = []
        for i in range(days - 1, -1, -1):
            d = (today - timedelta(days=i)).isoformat()
            daily.append(counts_by_date.get(d, 0))

        recent = daily[-3:]
        baseline = daily[-10:-3] if len(daily) >= 10 else daily[:-3]
        recent_avg = sum(recent) / len(recent) if recent else 0
        baseline_avg = (sum(baseline) / len(baseline)) if baseline else 0
        raw_ratio = recent_avg / max(baseline_avg, 0.5)
        recent_total = round(sum(recent), 1)

        mult = seasonal_multiplier(disease, today.month)
        adjusted_ratio = raw_ratio / mult
        adjusted_sustained_threshold = SUSTAINED_HIGH_ABS * mult

        predicted, risk = False, None
        if recent_total >= MIN_CASES_FOR_PREDICTION and adjusted_ratio >= GROWTH_RATIO_MODERATE:
            predicted = True
            risk = "high" if adjusted_ratio >= GROWTH_RATIO_HIGH else "moderate"
        elif recent_total >= adjusted_sustained_threshold:
            predicted = True
            risk = "moderate"

        if predicted:
            predictions.append(
                {
                    "disease": disease,
                    "locality": locality["name"],
                    "locality_id": locality_id,
                    "lat": float(locality["lat"]),
                    "lng": float(locality["lng"]),
                    "recent_case_equivalent": recent_total,
                    "growth_ratio": round(adjusted_ratio, 2),
                    "risk": risk,
                    "in_season": mult > 1.0,
                }
            )

    order = {"high": 0, "moderate": 1}
    predictions.sort(key=lambda p: (order.get(p["risk"], 2), -p["growth_ratio"]))
    return predictions


@app.route("/api/predictions")
def predictions_route():
    return jsonify(compute_predictions())


# ---------------------------------------------------------------------------
# Case counts — aggregate only, no individual records
# ---------------------------------------------------------------------------
@app.route("/api/case-counts")
def case_counts():
    days = request.args.get("days", default=14, type=int)
    localities_df = get_localities_df()
    conn = get_db()
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    cutoff_ts = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

    news_totals = conn.execute(
        """
        SELECT locality_id, disease, SUM(case_equivalent) AS total
        FROM news_signal_cache WHERE signal_date >= ? GROUP BY locality_id, disease
        """,
        (cutoff,),
    ).fetchall()

    citizen_rows = conn.execute(
        "SELECT lat, lng, disease FROM citizen_reports WHERE reported_at >= ?",
        (cutoff_ts,),
    ).fetchall()
    citizen_totals = {}
    for r in citizen_rows:
        if localities_df.empty:
            continue
        lid = nearest_locality_id(r["lat"], r["lng"], localities_df)
        key = (lid, r["disease"])
        citizen_totals[key] = citizen_totals.get(key, 0) + 1

    result = []
    for _, loc in localities_df.iterrows():
        lid = int(loc["id"])
        by_disease = {}
        for row in news_totals:
            if row["locality_id"] == lid and row["total"]:
                by_disease.setdefault(row["disease"], {"media_reported": 0, "citizen_reported": 0})
                by_disease[row["disease"]]["media_reported"] = round(row["total"], 1)
        for (clid, disease), c in citizen_totals.items():
            if clid == lid:
                by_disease.setdefault(disease, {"media_reported": 0, "citizen_reported": 0})
                by_disease[disease]["citizen_reported"] = c
        if by_disease:
            result.append(
                {"locality": loc["name"], "lat": float(loc["lat"]), "lng": float(loc["lng"]), "by_disease": by_disease}
            )
    return jsonify(result)


@app.route("/api/summary")
def summary():
    conn = get_db()
    today_str = date.today().isoformat()
    today_start_ts = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).isoformat()

    media_signals_today = conn.execute(
        "SELECT COALESCE(SUM(case_equivalent), 0) AS c FROM news_signal_cache WHERE signal_date = ?",
        (today_str,),
    ).fetchone()["c"]
    citizen_reports_today = conn.execute(
        "SELECT COUNT(*) AS c FROM citizen_reports WHERE reported_at >= ?", (today_start_ts,)
    ).fetchone()["c"]
    predictions = compute_predictions()

    return jsonify(
        {
            "predicted_outbreaks": len(predictions),
            "localities_monitored": len(LOCALITIES),
            "media_signals_today": round(media_signals_today, 1),
            "citizen_reports_today": citizen_reports_today,
        }
    )


# ---------------------------------------------------------------------------
# News — live fetch for on-screen display (separate from the cached signal
# used for prediction; this always hits the network so headlines shown to
# a person are current, not whatever was last cached).
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
