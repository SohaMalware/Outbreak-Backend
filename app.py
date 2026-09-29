"""
Disease Outbreak Early Warning System — backend API (v5).

Major change from v4: this is now global and GPS is used only to tell a
visitor what's relevant near them — never to submit a report. The
citizen-symptom-report feature is gone entirely, per the project's
direction. There is exactly one data source now: news coverage, monitored
continuously across a fixed set of major cities spanning every inhabited
continent (see LOCALITIES), for a fixed list of diseases (see DISEASES —
now includes Ebola).

Continuous background refresh: rather than only refreshing when someone's
browser happens to hit /api/refresh-news, a background thread
(background_news_worker) cycles through every (locality, disease)
combination on its own, all the time the process is alive, respecting
REFRESH_MAX_AGE_HOURS so it doesn't re-fetch something already fresh. This
is what makes the map "live" without requiring a page visit to trigger it.
Two honest limits on that claim: (1) it can only run while the process is
alive — most free hosting tiers (including Render's free plan) spin the
process down after a period with no incoming requests, which pauses the
background thread too, and it resumes on the next visit; (2) a full sweep
across every combination takes a while by design (a small delay between
requests, see BACKGROUND_FETCH_DELAY_SECONDS, to avoid hammering Google
News) — expect coverage to fill in gradually after each deploy, not
instantly.

Two different outputs, on purpose:
- /api/predictions — the strict, threshold-gated list: a locality+disease
  only appears here once its trend clears the bar (see compute_predictions
  for the exact logic, unchanged in spirit from v4, now globally scoped
  and with Ebola added to the seasonal profile — flat, since Ebola
  outbreaks are tied to spillover events, not a calendar season). This can
  legitimately be empty. That's not a bug — it means nothing currently
  monitored has crossed the bar.
- /api/headlines — an unfiltered, always-fresh-as-available feed of actual
  fetched articles, so the page has real content to show even when nothing
  has cleared the stricter prediction bar. This is coverage, not a claim
  that any given headline represents a confirmed outbreak.

What this does NOT do, stated plainly: it does not forecast an outbreak
appearing somewhere with zero current signal. "Prediction" here means
catching a real, current rise early — not genuine spatial forecasting to
unaffected areas. That would need a different, much more data-hungry
model than a heuristic project like this can responsibly claim.

Endpoints:
  GET  /api/health          -> liveness check
  GET  /api/localities      -> monitored cities worldwide (id, name, lat, lng)
  POST /api/refresh-news    -> manually trigger a bounded batch refresh (on-demand, in addition to the background worker)
  GET  /api/predictions     -> threshold-gated predicted outbreaks (may be empty)
  GET  /api/headlines       -> recent fetched articles, unfiltered (?limit=N)
  GET  /api/case-counts     -> per-locality case-equivalent totals by disease
  GET  /api/news            -> ?disease=X&locality=Y -> live headlines for display (on-demand, not cached)
  GET  /api/summary         -> dashboard counters

Run:
  pip install -r requirements.txt
  python app.py
  -> serves on http://127.0.0.1:5000
"""

import math
import os
import re
import sqlite3
import threading
import time
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote
from xml.etree import ElementTree

import pandas as pd
import requests
from flask import Flask, g, jsonify, request

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outbreak.db")

app = Flask(__name__)

DISEASES = ["Dengue", "Malaria", "Cholera", "Typhoid", "Influenza", "COVID-19", "Chikungunya", "Ebola"]

# Major cities spanning every inhabited continent — chosen for realistic
# news coverage, not tied to any hospital or institution. GPS from a
# visitor is only ever matched against this list to find what's nearest to
# them; nothing is ever submitted about a visitor's own location.
LOCALITIES = [
    {"name": "Lagos", "lat": 6.5244, "lng": 3.3792},
    {"name": "Kinshasa", "lat": -4.4419, "lng": 15.2663},
    {"name": "Nairobi", "lat": -1.2921, "lng": 36.8219},
    {"name": "Cairo", "lat": 30.0444, "lng": 31.2357},
    {"name": "Johannesburg", "lat": -26.2041, "lng": 28.0473},
    {"name": "Mumbai", "lat": 19.0760, "lng": 72.8777},
    {"name": "Delhi", "lat": 28.7041, "lng": 77.1025},
    {"name": "Dhaka", "lat": 23.8103, "lng": 90.4125},
    {"name": "Jakarta", "lat": -6.2088, "lng": 106.8456},
    {"name": "Manila", "lat": 14.5995, "lng": 120.9842},
    {"name": "Beijing", "lat": 39.9042, "lng": 116.4074},
    {"name": "Tokyo", "lat": 35.6762, "lng": 139.6503},
    {"name": "Bangkok", "lat": 13.7563, "lng": 100.5018},
    {"name": "Istanbul", "lat": 41.0082, "lng": 28.9784},
    {"name": "Riyadh", "lat": 24.7136, "lng": 46.6753},
    {"name": "London", "lat": 51.5074, "lng": -0.1278},
    {"name": "Paris", "lat": 48.8566, "lng": 2.3522},
    {"name": "Berlin", "lat": 52.5200, "lng": 13.4050},
    {"name": "Rome", "lat": 41.9028, "lng": 12.4964},
    {"name": "Madrid", "lat": 40.4168, "lng": -3.7038},
    {"name": "New York", "lat": 40.7128, "lng": -74.0060},
    {"name": "Mexico City", "lat": 19.4326, "lng": -99.1332},
    {"name": "Sao Paulo", "lat": -23.5505, "lng": -46.6333},
    {"name": "Lima", "lat": -12.0464, "lng": -77.0428},
    {"name": "Sydney", "lat": -33.8688, "lng": 151.2093},
]

NEWS_MENTION_WEIGHT = 3  # case-equivalent credited to a relevant headline with no extractable number

MIN_CASES_FOR_PREDICTION = 4
GROWTH_RATIO_MODERATE = 1.5
GROWTH_RATIO_HIGH = 2.5
SUSTAINED_HIGH_ABS = 15

REFRESH_MAX_AGE_HOURS = 2        # background worker won't re-fetch a combo more often than this
REFRESH_BATCH_LIMIT = 8          # cap for a single POST /api/refresh-news call
BACKGROUND_FETCH_DELAY_SECONDS = 4   # politeness delay between requests in the background worker
BACKGROUND_SWEEP_PAUSE_SECONDS = 60  # pause between full passes over the locality x disease list

# Peak months (1=Jan..12=Dec) and how much higher the "normal" baseline is
# expected to run during them. Outside peak months, multiplier is 1.0.
# Ebola and COVID-19 are left flat (1.0 year-round): neither has a
# reliable calendar-season pattern — Ebola outbreaks follow zoonotic
# spillover events, not seasons, so giving it one would be inventing a
# pattern that isn't real.
SEASONAL_PROFILE = {
    "Dengue":       {"peak_months": {6, 7, 8, 9, 10}, "peak_multiplier": 1.8},
    "Cholera":      {"peak_months": {6, 7, 8, 9},      "peak_multiplier": 1.6},
    "Malaria":      {"peak_months": {6, 7, 8, 9, 10},  "peak_multiplier": 1.5},
    "Chikungunya":  {"peak_months": {6, 7, 8, 9, 10},  "peak_multiplier": 1.5},
    "Typhoid":      {"peak_months": {6, 7, 8, 9},      "peak_multiplier": 1.3},
    "Influenza":    {"peak_months": {11, 12, 1, 2},    "peak_multiplier": 1.7},
    "COVID-19":     {"peak_months": set(),             "peak_multiplier": 1.0},
    "Ebola":        {"peak_months": set(),              "peak_multiplier": 1.0},
}

NUM_CASE_RE = re.compile(
    r"\b(\d{1,6})\b(?:\s+\S+){0,3}?\s+(cases|infections|patients|infected|hospitalisations|hospitalizations|deaths)\b",
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


def haversine_km(lat1, lng1, lat2, lng2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


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
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS news_headlines (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            locality_id INTEGER NOT NULL,
            disease TEXT NOT NULL,
            title TEXT NOT NULL,
            link TEXT UNIQUE NOT NULL,
            source TEXT,
            published_at TEXT NOT NULL,
            fetched_at TEXT NOT NULL,
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


def read_localities_df(conn):
    rows = conn.execute("SELECT * FROM localities").fetchall()
    return pd.DataFrame([dict(r) for r in rows])


def get_localities_df():
    return read_localities_df(get_db())


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


# ---------------------------------------------------------------------------
# News signal — event-based surveillance via Google News RSS (no API key)
# ---------------------------------------------------------------------------
def fetch_and_store_news_signal(conn, locality_id, locality_name, disease):
    """Best-effort: fetches recent headlines for this locality+disease,
    stores each individual article (deduped by link) in news_headlines for
    the always-on live feed, and buckets any extractable case-equivalent by
    the headline's publish date into news_signal_cache for the stricter
    prediction math. Any failure is swallowed — a stale or empty cache
    entry is fine; it just means fewer/no results this cycle."""
    query = f"{disease} cases {locality_name}"
    url = f"https://news.google.com/rss/search?q={quote(query)}&hl=en-US&gl=US&ceid=US:en"
    now_iso = datetime.now(timezone.utc).isoformat()
    cutoff = datetime.now(timezone.utc) - timedelta(days=5)

    counts_by_date = {}
    fetch_succeeded = False
    try:
        resp = requests.get(url, timeout=6, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        root = ElementTree.fromstring(resp.content)
        fetch_succeeded = True
        for item in root.findall(".//item")[:8]:
            title = item.findtext("title") or ""
            link = item.findtext("link") or ""
            pub_date_raw = item.findtext("pubDate") or ""
            source_el = item.find("source")
            source = source_el.text if source_el is not None else ""
            try:
                pub_dt = parsedate_to_datetime(pub_date_raw)
                if pub_dt.tzinfo is None:
                    pub_dt = pub_dt.replace(tzinfo=timezone.utc)
            except Exception:
                pub_dt = datetime.now(timezone.utc)
            if pub_dt < cutoff:
                continue

            if link:
                try:
                    conn.execute(
                        """
                        INSERT INTO news_headlines
                            (locality_id, disease, title, link, source, published_at, fetched_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(link) DO NOTHING
                        """,
                        (locality_id, disease, title, link, source, pub_dt.isoformat(), now_iso),
                    )
                except Exception:
                    pass

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
    elif fetch_succeeded:
        # Genuinely fetched successfully and found nothing relevant — safe
        # to mark as checked so we don't re-fetch again within the window.
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
    # else: the fetch itself failed (network error, blocked, malformed feed)
    # — leave the cache untouched so this combo is retried on the next pass
    # instead of being wrongly marked as checked.
    conn.commit()


def _stale_combos(conn, limit=None):
    localities_df = read_localities_df(conn)
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=REFRESH_MAX_AGE_HOURS)).isoformat()
    combos = []
    for _, loc in localities_df.iterrows():
        for disease in DISEASES:
            row = conn.execute(
                """
                SELECT fetched_at FROM news_signal_cache
                WHERE locality_id = ? AND disease = ? ORDER BY fetched_at DESC LIMIT 1
                """,
                (int(loc["id"]), disease),
            ).fetchone()
            if row and row["fetched_at"] >= cutoff:
                continue
            combos.append((int(loc["id"]), loc["name"], disease))
            if limit and len(combos) >= limit:
                return combos
    return combos


@app.route("/api/refresh-news", methods=["POST"])
def refresh_news():
    conn = get_db()
    combos = _stale_combos(conn, limit=REFRESH_BATCH_LIMIT)
    for locality_id, locality_name, disease in combos:
        fetch_and_store_news_signal(conn, locality_id, locality_name, disease)
    return jsonify({"refreshed": len(combos)})


def background_news_worker():
    """Runs for the lifetime of the process, cycling through every
    (locality, disease) combination and refreshing whichever ones have
    gone stale, with a small delay between requests. See the module
    docstring for what this can and can't guarantee on free hosting."""
    while True:
        try:
            conn = sqlite3.connect(DB_PATH, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            combos = _stale_combos(conn)
            for locality_id, locality_name, disease in combos:
                fetch_and_store_news_signal(conn, locality_id, locality_name, disease)
                time.sleep(BACKGROUND_FETCH_DELAY_SECONDS)
            conn.close()
        except Exception:
            pass
        time.sleep(BACKGROUND_SWEEP_PAUSE_SECONDS)


# ---------------------------------------------------------------------------
# Prediction engine — reads the cache only, so it's always fast regardless
# of how the cache gets populated (background worker or manual refresh).
# ---------------------------------------------------------------------------
def build_daily_series(days=14):
    localities_df = get_localities_df()
    conn = get_db()
    cutoff = (date.today() - timedelta(days=days)).isoformat()

    series = {}
    rows = conn.execute(
        "SELECT locality_id, disease, case_equivalent, signal_date FROM news_signal_cache WHERE signal_date >= ?",
        (cutoff,),
    ).fetchall()
    for r in rows:
        key = (r["locality_id"], r["disease"])
        series.setdefault(key, {})
        series[key][r["signal_date"]] = series[key].get(r["signal_date"], 0) + r["case_equivalent"]

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


@app.route("/api/headlines")
def headlines_route():
    limit = request.args.get("limit", default=25, type=int)
    conn = get_db()
    rows = conn.execute(
        """
        SELECT news_headlines.title, news_headlines.link, news_headlines.source,
               news_headlines.published_at, news_headlines.disease,
               localities.name AS locality, localities.lat AS lat, localities.lng AS lng
        FROM news_headlines JOIN localities ON news_headlines.locality_id = localities.id
        ORDER BY news_headlines.published_at DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return jsonify([dict(r) for r in rows])


# ---------------------------------------------------------------------------
# Case counts — aggregate only
# ---------------------------------------------------------------------------
@app.route("/api/case-counts")
def case_counts():
    days = request.args.get("days", default=14, type=int)
    localities_df = get_localities_df()
    conn = get_db()
    cutoff = (date.today() - timedelta(days=days)).isoformat()

    news_totals = conn.execute(
        """
        SELECT locality_id, disease, SUM(case_equivalent) AS total
        FROM news_signal_cache WHERE signal_date >= ? GROUP BY locality_id, disease
        """,
        (cutoff,),
    ).fetchall()

    result = []
    for _, loc in localities_df.iterrows():
        lid = int(loc["id"])
        by_disease = {}
        for row in news_totals:
            if row["locality_id"] == lid and row["total"]:
                by_disease[row["disease"]] = round(row["total"], 1)
        if by_disease:
            result.append(
                {"locality": loc["name"], "lat": float(loc["lat"]), "lng": float(loc["lng"]), "by_disease": by_disease}
            )
    return jsonify(result)


@app.route("/api/summary")
def summary():
    conn = get_db()
    today_str = date.today().isoformat()

    media_signals_today = conn.execute(
        "SELECT COALESCE(SUM(case_equivalent), 0) AS c FROM news_signal_cache WHERE signal_date = ?",
        (today_str,),
    ).fetchone()["c"]
    headlines_tracked = conn.execute("SELECT COUNT(*) AS c FROM news_headlines").fetchone()["c"]
    predictions = compute_predictions()

    return jsonify(
        {
            "predicted_outbreaks": len(predictions),
            "localities_monitored": len(LOCALITIES),
            "media_signals_today": round(media_signals_today, 1),
            "headlines_tracked": headlines_tracked,
        }
    )


# ---------------------------------------------------------------------------
# News — live fetch for on-screen display when someone taps a prediction
# ---------------------------------------------------------------------------
@app.route("/api/news")
def news():
    disease = request.args.get("disease", "").strip()
    locality = request.args.get("locality", "").strip()
    if not disease:
        return jsonify({"error": "disease query param required"}), 400

    query = f"{disease} outbreak {locality}".strip()
    url = f"https://news.google.com/rss/search?q={quote(query)}&hl=en-US&gl=US&ceid=US:en"

    try:
        resp = requests.get(url, timeout=6, headers={"User-Agent": "Mozilla/5.0"})
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


# Initialize the database at import time so a production WSGI server like
# gunicorn (which imports this module and never runs the block below)
# still gets a ready database and a running background worker.
init_db()
threading.Thread(target=background_news_worker, daemon=True).start()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)
