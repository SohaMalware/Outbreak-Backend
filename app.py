"""
Outbreak Early Warning System — backend API (v6).

What this system does
---------------------
Healthcare workers (role "admin") report how many cases of each diagnosed
disease or potential threat they have seen, with a location. The server
groups those reports into areas and flags an area when its case counts are
rising abnormally. That is an early WARNING: it reacts to real, reported
counts as they change. It does not forecast anything.

Only healthcare-worker reports feed the warnings. News headlines are shown
for information (see /api/news) but are never used in any calculation,
because headlines can be misleading.

Who can do what
---------------
- Anyone (no account): view warning zones, case counts per area, news.
- Signed-in user (role "citizen"): everything above, plus receive alerts
  (in the site's alert inbox, and by email if they keep email alerts on).
- Healthcare worker (role "admin"): everything above, plus report cases,
  delete their own reports, and issue alerts to all signed-in users.

Becoming a healthcare worker: sign up with the access code stored in the
ADMIN_SIGNUP_CODE environment variable (share it only with real staff), and
give a facility name. This is a shared-secret check, not identity
verification — a real rollout should have a person approve each account.

Storage
-------
Set DATABASE_URL (a Postgres connection string, e.g. from Neon) and data is
stored there and survives restarts. With no DATABASE_URL it falls back to a
local SQLite file, which free hosts such as Render wipe on every restart or
idle spin-down — fine for local testing, not for real use. /api/health
reports which one is active.

Email alerts
------------
Sent through Brevo's HTTPS API (BREVO_API_KEY, MAIL_SENDER_EMAIL). SMTP is
deliberately not used: Render's free tier blocks the SMTP ports. If those
variables are not set, alerts still reach the in-site inbox and the email
step is skipped.

Endpoints
---------
  GET  /api/health
  GET  /api/diseases
  GET  /api/warnings                 public: warning zones + headline numbers
  GET  /api/news                     public: ?place=&disease= (information only)

  POST /api/auth/signup              {name, email, password, email_alerts?, facility?, access_code?}
  POST /api/auth/login               {email, password}
  GET  /api/auth/me
  POST /api/auth/preferences         {email_alerts}

  GET  /api/alerts                   signed-in: alert inbox
  POST /api/alerts/<id>/read
  POST /api/alerts/read-all

  GET  /api/admin/reports            healthcare workers: recent reports
  POST /api/admin/reports            {disease, case_count, area, lat, lng, report_date?, notes?}
  DELETE /api/admin/reports/<id>     (own reports only)
  GET  /api/admin/alerts             alert history
  POST /api/admin/alerts             {title, message, severity, disease?, area?}
  GET  /api/admin/summary
"""

import hmac
import html as htmllib
import math
import os
import re
import sqlite3
import threading
import time
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from functools import wraps
from urllib.parse import quote
from xml.etree import ElementTree

import requests
from flask import Flask, g, jsonify, request
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from werkzeug.security import check_password_hash, generate_password_hash

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.environ.get("SQLITE_PATH") or os.path.join(BASE_DIR, "outbreak.db")
DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()
USE_PG = DATABASE_URL.startswith(("postgres://", "postgresql://"))

SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-change-me")
ADMIN_SIGNUP_CODE = os.environ.get("ADMIN_SIGNUP_CODE", "")
SEED_ADMIN_EMAIL = os.environ.get("SEED_ADMIN_EMAIL", "").strip().lower()
SEED_ADMIN_PASSWORD = os.environ.get("SEED_ADMIN_PASSWORD", "")
SEED_ADMIN_FACILITY = os.environ.get("SEED_ADMIN_FACILITY", "Health Department")

BREVO_API_KEY = os.environ.get("BREVO_API_KEY", "")
MAIL_SENDER_EMAIL = os.environ.get("MAIL_SENDER_EMAIL", "")
MAIL_SENDER_NAME = os.environ.get("MAIL_SENDER_NAME", "Outbreak Early Warning")
SITE_URL = os.environ.get("SITE_URL", "")
EMAIL_MAX_RECIPIENTS = int(os.environ.get("EMAIL_MAX_RECIPIENTS", "250"))  # stays under Brevo's 300/day free cap

TOKEN_MAX_AGE = 60 * 60 * 24 * 7  # 7 days

app = Flask(__name__)
app.config["SECRET_KEY"] = SECRET_KEY
serializer = URLSafeTimedSerializer(SECRET_KEY)

if SECRET_KEY == "dev-secret-change-me":
    print("WARNING: SECRET_KEY is not set — using an insecure development key.")

# ---------------------------------------------------------------------------
# Diseases and early-warning rules
# ---------------------------------------------------------------------------
DISEASES = [
    "Dengue", "Malaria", "Chikungunya", "Cholera", "Typhoid", "Hepatitis A",
    "Leptospirosis", "Acute diarrhoeal disease", "Influenza", "COVID-19",
    "Measles", "Mpox", "Tuberculosis", "Nipah virus", "Ebola",
]

# How much higher than usual a disease's case counts normally run in its
# peak months (1=Jan..12=Dec). Outside the peak months the multiplier is 1.
# These are tuned to the Indian monsoon / winter pattern — change them if
# the site is used in a region with a different calendar. Diseases that
# are not listed here (including anything typed in as "Other") get no
# seasonal adjustment, rather than an invented one.
SEASONAL_PROFILE = {
    "Dengue":                   {"peak_months": {6, 7, 8, 9, 10}, "peak_multiplier": 1.8},
    "Malaria":                  {"peak_months": {6, 7, 8, 9, 10}, "peak_multiplier": 1.5},
    "Chikungunya":              {"peak_months": {6, 7, 8, 9, 10}, "peak_multiplier": 1.5},
    "Cholera":                  {"peak_months": {6, 7, 8, 9},     "peak_multiplier": 1.6},
    "Typhoid":                  {"peak_months": {6, 7, 8, 9},     "peak_multiplier": 1.3},
    "Hepatitis A":              {"peak_months": {6, 7, 8, 9},     "peak_multiplier": 1.3},
    "Leptospirosis":            {"peak_months": {6, 7, 8, 9},     "peak_multiplier": 1.6},
    "Acute diarrhoeal disease": {"peak_months": {6, 7, 8, 9},     "peak_multiplier": 1.4},
    "Influenza":                {"peak_months": {11, 12, 1, 2},   "peak_multiplier": 1.7},
}

# For a few high-consequence diseases, a single reported case is already a
# warning. These are illustrative defaults — the health authority running
# the site should confirm them against its own notifiable-disease rules.
SENTINEL_LEVEL = {
    "Ebola": "high",
    "Nipah virus": "high",
    "Cholera": "moderate",
}

WINDOW_DAYS = 14
CLUSTER_RADIUS_KM = 5.0
MIN_CASES = 4              # fewest recent cases before a growth ratio is trusted
GROWTH_LOW = 1.2
GROWTH_MODERATE = 1.5
GROWTH_HIGH = 2.5
SUSTAINED_ABS = 15         # recent cases that are a warning on their own, before seasonal scaling

LEVEL_RANK = {"low": 1, "moderate": 2, "high": 3}
# Bullseye ring radii in km at severity 1.0, outermost first.
RING_BASE_KM = [("low", 4.0), ("moderate", 2.0), ("high", 0.8)]
RING_SCALE = {"low": 0.6, "moderate": 1.0, "high": 1.5}

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
DISEASE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 \-()'./]{1,59}$")


def seasonal_multiplier(disease, month):
    profile = SEASONAL_PROFILE.get(disease)
    if not profile:
        return 1.0
    return profile["peak_multiplier"] if month in profile["peak_months"] else 1.0


def haversine_km(lat1, lng1, lat2, lng2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def utc_now():
    return datetime.now(timezone.utc)


def utc_today():
    return utc_now().date()


# ---------------------------------------------------------------------------
# CORS (the site is a static page on a different origin; auth uses a bearer
# token in a header, not cookies, so a permissive origin is acceptable)
# ---------------------------------------------------------------------------
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response


@app.route("/api/<path:_any>", methods=["OPTIONS"])
def cors_preflight(_any):
    return "", 204


# ---------------------------------------------------------------------------
# Database layer — one small wrapper so the same SQL runs on SQLite (local /
# tests) and Postgres (hosted). Queries are written with "?" placeholders.
# ---------------------------------------------------------------------------
def to_pg_sql(sql):
    return sql.replace("?", "%s")


class Conn:
    def __init__(self):
        if USE_PG:
            import psycopg2
            import psycopg2.extras

            self.raw = psycopg2.connect(DATABASE_URL, connect_timeout=10)
            self._cursor_factory = psycopg2.extras.RealDictCursor
        else:
            self.raw = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=15)
            self.raw.row_factory = sqlite3.Row

    def execute(self, sql, params=()):
        if USE_PG:
            cur = self.raw.cursor(cursor_factory=self._cursor_factory)
            cur.execute(to_pg_sql(sql), params)
            return cur
        return self.raw.execute(sql, params)

    def all(self, sql, params=()):
        return [dict(r) for r in self.execute(sql, params).fetchall()]

    def one(self, sql, params=()):
        rows = self.all(sql, params)
        return rows[0] if rows else None

    def insert(self, sql, params=()):
        """Runs an INSERT and returns the new row's id."""
        rows = self.execute(sql + " RETURNING id", params).fetchall()
        return dict(rows[0])["id"]

    def commit(self):
        self.raw.commit()

    def close(self):
        try:
            self.raw.close()
        except Exception:
            pass


def get_db():
    if "db" not in g:
        g.db = Conn()
    return g.db


@app.teardown_appcontext
def close_db(_exc):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    pk = "SERIAL PRIMARY KEY" if USE_PG else "INTEGER PRIMARY KEY AUTOINCREMENT"
    real = "DOUBLE PRECISION"
    conn = Conn()
    conn.execute(
        f"""CREATE TABLE IF NOT EXISTS users (
            id {pk},
            email TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'citizen',
            facility TEXT,
            email_alerts INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL
        )"""
    )
    conn.execute(
        f"""CREATE TABLE IF NOT EXISTS reports (
            id {pk},
            user_id INTEGER NOT NULL REFERENCES users (id),
            disease TEXT NOT NULL,
            case_count INTEGER NOT NULL,
            lat {real} NOT NULL,
            lng {real} NOT NULL,
            area TEXT NOT NULL,
            report_date TEXT NOT NULL,
            notes TEXT,
            created_at TEXT NOT NULL
        )"""
    )
    conn.execute(
        f"""CREATE TABLE IF NOT EXISTS alerts (
            id {pk},
            title TEXT NOT NULL,
            message TEXT NOT NULL,
            severity TEXT NOT NULL,
            disease TEXT,
            area TEXT,
            created_by INTEGER REFERENCES users (id),
            created_at TEXT NOT NULL,
            email_status TEXT NOT NULL DEFAULT 'not sent',
            email_sent INTEGER NOT NULL DEFAULT 0,
            email_failed INTEGER NOT NULL DEFAULT 0
        )"""
    )
    conn.execute(
        f"""CREATE TABLE IF NOT EXISTS alert_reads (
            id {pk},
            alert_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            read_at TEXT NOT NULL,
            UNIQUE (alert_id, user_id)
        )"""
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_reports_date ON reports (report_date)")
    conn.commit()

    # Optional: create the first healthcare-worker account from environment
    # variables, so the owner never has to share an access code with
    # themselves — and it comes back after a wipe on hosts without a database.
    if SEED_ADMIN_EMAIL and SEED_ADMIN_PASSWORD:
        existing = conn.one("SELECT id FROM users WHERE email = ?", (SEED_ADMIN_EMAIL,))
        if not existing:
            conn.insert(
                "INSERT INTO users (email, name, password_hash, role, facility, email_alerts, created_at) "
                "VALUES (?, ?, ?, 'admin', ?, 1, ?)",
                (
                    SEED_ADMIN_EMAIL,
                    "Administrator",
                    generate_password_hash(SEED_ADMIN_PASSWORD),
                    SEED_ADMIN_FACILITY,
                    utc_now().isoformat(),
                ),
            )
            conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------
def make_token(user_id):
    return serializer.dumps({"uid": user_id})


def verify_token(token):
    try:
        return serializer.loads(token, max_age=TOKEN_MAX_AGE).get("uid")
    except (BadSignature, SignatureExpired):
        return None


def current_user():
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return None
    uid = verify_token(auth[len("Bearer "):])
    if uid is None:
        return None
    return get_db().one(
        "SELECT id, email, name, role, facility, email_alerts, created_at FROM users WHERE id = ?",
        (uid,),
    )


def public_user(u):
    return {
        "id": u["id"],
        "name": u["name"],
        "email": u["email"],
        "role": u["role"],
        "facility": u["facility"],
        "email_alerts": bool(u["email_alerts"]),
    }


def require_auth(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        user = current_user()
        if user is None:
            return jsonify({"error": "Please sign in."}), 401
        g.user = user
        return f(*args, **kwargs)
    return wrapper


def require_admin(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        user = current_user()
        if user is None:
            return jsonify({"error": "Please sign in."}), 401
        if user["role"] != "admin":
            return jsonify({"error": "This action is only for healthcare workers."}), 403
        g.user = user
        return f(*args, **kwargs)
    return wrapper


_login_attempts = {}
LOGIN_LIMIT = 8
LOGIN_WINDOW_SECONDS = 600


def client_ip():
    fwd = request.headers.get("X-Forwarded-For", "")
    return (fwd.split(",")[0].strip() if fwd else request.remote_addr) or "unknown"


def _recent_attempts(key):
    now = time.time()
    attempts = [t for t in _login_attempts.get(key, []) if now - t < LOGIN_WINDOW_SECONDS]
    _login_attempts[key] = attempts
    return attempts


def _json():
    return request.get_json(force=True, silent=True) or {}


# ---------------------------------------------------------------------------
# Auth routes
# ---------------------------------------------------------------------------
@app.route("/api/auth/signup", methods=["POST"])
def signup():
    data = _json()
    name = (data.get("name") or "").strip()[:80]
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""
    facility = (data.get("facility") or "").strip()[:120]
    access_code = data.get("access_code") or ""
    email_alerts = 1 if data.get("email_alerts", True) else 0

    if len(name) < 2:
        return jsonify({"error": "Please enter your name."}), 400
    if not EMAIL_RE.match(email) or len(email) > 254:
        return jsonify({"error": "Please enter a valid email address."}), 400
    if len(password) < 8:
        return jsonify({"error": "Password must be at least 8 characters."}), 400

    role = "citizen"
    if access_code:
        expected = ADMIN_SIGNUP_CODE.encode("utf-8")
        if not expected or not hmac.compare_digest(access_code.encode("utf-8"), expected):
            return jsonify({"error": "That healthcare worker access code is not valid."}), 403
        if len(facility) < 2:
            return jsonify({"error": "Please enter your hospital or facility name."}), 400
        role = "admin"
    else:
        facility = ""

    conn = get_db()
    if conn.one("SELECT id FROM users WHERE email = ?", (email,)):
        return jsonify({"error": "An account with this email already exists."}), 409

    uid = conn.insert(
        "INSERT INTO users (email, name, password_hash, role, facility, email_alerts, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (email, name, generate_password_hash(password), role, facility or None, email_alerts, utc_now().isoformat()),
    )
    conn.commit()
    user = conn.one("SELECT id, email, name, role, facility, email_alerts FROM users WHERE id = ?", (uid,))
    return jsonify({"token": make_token(uid), "user": public_user(user)}), 201


@app.route("/api/auth/login", methods=["POST"])
def login():
    data = _json()
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""
    key = f"{client_ip()}|{email}"
    if len(_recent_attempts(key)) >= LOGIN_LIMIT:
        return jsonify({"error": "Too many attempts. Please wait a few minutes and try again."}), 429

    conn = get_db()
    user = conn.one("SELECT * FROM users WHERE email = ?", (email,))
    if not user or not check_password_hash(user["password_hash"], password):
        _login_attempts.setdefault(key, []).append(time.time())
        return jsonify({"error": "Incorrect email or password."}), 401

    _login_attempts.pop(key, None)
    return jsonify({"token": make_token(user["id"]), "user": public_user(user)})


@app.route("/api/auth/me")
@require_auth
def me():
    return jsonify({"user": public_user(g.user)})


@app.route("/api/auth/preferences", methods=["POST"])
@require_auth
def preferences():
    data = _json()
    flag = 1 if data.get("email_alerts") else 0
    conn = get_db()
    conn.execute("UPDATE users SET email_alerts = ? WHERE id = ?", (flag, g.user["id"]))
    conn.commit()
    return jsonify({"email_alerts": bool(flag)})


# ---------------------------------------------------------------------------
# Healthcare-worker reports
# ---------------------------------------------------------------------------
def clean_disease(raw):
    name = re.sub(r"\s+", " ", (raw or "").strip())
    if not DISEASE_RE.match(name):
        return None
    for known in DISEASES:
        if known.lower() == name.lower():
            return known
    return name


@app.route("/api/diseases")
def diseases():
    return jsonify(DISEASES)


@app.route("/api/admin/reports", methods=["POST"])
@require_admin
def admin_add_report():
    data = _json()
    disease = clean_disease(data.get("disease"))
    if not disease:
        return jsonify({"error": "Please choose a disease or enter its name."}), 400
    try:
        case_count = int(data.get("case_count"))
        lat = float(data.get("lat"))
        lng = float(data.get("lng"))
    except (TypeError, ValueError):
        return jsonify({"error": "Case count and location are required."}), 400
    if not 0 <= case_count <= 100000:
        return jsonify({"error": "Case count must be between 0 and 100,000."}), 400
    if not (-90 <= lat <= 90 and -180 <= lng <= 180):
        return jsonify({"error": "That location is not valid."}), 400
    area = (data.get("area") or "").strip()[:80]
    if len(area) < 2:
        return jsonify({"error": "Please enter the area or locality name."}), 400
    notes = (data.get("notes") or "").strip()[:300] or None

    today = utc_today()
    try:
        report_date = date.fromisoformat(data.get("report_date") or today.isoformat())
    except ValueError:
        return jsonify({"error": "That date is not valid."}), 400
    if report_date > today + timedelta(days=1):
        return jsonify({"error": "The date cannot be in the future."}), 400
    if report_date < today - timedelta(days=60):
        return jsonify({"error": "Reports can only be dated within the last 60 days."}), 400

    conn = get_db()
    rid = conn.insert(
        "INSERT INTO reports (user_id, disease, case_count, lat, lng, area, report_date, notes, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (g.user["id"], disease, case_count, lat, lng, area, report_date.isoformat(), notes, utc_now().isoformat()),
    )
    conn.commit()
    return jsonify({"id": rid, "status": "recorded"}), 201


@app.route("/api/admin/reports")
@require_admin
def admin_list_reports():
    days = max(1, min(request.args.get("days", default=30, type=int), 90))
    cutoff = (utc_today() - timedelta(days=days)).isoformat()
    rows = get_db().all(
        """SELECT r.id, r.disease, r.case_count, r.lat, r.lng, r.area, r.report_date, r.notes,
                  r.created_at, r.user_id, u.name AS reporter, u.facility AS facility
           FROM reports r JOIN users u ON r.user_id = u.id
           WHERE r.report_date >= ?
           ORDER BY r.report_date DESC, r.id DESC
           LIMIT 200""",
        (cutoff,),
    )
    for r in rows:
        r["mine"] = r["user_id"] == g.user["id"]
        del r["user_id"]
    return jsonify(rows)


@app.route("/api/admin/reports/<int:report_id>", methods=["DELETE"])
@require_admin
def admin_delete_report(report_id):
    conn = get_db()
    row = conn.one("SELECT id, user_id FROM reports WHERE id = ?", (report_id,))
    if not row:
        return jsonify({"error": "Report not found."}), 404
    if row["user_id"] != g.user["id"]:
        return jsonify({"error": "You can only delete reports you submitted."}), 403
    conn.execute("DELETE FROM reports WHERE id = ?", (report_id,))
    conn.commit()
    return jsonify({"status": "deleted"})


# ---------------------------------------------------------------------------
# Early-warning engine. Reads healthcare-worker reports only.
# ---------------------------------------------------------------------------
def build_warning(disease, items, today):
    """Turns one geographic group of reports for one disease into a warning
    (or None if nothing about it deserves one)."""
    today_iso = today.isoformat()
    daily = {}
    for it in items:
        d = min(it["report_date"], today_iso)  # a report dated "tomorrow" in another timezone counts as today
        daily[d] = daily.get(d, 0) + it["case_count"]
    series = [daily.get((today - timedelta(days=i)).isoformat(), 0) for i in range(WINDOW_DAYS - 1, -1, -1)]

    recent = series[-3:]
    baseline = series[-10:-3]
    recent_total = sum(recent)
    if recent_total <= 0:
        return None
    recent_avg = recent_total / 3
    baseline_avg = sum(baseline) / len(baseline)
    raw_ratio = recent_avg / max(baseline_avg, 0.5)

    mult = seasonal_multiplier(disease, today.month)
    adjusted = raw_ratio / mult  # in a disease's peak season the same rise counts for less

    candidates = []
    sentinel = SENTINEL_LEVEL.get(disease)
    if sentinel:
        candidates.append(sentinel)
    if recent_total >= MIN_CASES and adjusted >= GROWTH_HIGH:
        candidates.append("high")
    elif recent_total >= MIN_CASES and adjusted >= GROWTH_MODERATE:
        candidates.append("moderate")
    if recent_total >= SUSTAINED_ABS * mult * 2:
        candidates.append("high")
    elif recent_total >= SUSTAINED_ABS * mult:
        candidates.append("moderate")
    if recent_total >= 2 and adjusted >= GROWTH_LOW:
        candidates.append("low")
    if not candidates:
        return None
    level = max(candidates, key=LEVEL_RANK.get)

    weights = [max(it["case_count"], 0) for it in items]
    total_w = sum(weights)
    if total_w > 0:
        lat = sum(it["lat"] * w for it, w in zip(items, weights)) / total_w
        lng = sum(it["lng"] * w for it, w in zip(items, weights)) / total_w
    else:
        lat = sum(it["lat"] for it in items) / len(items)
        lng = sum(it["lng"] for it in items) / len(items)

    area_counts = Counter()
    for it in items:
        area_counts[it["area"].strip()] += max(it["case_count"], 1)
    area = area_counts.most_common(1)[0][0]

    week_cutoff = (today - timedelta(days=6)).isoformat()
    facilities = sorted({it["facility"] for it in items if it["report_date"] >= week_cutoff and it["facility"]})
    scale = RING_SCALE[level]

    return {
        "id": f"{disease}|{round(lat, 3)}|{round(lng, 3)}",
        "disease": disease,
        "level": level,
        "area": area,
        "lat": round(lat, 5),
        "lng": round(lng, 5),
        "cases_3d": recent_total,
        "cases_7d": sum(series[-7:]),
        "growth": round(adjusted, 1),
        "in_season": mult > 1.0,
        "facilities": facilities[:5],
        "facility_count": len(facilities),
        "last_report_at": max(it["created_at"] for it in items),
        "rings": [{"level": lv, "radius_m": int(km * scale * 1000)} for lv, km in RING_BASE_KM],
    }


def compute_warnings(conn):
    today = utc_today()
    cutoff = (today - timedelta(days=WINDOW_DAYS)).isoformat()
    rows = conn.all(
        """SELECT r.id, r.disease, r.case_count, r.lat, r.lng, r.area, r.report_date, r.created_at,
                  u.facility AS facility
           FROM reports r JOIN users u ON r.user_id = u.id
           WHERE r.report_date >= ?""",
        (cutoff,),
    )
    by_disease = {}
    for r in rows:
        by_disease.setdefault(r["disease"], []).append(r)

    warnings = []
    for disease, items in by_disease.items():
        clusters = []
        for r in sorted(items, key=lambda x: (x["report_date"], x["id"])):
            best, best_d = None, None
            for c in clusters:
                d = haversine_km(r["lat"], r["lng"], c["lat"], c["lng"])
                if d <= CLUSTER_RADIUS_KM and (best_d is None or d < best_d):
                    best, best_d = c, d
            if best is None:
                clusters.append({"lat": r["lat"], "lng": r["lng"], "items": [r]})
            else:
                best["items"].append(r)
        for c in clusters:
            w = build_warning(disease, c["items"], today)
            if w:
                warnings.append(w)

    warnings.sort(key=lambda w: (-LEVEL_RANK[w["level"]], -w["cases_3d"]))

    week_cutoff = (today - timedelta(days=6)).isoformat()
    totals = conn.one(
        """SELECT COALESCE(SUM(r.case_count), 0) AS cases, COUNT(DISTINCT u.facility) AS facilities
           FROM reports r JOIN users u ON r.user_id = u.id WHERE r.report_date >= ?""",
        (week_cutoff,),
    )
    stats = {
        "active": len(warnings),
        "high": sum(1 for w in warnings if w["level"] == "high"),
        "moderate": sum(1 for w in warnings if w["level"] == "moderate"),
        "low": sum(1 for w in warnings if w["level"] == "low"),
        "cases_7d": int(totals["cases"] or 0),
        "facilities_reporting": int(totals["facilities"] or 0),
    }
    return warnings, stats


@app.route("/api/warnings")
def warnings_route():
    warnings, stats = compute_warnings(get_db())
    return jsonify({"warnings": warnings, "stats": stats, "updated_at": utc_now().isoformat()})


@app.route("/api/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "storage": "postgres" if USE_PG else "sqlite",
            "email_configured": bool(BREVO_API_KEY and MAIL_SENDER_EMAIL),
        }
    )


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------
def deliver_alert_emails(alert_id, recipients, title, message, severity, disease, area):
    """Sends one email per recipient through Brevo's HTTPS API, then records
    how many went out. Runs in a background thread so issuing an alert
    returns immediately."""
    sent = failed = 0
    status = "sent"
    if not (BREVO_API_KEY and MAIL_SENDER_EMAIL):
        status = "email not configured"
    else:
        label = {"low": "Low", "moderate": "Moderate", "high": "High"}.get(severity, severity)
        where = " · ".join(x for x in [disease, area] if x)
        link = f'<p><a href="{htmllib.escape(SITE_URL)}">Open the early warning map</a></p>' if SITE_URL else ""
        body = (
            f'<div style="font-family:Arial,sans-serif;max-width:560px">'
            f'<p style="margin:0 0 4px;color:#555">{htmllib.escape(label)} level alert'
            f'{" · " + htmllib.escape(where) if where else ""}</p>'
            f'<h2 style="margin:0 0 12px">{htmllib.escape(title)}</h2>'
            f'<p style="white-space:pre-wrap">{htmllib.escape(message)}</p>{link}'
            f'<hr><p style="font-size:12px;color:#777">You are receiving this because you have an account with '
            f"email alerts turned on. Sign in to the site to turn them off.</p></div>"
        )
        for email in recipients[:EMAIL_MAX_RECIPIENTS]:
            try:
                resp = requests.post(
                    "https://api.brevo.com/v3/smtp/email",
                    headers={"api-key": BREVO_API_KEY, "accept": "application/json", "content-type": "application/json"},
                    json={
                        "sender": {"name": MAIL_SENDER_NAME, "email": MAIL_SENDER_EMAIL},
                        "to": [{"email": email}],
                        "subject": f"[{label}] {title}",
                        "htmlContent": body,
                    },
                    timeout=10,
                )
                if resp.status_code in (200, 201):
                    sent += 1
                else:
                    failed += 1
            except Exception:
                failed += 1
            time.sleep(0.1)
        if len(recipients) > EMAIL_MAX_RECIPIENTS:
            failed += len(recipients) - EMAIL_MAX_RECIPIENTS
        if failed and not sent:
            status = "failed"
        elif failed:
            status = "partly sent"

    conn = Conn()
    try:
        conn.execute(
            "UPDATE alerts SET email_status = ?, email_sent = ?, email_failed = ? WHERE id = ?",
            (status, sent, failed, alert_id),
        )
        conn.commit()
    finally:
        conn.close()


@app.route("/api/admin/alerts", methods=["POST"])
@require_admin
def admin_issue_alert():
    data = _json()
    title = (data.get("title") or "").strip()[:120]
    message = (data.get("message") or "").strip()[:1500]
    severity = data.get("severity")
    disease = (data.get("disease") or "").strip()[:60] or None
    area = (data.get("area") or "").strip()[:80] or None
    if len(title) < 3 or len(message) < 5:
        return jsonify({"error": "Please enter a title and a message."}), 400
    if severity not in LEVEL_RANK:
        return jsonify({"error": "Please choose a severity level."}), 400

    conn = get_db()
    hour_ago = (utc_now() - timedelta(hours=1)).isoformat()
    recent = conn.one(
        "SELECT COUNT(*) AS c FROM alerts WHERE created_by = ? AND created_at >= ?", (g.user["id"], hour_ago)
    )["c"]
    if recent >= 10:
        return jsonify({"error": "You have issued 10 alerts in the last hour. Please wait before sending more."}), 429

    aid = conn.insert(
        "INSERT INTO alerts (title, message, severity, disease, area, created_by, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (title, message, severity, disease, area, g.user["id"], utc_now().isoformat()),
    )
    conn.commit()

    recipients = [r["email"] for r in conn.all("SELECT email FROM users WHERE email_alerts = 1")]
    total_users = conn.one("SELECT COUNT(*) AS c FROM users")["c"]
    email_ready = bool(BREVO_API_KEY and MAIL_SENDER_EMAIL)
    threading.Thread(
        target=deliver_alert_emails,
        args=(aid, recipients, title, message, severity, disease, area),
        daemon=True,
    ).start()
    return (
        jsonify(
            {
                "id": aid,
                "inbox_recipients": total_users,
                "email_recipients": len(recipients) if email_ready else 0,
                "email_configured": email_ready,
            }
        ),
        201,
    )


@app.route("/api/admin/alerts")
@require_admin
def admin_list_alerts():
    rows = get_db().all(
        """SELECT a.id, a.title, a.message, a.severity, a.disease, a.area, a.created_at,
                  a.email_status, a.email_sent, a.email_failed, u.name AS issued_by
           FROM alerts a LEFT JOIN users u ON a.created_by = u.id
           ORDER BY a.created_at DESC LIMIT 50"""
    )
    return jsonify(rows)


@app.route("/api/admin/summary")
@require_admin
def admin_summary():
    conn = get_db()
    return jsonify(
        {
            "users": conn.one("SELECT COUNT(*) AS c FROM users")["c"],
            "healthcare_workers": conn.one("SELECT COUNT(*) AS c FROM users WHERE role = 'admin'")["c"],
            "email_subscribers": conn.one("SELECT COUNT(*) AS c FROM users WHERE email_alerts = 1")["c"],
            "reports": conn.one("SELECT COUNT(*) AS c FROM reports")["c"],
            "alerts": conn.one("SELECT COUNT(*) AS c FROM alerts")["c"],
        }
    )


ALERT_WINDOW_DAYS = 30


@app.route("/api/alerts")
@require_auth
def my_alerts():
    cutoff = (utc_now() - timedelta(days=ALERT_WINDOW_DAYS)).isoformat()
    rows = get_db().all(
        """SELECT a.id, a.title, a.message, a.severity, a.disease, a.area, a.created_at,
                  (SELECT COUNT(*) FROM alert_reads x WHERE x.alert_id = a.id AND x.user_id = ?) AS is_read
           FROM alerts a WHERE a.created_at >= ? ORDER BY a.created_at DESC LIMIT 50""",
        (g.user["id"], cutoff),
    )
    for r in rows:
        r["read"] = bool(r.pop("is_read"))
    return jsonify({"alerts": rows, "unread": sum(1 for r in rows if not r["read"])})


@app.route("/api/alerts/<int:alert_id>/read", methods=["POST"])
@require_auth
def mark_read(alert_id):
    conn = get_db()
    if not conn.one("SELECT id FROM alerts WHERE id = ?", (alert_id,)):
        return jsonify({"error": "Alert not found."}), 404
    conn.execute(
        "INSERT INTO alert_reads (alert_id, user_id, read_at) VALUES (?, ?, ?) ON CONFLICT DO NOTHING",
        (alert_id, g.user["id"], utc_now().isoformat()),
    )
    conn.commit()
    return jsonify({"status": "ok"})


@app.route("/api/alerts/read-all", methods=["POST"])
@require_auth
def mark_all_read():
    conn = get_db()
    cutoff = (utc_now() - timedelta(days=ALERT_WINDOW_DAYS)).isoformat()
    for a in conn.all("SELECT id FROM alerts WHERE created_at >= ?", (cutoff,)):
        conn.execute(
            "INSERT INTO alert_reads (alert_id, user_id, read_at) VALUES (?, ?, ?) ON CONFLICT DO NOTHING",
            (a["id"], g.user["id"], utc_now().isoformat()),
        )
    conn.commit()
    return jsonify({"status": "ok"})


# ---------------------------------------------------------------------------
# News — information only. Never used in any warning calculation.
# ---------------------------------------------------------------------------
_news_cache = {}
NEWS_TTL_SECONDS = 900


@app.route("/api/news")
def news():
    place = (request.args.get("place") or "").strip()[:80]
    disease = (request.args.get("disease") or "").strip()[:60]
    if disease:
        query = f"{disease} outbreak {place}".strip()
    elif place:
        query = f"{place} disease outbreak health alert"
    else:
        query = "disease outbreak health alert"

    cached = _news_cache.get(query)
    if cached and time.time() - cached[0] < NEWS_TTL_SECONDS:
        return jsonify({"query": query, "ok": True, "items": cached[1]})

    url = f"https://news.google.com/rss/search?q={quote(query)}&hl=en-IN&gl=IN&ceid=IN:en"
    try:
        resp = requests.get(url, timeout=6, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        root = ElementTree.fromstring(resp.content)
        items = []
        for item in root.findall(".//item")[:8]:
            source_el = item.find("source")
            items.append(
                {
                    "title": item.findtext("title") or "",
                    "link": item.findtext("link") or "",
                    "source": source_el.text if source_el is not None else "",
                    "published": item.findtext("pubDate") or "",
                }
            )
        _news_cache[query] = (time.time(), items)
        return jsonify({"query": query, "ok": True, "items": items})
    except Exception:
        return jsonify({"query": query, "ok": False, "items": []})


# Create tables at import time so a production server such as gunicorn
# (which imports this module and never runs the block below) is ready.
init_db()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("FLASK_DEBUG") == "1")
