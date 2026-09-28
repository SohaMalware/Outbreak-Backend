"""
Disease Outbreak Early Warning System — backend API (v2).

Key change from v1: this version has real accounts and a real privacy split.
- The public (no login) only ever sees aggregated warning zones — never
  individual reports, never who reported what.
- Any signed-in user can submit a report; the server takes their GPS
  coordinates from the request body (the browser/device supplies them via
  the Geolocation API — the frontend handles that), not a manually chosen
  zone.
- Only accounts with role="admin" can see raw report data, counts, and the
  full clustering/anomaly detail.
- The database starts EMPTY. There is no seed/mock data — hotspots only
  appear once real reports come in. See compute_hotspots() for how "enough
  reports" is defined while there's little history to build a baseline from.

Becoming an admin: signup accepts an optional `admin_code` field. If it
matches the ADMIN_SIGNUP_CODE environment variable, the new account is
created with role="admin". Set that env var on your host (e.g. Render →
Environment) to something private, then sign up once using that code to
create your own admin account. Leave the env var unset and nobody can
create an admin account via signup.

Endpoints:
  POST /api/auth/signup      -> {username, password, admin_code?} -> {token, username, role}
  POST /api/auth/login       -> {username, password} -> {token, username, role}
  GET  /api/auth/me          -> current user from Authorization: Bearer <token>

  POST /api/reports          -> (auth required) {lat, lng, symptom} -> creates a report
  GET  /api/public/hotspots  -> (no auth) aggregated warning zones only

  GET  /api/admin/reports    -> (admin only) raw report rows
  GET  /api/admin/clusters   -> (admin only) full cluster detail (counts + centers)
  GET  /api/admin/summary    -> (admin only) dashboard counters

  GET  /api/health           -> liveness check

Run:
  pip install -r requirements.txt
  python app.py
  -> serves on http://127.0.0.1:5000
"""

import os
import sqlite3
from datetime import datetime, timedelta, timezone
from functools import wraps

import pandas as pd
from flask import Flask, g, jsonify, request
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from sklearn.cluster import DBSCAN
from werkzeug.security import check_password_hash, generate_password_hash

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outbreak.db")

# In production, set SECRET_KEY as an environment variable on your host.
# Falling back to a fixed value is fine for local dev only.
SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-change-me")
ADMIN_SIGNUP_CODE = os.environ.get("ADMIN_SIGNUP_CODE")  # unset = admin signup disabled
TOKEN_MAX_AGE = 60 * 60 * 24 * 7  # 7 days

app = Flask(__name__)
app.config["SECRET_KEY"] = SECRET_KEY
serializer = URLSafeTimedSerializer(SECRET_KEY)


# ---------------------------------------------------------------------------
# CORS (manual — the frontend is a static file on a different origin)
# ---------------------------------------------------------------------------
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
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
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'citizen',
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            lat REAL NOT NULL,
            lng REAL NOT NULL,
            symptom TEXT NOT NULL,
            reported_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        """
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
        data = serializer.loads(token, max_age=TOKEN_MAX_AGE)
        return data.get("uid")
    except (BadSignature, SignatureExpired):
        return None


def get_current_user():
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return None
    uid = verify_token(auth[len("Bearer "):])
    if uid is None:
        return None
    row = get_db().execute("SELECT * FROM users WHERE id = ?", (uid,)).fetchone()
    return dict(row) if row else None


def require_auth(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        user = get_current_user()
        if user is None:
            return jsonify({"error": "authentication required"}), 401
        request.current_user = user
        return f(*args, **kwargs)
    return wrapper


def require_admin(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        user = get_current_user()
        if user is None:
            return jsonify({"error": "authentication required"}), 401
        if user["role"] != "admin":
            return jsonify({"error": "admin access required"}), 403
        request.current_user = user
        return f(*args, **kwargs)
    return wrapper


# ---------------------------------------------------------------------------
# Auth routes
# ---------------------------------------------------------------------------
@app.route("/api/auth/signup", methods=["POST"])
def signup():
    data = request.get_json(force=True, silent=True) or {}
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""
    if len(username) < 3 or len(password) < 6:
        return jsonify({"error": "username must be 3+ chars, password 6+ chars"}), 400

    role = "citizen"
    if data.get("admin_code") and ADMIN_SIGNUP_CODE and data["admin_code"] == ADMIN_SIGNUP_CODE:
        role = "admin"

    conn = get_db()
    existing = conn.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
    if existing:
        return jsonify({"error": "username already taken"}), 409

    conn.execute(
        "INSERT INTO users (username, password_hash, role, created_at) VALUES (?, ?, ?, ?)",
        (username, generate_password_hash(password), role, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    token = make_token(user["id"])
    return jsonify({"token": token, "username": user["username"], "role": user["role"]}), 201


@app.route("/api/auth/login", methods=["POST"])
def login():
    data = request.get_json(force=True, silent=True) or {}
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""

    conn = get_db()
    user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    if not user or not check_password_hash(user["password_hash"], password):
        return jsonify({"error": "invalid username or password"}), 401

    token = make_token(user["id"])
    return jsonify({"token": token, "username": user["username"], "role": user["role"]})


@app.route("/api/auth/me")
@require_auth
def me():
    u = request.current_user
    return jsonify({"username": u["username"], "role": u["role"]})


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------
@app.route("/api/reports", methods=["POST"])
@require_auth
def add_report():
    data = request.get_json(force=True, silent=True) or {}
    required = ["lat", "lng", "symptom"]
    missing = [k for k in required if k not in data]
    if missing:
        return jsonify({"error": f"missing fields: {missing}"}), 400
    try:
        lat, lng = float(data["lat"]), float(data["lng"])
    except (TypeError, ValueError):
        return jsonify({"error": "lat/lng must be numbers"}), 400
    if not (-90 <= lat <= 90 and -180 <= lng <= 180):
        return jsonify({"error": "lat/lng out of range"}), 400

    conn = get_db()
    conn.execute(
        "INSERT INTO reports (user_id, lat, lng, symptom, reported_at) VALUES (?, ?, ?, ?, ?)",
        (request.current_user["id"], lat, lng, data["symptom"], datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    return jsonify({"status": "created"}), 201


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------
def fetch_reports_df(days=None):
    conn = get_db()
    if days is not None:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        rows = conn.execute(
            "SELECT * FROM reports WHERE reported_at >= ? ORDER BY reported_at", (cutoff,)
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM reports ORDER BY reported_at").fetchall()
    return pd.DataFrame([dict(r) for r in rows])


def compute_hotspots(days=14, min_reports=5, window_hours=72, public=True):
    """Geographic clustering of recent reports (DBSCAN).

    A cluster becomes a "hotspot" once it has at least `min_reports` reports
    within the last `window_hours`. This simple count threshold — rather
    than a statistical baseline comparison — is intentional: a system with
    no historical data yet has no baseline to compare against, so early on
    this is the honest way to flag a real, current concentration of
    reports. If `public` is True, each hotspot returns only a center point,
    radius, and severity label — never a report count or any individual
    report data, so the public map can warn people away from an area
    without exposing who reported what.
    """
    df = fetch_reports_df(days=days)
    if df.empty:
        return []
    df["reported_at"] = pd.to_datetime(df["reported_at"], utc=True)
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=window_hours)
    recent = df[df["reported_at"] >= cutoff]
    if recent.empty:
        return []

    coords = recent[["lat", "lng"]].to_numpy()
    labels = DBSCAN(eps=0.01, min_samples=3).fit(coords).labels_
    recent = recent.copy()
    recent["cluster"] = labels

    hotspots = []
    for label, group in recent[recent["cluster"] != -1].groupby("cluster"):
        count = len(group)
        if count < min_reports:
            continue
        entry = {
            "id": int(label),
            "center_lat": round(float(group["lat"].mean()), 5),
            "center_lng": round(float(group["lng"].mean()), 5),
            "radius_m": 1200,
            "severity": "high" if count >= min_reports * 2 else "moderate",
        }
        if not public:
            entry["report_count"] = count
            entry["symptoms"] = group["symptom"].value_counts().to_dict()
        hotspots.append(entry)
    hotspots.sort(key=lambda h: (h["severity"] != "high"))
    return hotspots


# ---------------------------------------------------------------------------
# Public routes (no auth — aggregated only)
# ---------------------------------------------------------------------------
@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/api/public/hotspots")
def public_hotspots():
    return jsonify(compute_hotspots(public=True))


# ---------------------------------------------------------------------------
# Admin routes (raw data — never exposed to regular users)
# ---------------------------------------------------------------------------
@app.route("/api/admin/reports")
@require_admin
def admin_reports():
    days = request.args.get("days", default=30, type=int)
    conn = get_db()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    rows = conn.execute(
        """
        SELECT reports.id, reports.lat, reports.lng, reports.symptom, reports.reported_at,
               users.username
        FROM reports JOIN users ON reports.user_id = users.id
        WHERE reports.reported_at >= ?
        ORDER BY reports.reported_at DESC
        """,
        (cutoff,),
    ).fetchall()
    return jsonify([dict(r) for r in rows])


@app.route("/api/admin/clusters")
@require_admin
def admin_clusters():
    return jsonify(compute_hotspots(min_reports=1, public=False))


@app.route("/api/admin/summary")
@require_admin
def admin_summary():
    conn = get_db()
    total_reports = conn.execute("SELECT COUNT(*) AS c FROM reports").fetchone()["c"]
    total_users = conn.execute("SELECT COUNT(*) AS c FROM users").fetchone()["c"]
    today = datetime.now(timezone.utc).date().isoformat()
    reports_today = conn.execute(
        "SELECT COUNT(*) AS c FROM reports WHERE reported_at >= ?", (today,)
    ).fetchone()["c"]
    active_hotspots = len(compute_hotspots(public=True))
    return jsonify(
        {
            "total_reports": total_reports,
            "total_users": total_users,
            "reports_today": reports_today,
            "active_hotspots": active_hotspots,
        }
    )


# Initialize the database at import time (not just under __main__) so a
# production WSGI server like gunicorn — which imports this module and never
# runs the block below — still gets a ready database.
init_db()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
