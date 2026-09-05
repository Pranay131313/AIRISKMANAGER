import sqlite3
from contextlib import contextmanager

DB_PATH = "db/risk_manager.db"


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS alert_investigations (
                alert_id TEXT PRIMARY KEY,
                merchant_id TEXT,
                day TEXT,
                status TEXT DEFAULT 'Open',
                analyst_note TEXT DEFAULT '',
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()


def upsert_alert_status(alert_id, merchant_id, day, status, note=""):
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO alert_investigations (alert_id, merchant_id, day, status, analyst_note, updated_at)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(alert_id) DO UPDATE SET
                status=excluded.status,
                analyst_note=excluded.analyst_note,
                updated_at=CURRENT_TIMESTAMP
            """,
            (alert_id, merchant_id, str(day), status, note),
        )
        conn.commit()


def get_alert_status(alert_id):
    with get_conn() as conn:
        cur = conn.execute(
            "SELECT status, analyst_note FROM alert_investigations WHERE alert_id = ?",
            (alert_id,),
        )
        row = cur.fetchone()
        if row:
            return {"status": row[0], "analyst_note": row[1]}
        return {"status": "Open", "analyst_note": ""}


def get_all_statuses():
    with get_conn() as conn:
        cur = conn.execute("SELECT alert_id, status, analyst_note FROM alert_investigations")
        return {r[0]: {"status": r[1], "analyst_note": r[2]} for r in cur.fetchall()}


if __name__ == "__main__":
    init_db()
    print("Database initialised at", DB_PATH)
