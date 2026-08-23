import os
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv(dotenv_path="/etc/secrets/.env", override=False)
load_dotenv(dotenv_path=".env", override=False)


def get_conn():
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL not set in environment")
    conn = psycopg2.connect(url, cursor_factory=psycopg2.extras.RealDictCursor)
    return conn


def init_db():
    with get_conn() as conn:
        with conn.cursor() as cur:

            # users table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id          TEXT PRIMARY KEY,
                    username    TEXT UNIQUE NOT NULL,
                    email       TEXT UNIQUE NOT NULL,
                    hashed_pw   TEXT NOT NULL,
                    tenant_id   TEXT NOT NULL,
                    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)

            # report history table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS reports (
                    id          TEXT PRIMARY KEY,
                    user_id     TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    topic       TEXT NOT NULL,
                    filename    TEXT NOT NULL,
                    pdf_bytes   BYTEA,
                    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)

        conn.commit()
    print("[DB] Tables ready")
