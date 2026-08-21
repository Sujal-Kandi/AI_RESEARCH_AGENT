import os
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from schemas import UserData

# ── Bcrypt workaround for passlib version mismatch ────────────────────────────
import bcrypt
if not hasattr(bcrypt, "__about__"):
    class _BcryptAbout:
        __version__ = getattr(bcrypt, "__version__", "4.0.1")
    bcrypt.__about__ = _BcryptAbout()

from passlib.context import CryptContext

# ── Config ────────────────────────────────────────────────────────────────────
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "dev-fallback-secret-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 3  # 3 days

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/login")

# ── Database setup ────────────────────────────────────────────────────────────
DB_PATH = Path(__file__).parent / "users.db"


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create the users table if it doesn't exist."""
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id          TEXT PRIMARY KEY,
                username    TEXT UNIQUE NOT NULL COLLATE NOCASE,
                email       TEXT UNIQUE NOT NULL COLLATE NOCASE,
                hashed_pw   TEXT NOT NULL,
                tenant_id   TEXT NOT NULL,
                created_at  TEXT NOT NULL
            )
        """)
        conn.commit()


# Initialise on import
init_db()


# ── Password helpers ──────────────────────────────────────────────────────────
def hash_password(plain: str) -> str:
    return pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


# ── User CRUD ─────────────────────────────────────────────────────────────────
def get_user_by_username(username: str) -> sqlite3.Row | None:
    with _get_conn() as conn:
        return conn.execute(
            "SELECT * FROM users WHERE username = ? COLLATE NOCASE", (username,)
        ).fetchone()


def get_user_by_email(email: str) -> sqlite3.Row | None:
    with _get_conn() as conn:
        return conn.execute(
            "SELECT * FROM users WHERE email = ? COLLATE NOCASE", (email,)
        ).fetchone()


def create_user(username: str, email: str, plain_password: str) -> sqlite3.Row:
    """
    Insert a new user. Raises HTTPException on duplicate username / email.
    Returns the newly created row.
    """
    if get_user_by_username(username):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username already taken"
        )
    if get_user_by_email(email):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered"
        )

    user_id   = str(uuid.uuid4())
    tenant_id = f"org_{user_id[:8]}"
    hashed_pw = hash_password(plain_password)
    now       = datetime.now(timezone.utc).isoformat()

    with _get_conn() as conn:
        conn.execute(
            """
            INSERT INTO users (id, username, email, hashed_pw, tenant_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (user_id, username, email, hashed_pw, tenant_id, now),
        )
        conn.commit()

    return get_user_by_username(username)


# ── JWT helpers ───────────────────────────────────────────────────────────────
def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(token: str = Depends(oauth2_scheme)) -> UserData:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        tenant_id: str = payload.get("tenant_id")
        email: str = payload.get("email", "")
        if not username:
            raise credentials_exception
        # Verify user still exists in DB
        row = get_user_by_username(username)
        if row is None:
            raise credentials_exception
        return UserData(username=username, email=email, tenant_id=tenant_id)
    except JWTError:
        raise credentials_exception
