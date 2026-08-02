"""Cygnet SSO auth for the FastAPI backend.

Mirrors the CygnetResearchTerminal auth contract so the two apps work as
sister applications behind one login:

- **users** and **sessions** tables have the same shape as CRT's
  (`user_store.UserModel`, `session_store.Session`). Point
  ``AUTH_DATABASE_URL`` at the shared auth Postgres on Railway and both apps
  validate the same accounts and sessions; locally it defaults to the app DB
  and the tables are created on startup if missing.
- **Cookie**: this app sets its own ``qn_session`` cookie (cookies cannot
  cross ``*.up.railway.app`` subdomains — that suffix is on the Public
  Suffix List), signed with the SHARED ``SESSION_COOKIE_SECRET_KEY`` via
  itsdangerous, salt ``qn-session-cookie``.
- **SSO handoff** (portal contract): the future portal signs the RAW session
  token with the shared secret and salt ``cygnet-sso-handoff`` and redirects
  to ``/sso/login?token=<signed>&next=/``. This app unsigns (60s window),
  validates the session row in the shared store, and sets its local cookie.
  CRT can adopt the same route; the portal then logs a user in once and
  deep-links into either app.

Identity is exposed through a ContextVar set by the ASGI middleware, so sync
callbacks (threadpool) and ``asyncio.to_thread`` work inherit it. Anonymous
requests are allowed everywhere — data defaults to public; ownership only
attaches when someone is signed in.
"""

from __future__ import annotations

import hashlib
import logging
import os
import secrets
import threading
import time
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from itsdangerous import BadSignature, URLSafeTimedSerializer
from sqlalchemy import (
    Boolean, DateTime, Index, String, Text, create_engine, select, update,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

import config as _config  # noqa: F401 — imported for its load_dotenv() side effect

logger = logging.getLogger(__name__)

# Cookie identity. Local default is app-scoped (qn_session, host-only).
# Production SSO on the cygnetsystems.us domain family sets, on ALL apps
# (portal, terminal, quantnews):
#   SESSION_COOKIE_NAME=cygnet_session
#   SESSION_COOKIE_SALT=cygnet-session-cookie
#   COOKIE_DOMAIN=.cygnetsystems.us
# Same name + salt + secret + auth DB = one login works everywhere; the
# server-side session row means logout anywhere revokes everywhere.
COOKIE_NAME = os.environ.get("SESSION_COOKIE_NAME", "qn_session")
COOKIE_SALT = os.environ.get("SESSION_COOKIE_SALT", "qn-session-cookie")
COOKIE_DOMAIN = os.environ.get("COOKIE_DOMAIN") or None
SSO_HANDOFF_SALT = "cygnet-sso-handoff"
SSO_HANDOFF_MAX_AGE = 60  # seconds — the portal redirect is immediate
PERSISTENT_COOKIE_MAX_AGE = 7 * 24 * 60 * 60  # matches CRT
ABSOLUTE_TTL = timedelta(days=7)
TOUCH_DEBOUNCE = timedelta(seconds=30)
_SESSION_CACHE_TTL = 5.0  # matches CRT's per-callback-burst cache


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _hash_token(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _aware(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is None:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Tables — same shapes as CRT's user_store / session_store, so a shared
# AUTH_DATABASE_URL means shared accounts and shared sessions.
# ---------------------------------------------------------------------------

class AuthBase(DeclarativeBase):
    pass


class AuthUser(AuthBase):
    __tablename__ = "users"

    uid: Mapped[str] = mapped_column(String(32), primary_key=True)
    first: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    middle: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    last: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    role: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="Active")
    password_hash: Mapped[str] = mapped_column(Text, nullable=False, default="")
    cygnet_email: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    private_email: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    mobile: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    builtin: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now, onupdate=_now)

    __table_args__ = (Index("ix_users_status", "status"),)


class AuthSession(AuthBase):
    __tablename__ = "sessions"

    token_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    uid: Mapped[str] = mapped_column(String(64), nullable=False)
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    login_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    last_activity: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    absolute_expiry: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    ip: Mapped[Optional[str]] = mapped_column(String(45), nullable=True)
    user_agent: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    revoked_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now)

    __table_args__ = (Index("ix_sessions_absolute_expiry", "absolute_expiry"),)


# ---------------------------------------------------------------------------
# Auth DB engine — separate from the app engine so it can point at the shared
# Cygnet auth database without touching app data.
# ---------------------------------------------------------------------------

_auth_engine = None
_AuthSessionLocal = None
_engine_lock = threading.Lock()


def _get_auth_engine():
    global _auth_engine, _AuthSessionLocal
    if _auth_engine is None:
        with _engine_lock:
            if _auth_engine is None:
                from config import DB
                url = os.getenv("AUTH_DATABASE_URL", DB.URL)
                _auth_engine = create_engine(
                    url, pool_size=3, max_overflow=5, pool_pre_ping=True)
                _AuthSessionLocal = sessionmaker(
                    bind=_auth_engine, expire_on_commit=False)
    return _auth_engine


def _auth_session():
    _get_auth_engine()
    return _AuthSessionLocal()


def ensure_auth_tables() -> None:
    """Create users/sessions if absent (no-op against the shared DB, which
    already has them from CRT). Called from app startup, never at import."""
    try:
        AuthBase.metadata.create_all(_get_auth_engine(), checkfirst=True)
    except Exception as e:
        logger.error(f"auth tables unavailable: {e}")


# ---------------------------------------------------------------------------
# Signing
# ---------------------------------------------------------------------------

def _secret() -> str:
    secret = os.environ.get("SESSION_COOKIE_SECRET_KEY")
    if not secret:
        raise RuntimeError(
            "SESSION_COOKIE_SECRET_KEY is required (shared across Cygnet "
            "apps). Generate with `python -c 'import secrets; "
            "print(secrets.token_urlsafe(48))'`.")
    return secret


def sign_token(raw_token: str, salt: str = COOKIE_SALT) -> str:
    return URLSafeTimedSerializer(_secret(), salt=salt).dumps(raw_token)


def unsign_token(signed: str, salt: str = COOKIE_SALT,
                 max_age: int = PERSISTENT_COOKIE_MAX_AGE) -> Optional[str]:
    try:
        return URLSafeTimedSerializer(_secret(), salt=salt).loads(
            signed, max_age=max_age)
    except (BadSignature, Exception):
        return None


# ---------------------------------------------------------------------------
# Current-user identity (ContextVar — inherited by threadpool + to_thread)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CurrentUser:
    uid: str
    role: str
    first: str = ""
    last: str = ""
    token_hash: str = ""

    @property
    def is_admin(self) -> bool:
        return self.role == "Administrator"

    @property
    def display_name(self) -> str:
        return (f"{self.first} {self.last}".strip()) or self.uid


_current_user: ContextVar[Optional[CurrentUser]] = ContextVar(
    "qn_current_user", default=None)


def current_user() -> Optional[CurrentUser]:
    return _current_user.get()


def current_uid() -> Optional[str]:
    u = _current_user.get()
    return u.uid if u else None


def set_current_user(user: Optional[CurrentUser]):
    return _current_user.set(user)


# ---------------------------------------------------------------------------
# Session lifecycle (CRT session_store semantics)
# ---------------------------------------------------------------------------

_session_cache: dict[str, tuple] = {}  # signed_cookie -> (expiry_monotonic, CurrentUser)
_users_cache: dict = {"loaded_at": 0.0, "data": {}}
_USERS_TTL = 60.0


def _load_users_cached() -> dict:
    now = time.time()
    if now - _users_cache["loaded_at"] < _USERS_TTL:
        return _users_cache["data"]
    try:
        with _auth_session() as s:
            rows = s.scalars(select(AuthUser)).all()
            _users_cache["data"] = {
                r.uid: {"first": r.first or "", "last": r.last or "",
                        "role": r.role, "status": r.status,
                        "password": r.password_hash or ""}
                for r in rows
            }
        _users_cache["loaded_at"] = now
    except Exception as e:
        logger.debug(f"user load failed, serving last snapshot: {e}")
    return _users_cache["data"]


def invalidate_caches() -> None:
    _session_cache.clear()
    _users_cache["loaded_at"] = 0.0


def create_session(uid: str, role: str, ip: Optional[str],
                   user_agent: Optional[str]) -> str:
    """Mint a raw token + session row; return the raw token."""
    raw_token = secrets.token_urlsafe(32)
    now = _now()
    with _auth_session() as s:
        s.add(AuthSession(
            token_hash=_hash_token(raw_token), uid=uid, role=role,
            login_time=now, last_activity=now,
            absolute_expiry=now + ABSOLUTE_TTL,
            ip=ip, user_agent=(user_agent or "")[:512] or None,
        ))
        s.commit()
    return raw_token


def _read_session(raw_token: str) -> Optional[AuthSession]:
    with _auth_session() as s:
        row = s.get(AuthSession, _hash_token(raw_token))
        if row is None or row.revoked_at is not None:
            return None
        if _now() >= _aware(row.absolute_expiry):
            return None
        # touch (debounced) — best-effort
        if _now() - _aware(row.last_activity) >= TOUCH_DEBOUNCE:
            try:
                s.execute(update(AuthSession)
                          .where(AuthSession.token_hash == row.token_hash)
                          .values(last_activity=_now()))
                s.commit()
            except Exception:
                pass
        return row


def revoke_session(token_hash: str) -> None:
    try:
        with _auth_session() as s:
            s.execute(update(AuthSession)
                      .where(AuthSession.token_hash == token_hash,
                             AuthSession.revoked_at.is_(None))
                      .values(revoked_at=_now()))
            s.commit()
    except Exception:
        pass


def resolve_cookie(signed_cookie: Optional[str]) -> Optional[CurrentUser]:
    """Signed cookie -> CurrentUser, or None (anonymous). Short-TTL cached
    per cookie so a burst of Dash callbacks costs one session read."""
    if not signed_cookie:
        return None
    mono = time.monotonic()
    hit = _session_cache.get(signed_cookie)
    if hit is not None and hit[0] > mono:
        user = hit[1]
    else:
        raw = unsign_token(signed_cookie)
        if raw is None:
            _session_cache.pop(signed_cookie, None)
            return None
        try:
            row = _read_session(raw)
        except Exception:
            return None
        if row is None:
            _session_cache.pop(signed_cookie, None)
            return None
        user = CurrentUser(uid=row.uid, role=row.role,
                           token_hash=row.token_hash)
        _session_cache[signed_cookie] = (mono + _SESSION_CACHE_TTL, user)
        if len(_session_cache) > 512:
            for k in [k for k, v in _session_cache.items() if v[0] <= mono]:
                _session_cache.pop(k, None)

    # Kill-switch check every request (in-process user cache, like CRT)
    profile = _load_users_cached().get(user.uid)
    if profile is None or profile.get("status") != "Active" \
            or profile.get("role") != user.role:
        _session_cache.pop(signed_cookie, None)
        revoke_session(user.token_hash)
        return None
    return CurrentUser(uid=user.uid, role=user.role,
                       first=profile.get("first", ""),
                       last=profile.get("last", ""),
                       token_hash=user.token_hash)


def attempt_login(userid: str, password: str, ip: Optional[str],
                  user_agent: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """Validate credentials against the shared user store.
    Returns (raw_token, error_message)."""
    uid = (userid or "").strip()
    pwd = (password or "").strip()
    if not uid or not pwd:
        return None, "Please enter User ID and Password."

    users = _load_users_cached()
    user = users.get(uid)
    if not user:  # case-insensitive, canonical uid wins (CRT behavior)
        up = uid.upper()
        for k, v in users.items():
            if k.upper() == up:
                uid, user = k, v
                break
    if not user:
        _audit("login_failure_unknown_user", uid, ip)
        return None, "Invalid User ID or Password."
    if user.get("status") != "Active":
        _audit("login_failure_inactive", uid, ip)
        return None, "Account is not active."
    if hashlib.sha256(pwd.encode()).hexdigest() != user.get("password", ""):
        _audit("login_failure_wrong_password", uid, ip)
        return None, "Invalid User ID or Password."

    raw_token = create_session(uid, user.get("role", ""), ip, user_agent)
    _audit("login_success", uid, ip)
    return raw_token, None


def _audit(event: str, uid: str, ip: Optional[str]) -> None:
    """Auth events go to the app's activity_log (best-effort)."""
    try:
        from db.session import get_session
        from db.models import ActivityLog
        with get_session() as s:
            s.add(ActivityLog(user_id=uid or "unknown", run_id="auth",
                              stage="auth", message=f"{event} ip={ip or '?'}"))
    except Exception:
        pass
