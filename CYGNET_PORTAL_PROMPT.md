# Prompt: Build the Cygnet SSO Portal + align Cygnet Research Terminal

> Run as two phases, in order. **Phase A** modifies `~/Projects/CygnetResearchTerminal`
> (work on a fresh feature branch — the tree may carry unrelated in-flight
> changes; don't commit those). **Phase B** creates a new repo. Everything in
> Context is the established, **already-deployed** contract — QuantNews is the
> live reference implementation. Do not redesign it; implement against it.

---

## Context (already live — do not change)

Cygnet Systems runs sister apps on the Railway project **cygnet-systems**, all
under one parent domain so a single domain-scoped cookie is the SSO:

| App | Domain | Stack | Status |
|---|---|---|---|
| Cygnet Research Terminal | terminal.cygnetsystems.us | Dash 2.x on Flask, flask-login | live |
| QuantNews | quantnews.cygnetsystems.us | Dash on FastAPI/uvicorn | live (**reference implementation**) |
| Portal (this task) | portal.cygnetsystems.us | React SPA + FastAPI auth backend | to build |

**Shared auth database** = the Terminal's own Postgres (Railway internal
`postgres.railway.internal:5432/railway`). The Terminal reaches it via its
normal `DATABASE_URL`; QuantNews and the Portal point `AUTH_DATABASE_URL` at
it. Two tables, both already in production use by both live apps:

- `users` (Terminal owner: `user_store.py`): `uid` PK String(32), first/middle/last,
  `role`, `status` ('Active'/…), `password_hash` (**sha256 hex of plaintext** —
  legacy scheme, keep for now), cygnet_email, private_email, mobile, builtin,
  created_at/updated_at. Note: `user_store.load_all()` returns dicts keyed
  `password` (mapped from `password_hash`) — QuantNews's copy mirrors this.
- `sessions` (Terminal owner: `session_store.py`): `token_hash` PK (sha256 hex of
  the raw token — the raw token is never stored), uid, role, login_time,
  last_activity (30s-debounced touch), `absolute_expiry` (**7 days**, no idle
  timeout), ip, user_agent, `revoked_at` (NULL = active), created_at.
- `auth_events` (Terminal owner: `audit.py`): append-only audit — id BigInt PK,
  event_type String(48), uid, session_token_hash, ip, user_agent,
  `metadata` JSONB, created_at. The Portal writes its auth events **here** so
  all SSO logins land in one audit trail.

**The SSO cookie contract** (reference: `quant-news/services/auth_service.py`
plus the route layer at `quant-news/app.py:160-229` — copy the semantics
exactly, don't reinvent):

- Cookie name `cygnet_session`, `Domain=.cygnetsystems.us`, HttpOnly, Secure,
  **SameSite=Lax**, `Path=/`, 7-day max_age.
  - *Lax, not Strict*: the portal→app handoff and any external deep link
    arrive as top-level navigations; Strict drops the cookie on exactly those
    hops. (See the comment block at `quant-news/app.py:168-171`.)
- Value = `itsdangerous.URLSafeTimedSerializer(SESSION_COOKIE_SECRET_KEY,
  salt=SESSION_COOKIE_SALT).dumps(raw_session_token)` with
  `SESSION_COOKIE_SALT=cygnet-session-cookie`. Unsign window 7 days.
- Validation per request: unsign cookie → sha256 the raw token → `sessions`
  row (not revoked, not past absolute_expiry) → users kill-switch (`status ==
  'Active'` AND `role` matches the session row, else revoke) → identity.
  Cache the validated session ~5s keyed by the signed cookie string; cache the
  users table ~60s in-process.
- Because the cookie is domain-scoped and sessions are server-side rows:
  **login anywhere = logged in everywhere; revoking the session row = logged
  out everywhere** (within the ≤5s/60s cache TTLs on other devices).

**Token handoff (fallback / deep links, e.g. before DNS or cross-domain):**
sign the **raw** session token with the same secret, salt
`cygnet-sso-handoff`, 60-second validity →
`GET https://<app>/sso/login?token=<signed>&next=/`. QuantNews already serves
this route (`app.py:207-219`): unsign → validate the session row → set the
local cookie → 303 to `next`. Forged/expired tokens → 401. `next` must pass
`next.startswith("/") and not next.startswith("//")`, else fall back to `/`
(open-redirect guard).

**Identity probe** every app exposes: `GET /auth/whoami` →
`{"authenticated": bool, "uid": str|null, "name": str|null, "role": str|null}`
where `name` = `f"{first} {last}".strip() or uid`.

**Shared environment matrix** — identical on portal, terminal, quantnews
(QuantNews already has them set; read live values with
`railway variables --service <name>`):

```
SESSION_COOKIE_SECRET_KEY=<existing shared secret — read from the
                           cygnet-research-terminal Railway service>
SESSION_COOKIE_NAME=cygnet_session
SESSION_COOKIE_SALT=cygnet-session-cookie
COOKIE_DOMAIN=.cygnetsystems.us
COOKIE_SECURE=true
```

Plus, on **QuantNews and Portal only**: `AUTH_DATABASE_URL=<Terminal DB
internal URL>`. The Terminal does **not** need it — its `DATABASE_URL` already
is the auth DB.

Local dev: leave the env unset → each app falls back to its app-scoped
host-only cookie (`crt_session` / `qn_session`), no domain attribute. This is
also the production rollback: unset the three cookie vars and the Terminal
reverts to standalone auth.

---

## Task A — Modify CygnetResearchTerminal (repo: `~/Projects/CygnetResearchTerminal`)

Goal: the Terminal joins the shared cookie without logging anyone out and
without touching its 30+ `@auth_guard`/`@admin_guard` callbacks (they read
`current_user` and need no changes).

**Verified call-site map** (line numbers as of this writing — trust the
symbol names over the numbers if they've drifted):

| Site | What's there today |
|---|---|
| `auth.py:32` | `COOKIE_NAME = "crt_session"` hardcoded |
| `auth.py:38` | `PERSISTENT_COOKIE_MAX_AGE = 7d` — keep, already matches the contract |
| `auth.py:134-154` | `_serializer()` hardcodes salt `"crt-session-cookie"`; `sign_token`/`unsign_token` are single-salt |
| `auth.py:173` | `_request_loader` — the legacy-cookie fallback goes here |
| `auth.py:357-375` | `init_app` — sets `SESSION_COOKIE_SAMESITE = "Strict"` and Flask's `SESSION_COOKIE_NAME` |
| `app.py:11054` | `_attach_session_cookie` — `samesite="Strict"`, no `domain` |
| `app.py:11074` | `_clear_session_cookie` — same, also called by `check_session_expiry` at `app.py:11182` |
| `app.py:92-103` | before_request allowlist hook (`_AUTH_PATHS`) — **no change needed**, see step 4 |
| `tests/test_auth_flow.py` | 16 integration tests; `test_keep_me_*` assert cookie attributes and will need updating |

1. **Env-driven cookie identity** in `auth.py`:
   `COOKIE_NAME = os.environ.get("SESSION_COOKIE_NAME", "crt_session")`,
   `COOKIE_SALT = os.environ.get("SESSION_COOKIE_SALT", "crt-session-cookie")`,
   `COOKIE_DOMAIN = os.environ.get("COOKIE_DOMAIN") or None`. Give
   `sign_token`/`unsign_token` optional `salt=`/`max_age=` params (mirror the
   signatures at `quant-news/services/auth_service.py:185-195`) so the same
   helpers serve the cookie salt and the `cygnet-sso-handoff` salt.
2. **Fix the cookie attributes, once.** Refactor `_attach_session_cookie` /
   `_clear_session_cookie` into `auth.py` helpers that take a response object
   (`set_session_cookie(resp, raw_token, persistent)` /
   `clear_session_cookie(resp)`); Dash callbacks pass `dash.ctx.response`, the
   new Flask routes pass their own. In them: `samesite="Lax"` (was Strict —
   see contract rationale), `domain=COOKIE_DOMAIN` on **both set and delete**
   (a `Domain=` cookie cannot be cleared without a matching `domain` kwarg),
   keep httponly/secure/path and the persistent-vs-session `max_age` branch
   (`login-keep-me` checkbox behavior is unchanged). In `init_app`, change
   `SESSION_COOKIE_SAMESITE` to `"Lax"` and **stop setting Flask's
   `SESSION_COOKIE_NAME`** — that config names Flask's *own* `flask.session`
   cookie; pointing it at `cygnet_session` means any accidental
   `flask.session` write would clobber the family-wide SSO cookie.
3. **Migration window** in `_request_loader`: if the configured cookie is
   absent from the request, also try the legacy `crt_session` cookie unsigned
   with the legacy `crt-session-cookie` salt, so already-signed-in users keep
   their session; order the checks so a legacy-salt miss does **not** emit a
   `cookie_tamper` audit event. On successful login and on logout,
   additionally delete the legacy cookie so stale copies don't linger. Remove
   the fallback after a week — `ABSOLUTE_TTL` is 7 days, so no legacy session
   can outlive it.
4. **Two Flask routes** matching the QuantNews contract exactly (these are the
   first plain `@server.route` endpoints in `app.py`; put them next to the
   before_request hook). Do **not** extend `_AUTH_PATHS`: it only pre-warms
   the loader for Dash paths — `current_user` resolves lazily on access
   inside any view.
   - `GET /sso/login?token=&next=` — unsign with salt `cygnet-sso-handoff`,
     `max_age=60`; `session_store.read(raw)` must return a row, else 401;
     validate `next` (relative-path guard above); 303 redirect with the
     cookie set persistent (mirror `quant-news/app.py:207-219`).
   - `GET /auth/whoami` — `flask.jsonify` of the contract shape;
     `name = f"{first} {last}".strip() or uid`.
5. **De-fang the heartbeat.** Remove the `_clear_session_cookie()` call from
   `check_session_expiry` (`app.py:11182`). Today it deletes the app-local
   cookie whenever a 60s tick finds the request unauthenticated; under a
   domain-scoped cookie, one app's transient DB failure would sign the user
   out of the entire family. Cookie deletion belongs to explicit logout (and
   the login/logout legacy cleanup) only — the overlay flip is still correct.
6. **Tests** (`pytest tests/test_auth_flow.py tests/test_session_store.py`):
   update the `test_keep_me_*` attribute asserts (SameSite → Lax); add:
   legacy-cookie fallback authenticates (and stops after the env flip + fresh
   login), `/sso/login` happy path + expired token + tampered token + `next`
   validation, `/auth/whoami` authenticated and anonymous shapes.
7. **Deploy**: set `SESSION_COOKIE_NAME`/`SESSION_COOKIE_SALT`/`COOKIE_DOMAIN`
   on the `cygnet-research-terminal` Railway service (secret + DB are already
   there). Verify: existing session survives the deploy (legacy fallback);
   fresh login sets `cygnet_session` on `.cygnetsystems.us`; after logging
   into the Terminal, `https://quantnews.cygnetsystems.us/auth/whoami` reports
   the same user without a second login (and vice versa); logout in either
   app logs out both.

## Task B — Create the Cygnet Portal (new repo, deploy as `portal` service)

A React-based website and team login portal at **portal.cygnetsystems.us** —
the SSO entry point and application catalog for Cygnet Systems.

**Stack**: Vite + React + TypeScript frontend; FastAPI backend that serves the
built SPA and the auth endpoints. Keep it small: two client routes, no state
library, `fetch()` only.

**Backend auth module**: copy `quant-news/services/auth_service.py` (tables,
signing, session lifecycle, caches, kill-switch — all battle-tested) with
exactly three adaptations:
1. Drop `import config as _config` (QuantNews-specific dotenv side effect) —
   call `load_dotenv()` directly.
2. `AUTH_DATABASE_URL` is **required** (no fallback to an app DB — the portal
   has no other database).
3. Replace `_audit()` — it writes to QuantNews's app-local `activity_log`.
   The portal instead inserts into the shared **`auth_events`** table
   (schema in Context; column `metadata` is JSONB), emitting
   `login_success`, `login_failure_unknown_user`,
   `login_failure_wrong_password`, `login_failure_inactive`, `logout_user`
   with ip/user_agent, so portal logins appear in the same audit trail the
   Terminal admins already query.

**Backend endpoints**:
- `POST /auth/login` — JSON `{userid, password}` → validates via the shared
  store (case-insensitive uid match is already in `attempt_login` — keep it),
  creates a `sessions` row, sets the `cygnet_session` cookie (persistent,
  7-day), audit-logs → `200` with the whoami shape. Failures → `401
  {"error": msg}` ("Invalid User ID or Password." / "Account is not active.").
- `GET /auth/logout` → revoke the row, clear the cookie **with
  `domain=COOKIE_DOMAIN`**, invalidate caches, 303 to `/` (domain-scoped ⇒
  logs out every app).
- `GET /auth/whoami` → contract shape.
- `GET /handoff?app=<key>&next=/` → requires auth; recover the **raw** session
  token by unsigning the caller's own cookie (the resolved user object only
  carries `token_hash` — the hash cannot be re-signed); re-sign it with salt
  `cygnet-sso-handoff`; 302 to `https://<app-domain>/sso/login?token=…&next=…`.
  Unknown `app` key → 404. (Fallback path only — with the shared cookie,
  plain links to the apps are already authenticated.)
- Serve the SPA build for everything else.

**Frontend**:
- Login page: userid + password → `POST /auth/login`; inline error display;
  already-authenticated visitors (per whoami on load) go straight to the
  catalog.
- Catalog page: cards from a static config —
  ```ts
  [{ key: "terminal",  name: "Cygnet Research Terminal",
     url: "https://terminal.cygnetsystems.us",
     description: "Screeners, baskets, pairs & research tooling" },
   { key: "quantnews", name: "QuantNews",
     url: "https://quantnews.cygnetsystems.us",
     description: "Stock dashboard with news, ML signals & AI research reports" }]
  ```
  Each card: name, description, Open button (plain `<a>` — the shared cookie
  does the rest). Header: user name + Sign out. Session state comes from the
  **portal's own** `/auth/whoami` once per load — do **not** fetch the other
  apps' whoami cross-origin (needless CORS surface; one shared session means
  one state for all cards).
- **Branding**: dark theme matching the sister apps. Use the Terminal's core
  tokens — bg `#0a0d14`, panel `#12172a`, card `#1a2130`, text `#dde3ed`,
  bright `#f5f8fc`, muted `#a5b0c2`, accent `#38bdf8` (hover `#7dd3fc`) —
  fonts Inter (UI) / JetBrains Mono (ids, data). "Cygnet Systems" wordmark.

**Deployment**: new service `portal` in the `cygnet-systems` Railway project;
two-stage Dockerfile (node build → python runtime serving SPA + API); the
shared env matrix + `AUTH_DATABASE_URL`; custom domain portal.cygnetsystems.us
(CNAME per Railway dashboard).

**Acceptance**:
1. Login at the portal → open Terminal and QuantNews → both already signed in
   (whoami shows the uid; QuantNews sidebar chip shows the name; the Terminal
   lands past the login overlay via its server-rendered `serve_layout`).
2. Sign out anywhere → all three anonymous: immediately in that browser
   (cookie cleared domain-wide), within ~5s elsewhere (session-cache TTL).
3. Wrong password → inline error; inactive user → "Account is not active.";
   flipping `users.status` in the DB revokes access everywhere within ~60s.
4. `/handoff?app=quantnews` deep-link works; a tampered or >60s-old handoff
   token → 401; `next=//evil.example` is coerced to `/`.
5. Anonymous visit to portal → login page; the apps remain publicly browsable
   (their data is public by default — intended).
6. Terminal suite green: `pytest tests/test_auth_flow.py tests/test_session_store.py`.

**Security notes to preserve**: HttpOnly cookie (UI reads identity only via
whoami); raw tokens never persisted (sha256 in DB); kill-switch = flipping
`users.status`/`role` revokes everywhere within the cache TTLs; `next` and
handoff tokens validated as specified. Keep sha256 password hashing for
compatibility now, but leave a TODO to migrate to argon2 with rehash-on-login
across all apps simultaneously, and a TODO for rate-limiting the portal's
login endpoint (it is the family's public credential surface).
