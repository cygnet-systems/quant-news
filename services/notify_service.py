"""Email notifications for scheduled runs — morning calls, evening results.

The transport is inherited from CygnetResearchTerminal's ``cron_jobs/notify.py``:
Microsoft Graph ``sendMail`` with client-credentials auth, reading the SAME
environment variables (``AZURE_TENANT_ID``, ``AZURE_CLIENT_ID``,
``AZURE_CLIENT_SECRET``, ``NOTIFY_FROM_EMAIL``, ``NOTIFY_TO_EMAIL``). One Azure
app registration and one set of Railway variables serve both apps.

What is NOT inherited is the message shape. CRT reports whether a data-loading
job moved bytes; the interesting content here is what the platform decided and
whether it was right, so the two mails are:

* **Prediction** (after the morning analysis) — the call per symbol, what the
  models disagreed about, and what the run cost.
* **Result** (after the evening evaluation) — how those calls actually landed:
  hit rate, P&L, and which names were wrong.

Every function is best-effort. A mail server problem must never fail a run
that already produced its analysis.
"""

from __future__ import annotations

import logging
import os
from datetime import date
from html import escape as html_escape
from typing import Optional

import requests

logger = logging.getLogger(__name__)

_TOKEN_URL = "https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token"
_SEND_URL = "https://graph.microsoft.com/v1.0/users/{sender}/sendMail"

_REQUIRED = (
    "AZURE_TENANT_ID", "AZURE_CLIENT_ID", "AZURE_CLIENT_SECRET",
    "NOTIFY_FROM_EMAIL", "NOTIFY_TO_EMAIL",
)

_ACTION_COLOR = {"BUY": "#00C805", "SELL": "#FF5000", "HOLD": "#A0A0A0"}


def _config() -> Optional[dict]:
    cfg = {k: os.environ.get(k) for k in _REQUIRED}
    missing = [k for k, v in cfg.items() if not v]
    if missing:
        logger.debug(f"Email notifications disabled — missing: {', '.join(missing)}")
        return None
    return cfg


def enabled() -> bool:
    return _config() is not None


def _send(subject: str, html: str) -> bool:
    cfg = _config()
    if cfg is None:
        return False
    try:
        token = requests.post(
            _TOKEN_URL.format(tenant=cfg["AZURE_TENANT_ID"]),
            data={
                "client_id": cfg["AZURE_CLIENT_ID"],
                "client_secret": cfg["AZURE_CLIENT_SECRET"],
                "scope": "https://graph.microsoft.com/.default",
                "grant_type": "client_credentials",
            },
            timeout=15,
        )
        token.raise_for_status()
        access = token.json()["access_token"]

        resp = requests.post(
            _SEND_URL.format(sender=cfg["NOTIFY_FROM_EMAIL"]),
            headers={"Authorization": f"Bearer {access}",
                     "Content-Type": "application/json"},
            json={
                "message": {
                    "subject": subject,
                    "body": {"contentType": "HTML", "content": html},
                    "toRecipients": [
                        {"emailAddress": {"address": addr.strip()}}
                        for addr in cfg["NOTIFY_TO_EMAIL"].split(",") if addr.strip()
                    ],
                },
                "saveToSentItems": "false",
            },
            timeout=20,
        )
        if resp.status_code >= 400:
            logger.error(
                f"Notification send failed ({resp.status_code}): "
                f"{_diagnose(resp.status_code, cfg['NOTIFY_FROM_EMAIL'])}"
            )
            return False
        logger.info(f"Notification sent: {subject}")
        return True
    except Exception:
        logger.exception("Failed to send notification email — continuing")
        return False


def _diagnose(status: int, sender: str) -> str:
    """Turn a Graph status into the thing to actually go and fix.

    These three are indistinguishable in a raw traceback and have completely
    different fixes, so name them rather than making the reader guess.
    """
    if status == 401:
        return ("the token was rejected — check AZURE_TENANT_ID, "
                "AZURE_CLIENT_ID and AZURE_CLIENT_SECRET")
    if status == 403:
        return ("authenticated but not allowed — the app registration needs "
                "Mail.Send as an APPLICATION permission with admin consent, "
                "and any application access policy must include this mailbox")
    if status == 404:
        return (f"the tenant has no mailbox at {sender} — Graph resolves "
                f"/users/{{id}} by primary UPN or object id, so an alias or "
                f"proxy address will 404. Create it (a shared mailbox needs "
                f"no licence) or set NOTIFY_FROM_EMAIL to the primary UPN")
    if status == 429:
        return "throttled by Graph — the next scheduled run will try again"
    return "see the Graph response above"


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_STYLE = (
    "font-family:-apple-system,Segoe UI,Roboto,sans-serif;"
    "font-size:14px;color:#1a1a1a;line-height:1.5"
)
_TH = ("text-align:left;padding:6px 10px;border-bottom:2px solid #ddd;"
       "font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:#666")
_TD = "padding:6px 10px;border-bottom:1px solid #eee"


def _wrap(title: str, subtitle: str, body: str) -> str:
    return (
        f'<div style="{_STYLE}">'
        f'<h2 style="margin:0 0 2px">{title}</h2>'
        f'<div style="color:#666;font-size:13px;margin-bottom:16px">{subtitle}</div>'
        f'{body}'
        '<p style="color:#999;font-size:12px;margin-top:22px">'
        'Sent by quant-news. Predictions are model output, not advice — the '
        'platform has shown no demonstrable alpha at scale.</p></div>'
    )


def run_cost(since, until=None) -> Optional[float]:
    """LLM spend inside a time window.

    Attributed by time rather than by ``run_id``: the job executes the
    pipeline in a subprocess, which opens its own progress run, so the
    parent's id never appears on the usage rows the work actually wrote.
    """
    from datetime import datetime as _dt

    try:
        from db.session import get_session
        from sqlalchemy import text
        with get_session() as session:
            return session.execute(
                text("select sum(cost_usd) from llm_usage "
                     "where created_at >= :a and created_at <= :b"),
                {"a": since, "b": until or _dt.now()},
            ).scalar()
    except Exception as e:
        logger.debug(f"cost lookup failed: {e}")
        return None


def notify_analysis(summary: dict, cost: float | None = None,
                    duration_ms: int | None = None, log: str = "") -> bool:
    """The morning mail: what the platform is calling for today's session."""
    actions = summary.get("actions") or {}
    if not actions:
        # Two different things, and calling both "no recommendations" sent a
        # failure mail for eleven good runs: the pipeline can genuinely produce
        # no calls, or the caller can fail to read the summary it printed. Say
        # which, and show the run's own output either way.
        reason = ("The run reported success but its summary could not be read, "
                  "so what it produced is unknown. The predictions may well be "
                  "stored — check the Schedule page."
                  if not summary else
                  "The run completed and produced no per-symbol calls.")
        return notify_job_failure("daily_analysis", f"{reason}\n\n{log}".strip())

    order = {"BUY": 0, "SELL": 1, "HOLD": 2}
    rows = ""
    for sym, action in sorted(actions.items(),
                              key=lambda kv: (order.get(kv[1], 3), kv[0])):
        color = _ACTION_COLOR.get(action, "#666")
        rows += (f'<tr><td style="{_TD};font-weight:600">{sym}</td>'
                 f'<td style="{_TD};color:{color};font-weight:600">{action}</td></tr>')

    counts = {a: sum(1 for v in actions.values() if v == a)
              for a in ("BUY", "SELL", "HOLD")}
    meta = [f"{summary.get('predictions_stored', 0)} predictions stored"]
    if duration_ms:
        meta.append(f"{duration_ms // 1000}s")
    if cost:
        meta.append(f"${cost:.2f} in LLM calls")
    if summary.get("skipped"):
        meta.append(f"skipped: {', '.join(summary['skipped'])}")

    body = (
        f'<table style="border-collapse:collapse;min-width:280px">'
        f'<tr><th style="{_TH}">Symbol</th><th style="{_TH}">Call</th></tr>'
        f'{rows}</table>'
        f'<p style="color:#666;font-size:13px;margin-top:14px">'
        f'{counts["BUY"]} buy · {counts["SELL"]} sell · {counts["HOLD"]} hold<br>'
        f'{" · ".join(meta)}</p>'
    )
    subject = (f"Pre-open calls {summary.get('target_date', '')} — "
               f"{counts['BUY']}B/{counts['SELL']}S/{counts['HOLD']}H")
    return _send(subject, _wrap(
        "Pre-open calls",
        f"Target session {summary.get('target_date', '?')} · "
        f"data through {summary.get('as_of', '?')}",
        body,
    ))


def notify_evaluation(trade_date: str | None = None) -> bool:
    """The evening mail: how the calls actually landed."""
    from db.session import get_session
    from sqlalchemy import text

    target = trade_date or date.today().isoformat()
    try:
        with get_session() as session:
            per_model = session.execute(text("""
                select model_name,
                       count(*) filter (where decision <> 'HOLD') as active,
                       count(*) filter (where decision <> 'HOLD' and was_correct) as hits,
                       round(sum(pnl_dollars)::numeric, 2) as pnl
                from model_predictions
                where target_date = :d and was_correct is not null
                group by 1 order by 1
            """), {"d": target}).mappings().all()

            synthesis = session.execute(text("""
                select symbol, decision, was_correct,
                       round(pnl_dollars::numeric, 2) as pnl
                from model_predictions
                where target_date = :d and model_name = 'recommendation_synthesis'
                  and was_correct is not null
                order by symbol
            """), {"d": target}).mappings().all()
    except Exception as e:
        logger.warning(f"evaluation notification query failed: {e}")
        return False

    # "Nothing scored" has two very different causes — a holiday, or an
    # evaluator that keeps skipping the same rows. Check which one before
    # calling it normal.
    backlog = {}
    try:
        from services.cache_service import get_cache
        backlog = get_cache().evaluation_backlog()
    except Exception as e:
        logger.warning(f"evaluation backlog query failed: {e}")

    if not per_model:
        pending = backlog.get("pending_mature", 0)
        if pending:
            by_date = ", ".join(f"{d}: {n}" for d, n in
                                (backlog.get("by_target_date") or {}).items())
            return _send(
                f"EVALUATION STALLED — {pending} mature predictions unscored",
                _wrap("Nothing scored, but the backlog is not empty",
                      f"Target session {target}",
                      f"<p>{pending} predictions whose target session has "
                      f"closed are still unscored ({by_date}). The evaluator "
                      f"is skipping them — most likely a missing close for "
                      f"the target date or a NULL previous_close. This will "
                      f"not fix itself.</p>"))
        return _send(
            f"No results to score for {target}",
            _wrap("Nothing scored", f"Target session {target}",
                  "<p>No predictions were ready to evaluate, and no mature "
                  "prediction is waiting. Normal on a non-trading day, or "
                  "when the vendor has not published the session's close "
                  "yet.</p>"))

    model_rows = ""
    for r in per_model:
        acc = (r["hits"] / r["active"] * 100) if r["active"] else None
        pnl = float(r["pnl"] or 0)
        pnl_color = "#00A004" if pnl > 0 else "#D93900" if pnl < 0 else "#666"
        model_rows += (
            f'<tr><td style="{_TD}">{r["model_name"]}</td>'
            f'<td style="{_TD}">{r["active"]}</td>'
            f'<td style="{_TD}">{f"{acc:.0f}%" if acc is not None else "—"}</td>'
            f'<td style="{_TD};color:{pnl_color}">${pnl:,.2f}</td></tr>'
        )

    miss_rows = ""
    for r in synthesis:
        mark = "✓" if r["was_correct"] else "✗"
        color = "#00A004" if r["was_correct"] else "#D93900"
        miss_rows += (
            f'<tr><td style="{_TD};font-weight:600">{r["symbol"]}</td>'
            f'<td style="{_TD}">{r["decision"]}</td>'
            f'<td style="{_TD};color:{color}">{mark}</td>'
            f'<td style="{_TD}">${float(r["pnl"] or 0):,.2f}</td></tr>'
        )

    total_active = sum(r["active"] for r in per_model)
    total_hits = sum(r["hits"] for r in per_model)
    total_pnl = sum(float(r["pnl"] or 0) for r in per_model)
    overall = (total_hits / total_active * 100) if total_active else 0

    body = (
        f'<table style="border-collapse:collapse;min-width:380px">'
        f'<tr><th style="{_TH}">Model</th><th style="{_TH}">Active</th>'
        f'<th style="{_TH}">Hit rate</th><th style="{_TH}">P&amp;L</th></tr>'
        f'{model_rows}</table>'
    )
    if miss_rows:
        body += (
            f'<h3 style="font-size:14px;margin:20px 0 6px">Synthesis calls</h3>'
            f'<table style="border-collapse:collapse;min-width:320px">'
            f'<tr><th style="{_TH}">Symbol</th><th style="{_TH}">Call</th>'
            f'<th style="{_TH}">Right?</th><th style="{_TH}">P&amp;L</th></tr>'
            f'{miss_rows}</table>'
        )
    body += (
        f'<p style="color:#666;font-size:13px;margin-top:14px">'
        f'{total_hits}/{total_active} directional calls correct ({overall:.0f}%) · '
        f'${total_pnl:,.2f} gross, before costs.<br>'
        f'A single day is far too small a sample to read as skill.</p>'
    )
    if backlog.get("pending_mature"):
        by_date = ", ".join(f"{d}: {n}" for d, n in
                            (backlog.get("by_target_date") or {}).items())
        body += (
            f'<p style="color:#D93900;font-size:13px;margin-top:8px">'
            f'⚠ {backlog["pending_mature"]} mature prediction(s) remain '
            f'unscored ({by_date}) — the evaluator is skipping them.</p>'
        )
    return _send(
        f"Results {target} — {overall:.0f}% on {total_active} calls, ${total_pnl:,.2f}",
        _wrap("Results", f"Target session {target}", body),
    )


def send_test() -> bool:
    """Send a probe so the mail path can be proven before a run depends on it.

    Exercises the same token + sendMail call the scheduled notifications use,
    so a success here means the app registration, the sender mailbox and the
    recipients are all genuinely working — not merely configured.
    """
    cfg = _config()
    if cfg is None:
        return False
    body = (
        "<p>If you are reading this, quant-news can send mail.</p>"
        f"<p style='color:#666;font-size:13px'>Sent as "
        f"<code>{cfg['NOTIFY_FROM_EMAIL']}</code> via Microsoft Graph, using "
        f"the same path as the pre-open and results notifications.</p>"
    )
    return _send("quant-news: notification test",
                 _wrap("Notification test", "Manual probe", body))


def notify_partial(job_id: str, reasons: list[str], summary: dict) -> bool:
    """The run finished and stored data, but not all of what it was asked for.

    Kept distinct from a failure because the two need different responses: a
    failure means today has no analysis, a partial means today has one you
    should not fully trust.
    """
    coverage = summary.get("model_coverage") or {}
    rows = "".join(
        f'<tr><td style="{_TD}">{m}</td>'
        f'<td style="{_TD};color:{"#D93900" if n == 0 else "#666"}">{n}</td></tr>'
        for m, n in sorted(coverage.items())
    )
    body = (
        "<p>The run completed and stored predictions, but did not produce "
        "everything it was asked for:</p><ul>"
        + "".join(f"<li>{r}</li>" for r in reasons)
        + "</ul>"
    )
    if rows:
        body += (
            f'<h3 style="font-size:14px;margin:18px 0 6px">Symbols scored per model</h3>'
            f'<table style="border-collapse:collapse;min-width:300px">'
            f'<tr><th style="{_TH}">Model</th><th style="{_TH}">Scored</th></tr>'
            f'{rows}</table>'
        )
    body += ('<p style="color:#666;font-size:13px;margin-top:14px">'
             'Today is recorded as not-yet-successful, so /healthz reports it '
             'overdue. It will not retry on its own — re-run it from the '
             'Schedule page once the cause is fixed.</p>')
    return _send(f"⚠️ quant-news: {job_id} partial — {reasons[0] if reasons else 'incomplete'}",
                 _wrap("Partial run", summary.get("target_date", job_id), body))


def notify_job_failure(job_id: str, detail: str) -> bool:
    """The failure mail — carries the run's own output, not a pointer to it.

    A mail that only says "check the app" is a mail that costs a login before
    it tells you anything, and the tail of the run log is usually the whole
    diagnosis. The last lines are the ones that matter, so a long log is cut
    from the front.
    """
    tail = (detail or "").strip()
    if len(tail) > 6000:
        tail = "…earlier output omitted…\n" + tail[-6000:]
    body = (f'<p>The scheduled job <code>{job_id}</code> did not complete.</p>'
            f'<pre style="background:#f6f6f6;padding:10px;border-radius:4px;'
            f'font-size:12px;white-space:pre-wrap;overflow-x:auto">'
            f'{html_escape(tail)}</pre>'
            f'<p style="color:#666;font-size:13px">The full log is on the '
            f'Schedule page under this job&rsquo;s recent runs; '
            f'<code>/healthz</code> reports whether the day is still '
            f'outstanding.</p>')
    return _send(f"❌ quant-news: {job_id} failed", _wrap("Job failed", job_id, body))


def notify_alpha_lab(results: dict) -> bool:
    """Alpha Lab digest: one line per standing hypothesis, loud only when
    something crosses the pre-registered bar. A quiet week mails a quiet
    email — the absence of edge is a result, not a failure."""
    tests = results.get("tests") or []
    passing = results.get("passing") or []

    rows = []
    for t in tests:
        name = t.get("hypothesis", "?")
        sig = t.get("significant")
        stat = {
            "cross_sectional": (f"spread {t.get('mean_spread_pct')}%/day, "
                                f"t={t.get('t_spread')}"),
            "event_drift": (f"5d {t.get('drift_5d_pct')}% / 10d "
                            f"{t.get('drift_10d_pct')}%, "
                            f"t={t.get('t_5d')}/{t.get('t_10d')}"),
            "calibration_gate": (f"gated {t.get('hit_gated')} vs ungated "
                                 f"{t.get('hit_ungated')} "
                                 f"({t.get('edge_pp')}pp)"),
        }.get(name, "")
        color = "#2e7d32" if sig else "#777"
        verdict = "SIGNIFICANT" if sig else "not significant"
        rows.append(
            f"<tr><td style='padding:4px 12px 4px 0'><b>{name}</b></td>"
            f"<td style='padding:4px 12px 4px 0'>{stat}</td>"
            f"<td style='padding:4px 0;color:{color}'>{verdict} "
            f"<span style='color:#999'>({t.get('power', '')})</span></td></tr>")

    headline = (f"{len(passing)} hypothesis(es) crossed the bar: "
                f"{', '.join(passing)} — review before acting"
                if passing else
                "No standing hypothesis is significant yet — power is "
                "accruing with every scheduled day")
    body = (f"<p>{headline}</p>"
            f"<table style='border-collapse:collapse'>{''.join(rows)}</table>"
            f"<p style='color:#999;font-size:12px'>Criteria are "
            f"pre-registered in services/alpha_lab.py; n grows daily via the "
            f"schedule. {results.get('n_evaluated_predictions', '?')} "
            f"evaluated predictions in the pool.</p>")
    subject = ("Alpha Lab: " + (f"{len(passing)} SIGNIFICANT" if passing
                                else "no edge yet"))
    return _send(subject, _wrap("Alpha Lab", results.get("as_of", ""), body))


def notify_overdue(job_ids: list[str]) -> bool:
    names = ", ".join(job_ids)
    body = (f'<p>These jobs have no successful run today and their scheduled '
            f'window has passed: <strong>{names}</strong>.</p>'
            f'<p style="color:#666;font-size:13px">The app is reachable — this '
            f'is a job problem, not an uptime one. Check <code>/healthz</code>.</p>')
    return _send(f"⚠️ quant-news: {names} overdue",
                 _wrap("Job overdue", names, body))
