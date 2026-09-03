"""The Runs section on the Reports page.

One row per analysis run, newest first, above the symbol-grouped archive.
The page's filter bar applies to it the way it applies to recommendations:
a run stays visible if any of its symbols matches, and dates are the ET day
the run started (started_at is stored in UTC). Pure rendering over the dicts
run_service.list_runs returns; no database, no model, no LLM.
"""

from datetime import datetime, timedelta, timezone

from dash import dcc

from layouts import history_sections as hs
from layouts.pages import reports as reports_page


def _text(node) -> str:
    if node is None:
        return ""
    if isinstance(node, (str, int, float)):
        return str(node)
    if isinstance(node, (list, tuple)):
        return " ".join(t for t in (_text(c) for c in node) if t)
    return _text(getattr(node, "children", None))


def _find_all(node, **attrs) -> list:
    def matches(n):
        for k, v in attrs.items():
            got = getattr(n, k, None)
            if k == "className":
                if not got or v not in str(got).split():
                    return False
            elif got != v:
                return False
        return True

    out = []

    def walk(n):
        if n is None or isinstance(n, (str, int, float)):
            return
        if isinstance(n, (list, tuple)):
            for c in n:
                walk(c)
            return
        if matches(n):
            out.append(n)
        walk(getattr(n, "children", None))

    walk(node)
    return out


def _find(node, **attrs):
    hits = _find_all(node, **attrs)
    return hits[0] if hits else None


def _rows(section) -> list:
    body = _find(section, className="history-section-body")
    return [n for n in _find_all(body) if type(n).__name__ == "Tr"
            and not any(type(c).__name__ == "Th" for c in (n.children or []))]


def run(run_id="11111111-2222-4333-8444-555555555555", kind="manual",
        status="done", symbols="NVDA,AMD", started=None, finished=None,
        preset="standard", owner="u1", error=None):
    started = started or datetime(2026, 9, 2, 11, 0, tzinfo=timezone.utc)
    if finished is None and status in ("done", "failed", "cancelled"):
        finished = started + timedelta(seconds=252)
    return {
        "run_id": run_id, "owner_uid": owner, "kind": kind, "status": status,
        "preset": preset, "config": {}, "symbols_csv": symbols,
        "symbols": symbols.split(","), "prediction_date": "2026-09-02",
        "target_date": "2026-09-03", "stages": {}, "counters": {},
        "estimate_s": 120, "started_at": started.isoformat(),
        "finished_at": finished.isoformat() if finished else None,
        "error": error, "job_run_id": None, "is_public": True,
        "active": status in ("queued", "running"),
    }


class TestBuildRunsSection:
    def test_nothing_to_list_renders_nothing(self):
        assert hs.build_runs_section([]) is None
        assert hs.build_runs_section(None) is None

    def test_row_carries_every_column_and_links_to_the_run_page(self):
        section = hs.build_runs_section([run()])
        rows = _rows(section)
        assert len(rows) == 1
        text = _text(rows[0])
        # 11:00 UTC is 07:00 ET on that date.
        assert "2026-09-02 07:00 ET" in text
        assert "manual" in text
        assert "Standard" in text
        assert "NVDA, AMD" in text
        assert "Complete" in text
        assert "4 min 12s" in text
        assert "u1" in text
        link = _find(rows[0], href="/runs/11111111-2222-4333-8444-555555555555")
        assert isinstance(link, dcc.Link)
        assert "Open" in _text(link)
        assert _find(rows[0], className="run-status-done") is not None
        assert _find(rows[0], className="run-kind-manual") is not None
        assert _text(_find(section, className="history-section-header")).startswith("Runs")

    def test_scheduled_badge_anonymous_owner_and_running_duration(self):
        section = hs.build_runs_section([
            run(kind="scheduled", status="running", owner=None, finished=None)])
        row = _rows(section)[0]
        text = _text(row)
        assert _find(row, className="run-kind-scheduled") is not None
        assert "scheduled" in text
        assert "anonymous" in text
        assert "running" in text
        assert "so far" not in text
        assert "Running" in text

    def test_failed_row_keeps_the_first_error_line_as_a_tooltip(self):
        section = hs.build_runs_section([
            run(status="failed", error="kronos_mini: weights missing\nTraceback")])
        pill = _find(_rows(section)[0], className="run-status-failed")
        assert pill.title == "kronos_mini: weights missing"
        assert "Failed" in _text(pill)

    def test_long_symbol_lists_collapse_after_five_with_the_full_list_in_title(self):
        syms = "AAPL,MSFT,NVDA,AMD,TSLA,BE,IOVA,REX"
        section = hs.build_runs_section([run(symbols=syms)])
        cell = _find(_rows(section)[0], className="runs-symbols")
        assert _text(cell) == "AAPL, MSFT, NVDA, AMD, TSLA +3"
        assert cell.title == "AAPL, MSFT, NVDA, AMD, TSLA, BE, IOVA, REX"
        # Five or fewer: no suffix.
        section = hs.build_runs_section([run(symbols="AAPL,MSFT,NVDA,AMD,TSLA")])
        assert "+" not in _text(_find(_rows(section)[0], className="runs-symbols"))

    def test_rows_keep_the_order_given(self):
        newest = run(run_id="a" * 36, started=datetime(2026, 9, 2, 15, tzinfo=timezone.utc))
        older = run(run_id="b" * 36, started=datetime(2026, 9, 1, 15, tzinfo=timezone.utc))
        rows = _rows(hs.build_runs_section([newest, older]))
        assert [r.children[-1].children.href for r in rows] == [
            "/runs/" + "a" * 36, "/runs/" + "b" * 36]

    def test_paginates_past_the_page_size(self):
        runs = [run(run_id=f"{i:036d}") for i in range(hs.PAGE_SIZE["runs"] + 5)]
        section = hs.build_runs_section(runs)
        assert len(_rows(section)) == hs.PAGE_SIZE["runs"]
        pager = _find(section, className="history-pager")
        assert pager is not None
        assert f"1–{hs.PAGE_SIZE['runs']} of {len(runs)}" in _text(pager)
        nxt = [n for n in _find_all(pager)
               if isinstance(getattr(n, "id", None), dict)
               and n.id.get("dir") == "next"][0]
        assert nxt.id["bucket"] == "runs"
        page2 = hs.build_runs_section(runs, page={"runs": hs.PAGE_SIZE["runs"]})
        assert len(_rows(page2)) == 5
        # The section badge counts everything, not the page.
        assert _text(_find(section, className="history-count-badge")) == str(len(runs))


class TestFilters:
    def test_symbol_filter_keeps_a_run_if_any_symbol_matches(self):
        runs = [run(run_id="a" * 36, symbols="NVDA,AMD"),
                run(run_id="b" * 36, symbols="TSLA")]
        kept = hs.filter_runs(runs, ["AMD"], "all")
        assert [r["run_id"] for r in kept] == ["a" * 36]
        assert hs.filter_runs(runs, ["BE"], "all") == []
        assert len(hs.filter_runs(runs, None, "all")) == 2

    def test_specific_date_is_the_et_day_the_run_started(self):
        # 02:30 UTC on the 3rd is 22:30 ET on the 2nd.
        late = run(started=datetime(2026, 9, 3, 2, 30, tzinfo=timezone.utc))
        assert hs.run_started_date(late) == "2026-09-02"
        assert len(hs.filter_runs([late], None, "all", "2026-09-02")) == 1
        assert hs.filter_runs([late], None, "all", "2026-09-03") == []

    def test_range_filter_drops_old_runs(self):
        fresh = run(run_id="a" * 36, started=datetime.now(timezone.utc))
        stale = run(run_id="b" * 36,
                    started=datetime.now(timezone.utc) - timedelta(days=40))
        kept = hs.filter_runs([fresh, stale], None, "30d")
        assert [r["run_id"] for r in kept] == ["a" * 36]
        assert len(hs.filter_runs([fresh, stale], None, "90d")) == 2

    def test_filter_history_data_carries_the_runs_bucket(self):
        data = {"runs": [run(symbols="NVDA"), run(run_id="b" * 36, symbols="AMD")]}
        buckets = hs.filter_history_data(data, ["AMD"], "all")
        assert [r["symbols_csv"] for r in buckets["runs"]] == ["AMD"]
        assert hs.filter_history_data({}, None, "all")["runs"] == []

    def test_filter_bar_offers_symbols_that_only_appear_in_runs(self):
        bar = hs.build_history_filter_bar({"runs": [run(symbols="ZZZQ,NVDA")]})
        dropdown = _find(bar, id="history-symbol-dropdown")
        assert {o["value"] for o in dropdown.options} == {"ZZZQ", "NVDA"}


class TestReportsPage:
    def test_body_leads_with_the_runs_section(self):
        data = {
            "runs": [run()],
            "trading_agent_reports": [{"id": "r1", "symbol": "NVDA",
                                       "trade_date": "2026-09-02",
                                       "decision": "BUY", "confidence": 0.6,
                                       "created_at": "2026-09-02T12:00:00"}],
            "reports": [], "recommendations": [],
        }
        body = reports_page.body(data)
        sections = _find(body, id="reports-sections").children
        assert sections[0].children[0].id == {
            "type": "history-section-toggle", "section": "runs"}
        assert sections[1].children[0].id == {
            "type": "history-section-toggle", "section": "ta"}

    def test_filter_bar_narrows_the_runs_section_too(self):
        data = {"runs": [run(symbols="NVDA"), run(run_id="b" * 36, symbols="AMD")],
                "trading_agent_reports": [], "reports": [], "recommendations": []}
        body = reports_page.body(data, filter_symbols=["AMD"])
        runs_section = _find(body, className="history-collapsible-section")
        rows = _rows(runs_section)
        assert len(rows) == 1
        assert "AMD" in _text(rows[0])
        # Nothing matches: the shared empty state, not a bare page.
        body = reports_page.body(data, filter_symbols=["BE"])
        assert "No matching reports" in _text(body)

    def test_action_bar_copy_matches_the_empty_dialog(self):
        text = _text(reports_page._action_bar())
        assert "any symbols" in text
        assert "past run or report" in text
        assert "for the watchlist" not in text
