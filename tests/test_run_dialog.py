"""The Run dialog: presets, additive symbol sources, the ticker typeahead.

The rules worth pinning: a preset fixes only the fields it names and the
row records the name only while the controls still match it; Customize
unfolds on divergence and never folds on its own; the toolbar button opens
EMPTY on the remembered preset while the report shortcuts pre-fill their
symbols and force the report scope; every symbol source adds and dedupes;
a typed name becomes a chip only after the cache or one price lookup has
vouched for it; the typeahead serves the cache's ranking untouched.

Importing app registers every callback; the decorated functions are still
plain callables, exercised here with a stubbed ctx and an in-memory SQLite.
Nothing fetches prices (validate_symbol's vendor call is stubbed), runs a
model or calls an LLM.
"""

import asyncio
import json
import os
from types import SimpleNamespace

import dash
import diskcache
import pytest
from dash.exceptions import PreventUpdate
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from db.models import AnalysisRun, Base, JobRun, Ticker
from layouts import modals
from services import progress_service as prog
from services import run_service as rs
from services import ticker_service as ts

_flag_before = os.environ.get(prog._ENV_FLAG)
import app as app_module  # noqa: E402

if _flag_before is None:
    os.environ.pop(prog._ENV_FLAG, None)

from tests.test_run_dispatch import (  # noqa: E402
    BTN_DISABLED, BTN_LABEL, CUSTOMIZE, IS_OPEN, PRESET, PREFS, VALIDATION)

SCOPE, SYMBOLS = 1, 2
ALL_MODELS = [mid for mid, _, _ in modals.RUN_MODELS]
QUICK_MODELS = [m for m in ALL_MODELS if m != "trading_agents"]
CHECK_IDS = [{"type": "run-model-check", "model": m} for m in ALL_MODELS]


@compiles(JSONB, "sqlite")
def _jsonb_as_json(type_, compiler, **kw):
    return "JSON"


@pytest.fixture
def db(monkeypatch):
    import db.session as dbs

    # One shared connection: the symbol callbacks run their lookups in a
    # worker thread, and a per-thread in-memory SQLite would be empty there.
    eng = create_engine("sqlite://", poolclass=StaticPool,
                        connect_args={"check_same_thread": False})
    Base.metadata.create_all(eng, tables=[AnalysisRun.__table__, JobRun.__table__,
                                          Ticker.__table__])
    monkeypatch.setattr(dbs, "_engine", eng)
    monkeypatch.setattr(
        dbs, "_SessionLocal", sessionmaker(bind=eng, expire_on_commit=False))
    return dbs


@pytest.fixture
def feed(monkeypatch, tmp_path):
    monkeypatch.setenv(prog._ENV_FLAG, "1")
    monkeypatch.setattr(prog, "_cache", diskcache.Cache(str(tmp_path)))
    monkeypatch.setattr(prog, "_local_run", {"id": None, "title": None})
    monkeypatch.setattr(prog, "_write_audit", lambda *a, **kw: None)
    monkeypatch.setattr(prog, "_last_db_reap", 0.0)
    token = prog._run_ctx.set(None)
    yield prog
    prog._run_ctx.reset(token)


@pytest.fixture
def tickers(db):
    ts.upsert([
        {"symbol": "AAPL", "name": "Apple Inc.", "exchange": "NASDAQ"},
        {"symbol": "A", "name": "Agilent Technologies"},
        {"symbol": "AMD", "name": "Advanced Micro Devices"},
        {"symbol": "APLE", "name": "Apple Hospitality REIT"},
        {"symbol": "NVDA", "name": "NVIDIA Corp"},
        {"symbol": "SNAP", "name": "Snap Inc."},
    ], "index")


def _trigger(monkeypatch, triggered_id, value=1, **extra):
    prop = (triggered_id if isinstance(triggered_id, str)
            else json.dumps(triggered_id, sort_keys=True, separators=(",", ":")))
    monkeypatch.setattr(app_module, "ctx", SimpleNamespace(
        triggered_id=triggered_id,
        triggered=[{"prop_id": f"{prop}.n_clicks", "value": value}],
        **extra))


def _open(monkeypatch, triggered, watchlist=(), prefs=None, ctx_clicks=None):
    """toggle_run_modal's opener branch for one of the three entry points."""
    _trigger(monkeypatch, triggered)
    clicks = {"run-analysis-btn": 1} if triggered == "run-analysis-btn" else {}
    reports = 1 if triggered == "reports-new-btn" else None
    return app_module.toggle_run_modal(
        clicks.get("run-analysis-btn"), reports, ctx_clicks or [], None, None,
        {}, list(watchlist), {}, False, {}, "full", {},
        preset=None, prefs=prefs)


class TestPresetFields:
    def test_each_preset_maps_to_the_dialog_fields(self):
        quick = modals.preset_fields("quick")
        assert quick == {"scope": "models", "models": QUICK_MODELS,
                         "recs": "off"}
        standard = modals.preset_fields("standard")
        assert standard == {"scope": "full", "models": ALL_MODELS,
                            "recs": "auto", "tools": []}
        deep = modals.preset_fields("deep")
        assert deep["scope"] == "full" and deep["recs"] == "auto"
        assert deep["evidence"] == [o["value"] for o in modals.EVIDENCE_OPTIONS]
        assert deep["tools"] == ["web_research"]
        assert set(modals.RUN_PRESET_ORDER) == set(modals.RUN_PRESETS)

    def test_quick_is_the_cheapest_preset(self):
        # Quick used to keep TradingAgents; the models-only path runs that
        # research serially, so Quick quoted a longer run than Standard.
        def est(name, n):
            f = modals.preset_fields(name)
            return modals.estimate_run_seconds(
                f["scope"], n, f["models"], f["recs"] != "off")
        for n in (1, 3, 8):
            assert est("quick", n) < est("standard", n) <= est("deep", n)
        assert "trading_agents" not in modals.preset_fields("quick")["models"]

    def test_the_default_preset_produces_both_primary_outputs(self):
        # The regression this pins: Standard shipped with recs "off", so
        # every default run — and every schedule saved through the dialog,
        # which carries the dialog's values into params_json — produced a
        # research report and no synthesis. The default preset may never
        # switch either of the product's two outputs off.
        f = modals.preset_fields(modals.DEFAULT_RUN_PRESET)
        assert modals.DEFAULT_RUN_PRESET == "standard"
        assert f["scope"] == "full"
        assert "trading_agents" in f["models"]
        assert f["recs"] == "auto"

    def test_quick_is_the_only_preset_that_drops_them(self):
        quick = modals.preset_fields("quick")
        assert quick["scope"] == "models" and quick["recs"] == "off"
        assert "trading_agents" not in quick["models"]
        assert quick != modals.preset_fields(modals.DEFAULT_RUN_PRESET)
        # An explicit choice has to say what it costs.
        hint = modals.RUN_PRESETS["quick"]["hint"]
        assert "no research report" in hint
        assert "no recommendation synthesis" in hint
        for name in ("standard", "deep"):
            f = modals.preset_fields(name)
            assert f["recs"] == "auto" and "trading_agents" in f["models"]

    def test_unknown_or_stale_name_is_the_default(self):
        assert modals.preset_fields(None) == modals.preset_fields("standard")
        assert modals.preset_fields("gone") == modals.preset_fields("standard")
        # A copy: mutating the answer never edits the preset.
        modals.preset_fields("quick")["models"].clear()
        assert modals.preset_fields("quick")["models"] == QUICK_MODELS

    def test_dialog_defaults_match_the_default_preset(self):
        # A report shortcut leaves the preset alone; the controls it finds
        # must already sit on Standard or Customize would unfold on open.
        modal = modals.create_run_modal()

        def find(node, wanted):
            if getattr(node, "id", None) == wanted:
                return node
            ch = getattr(node, "children", None)
            for c in (ch if isinstance(ch, (list, tuple)) else [ch]):
                if hasattr(c, "to_plotly_json"):
                    hit = find(c, wanted)
                    if hit is not None:
                        return hit
            return None

        std = modals.preset_fields("standard")
        assert find(modal, "run-recs").value == std["recs"]
        assert find(modal, "run-scope").value == std["scope"]
        assert find(modal, "run-preset").value == "standard"
        assert find(modal, "run-customize-collapse").is_open is False
        # Standard's tools are what makes it cheaper than Deep; the
        # checklist has to open on them, not on the checklist's own
        # default (apply_run_preset never fires for an untouched preset).
        assert find(modal, "run-tools").value == std["tools"]
        # A plain text box, not a Dropdown: typing is the whole interaction.
        search = find(modal, "run-symbol-search")
        assert search.type == "text" and search.debounce is False
        assert find(modal, "run-symbol-suggest") is not None

        # Every model box, TradingAgents included, is ticked on open.
        boxes = {}

        def walk(node):
            nid = getattr(node, "id", None)
            if isinstance(nid, dict) and nid.get("type") == "run-model-check":
                boxes[nid["model"]] = node.value
            ch = getattr(node, "children", None)
            for c in (ch if isinstance(ch, (list, tuple)) else [ch]):
                if hasattr(c, "to_plotly_json"):
                    walk(c)

        walk(modal)
        assert boxes == {m: True for m in std["models"]}

    def test_apply_writes_only_what_the_preset_names(self, monkeypatch):
        _trigger(monkeypatch, "run-preset")
        scope, checks, recs, evidence, tools = app_module.apply_run_preset(
            "quick", "2099-01-05", CHECK_IDS)
        assert scope == "models" and recs == "off"
        # TradingAgents is the one box Quick unticks.
        assert checks == [c["model"] != "trading_agents" for c in CHECK_IDS]
        # Quick fixes neither evidence nor tools: the date default stands.
        assert evidence is dash.no_update
        assert tools == ["web_research"]

        scope, checks, recs, evidence, tools = app_module.apply_run_preset(
            "deep", "2099-01-05", CHECK_IDS)
        assert scope == "full" and recs == "auto"
        assert evidence == [o["value"] for o in modals.EVIDENCE_OPTIONS]
        assert tools == ["web_research"]

        *_, tools = app_module.apply_run_preset("standard", "2099-01-05", CHECK_IDS)
        assert tools == []

    def test_backtest_date_never_turns_the_web_on(self, monkeypatch):
        _trigger(monkeypatch, "run-preset")
        *_, tools = app_module.apply_run_preset("deep", "2024-03-05", CHECK_IDS)
        assert tools == []
        # A date change alone touches only the tools. The ALL output still
        # gets a list (a bare no_update for a wildcard output is a 500).
        _trigger(monkeypatch, "run-date-picker")
        out = app_module.apply_run_preset("deep", "2024-03-05", CHECK_IDS)
        assert out[0] is dash.no_update and out[2] is dash.no_update
        assert out[1] == [dash.no_update] * 5
        assert out[3] is dash.no_update and out[4] == []


class TestDivergence:
    def test_matching_controls_do_not_diverge(self):
        assert modals.preset_divergence("standard", {
            "scope": "full", "models": list(reversed(ALL_MODELS)),
            "recs": "auto", "tools": [], "evidence": ["options"]}) == []

    def test_each_named_field_is_reported(self):
        got = modals.preset_divergence("deep", {
            "scope": "report", "models": ALL_MODELS[:2], "recs": "off",
            "evidence": ["options"], "tools": []})
        assert got == ["scope", "models", "recs", "evidence", "tools"]

    def test_unmounted_values_are_not_divergence(self):
        assert modals.preset_divergence("deep", {"scope": None, "models": None,
                                                 "recs": None}) == []
        assert modals.preset_divergence("quick", {"scope": "models",
                                                  "tools": ["web_research"]}) == []

    def _preflight(self, monkeypatch, preset, scope, checks, recs, evidence,
                   tools, is_open=True, opened=True, customize_open=False,
                   auto=None):
        """run_preflight as one batch calls it: ``opened`` puts the modal's
        is_open among the triggers (the dialog opening), otherwise the
        trigger is a control change inside the open dialog."""
        trigger = "run-modal.is_open" if opened else "run-scope.value"
        monkeypatch.setattr(app_module, "ctx", SimpleNamespace(
            triggered=[{"prop_id": trigger, "value": is_open if opened
                        else scope}],
            inputs_list=[
                {"id": "run-modal", "property": "is_open"},
                {"id": "run-preset", "property": "value"},
                {"id": "run-scope", "property": "value"},
                {"id": "run-symbols-store", "property": "data"},
                [{"id": {"type": "run-model-check", "model": m}, "property": "value",
                  "value": on} for m, on in zip(ALL_MODELS, checks)],
            ]))
        return app_module.run_preflight(
            is_open, preset, scope, {"symbols": ["NVDA"]}, checks,
            "claude-haiku-4-5", recs, "claude-sonnet-5", evidence, tools,
            customize_open, auto)

    def test_customize_unfolds_only_on_divergence(self, monkeypatch):
        preflight, hint, collapse, auto = self._preflight(
            monkeypatch, "standard", "full", [True] * 5, "auto",
            ["options"], [])
        assert collapse is dash.no_update
        assert auto == {"diverged": []}
        assert hint == [modals.RUN_PRESETS["standard"]["hint"]]
        assert preflight and "1 symbol" in str(preflight[-1].to_plotly_json())

        preflight, hint, collapse, auto = self._preflight(
            monkeypatch, "standard", "report", [True] * 5, "off",
            None, [])
        assert collapse is True
        assert auto == {"diverged": ["scope", "recs"]}
        assert "what to run, recommendations" in hint[1].children

    def test_customize_unfolds_once_per_divergence_set(self, monkeypatch):
        # A report shortcut: the scope diverges from the moment it opens.
        *_, collapse, auto = self._preflight(
            monkeypatch, "standard", "report", [True] * 5, "auto", None, [])
        assert collapse is True and auto == {"diverged": ["scope"]}
        # The user folds Customize and keeps editing: the same divergence
        # is not a reason to unfold it again, and nothing is re-recorded.
        *_, collapse, again = self._preflight(
            monkeypatch, "standard", "report", [True] * 5, "auto", None, [],
            opened=False, customize_open=False, auto=auto)
        assert collapse is dash.no_update and again is dash.no_update
        # A new difference is: it unfolds once more and remembers the set.
        *_, collapse, auto = self._preflight(
            monkeypatch, "standard", "report", [True] * 5, "off", None, [],
            opened=False, customize_open=False, auto=auto)
        assert collapse is True and auto == {"diverged": ["scope", "recs"]}
        # Folded again, the difference removed: the set moved but nothing
        # new is on screen to show, so it stays folded; the set is kept
        # current so the same field diverging again is news.
        *_, collapse, auto = self._preflight(
            monkeypatch, "standard", "report", [True] * 5, "auto", None, [],
            opened=False, customize_open=False, auto=auto)
        assert collapse is dash.no_update and auto == {"diverged": ["scope"]}
        # While it is open a new difference needs no unfold, only the record.
        *_, collapse, auto = self._preflight(
            monkeypatch, "standard", "report", [True] * 4 + [False], "auto",
            None, [], opened=False, customize_open=True, auto=auto)
        assert collapse is dash.no_update
        assert auto == {"diverged": ["scope", "models"]}
        # Reopening the dialog starts over: the shortcut's divergence is
        # shown again even though it is the set recorded last time.
        *_, collapse, auto = self._preflight(
            monkeypatch, "standard", "report", [True] * 5, "auto", None, [],
            opened=True, customize_open=False, auto={"diverged": ["scope"]})
        assert collapse is True and auto == {"diverged": ["scope"]}

    def test_unmounted_recs_is_no_divergence(self, monkeypatch):
        # run-recs arrives as None until its Select has rendered once; the
        # raw value reaches preset_divergence, which skips it, so an
        # untouched Standard dialog is not "customized: recommendations".
        _, hint, collapse, auto = self._preflight(
            monkeypatch, "standard", "full", [True] * 5, None, None, [])
        assert collapse is dash.no_update and auto == {"diverged": []}
        assert hint == [modals.RUN_PRESETS["standard"]["hint"]]

    def test_closed_dialog_changes_nothing(self, monkeypatch):
        out = self._preflight(monkeypatch, "standard", "full", [True] * 5,
                              "auto", None, [], is_open=False)
        assert out == (dash.no_update,) * 4

    def test_customize_button_toggles(self):
        assert app_module.toggle_run_customize(1, False) is True
        assert app_module.toggle_run_customize(2, True) is False
        with pytest.raises(PreventUpdate):
            app_module.toggle_run_customize(None, False)


class TestOpeners:
    @pytest.fixture(autouse=True)
    def owner(self, db, feed, monkeypatch):
        monkeypatch.setattr(app_module, "_run_owner_uid", lambda: "u1")

    def test_toolbar_opens_empty_on_the_remembered_preset(self, monkeypatch):
        rs.create_run("manual", ["tsla", "AMD"], "u1")
        rs.create_run("manual", ["NVDA"], "u2")      # someone else's

        out = _open(monkeypatch, "run-analysis-btn", watchlist=["AAPL", "BE"],
                    prefs={"preset": "deep", "symbols": ["XYZ"]})

        assert out[IS_OPEN] is True
        assert out[SYMBOLS] == {"symbols": [], "watchlist": ["AAPL", "BE"],
                                "lastrun": ["TSLA", "AMD"]}
        assert out[SCOPE] is dash.no_update      # the preset sets it
        assert out[PRESET] == "deep"
        assert out[CUSTOMIZE] is False
        assert out[VALIDATION] == ""
        assert out[BTN_DISABLED] is False and out[BTN_LABEL][-1] == "Run"
        assert out[PREFS] is dash.no_update

    def test_stale_preset_and_no_runs_fall_back(self, monkeypatch):
        out = _open(monkeypatch, "run-analysis-btn",
                    prefs={"preset": "turbo", "symbols": ["nvda", ""]})
        assert out[PRESET] == modals.DEFAULT_RUN_PRESET
        # No run on record for this owner: the browser's list stands in.
        assert out[SYMBOLS]["lastrun"] == ["nvda"]
        out = _open(monkeypatch, "run-analysis-btn", prefs="junk")
        assert out[PRESET] == modals.DEFAULT_RUN_PRESET
        assert out[SYMBOLS] == {"symbols": [], "watchlist": [], "lastrun": []}

    def test_reports_entry_prefills_the_watchlist_and_forces_report(self, monkeypatch):
        out = _open(monkeypatch, "reports-new-btn", watchlist=["AAPL", "BE"])
        assert out[SYMBOLS]["symbols"] == ["AAPL", "BE"]
        assert out[SCOPE] == "report"
        # The preset is left alone: the report scope diverges from it and
        # the preflight callback unfolds Customize to show that.
        assert out[PRESET] is dash.no_update
        assert out[CUSTOMIZE] is False

    def test_per_symbol_entry_prefills_that_symbol(self, monkeypatch):
        out = _open(monkeypatch, {"type": "new-report-btn", "symbol": "BE"},
                    watchlist=["AAPL"], ctx_clicks=[1])
        assert out[SYMBOLS]["symbols"] == ["BE"]
        assert out[SCOPE] == "report"
        assert out[PRESET] is dash.no_update

    def test_mount_echo_of_a_context_button_does_not_open(self, monkeypatch):
        _trigger(monkeypatch, {"type": "new-report-btn", "symbol": "BE"}, value=None)
        with pytest.raises(PreventUpdate):
            app_module.toggle_run_modal(None, None, [None], None, None,
                                        {}, [], {}, False, {}, "full", {})


class TestSymbols:
    def _set(self, monkeypatch, triggered, store, picked=None, clicks=1,
             remove=(), typed=None):
        """One firing of set_run_symbols. ``picked`` is a suggestion row's
        value (the row's pattern id becomes the trigger); ``typed`` is the
        box's text with Enter as the trigger."""
        if picked is not None:
            triggered = {"type": "run-sym-pick", "value": picked}
        _trigger(monkeypatch, triggered)
        add_wl = clicks if triggered == "run-add-watchlist" else None
        add_lr = clicks if triggered == "run-add-lastrun" else None
        picks = [clicks] if picked is not None else []
        submit = clicks if typed is not None else None
        return asyncio.run(app_module.set_run_symbols(
            add_wl, add_lr, list(remove), picks, submit, typed, store))

    def test_sources_add_and_dedupe(self, monkeypatch):
        store = {"symbols": ["NVDA"], "watchlist": ["AAPL", "NVDA"],
                 "lastrun": ["NVDA", "AMD"]}
        out, value, msg = self._set(monkeypatch, "run-add-watchlist", store)
        assert out["symbols"] == ["NVDA", "AAPL"] and msg == ""
        assert value is dash.no_update
        with pytest.raises(PreventUpdate):      # nothing left to add
            self._set(monkeypatch, "run-add-watchlist", out)
        out, _, _ = self._set(monkeypatch, "run-add-lastrun", out)
        assert out["symbols"] == ["NVDA", "AAPL", "AMD"]
        assert out["watchlist"] == ["AAPL", "NVDA"]   # snapshots untouched
        with pytest.raises(PreventUpdate):      # a mount echo, not a click
            self._set(monkeypatch, "run-add-lastrun", store, clicks=None)

    def test_remove_chip(self, monkeypatch):
        monkeypatch.setattr(app_module, "ctx", SimpleNamespace(
            triggered_id={"type": "run-sym-remove", "symbol": "AAPL"},
            triggered=[{"prop_id": "x.n_clicks", "value": 1}]))
        out, _, _ = asyncio.run(app_module.set_run_symbols(
            None, None, [1, None], [], None, "",
            {"symbols": ["NVDA", "AAPL"], "watchlist": ["AAPL"]}))
        assert out["symbols"] == ["NVDA"]
        with pytest.raises(PreventUpdate):
            asyncio.run(app_module.set_run_symbols(
                None, None, [None], [], None, "", {"symbols": ["AAPL"]}))

    def test_known_pick_becomes_a_chip_and_clears_the_box(self, monkeypatch,
                                                         tickers):
        def _never(sym):
            raise AssertionError("cached symbol must not be fetched")
        monkeypatch.setattr(ts, "_fetch_info", _never)
        out, value, msg = self._set(monkeypatch, None,
                                    {"symbols": ["NVDA"]}, picked="AAPL")
        assert out["symbols"] == ["NVDA", "AAPL"]
        assert value == "" and msg == ""
        # Rows mounting report n_clicks of 0/None: not a pick.
        with pytest.raises(PreventUpdate):
            self._set(monkeypatch, None, out, picked="AAPL", clicks=0)

    def test_enter_takes_the_top_suggestion_not_the_text(self, monkeypatch,
                                                        tickers):
        def _never(sym):
            raise AssertionError("a cache hit must not be fetched")
        monkeypatch.setattr(ts, "_fetch_info", _never)
        # "nvi" is nobody's ticker; the top row for it is NVDA.
        out, value, msg = self._set(monkeypatch, "run-symbol-search",
                                    {"symbols": []}, typed="nvi")
        assert out["symbols"] == ["NVDA"] and value == "" and msg == ""
        # Enter on text no row matches adds nothing and says so; the text
        # stays in the box for the user to fix.
        out, value, msg = self._set(monkeypatch, "run-symbol-search",
                                    {"symbols": []}, typed="not a ticker!")
        assert out is dash.no_update and value is dash.no_update
        assert msg == "No listed symbol matches \u201cnot a ticker!\u201d"
        with pytest.raises(PreventUpdate):     # Enter on an empty box
            self._set(monkeypatch, "run-symbol-search", {"symbols": []},
                      typed="  ")

    def test_verify_row_is_validated_once_then_cached(self, monkeypatch,
                                                     tickers):
        calls = []

        def _info(sym):
            calls.append(sym)
            return SimpleNamespace(name="Bloom Energy", current_price=21.5)
        monkeypatch.setattr(ts, "_fetch_info", _info)
        out, _, msg = self._set(monkeypatch, None, {"symbols": []}, picked="BE")
        assert out["symbols"] == ["BE"] and msg == ""
        assert calls == ["BE"]
        assert ts.get("BE")["source"] == "validated"
        assert ts.get("BE")["name"] == "Bloom Energy"

    def test_no_price_data_is_refused_with_a_message(self, monkeypatch, tickers):
        def _info(sym):
            raise ValueError("no data")
        monkeypatch.setattr(ts, "_fetch_info", _info)
        # Picking the verify row for a name the vendor has nothing on.
        out, value, msg = self._set(monkeypatch, None,
                                    {"symbols": ["NVDA"]}, picked="ZZZQ")
        assert out is dash.no_update and value == ""
        assert msg == "No price data for ZZZQ"
        assert ts.get("ZZZQ") is None
        # Enter on the same text goes through the same gate.
        out, _, msg = self._set(monkeypatch, "run-symbol-search",
                                {"symbols": ["NVDA"]}, typed="zzzq")
        assert out is dash.no_update and msg == "No price data for ZZZQ"
        # A paste with one bad name still adds the good ones and says why.
        out, _, msg = self._set(monkeypatch, None, {"symbols": []},
                                picked="AMD ZZZQ NVDA")
        assert out["symbols"] == ["AMD", "NVDA"]
        assert msg == "No price data for ZZZQ"

    def test_profile_without_a_price_is_refused(self, monkeypatch, tickers):
        # The vendor knows the name but quotes nothing: not a tradable symbol.
        monkeypatch.setattr(ts, "_fetch_info",
                            lambda s: SimpleNamespace(name="Gone", current_price=0))
        out, _, msg = self._set(monkeypatch, None, {"symbols": []}, picked="GONE")
        assert out is dash.no_update and msg == "No price data for GONE"
        assert ts.get("GONE") is None

    def test_paste_adds_every_symbol_once(self, monkeypatch, tickers):
        # Enter on a comma-separated paste: the single paste row is the top
        # (and only) suggestion.
        out, value, msg = self._set(monkeypatch, "run-symbol-search",
                                    {"symbols": ["AMD"]},
                                    typed="nvda, AMD;aapl nvda")
        assert out["symbols"] == ["AMD", "NVDA", "AAPL"]
        assert value == "" and msg == ""

    def test_chips_and_source_buttons(self):
        chips, wl_label, wl_off, lr_label, lr_off = app_module.render_run_symbol_chips(
            {"symbols": ["NVDA", "AMD"], "watchlist": ["AAPL", "BE"], "lastrun": []})
        assert [c.children[0].children for c in chips] == ["NVDA", "AMD"]
        assert chips[1].children[1].id == {"type": "run-sym-remove", "symbol": "AMD"}
        assert (wl_label, wl_off) == ("+ Watchlist (2)", False)
        assert (lr_label, lr_off) == ("+ Last run (0)", True)
        chips, *_ = app_module.render_run_symbol_chips({})
        assert chips[0].className == "run-symbols-empty"

    def test_data_summary_follows_the_chip_set(self):
        out = app_module.render_run_data_summary({"symbols": ["NVDA"]}, {}, {})
        assert "NVDA" in str(out.to_plotly_json())
        empty = app_module.render_run_data_summary({}, {}, {})
        assert "No symbols selected" in str(empty.to_plotly_json())
        from dash._callback import GLOBAL_CALLBACK_MAP
        cb = GLOBAL_CALLBACK_MAP["run-data-summary.children"]
        assert [i["id"] for i in cb["inputs"]] == ["run-symbols-store"]


def _row_values(rows):
    return [r.id["value"] for r in rows if getattr(r, "id", None)]


class TestSearch:
    def _search(self, q):
        return asyncio.run(app_module.search_run_symbols(q))

    def test_cache_ranking_reaches_the_rows_untouched(self, tickers):
        got = self._search("ap")
        assert _row_values(got) == [r["symbol"] for r in ts.search("ap")]
        assert got[0].id == {"type": "run-sym-pick", "value": "APLE"}
        assert got[0].className == "run-sym-suggest-row run-sym-suggest-hit"
        rows = self._search("nvi")
        assert _row_values(rows) == ["NVDA"]
        sym, name = rows[0].children
        assert (sym.children, name.children) == ("NVDA", "NVIDIA Corp")

    def test_under_two_characters_shows_nothing(self, tickers, monkeypatch):
        def _never(*a, **kw):
            raise AssertionError("no lookup under the minimum")
        monkeypatch.setattr(ts, "search", _never)
        for q in (None, "", " ", "a", "  ,"):
            assert self._search(q) == []
        assert modals.MIN_SEARCH_CHARS == 2

    def test_no_hit_offers_a_verify_row_never_an_add(self, tickers):
        rows = self._search("zzzq")
        assert _row_values(rows) == ["ZZZQ"]
        assert "run-sym-suggest-verify" in rows[0].className
        assert "Checks for price data" in rows[0].children[1].children
        # Text that cannot be a ticker gets one inert line, no button.
        rows = self._search("not a ticker!")
        assert len(rows) == 1 and rows[0].className == "run-sym-suggest-empty"
        assert _row_values(rows) == []
        assert rows[0].children == "No listed symbol matches \u201cnot a ticker!\u201d"

    def test_spaces_are_a_company_name_first(self, tickers):
        # "advanced micro" is a name, not two tickers; only when the cache
        # has nothing and every word is a ticker does it read as a paste.
        assert _row_values(self._search("advanced micro")) == ["AMD"]
        paste = self._search("nvda amd")[0]
        assert paste.id["value"] == "NVDA AMD"
        assert paste.children[0].children == "Add 2 symbols: NVDA, AMD"
        assert _row_values(self._search("apple hosp")) == ["APLE"]

    def test_paste_is_one_row_without_a_lookup(self, monkeypatch):
        def _never(*a, **kw):
            raise AssertionError("a paste needs no search")
        monkeypatch.setattr(ts, "search", _never)
        rows = self._search("nvda, amd amd")
        assert _row_values(rows) == ["NVDA AMD"]
        assert "run-sym-suggest-paste" in rows[0].className

    def test_search_failure_degrades_to_the_verify_row(self, monkeypatch):
        def _boom(*a, **kw):
            raise RuntimeError("db down")
        monkeypatch.setattr(ts, "search", _boom)
        rows = self._search("nvda")
        assert _row_values(rows) == ["NVDA"]
        assert "run-sym-suggest-verify" in rows[0].className

    def test_cap_is_twelve(self, monkeypatch):
        seen = {}

        def _search(q, limit):
            seen["limit"] = limit
            return []
        monkeypatch.setattr(ts, "search", _search)
        self._search("ab")
        assert seen["limit"] == 12

    def test_option_shapes(self, tickers):
        # The pure builder the rows and Enter both read.
        assert modals.run_symbol_options("nvi", ts.search("nvi")) == [
            {"label": "NVDA · NVIDIA Corp", "value": "NVDA", "kind": "hit",
             "name": "NVIDIA Corp"}]
        assert modals.run_symbol_options("zzzq", []) == [
            {"label": "Verify ZZZQ", "value": "ZZZQ", "kind": "verify"}]
        assert modals.run_symbol_options("zz!!", []) == []


class TestRetryScope:
    def test_scope_comes_from_config_then_legacy_preset(self):
        assert app_module._run_scope_of(
            {"preset": "custom", "config": {"scope": "models"}}) == "models"
        assert app_module._run_scope_of({"preset": "report", "config": {}}) == "report"
        assert app_module._run_scope_of({"preset": "quick", "config": {}},
                                        {"scope": "models"}) == "models"
        assert app_module._run_scope_of({}, None) == "full"


class TestScheduleModalDefaults:
    """The Schedule modal renders the Run dialog's field builders under the
    "sj" prefix, so a preset change can leak into a saved job. A job created
    from the modal's own defaults must run every model and the synthesis."""

    def _control_values(self):
        """What open_schedule_modal writes into the controls for a new job."""
        return app_module._sj_values_from_params(
            "analysis", {}, ALL_MODELS, ALL_MODELS, [])

    def test_new_job_form_opens_on_every_model_and_recs_auto(self):
        vals = self._control_values()
        assert vals["models"] == [True] * len(ALL_MODELS)
        assert vals["sj-recs"] == "auto"

    def test_saving_the_defaults_writes_them_into_params_json(self, monkeypatch):
        from services import scheduler_service

        vals = self._control_values()
        scalars = {
            "sj-kind": "analysis", "sj-name": "Morning", "sj-hour": 7,
            "sj-minute": 0, "sj-days": "mon-fri", "sj-tz": "US/Eastern",
            "sj-visibility": "private", "sj-enabled": True,
            "sj-symbols": "nvda, amd",
        }
        scalars.update({k: v for k, v in vals.items() if k.startswith("sj-")})
        created = {}

        def _create(**kw):
            created.update(kw)
            return "job-1"
        monkeypatch.setattr(scheduler_service, "create_job", _create)

        member_ids = [{"type": "sj-ens-member", "model": m} for m in ALL_MODELS]
        out = app_module.save_schedule_job(
            1, None,
            *[scalars[cid] for cid, _ in app_module._SJ_SCALARS],
            vals["models"],
            [{"type": "sj-model-check", "model": m} for m in ALL_MODELS],
            vals["members"], vals["weights"], member_ids, [], [])

        assert out[1] is False and out[2] is None
        params = created["params"]
        assert "trading_agents" in params["models"]
        assert params["models"] == ALL_MODELS
        assert params["recs"] == "auto"
        assert created["symbols_csv"] == "NVDA,AMD"
