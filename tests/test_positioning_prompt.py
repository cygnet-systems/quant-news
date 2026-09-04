"""What the research prompt is required to do with the disclosure blocks.

Fetching Form 4 lines and congressional filings buys nothing if the report
treats them as decoration. Three things are pinned here:

* the reasoning step names the executives and the members, the sizes, the
  dates and the filing lag, and forbids the over-readings this repo has
  already paid for: a disclosure is not a timing signal, one member's single
  trade is noise, only a cluster of distinct executives is worth weighting, a
  committee seat is context rather than causation, and a row priced 0.00 is a
  grant or an exercise, never a purchase;
* the blocks reach the prompt with the people named in them, and a run that
  gathered neither leaves no dangling heading behind;
* every figure the blocks introduce survives the figure auditor when the
  report quotes it, in the compressed form the block prints ("$18.05M") and
  in the expanded form a model reasonably writes instead ("$15,001").

In-memory SQLite, HTTP stubbed, no network.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import sessionmaker

from db.models import (
    AvFetchLog,
    Base,
    CongressTrade,
    InsiderTransaction,
    Politician,
    PoliticianAlias,
)
from models.single_agent import SINGLE_AGENT_PROMPT
from services import av_store, insider_service
from services import politician_dossier as pd_
from services.alpha_vantage import AlphaVantageUnavailable
from utils.figure_check import check_figures

pytestmark = pytest.mark.filterwarnings(
    r"ignore:.*does \*not\* support Decimal.*")

TABLES = [InsiderTransaction.__table__, CongressTrade.__table__,
          Politician.__table__, PoliticianAlias.__table__,
          AvFetchLog.__table__]

AS_OF = "2026-08-20"


@compiles(JSONB, "sqlite")
def _jsonb_as_json(type_, compiler, **kw):
    return "JSON"


@pytest.fixture
def db(monkeypatch):
    import db.session as dbs

    eng = create_engine("sqlite://")
    Base.metadata.create_all(eng, tables=TABLES)
    monkeypatch.setattr(dbs, "_engine", eng)
    monkeypatch.setattr(
        dbs, "_SessionLocal", sessionmaker(bind=eng, expire_on_commit=False))
    return dbs


# Two officers selling in the same fortnight (the one insider pattern worth
# weighting) plus one vesting grant priced at zero, which is the row a report
# must never call a purchase.
INSIDERS = {"data": [
    {"transaction_date": "2026-08-10", "ticker": "NVDA",
     "executive": "Jensen Huang", "executive_title": "President and CEO",
     "security_type": "Common Stock", "acquisition_or_disposal": "D",
     "shares": "100000", "share_price": "180.50"},
    {"transaction_date": "2026-08-05", "ticker": "NVDA",
     "executive": "Colette Kress", "executive_title": "EVP and CFO",
     "security_type": "Common Stock", "acquisition_or_disposal": "D",
     "shares": "40000", "share_price": "176.25"},
    {"transaction_date": "2026-08-03", "ticker": "NVDA",
     "executive": "Colette Kress", "executive_title": "EVP and CFO",
     "security_type": "Restricted Stock Units",
     "acquisition_or_disposal": "A", "shares": "25000", "share_price": "0.0"},
]}

CONGRESS = {"symbol": "NVDA", "trades": [
    {"bioguide_id": "N000189", "politician_canonical": "Dan Newhouse",
     "politician": "Dan Newhouse", "symbol": "NVDA", "party": "R",
     "chamber": "HOUSE", "state": "WA", "state_district": "WA04",
     "asset_name": "NVDA common", "asset_type_code": "ST",
     "transaction_type": "BUY", "transaction_date": "2026-07-10",
     "filed_date": "2026-07-17", "amount_min": "15001.00",
     "amount_max": "50000.00", "owner_code": "SELF", "filing_status": "NEW"},
]}

ROSTER = {"politicians": [
    {"bioguide_id": "N000189", "display_name": "Dan Newhouse",
     "aliases": ["dan newhouse"], "chamber": "HOUSE", "state": "WA",
     "district": "04", "party": "R",
     "terms": [{"chamber": "HOUSE", "state": "WA", "district": "04",
                "start_date": "2015-01-06", "end_date": "2027-01-03"}]},
]}

NEWS = [{"title": "Dan Newhouse presses NVIDIA on export licences",
         "summary": "The congressman wants a hearing.",
         "published_at": "2026-08-18"}]


def stub_fetch(monkeypatch, error=None):
    def _fetch(function, **params):
        if error:
            raise error
        return {av_store.INSIDER_FUNCTION: INSIDERS,
                av_store.CONGRESS_FUNCTION: CONGRESS,
                av_store.ROSTER_FUNCTION: ROSTER}[function]

    monkeypatch.setattr(av_store, "fetch", _fetch)


@pytest.fixture
def seeded(db, monkeypatch):
    stub_fetch(monkeypatch)
    av_store.sync_insider_transactions("NVDA")
    av_store.sync_politicians()
    av_store.sync_congress_trades("NVDA")
    return db


def build_prompt(extra_context: str) -> str:
    """The real template, filled the way `analyze` fills it. Only the blocks
    under test carry content: everything else is a placeholder, so any figure
    the auditor flags below came from these blocks or from nowhere."""
    return SINGLE_AGENT_PROMPT.format(
        ticker="NVDA", date=AS_OF, sector_etf="XLK",
        situation_line="momentum only.", track_record_block="no rate yet",
        business_block="chips", spy_block="spy", sector_block="sector",
        price_block="price", tech_block="tech", fundamentals_block="fund",
        news_block="news",
        extra_context=(
            "\n== PRECOMPUTED METRICS & EVENTS (validated. Prefer these numbers) ==\n"
            + extra_context + "\n") if extra_context else "")


def blocks(news=NEWS) -> str:
    insiders, insider_problems = insider_service.insider_block("NVDA", AS_OF)
    politicians, politician_problems = pd_.politician_block(
        "NVDA", AS_OF, news=news)
    assert not insider_problems and not politician_problems
    return "\n\n".join(b for b in (insiders, politicians) if b)


class TestPromptRules:
    def test_the_positioning_step_reads_both_disclosure_blocks(self):
        step = SINGLE_AGENT_PROMPT.split("**Step 2b")[1].split("**Step 3:")[0]
        assert "insider (Form 4)" in step and "congressional-dossier" in step
        assert "DISCLOSURE-LAGGED" in step
        assert "second trading day after the transaction" in step

    def test_it_forbids_the_four_over_readings(self):
        step = SINGLE_AGENT_PROMPT.split("**Step 2b")[1].split("**Step 3:")[0]
        # Disclosure lag is not timing.
        assert "read the gap between a transaction date and its" in step
        # One member is noise; a cluster of executives is the pattern.
        assert "One trade by one member of Congress is noise." in step
        assert "CLUSTER OF DISTINCT" in step
        # A zero price is a grant, not a purchase.
        assert "A SHARE PRICE OF 0.00 IS NOT A PURCHASE." in step
        # Committee relevance is context.
        assert "That is context, not causation" in step
        # Thin evidence is stated as thin rather than stretched.
        assert "give it the weight thin evidence deserves" in step

    def test_section_five_demands_the_names_the_sizes_and_the_lag(self):
        section = SINGLE_AGENT_PROMPT.split("5. Positioning & Flows")[1] \
            .split("6. Peer Comparison")[0]
        assert "NAME the executives and their titles" in section
        assert "NAME the members with their" in section
        assert "visible-from dates" in section
        assert "state the filing lag" in section
        assert "amount bands quoted as the block writes" in section
        # The bands print with a dash, the report's voice rule bans one.
        assert 'your prose writes it "X to Y"' in section
        assert "Rows priced at 0.00 are grants" in section
        assert "cluster of distinct sellers" in section
        assert "for which a block was provided" in section


class TestBlocksInThePrompt:
    def test_the_prompt_carries_the_named_executives_and_members(self, seeded):
        prompt = build_prompt(blocks())

        assert "Jensen Huang (President and CEO)" in prompt
        assert "Colette Kress (EVP and CFO)" in prompt
        assert "Dan Newhouse (R, HOUSE, WA-04)" in prompt
        assert "sitting HOUSE member, in office since 2015-01-06" in prompt
        assert "2026-07-10 BUY $15K–$50K, public from 2026-07-17" in prompt
        assert "$25.10M disposed" in prompt
        assert "visible from 2026-08-12" in prompt

    def test_a_run_that_gathered_neither_leaves_no_dangling_block(
            self, db, monkeypatch):
        # Nothing stored and the vendor down: both blocks decline to write,
        # and the ledger records the gap instead.
        stub_fetch(monkeypatch, error=AlphaVantageUnavailable("throttled"))
        insiders, insider_problems = insider_service.insider_block(
            "NVDA", AS_OF)
        politicians, politician_problems = pd_.politician_block(
            "NVDA", AS_OF, news=NEWS)
        assert insiders == "" and politicians == ""
        assert insider_problems and politician_problems

        prompt = build_prompt("")
        assert "insider transactions (SEC Form 4)" not in prompt
        assert "congressional dossier" not in prompt
        assert "== PRECOMPUTED METRICS & EVENTS" not in prompt
        # The output contract survives the absence: section 5 is still there
        # and still says to state what was not gathered.
        assert "5. Positioning & Flows" in prompt
        assert "If a block was not gathered at all, say" in prompt

    def test_an_empty_window_is_stated_rather_than_dropped(self, db,
                                                           monkeypatch):
        stub_fetch(monkeypatch)
        av_store.sync_insider_transactions("NVDA")
        # 2026-01-15 predates every seeded row, so the window is genuinely
        # empty: the block still writes, because "nobody sold" is evidence.
        block, problems = insider_service.insider_block("NVDA", "2026-01-15")
        assert not problems
        assert "No Form 4 lines were visible in this window." in block


# A report of the shape section 5 now demands: names, titles, share counts,
# dollar values, both dates and the lag, quoting the compressed figures the
# blocks print and expanding the amount band the way a model writing for a
# reader would.
GROUNDED_REPORT = """### 5. Positioning & Flows: two officers sold into the rally
Jensen Huang (President and CEO) disposed 100,000 shares at $180.50, $18.05M,
transacted 2026-08-10 and visible from 2026-08-12. Colette Kress (EVP and CFO)
disposed 40,000 shares at $176.25, $7.05M, transacted 2026-08-05. Over the
window $25.10M was disposed across the 2 rows carrying a real share price. The
25,000 shares Kress acquired came at a share price of 0.00, a restricted stock
grant, so no dollar value attaches to it and it is not a purchase.
Two distinct officers on the same side is the pattern worth weighting; the
visibility dates are filing deadlines, not the moment the tape learned it.
Dan Newhouse (R, HOUSE, WA-04) disclosed one buy of $15,001 to $50,000 on
2026-07-10, public from 2026-07-17. One member trading once is noise.
**Read:** the selling is real but disclosed late, so it prices nothing tomorrow.
"""


class TestFigureGrounding:
    def test_a_report_quoting_the_new_blocks_is_fully_grounded(self, seeded):
        prompt = build_prompt(blocks())
        result = check_figures(GROUNDED_REPORT, prompt, ignore_values=(0.6,))

        assert result.unmatched == []
        # The share prices, the per-executive dollars, the window total and
        # both edges of the amount band are all audited, not skipped.
        assert result.checked >= 7

    def test_an_invented_dollar_figure_is_still_caught(self, seeded):
        prompt = build_prompt(blocks())
        invented = GROUNDED_REPORT.replace("$25.10M", "$42.75M")
        result = check_figures(invented, prompt, ignore_values=(0.6,))

        assert result.unmatched == ["$42.75M"]

    def test_the_compressed_and_expanded_forms_both_ground(self, seeded):
        prompt = build_prompt(blocks())
        # The block prints "$15K–$50K"; a report may quote either shape.
        for text in ("the band was $15K–$50K",
                     "the band was $15,001 to $50,000",
                     "Huang sold $18.05M", "Huang sold $18,050,000"):
            assert check_figures(text, prompt).unmatched == [], text
