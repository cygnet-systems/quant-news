"""The report page: one parser, one Dash view, one PDF, same structure."""

from services.report_parse import parse_research_report

NEW = """## Verdict
FINAL TRANSACTION PROPOSAL: **SELL**
CONVICTION: **0.61** (this report's own probability that the direction is right, not a measured hit rate)
MEASURED ACCURACY: Not enough evaluated history to state a hit rate (0 scored non-HOLD calls).
SINCE LAST REPORT: no prior report on record
REASSESS_TO_BUY: close above the 50-day SMA, approximately $61.00 as of 2026-09-01
MOVE_TO_SELL: close below 20-day support, approximately $51.50 as of 2026-09-01
- The pending acquisition is still exposed to unresolved insurance approvals.
- The $70.00 cash offer creates substantial upside if approvals advance.
- The next session must hold above approximately $51.50.

### 1. Situation & Key Figures: pending acquisition with unresolved regulatory completion risk
BHF is classified as a pending acquisition. Aquarian agreed to pay $70.00 per share.

Second paragraph with **bold**.

**Read:** The next 1-5 sessions depend on Delaware approval progress.

### 2. Technicals (BHF): bearish trend, oversold conditions
Close $52.99 versus 20SMA $56.45.

**Read:** Trend is down; RSI 30 allows a bounce.

```json
{"stance": "BEARISH", "watch_items": ["$51.50"]}
```

---
*Compiled by gpt-5.6-luna (openai) · as-of 2026-09-01 · Sources: 251-bar OHLCV. **Written without expected evidence: peers.** Every figure above comes from these blocks.*
"""

LEGACY = """## Verdict
FINAL TRANSACTION PROPOSAL: **BUY**
CONVICTION: **0.7**
### 1. Technicals — bullish trend
Price above SMAs.
**Read:** Momentum favours longs.
"""


def test_parses_new_format_into_term_sheet_and_sections():
    p = parse_research_report(NEW)
    assert p.structured
    assert p.call == "SELL" and p.conviction == 0.61
    assert p.conviction_note.startswith("this report's own probability")
    assert p.measured.startswith("Not enough evaluated history")
    assert p.since == "no prior report on record"
    assert p.reassess.startswith("close above the 50-day SMA")
    assert p.move.startswith("close below 20-day support")
    assert len(p.why) == 3 and p.why[1].startswith("The $70.00 cash offer")
    assert [s.num for s in p.sections] == ["1", "2"]
    s1 = p.sections[0]
    assert s1.name == "Situation & Key Figures"
    assert s1.takeaway.startswith("pending acquisition with")
    assert "Second paragraph with **bold**." in s1.body_md
    assert "Read:" not in s1.body_md
    assert s1.read.startswith("The next 1-5 sessions")
    assert p.sections[1].name == "Technicals (BHF)"
    assert "stance" not in NEW.replace("```", "") or "json" not in p.sections[1].body_md
    assert p.footer.startswith("Compiled by gpt-5.6-luna")
    assert "Written without expected evidence: peers" in p.footer


def test_parses_legacy_dash_headings():
    p = parse_research_report(LEGACY)
    assert p.structured
    assert p.call == "BUY" and p.conviction == 0.7
    assert p.sections[0].name == "Technicals"
    assert p.sections[0].takeaway == "bullish trend"
    assert p.sections[0].read == "Momentum favours longs."


def test_free_form_text_falls_back_to_markdown():
    p = parse_research_report("Just some prose about a stock.\n\nMore prose.")
    assert not p.structured
    assert p.legacy_body.startswith("Just some prose")


def test_dash_view_builds_for_both_formats():
    from layouts.report_view import build_report_view
    view = build_report_view(NEW, symbol="BHF", decision="SELL", weight=0.5,
                             trade_date="2026-09-01", model_name="gpt-5.6-luna")
    assert "rr" in view.className
    text = str(view.to_plotly_json())
    assert "rr-verdict-sell" in text and "Situation & Key Figures" in text
    assert "rr-read" in text and "rr-footer" in text
    legacy = build_report_view("plain prose", symbol="X")
    assert "rr-legacy" in legacy.className


def test_pdf_renders_bytes_with_the_same_structure():
    from services.report_service import generate_ta_report_pdf
    pdf = generate_ta_report_pdf({
        "symbol": "BHF", "decision": "SELL", "confidence": 0.5,
        "trade_date": "2026-09-01", "report_text": NEW,
        "model_name": "gpt-5.6-luna", "created_at": "2026-09-02T01:00:00",
    })
    assert pdf and pdf[:4] == b"%PDF"
    legacy_pdf = generate_ta_report_pdf({"symbol": "X", "decision": "HOLD",
                                         "confidence": 0.5, "trade_date": "2026-01-01",
                                         "report_text": "Just prose."})
    assert legacy_pdf and legacy_pdf[:4] == b"%PDF"
