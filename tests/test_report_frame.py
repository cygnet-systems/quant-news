"""The report frame after the 2026-09-06 reorder.

Meat first: Situation (with trajectory and flags), the researched anomaly
sections, News, Technicals & Trade Plan, Bull vs Bear, Scenarios & Risk, and
the evidence sections at the bottom. An anomaly with no researched finding is
a flag bullet in section 1, not a section that says "not researched" at
length; a news-volume spike keeps its section because its own articles are
in the NEWS block. The research prompt reads every article the run kept.
"""
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

from models.single_agent import (
    FIXED_SECTIONS, _news_block, earns_section, render_output_sections,
    split_anomalies,
)
from services import anomaly_service

AS_OF = "2026-09-04"


def _anomaly(key, researched=False):
    return {"key": key, "title": f"{key} title", "researched": researched,
            "facts": ["a fact"], "question": "why?", "evidence_key": key}


class TestTheFrame:
    def test_meat_first_and_no_static_sections(self):
        names = [s.split(":")[0].strip() for s in FIXED_SECTIONS]
        assert names == [
            "Situation & Key Figures", "News & Catalysts",
            "Technicals & Trade Plan ({ticker})", "Bull vs Bear",
            "Scenarios & Risk", "Fundamentals", "Positioning & Flows",
            "Peer Comparison",
        ]
        flat = " ".join(FIXED_SECTIONS)
        assert "Business Context" not in flat
        assert "Market & Sector Backdrop" not in flat
        # The trajectory paragraph reads the developments list, never the
        # profile paragraph.
        situation = " ".join(FIXED_SECTIONS[0].split())
        assert "RECENT DEVELOPMENTS" in situation
        assert "never fill the gap from the profile paragraph" in situation

    def test_positioning_is_a_pattern_not_a_roster(self):
        section = FIXED_SECTIONS[6]
        assert "Lead each with the pattern, not the roster" in section
        assert "ONLY for the one or two filers who define the" in section
        assert "do not transcribe it" in section


class TestFlagsVersusSections:
    def test_unresearched_anomalies_are_flags_in_section_one(self):
        found = [_anomaly("quality_failures"), _anomaly("options_skew"),
                 _anomaly("congress_activity")]
        assert split_anomalies(found) == ([], found)
        sections = render_output_sections("ORCL", AS_OF, "XLK", found)
        assert "(d) Also flagged for ORCL today, not researched" in sections
        assert '"quality_failures title" (block "quality_failures")' in sections
        # No numbered section for them, and the frame is not renumbered.
        assert '"quality_failures". Three parts' not in sections
        assert "2. News & Catalysts" in sections
        assert "7. Positioning & Flows" in sections
        # Flagged is not quiet: the short-report wording must not appear.
        assert "Nothing stands out for ORCL" not in sections

    def test_a_researched_anomaly_and_a_news_spike_earn_sections(self):
        found = [_anomaly("options_skew", researched=True),
                 _anomaly("news_spike"), _anomaly("quality_failures")]
        assert earns_section(found[0]) and earns_section(found[1])
        assert not earns_section(found[2])
        sections = render_output_sections("ORCL", AS_OF, "XLK", found)
        assert "Sections 2 to 3 below are the reason this" in sections
        assert '"options_skew". Three parts' in sections
        assert '"news_spike". Three parts' in sections
        assert "4. News & Catalysts" in sections
        assert '"quality_failures title" (block "quality_failures")' in sections

    def test_no_anomalies_is_still_the_quiet_wording(self):
        sections = render_output_sections("ORCL", AS_OF, "XLK", [],
                                          screened=["news volume"])
        assert "Nothing stands out for ORCL" in sections
        assert "Also flagged" not in sections


class TestNewsSpikeBlock:
    def test_unresearched_spike_points_at_the_news_block(self):
        a = dict(_anomaly("news_spike"), unresearched="no_web")
        block = anomaly_service.format_anomaly_block("ORCL", a, None)
        assert "web research was off for this run" in block
        assert "articles from that day in the NEWS block" in block
        assert "do not supply one of your own" not in block

    def test_other_unresearched_anomalies_keep_the_no_cause_rule(self):
        a = dict(_anomaly("options_skew"), unresearched="no_web")
        block = anomaly_service.format_anomaly_block("ORCL", a, None)
        assert "do not supply one of your own" in block


class TestCongressRecency:
    def _row(self, transacted, public, kind="SALE"):
        return {"politician": "A Member", "bioguide_id": "M1", "type": kind,
                "transaction_date": transacted.isoformat(),
                "filed_date": public.isoformat(),
                "amount_min": 15001.0, "amount_max": 50000.0}

    def test_a_four_month_old_trade_is_not_an_anomaly_today(self):
        as_of = date(2026, 9, 4)
        old = [self._row(as_of - timedelta(days=120), as_of - timedelta(days=95))]
        assert anomaly_service.detect("ORCL", as_of, congress=old) == []

    def test_a_disclosure_that_just_became_public_is(self):
        as_of = date(2026, 9, 4)
        rows = [self._row(as_of - timedelta(days=40), as_of - timedelta(days=3))]
        found = anomaly_service.detect("ORCL", as_of, congress=rows)
        assert [a["key"] for a in found] == ["congress_activity"]
        flat = " ".join(found[0]["facts"])
        assert "within the last 30 days" in flat
        assert "most recent disclosure public 2026-09-01" in flat


class TestNewsBlockReadsEverything:
    def test_every_kept_article_is_in_the_block(self):
        arts = [SimpleNamespace(title=f"t{i}", summary="s", sentiment="neutral",
                                ticker_relevance_score=0.9, source="Wire",
                                published_at=datetime(2026, 8, 10 + i % 20,
                                                      tzinfo=timezone.utc))
                for i in range(50)]
        text, shown, span = _news_block(arts)
        assert shown == 50
        assert "all 50 articles in the window" in text
        assert text.count("\n- ") == 50
        assert "treat repetition as one fact" in text
