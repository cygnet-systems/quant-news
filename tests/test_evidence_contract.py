"""The evidence contract: required blocks raise, expected blocks are carried
as gaps into the prompt, the footer, the details and the run summary."""

import pytest

from services.evidence_contract import (
    EXPECTED, OPTIONAL, EvidenceLedger, MissingRequiredEvidence,
    gaps_from_details,
)


def test_required_block_raises_with_symbol_and_reason():
    ledger = EvidenceLedger("BHF")
    with pytest.raises(MissingRequiredEvidence) as exc:
        ledger.missing("news_source", "vendor throttled")
    assert exc.value.symbol == "BHF"
    assert exc.value.block == "news_source"
    assert "vendor throttled" in str(exc.value)
    # Nothing was recorded as a soft gap. The report is not written at all.
    assert ledger.gaps == []


def test_expected_gap_is_carried_not_raised():
    ledger = EvidenceLedger("BHF")
    ledger.have("metrics")
    ledger.missing("options", "no chain returned")
    ledger.missing("peers", "no peer set mapped for this symbol")
    ledger.missing("political", "AV throttled")  # optional: logged only
    assert ledger.degraded
    assert [g.block for g in ledger.expected_gaps()] == ["options", "peers"]
    summary = ledger.summary()
    assert "options positioning (no chain returned)" in summary
    assert "peer relative strength" in summary
    assert "political" not in summary  # optional gaps never mark the run


def test_prompt_block_names_each_expected_gap_for_the_model():
    ledger = EvidenceLedger("BHF")
    ledger.missing("options", "no chain returned")
    block = ledger.prompt_block()
    assert block.startswith("[Evidence NOT available")
    assert "options positioning: no chain returned" in block
    assert "rather than reasoning as if it were neutral" in block
    assert EvidenceLedger("X").prompt_block() == ""


def test_details_round_trip_keeps_only_expected_gaps():
    ledger = EvidenceLedger("BHF")
    ledger.missing("quality", "screen failed")
    ledger.missing("continuity", "first report")
    details = {"evidence": ledger.to_dict()}
    gaps = gaps_from_details(details)
    assert [g["block"] for g in gaps] == ["quality"]
    assert gaps[0]["severity"] == EXPECTED
    assert gaps_from_details({}) == []
    assert gaps_from_details(None) == []


def test_explicit_severity_override():
    ledger = EvidenceLedger("BHF")
    ledger.missing("options", "not selected this run", severity=OPTIONAL)
    assert not ledger.degraded


class TestContextBudget:
    """A block the ledger recorded as present has to reach the prompt.

    The old assembly sliced each block to a fixed width and then cut the
    joined string from the end, so the blocks appended last -- the insider
    and congressional disclosures -- were the first to disappear while the
    ledger still reported them present.
    """

    def _fit(self, blocks, **kw):
        from models.trading_agents_model import _fit_blocks

        return _fit_blocks(blocks, **kw)

    def test_no_block_is_dropped_when_the_total_overflows(self):
        blocks = [f"[block {i} header]\n" + "line of evidence\n" * 400
                  for i in range(12)]
        fitted = self._fit(blocks)
        for i in range(12):
            assert f"[block {i} header]" in fitted

    def test_the_longest_blocks_pay_for_the_overflow(self):
        short = "[short]\n" + "x\n" * 20
        long = "[long]\n" + "y\n" * 8000
        fitted = self._fit([short, long, short], budget=2000)
        assert fitted.count(short.strip()) == 2   # both short blocks whole
        assert "[long]" in fitted and "[truncated]" in fitted

    def test_a_disclosure_sized_block_is_not_cut_at_all(self):
        from models.trading_agents_model import PER_BLOCK_CHARS

        # The insider block runs to ~2.7 KB and the dossier to ~2.8 KB on a
        # busy name; both must arrive whole alongside the technical blocks.
        disclosure = "[NVDA: insider transactions]\n" + "filing line\n" * 200
        assert len(disclosure) < PER_BLOCK_CHARS
        fitted = self._fit([disclosure, "[metrics]\nrsi 51\n"])
        assert disclosure in fitted

    def test_the_result_fits_the_prompt_budget(self):
        from models.single_agent import MAX_EXTRA_CONTEXT_CHARS

        blocks = [f"[b{i}]\n" + "z\n" * 3000 for i in range(14)]
        assert len(self._fit(blocks)) <= MAX_EXTRA_CONTEXT_CHARS

    def test_empty_blocks_are_dropped_not_joined_as_blank_gaps(self):
        assert self._fit(["", "[a]\nrow", ""]) == "[a]\nrow"
        assert self._fit([]) == ""
