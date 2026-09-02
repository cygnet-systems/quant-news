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
