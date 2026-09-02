"""What a research report is allowed to be written without.

Every context block the research prompt carries is classed here as
required, expected or optional. The classes decide what happens when a
block cannot be built:

  required: the report is not written. ``MissingRequiredEvidence`` is
              raised, the model returns an error row, and the run reports a
              failed symbol instead of a call that looks like one.
  expected: the report is written, but the gap travels with it: into the
              prompt (so the model says what it could not see), the footer,
              the prediction details and the run's completeness check, which
              marks the run PARTIAL.
  optional: absence is normal (a symbol with no congressional trades, a
              first report with no prior stance) and is only logged.

Before this, an options fetch that hit the vendor's rate limit produced a
report with no positioning block and no trace that one was ever attempted;
a symbol with no mapped peers lost its peer section the same way. The
2026-09-01 BHF report is the reference case: it read like a complete
analysis and was written from a fraction of the evidence the run was
configured to gather.
"""

from dataclasses import dataclass, field

REQUIRED = "required"
EXPECTED = "expected"
OPTIONAL = "optional"

# Block key -> severity. Keys match the labels used in provenance/footers.
BLOCK_SEVERITY: dict[str, str] = {
    "ohlcv": REQUIRED,
    "technicals": REQUIRED,
    "metrics": REQUIRED,
    "spy": REQUIRED,
    # A news SOURCE failure (throttled, down) is not a quiet week. A
    # news-driven report written blind is the exact failure the platform
    # already refuses for the sentiment model; the research arm now refuses
    # too. An empty window with a responding source is fine.
    "news_source": REQUIRED,
    "business": EXPECTED,
    "sector": EXPECTED,
    "fundamentals": EXPECTED,
    "events": EXPECTED,
    "peers": EXPECTED,
    "options": EXPECTED,
    "quality": EXPECTED,
    "investigation": EXPECTED,
    "filings": OPTIONAL,
    "market_context": OPTIONAL,
    "political": OPTIONAL,
    "continuity": OPTIONAL,
    "track_record": OPTIONAL,
    "reflection": OPTIONAL,
}

# Human labels for footers and run summaries.
BLOCK_LABELS: dict[str, str] = {
    "ohlcv": "price bars",
    "technicals": "technical indicators",
    "metrics": "validated metrics",
    "spy": "SPY market context",
    "news_source": "news source",
    "business": "business profile",
    "sector": "sector ETF context",
    "fundamentals": "fundamentals",
    "events": "event calendar",
    "peers": "peer relative strength",
    "options": "options positioning",
    "quality": "quality screen",
    "investigation": "situation & investigation",
    "filings": "SEC filings",
    "market_context": "market breadth snapshot",
    "political": "political & institutional flows",
    "continuity": "prior stance",
    "track_record": "measured accuracy",
    "reflection": "reflection memory",
}


class MissingRequiredEvidence(RuntimeError):
    """A block the report cannot honestly be written without is absent."""

    def __init__(self, symbol: str, block: str, reason: str):
        self.symbol = symbol
        self.block = block
        self.reason = reason
        super().__init__(
            f"{symbol}: required evidence missing, "
            f"{BLOCK_LABELS.get(block, block)}: {reason}")


@dataclass
class EvidenceGap:
    block: str
    severity: str
    reason: str

    @property
    def label(self) -> str:
        return BLOCK_LABELS.get(self.block, self.block)

    def to_dict(self) -> dict:
        return {"block": self.block, "severity": self.severity,
                "reason": self.reason, "label": self.label}


@dataclass
class EvidenceLedger:
    """Which blocks a report was built from, and which it was built without.

    ``present`` and ``gaps`` are appended as the prompt is assembled; the
    ledger is then serialised into the prediction details and rendered in
    the report footer, so a reader of any surface can tell a complete
    report from a degraded one.
    """

    symbol: str
    present: list[str] = field(default_factory=list)
    gaps: list[EvidenceGap] = field(default_factory=list)

    def have(self, block: str) -> None:
        if block not in self.present:
            self.present.append(block)

    def missing(self, block: str, reason: str,
                severity: str | None = None) -> None:
        """Record an absent block. Raises for required blocks."""
        severity = severity or BLOCK_SEVERITY.get(block, OPTIONAL)
        if severity == REQUIRED:
            raise MissingRequiredEvidence(self.symbol, block, reason)
        self.gaps.append(EvidenceGap(block, severity, reason))

    def expected_gaps(self) -> list[EvidenceGap]:
        return [g for g in self.gaps if g.severity == EXPECTED]

    @property
    def degraded(self) -> bool:
        return bool(self.expected_gaps())

    def summary(self, severity: str | None = EXPECTED) -> str:
        """'options positioning (AV throttled); peer relative strength (…)'."""
        gaps = (self.expected_gaps() if severity == EXPECTED
                else [g for g in self.gaps if severity in (None, g.severity)])
        return "; ".join(f"{g.label} ({g.reason})" for g in gaps)

    def prompt_block(self) -> str:
        """The gaps, phrased for the model: name what it cannot see."""
        gaps = self.expected_gaps()
        if not gaps:
            return ""
        lines = ["[Evidence NOT available for this report. Do not infer it]"]
        for g in gaps:
            lines.append(f"- {g.label}: {g.reason}")
        lines.append("Where a section would rest on one of these, say the "
                     "evidence was unavailable rather than reasoning as if "
                     "it were neutral.")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "present": list(self.present),
            "gaps": [g.to_dict() for g in self.gaps],
            "degraded": self.degraded,
        }


def gaps_from_details(details: dict | None) -> list[dict]:
    """Expected-severity gaps stored on a prediction's details, if any."""
    ledger = (details or {}).get("evidence") or {}
    return [g for g in (ledger.get("gaps") or [])
            if g.get("severity") == EXPECTED]
