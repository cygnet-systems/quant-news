"""Parse a research report's markdown into the structure every renderer shows.

The report text the model writes is one markdown document: a Verdict block of
labelled field lines and three bullets, ten "### n. Name: takeaway" sections
each ending in a "**Read:**" line, a machine-read JSON epilogue, and a
"Compiled by" provenance footer. The in-app reader, the Home reading pane,
the Analyze tab and the PDF all used to hand that whole string to a generic
markdown renderer, so the verdict folded into one paragraph and the sections
read as undifferentiated headings.

This module reads the document once into ``ParsedReport``; ``layouts.
report_view`` builds the Dash page from it and ``services.report_service``
builds the PDF, so the two stay the same shape. Older reports (dash-separated
headings, or free-form text without a Verdict block) still parse: what cannot
be structured is kept as markdown in ``legacy_body``.
"""

import re
from dataclasses import dataclass, field
from typing import Optional

_FIELD_KEYS = (
    ("FINAL TRANSACTION PROPOSAL", "call"),
    ("CONVICTION", "conviction"),
    ("MEASURED ACCURACY", "measured"),
    ("SINCE LAST REPORT", "since"),
    ("REASSESS_TO_BUY", "reassess"),
    ("MOVE_TO_SELL", "move"),
)
_FIELD_RE = re.compile(
    r"^\s*(?:[-*]\s*)?\**\s*(" + "|".join(re.escape(k) for k, _ in _FIELD_KEYS)
    + r")\s*:?\**\s*:?\s*(.*)$", re.I)
_HEADING_RE = re.compile(r"^###\s+(\d+)\.\s*(.+?)(?:\s*:\s+|\s+[—–-]\s+)(.+)$")
_HEADING_PLAIN_RE = re.compile(r"^###\s+(?:(\d+)\.\s*)?(.+)$")
_READ_RE = re.compile(r"^\s*\**Read:?\**\s*:?\s*(.+)$", re.I)
_FOOTER_RE = re.compile(r"\n-{3,}\n\s*\*(.*?)\*\s*$", re.S)
_NUM_RE = re.compile(r"([01](?:\.\d+)?)")


@dataclass
class ReportSection:
    num: str
    name: str
    takeaway: str
    body_md: str
    read: str = ""


@dataclass
class ParsedReport:
    call: str = ""
    conviction: Optional[float] = None
    conviction_note: str = ""
    measured: str = ""
    since: str = ""
    reassess: str = ""
    move: str = ""
    why: list = field(default_factory=list)
    sections: list = field(default_factory=list)
    footer: str = ""
    legacy_body: str = ""

    @property
    def structured(self) -> bool:
        return bool(self.call) and bool(self.sections)


def _clean(value: str) -> str:
    value = value.strip()
    value = re.sub(r"^\*\*(.+?)\*\*", r"\1", value)      # leading bold value
    return value.replace("**", "").strip()


def parse_research_report(text: str) -> ParsedReport:
    from models.single_agent import strip_epilogue

    body = strip_epilogue(text or "").strip()
    parsed = ParsedReport()

    m = _FOOTER_RE.search(body)
    if m:
        parsed.footer = re.sub(r"\s+", " ", m.group(1)).strip()
        body = body[:m.start()].rstrip()

    chunks = re.split(r"\n(?=###\s)", "\n" + body)
    head = chunks[0].strip()
    for line in head.splitlines():
        fm = _FIELD_RE.match(line)
        if fm:
            key = next(k for k, _ in _FIELD_KEYS if k.lower() == fm.group(1).lower())
            attr = dict(_FIELD_KEYS)[key]
            value = _clean(fm.group(2))
            if attr == "conviction":
                nm = _NUM_RE.search(value)
                parsed.conviction = float(nm.group(1)) if nm else None
                note = value[nm.end():] if nm else value
                parsed.conviction_note = note.strip(" :()").strip()
            else:
                setattr(parsed, attr, value)
        elif line.lstrip().startswith(("- ", "* ")):
            parsed.why.append(_clean(line.lstrip()[2:]))

    for chunk in chunks[1:]:
        heading, _, rest = chunk.strip().partition("\n")
        hm = _HEADING_RE.match(heading)
        if hm:
            num, name, take = hm.group(1), hm.group(2).strip(), hm.group(3).strip()
        else:
            pm = _HEADING_PLAIN_RE.match(heading)
            num = (pm.group(1) or "") if pm else ""
            name = (pm.group(2).strip() if pm else heading.lstrip("# ").strip())
            take = ""
        read = ""
        kept = []
        for para in re.split(r"\n\s*\n", rest.strip()):
            # A Read line usually stands alone; older reports glued it to
            # the paragraph above with a single newline.
            lines = para.strip().splitlines()
            rm = _READ_RE.match(lines[-1]) if lines else None
            if rm and not read:
                read = _clean(rm.group(1))
                if len(lines) > 1:
                    kept.append("\n".join(lines[:-1]))
            else:
                kept.append(para)
        parsed.sections.append(ReportSection(
            num=num, name=name, takeaway=take,
            body_md="\n\n".join(kept).strip(), read=read))

    if not parsed.structured:
        parsed.legacy_body = body
    return parsed


DECISION_TONE = {"BUY": "buy", "SELL": "sell", "HOLD": "hold"}


def tone_for(call: str) -> str:
    return DECISION_TONE.get((call or "").upper(), "hold")
