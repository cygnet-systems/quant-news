"""Post-hoc numeric validation of LLM report text against its source data.

The report contract says every figure must come from the provided data blocks
("flagged 'not in data' where missing"), but a prompt rule is a request, not
a check. This is the check: extract the numbers the report actually cites and
verify each one appears somewhere in the data the model was shown.

Matching is deliberately tolerant, a report legitimately rounds 94.196 to
94.20 or "$94", is REQUIRED by the prompt to round price levels to a tick, and
may drop a minus sign or a percent marker, so a figure counts as grounded
when the source contains any number of the same magnitude within that
rounding. Compressed magnitudes ("$18.90M", "1.25M sh", the "$15K-$50K"
bands the disclosure blocks print) are read at both magnitudes on both sides,
so quoting one either way grounds. What this catches is not sloppy rounding
but invention: prices, percentages, and ratios that appear in no input at all.
"""

import re
from dataclasses import dataclass

# Numbers worth checking: attached to $, %, x-multiples, or ratios, plus bare
# decimals with 2+ significant digits. Bare small integers (step numbers,
# list indices, "50-day SMA") are noise, not claims.
_FIGURE_RE = re.compile(
    r"\$\s*([0-9][\d,]*\.?\d*)"      # $363.86, $1,000
    r"|([0-9][\d,]*\.?\d*)\s*%"      # 46.8%, 5%
    r"|\b([0-9][\d,]*\.\d{2,})\b"    # bare decimals: 94.196, 0.24
    r"|([0-9][\d,]*\.?\d*)\s*x\b"    # 0.78x, 12.05x
    r"|([0-9][\d,]*\.?\d*)(?=[KMBT]\b)",  # 1.25M sh, 105K shares
    re.IGNORECASE,
)

_SOURCE_NUM_RE = re.compile(r"-?[0-9][\d,]*\.?\d*")

# The evidence blocks write large figures compressed ("$18.90M disposed",
# "1.25M sh", "$15K–$50K" for a STOCK Act band) and the report is told to
# quote them that way, but a model legitimately expands one to $18,900,000.
# Both sides are read at both magnitudes so a correct quote in either shape
# grounds, and so "$15K" is audited as the $15,000 claim it is rather than
# skipped as a small integer.
_SCALES = {"k": 1e3, "m": 1e6, "b": 1e9, "t": 1e12}


def _scale_at(text: str, idx: int) -> float | None:
    """The magnitude multiplier of a suffix immediately at ``idx``.

    Deliberately strict: the suffix must touch the digits and must end a
    word, so "1.25M sh" scales while "3.5MB" and "1Billion" do not.
    """
    ch = text[idx:idx + 1].lower()
    if ch not in _SCALES:
        return None
    nxt = text[idx + 1:idx + 2]
    if nxt.isalnum() or nxt == "_":
        return None
    return _SCALES[ch]


# Integers below this are ignored as figures. They are almost always
# structure (step 3, top 5, 50-day) rather than data claims.
_MIN_BARE_INT = 100

# The report prompt REQUIRES trigger/invalidation levels to be rounded to a
# tick (nearest $0.05 under $50, $0.25 to $500, $1 above), because the long
# moving averages they are quoted against shift between runs. A checker that
# only tolerates rounding at the printed precision then flags every correctly
# rounded level: "$18.65" against a source 18.67 is the rule working, not
# invention. Half a tick is the largest gap correct rounding can open.
_PRICE_TICKS = ((50.0, 0.05), (500.0, 0.25), (float("inf"), 1.0))


def _price_tolerance(value: float) -> float:
    """Half the tick the prompt mandates at this price magnitude."""
    for ceiling, tick in _PRICE_TICKS:
        if abs(value) < ceiling:
            return tick / 2.0
    return _PRICE_TICKS[-1][1] / 2.0


@dataclass
class FigureCheckResult:
    checked: int
    unmatched: list[str]

    @property
    def grounded_ratio(self) -> float:
        if not self.checked:
            return 1.0
        return 1.0 - len(self.unmatched) / self.checked


def _parse(num_str: str) -> float | None:
    try:
        return float(num_str.replace(",", ""))
    except (TypeError, ValueError):
        return None


def _decimals(num_str: str) -> int:
    return len(num_str.split(".")[1]) if "." in num_str else 0


def check_figures(
    report_text: str,
    source_text: str,
    ignore_values: "tuple[float, ...] | list[float] | None" = None,
) -> FigureCheckResult:
    """Verify the figures in ``report_text`` against ``source_text``.

    Returns the count of figures checked and the distinct figures with no
    source match. Derived arithmetic the model performs (differences, ratios
    of two grounded numbers) will show as unmatched. Treat the result as a
    review signal, not a hard failure: a HIGH unmatched ratio means the
    report is inventing, a few unmatched entries usually mean arithmetic.

    Comparison is on MAGNITUDE. A source "MACD: -0.040" and a report writing
    "0.040" are the same reading, and this check exists to catch numbers that
    came from nowhere, not sign or percent-formatting differences, matching
    signed made a grounded figure look invented.

    ``ignore_values`` drops figures the report legitimately generates rather
    than sources, above all its own stated conviction: that number is the
    model's judgement and by definition appears in no input block.
    """
    source_values: set[float] = set()
    # A compressed source figure is itself rounded: "$15K" is any amount
    # between $14,500 and $15,500, so a report expanding it to the exact
    # disclosed band edge ($15,001) is quoting, not inventing. Only the
    # scaled readings carry a half-precision, because only they lost digits.
    source_rounded: list[tuple[float, float]] = []
    for m in _SOURCE_NUM_RE.finditer(source_text):
        v = _parse(m.group(0))
        if v is None:
            continue
        v = abs(v)
        source_values.add(round(v, 4))
        scale = _scale_at(source_text, m.end())
        if scale:
            source_values.add(round(v * scale, 4))
            source_rounded.append(
                (v * scale, 0.5 * (10 ** -_decimals(m.group(0))) * scale))

    ignored = {round(abs(float(v)), 4) for v in (ignore_values or [])
               if v is not None}

    checked = 0
    unmatched: list[str] = []
    seen: set[str] = set()

    for m in _FIGURE_RE.finditer(report_text):
        raw = next(g for g in m.groups() if g)
        # The suffix is part of the claim: "$18.90M" and "18.90" are two
        # different figures and only one of them is money.
        scale = _scale_at(report_text, m.end())
        key = f"{raw}{scale:.0f}" if scale else raw
        if key in seen:
            continue
        seen.add(key)
        value = _parse(raw)
        if value is None:
            continue
        dp = _decimals(raw)
        value = abs(value) * (scale or 1.0)
        # Precision follows the printed digits, then the magnitude: "$18.90M"
        # is stated to the nearest $10,000, not the nearest cent.
        precision = 0.5 * (10 ** -dp) * (scale or 1.0) + 1e-9
        if dp == 0 and value < _MIN_BARE_INT:
            continue
        if any(abs(value - iv) <= precision for iv in ignored):
            continue
        checked += 1

        # Grounded iff some source number equals it at the precision the
        # report used (so "$94.20" matches source 94.196, "$94" matches 94.4).
        # Dollar figures additionally get half a mandated tick, because the
        # prompt requires levels to be rounded to one.
        tolerance = precision
        if m.group(1):  # the "$..." alternative
            tolerance = max(tolerance, _price_tolerance(value))
        if any(abs(value - sv) <= tolerance
               or (sv != 0 and abs(value - round(sv, dp)) <= tolerance)
               for sv in source_values):
            continue
        if any(abs(value - sv) <= max(tolerance, half)
               for sv, half in source_rounded):
            continue
        suffix = report_text[m.end():m.end() + 1] if scale else ""
        unmatched.append(m.group(0).strip() + suffix)

    return FigureCheckResult(checked=checked, unmatched=unmatched)
