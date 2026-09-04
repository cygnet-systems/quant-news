"""One door to Alpha Vantage: quota, key, and the HTTP-200 error bodies.

Four call sites used to hand-roll ``requests.get`` against the same base URL
and each re-implemented the same two subtleties, one of them badly at least
once:

* the shared quota. Every caller on this box competes for the same key, so
  the call has to be paced through ``rate_limiter.alpha_vantage_bucket()``
  before it is spent, not after it fails.
* the vendor answers a throttled or malformed request with **HTTP 200** and
  an explanatory body under ``Note`` / ``Information`` / ``Error Message``.
  ``raise_for_status`` sees nothing wrong, so a caller that only checks the
  transport reads a rate limit as "this symbol has no data" — which is how a
  throttled news fetch once became "no company-specific news" in a report.

This module raises :class:`AlphaVantageUnavailable` for both the missing key
and those bodies. What a caller does with it stays the caller's decision:
news re-raises (its evidence is REQUIRED and a gap must be loud), options
returns None (no chain is a normal answer for an unoptioned name), political
lets it out so the run records an OPTIONAL gap.
"""

from __future__ import annotations

import logging

import requests

from config import API
from services.rate_limiter import alpha_vantage_bucket

logger = logging.getLogger(__name__)

# Keys the vendor uses to explain a 200 that carries no data.
ERROR_KEYS = ("Note", "Information", "Error Message")


class AlphaVantageUnavailable(RuntimeError):
    """The vendor answered with a throttle/limit/error body, not data."""


def fetch(function: str, **params) -> dict:
    """Call ``function`` and return the decoded body.

    Params with a None value are dropped, so a caller can pass an optional
    argument without branching. The bucket is acquired with the same
    generous timeout the hand-rolled callers used: a queue behind the quota
    is normal at 70 calls/minute and is not a failure.

    Raises:
        AlphaVantageUnavailable: no API key, or the body is a vendor
            message rather than data.
        requests.RequestException: transport failure or a non-2xx status.
    """
    if not API.ALPHA_VANTAGE_API_KEY:
        raise AlphaVantageUnavailable(f"{function}: no ALPHA_VANTAGE_API_KEY")

    alpha_vantage_bucket().acquire(timeout=API.DEFAULT_TIMEOUT * 4)
    response = requests.get(
        API.ALPHA_VANTAGE_BASE_URL,
        params={
            "function": function,
            **{k: v for k, v in params.items() if v is not None},
            "apikey": API.ALPHA_VANTAGE_API_KEY,
        },
        timeout=API.DEFAULT_TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise AlphaVantageUnavailable(
            f"{function}: expected a JSON object, got {type(data).__name__}")
    for key in ERROR_KEYS:
        if key in data:
            raise AlphaVantageUnavailable(
                f"{function} {key}: {str(data[key])[:200]}")
    return data
