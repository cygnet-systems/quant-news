"""Cache-correctness guards for the scheduled analysis pipeline.

A cache that returns a *nearly* right prediction is worse than no cache at
all: once stored, a stale verdict is indistinguishable from a fresh one, and
every downstream artifact (synthesis, scoreboard, backtest) inherits it. These
tests pin the conditions under which stored predictions may be reused.

Each case here corresponds to a way the cache was, or could have been, wrong:

* pipeline_epoch — a data-quality fix upstream of the models changes what they
  are fed while nothing else in the key moves (the sector-lookup race, 2026-08-05)
* previous_close — the vendor revises the cutoff bar after the run
* research model — the caller asks for a different report model
* news_count — the vendor indexes late articles for the same window
* recommendation_synthesis — Luna's own verdict is stored as a prediction and
  must never be handed back as independent model evidence
"""

import copy
from datetime import date
from unittest.mock import patch

import pandas as pd
import pytest

from services import analysis_runner as ar

CUTOFF = date(2026, 8, 4)
RESEARCH_MODEL = "gpt-5.6-luna"
NEWS_COUNT = 40


@pytest.fixture
def price_df():
    """Two bars; the last close is what stored rows must agree with."""
    return pd.DataFrame({"Close": [100.0, 110.0]})


@pytest.fixture
def stored():
    return {
        name: {
            "decision": "BUY",
            "previous_close": 110.0,
            "details": {
                "pipeline_epoch": ar.PIPELINE_EPOCH,
                "model": RESEARCH_MODEL,
                "news_count": NEWS_COUNT,
            },
        }
        for name in list(ar.ALL_MODELS) + ["ensemble"]
    }


def reuse(stored_rows, price_df, **overrides):
    kwargs = {"research_model": RESEARCH_MODEL, "news_count": NEWS_COUNT}
    kwargs.update(overrides)
    with patch("services.cache_service.get_cache") as get_cache:
        get_cache.return_value.get_predictions_for_today.return_value = stored_rows
        return ar._reusable_predictions(
            "TEST", CUTOFF, set(ar.ALL_MODELS), price_df, **kwargs)


def test_identical_inputs_are_reused(stored, price_df):
    assert reuse(stored, price_df) is not None


def test_older_pipeline_epoch_forces_rerun(stored, price_df):
    stored["kronos_mini"]["details"]["pipeline_epoch"] = "2026-01-01.0"
    assert reuse(stored, price_df) is None


def test_missing_pipeline_epoch_forces_rerun(stored, price_df):
    """Rows written before the epoch existed carry no claim about their inputs."""
    del stored["xgboost_shap"]["details"]["pipeline_epoch"]
    assert reuse(stored, price_df) is None


def test_revised_cutoff_bar_forces_rerun(stored, price_df):
    stored["lightgbm"]["previous_close"] = 109.5
    assert reuse(stored, price_df) is None


def test_different_research_model_forces_rerun(stored, price_df):
    assert reuse(stored, price_df, research_model="claude-sonnet-5") is None


def test_grown_news_window_forces_rerun(stored, price_df):
    assert reuse(stored, price_df, news_count=NEWS_COUNT + 10) is None


def test_incomplete_model_set_forces_rerun(stored, price_df):
    del stored["ensemble"]
    assert reuse(stored, price_df) is None


def test_synthesis_output_is_never_returned_as_a_model_signal(stored, price_df):
    """Luna's stored verdict must not come back as evidence for Luna."""
    stored["recommendation_synthesis"] = {
        "decision": "BUY",
        "previous_close": 110.0,
        "details": {"pipeline_epoch": ar.PIPELINE_EPOCH},
    }
    reused = reuse(stored, price_df)
    assert reused is not None
    assert "recommendation_synthesis" not in reused


def test_empty_store_forces_rerun(price_df):
    assert reuse({}, price_df) is None
