"""A record badge only when the scored history at a score is distinguishable
from a coin flip, and the light "levels" field on prediction rows."""
from services import calibration_service as cal
from services.cache_service import _levels_of


def _fit_with(bins):
    fit = cal._ModelFit(n=sum(n for _, _, n in bins))
    fit.bins = list(bins)
    fit.steps = [(x, v) for x, v, _ in bins]
    return fit


class TestRecordState:
    def test_noise_around_fifty_is_a_coin_flip(self, monkeypatch):
        monkeypatch.setattr(cal, "_fits_for", lambda _as_of: {
            "m": _fit_with([(0.5, 0.48, 133), (0.6, 0.55, 40)])})
        assert cal.record_state("m", 0.55) == {"state": "coin flip", "rate": 0.48, "n": 133}
        # 55% over 40 calls is under the sample floor: still a coin flip.
        assert cal.record_state("m", 0.62)["state"] == "coin flip"

    def test_two_standard_errors_over_fifty_calls_is_an_edge(self, monkeypatch):
        monkeypatch.setattr(cal, "_fits_for", lambda _as_of: {
            "m": _fit_with([(0.5, 0.61, 140)])})
        assert cal.record_state("m", 0.7) == {"state": "edge", "rate": 0.61, "n": 140}

    def test_the_mirror_is_fading(self, monkeypatch):
        monkeypatch.setattr(cal, "_fits_for", lambda _as_of: {
            "m": _fit_with([(0.5, 0.39, 90)])})
        assert cal.record_state("m", 0.5)["state"] == "fading"

    def test_no_fit_means_none(self, monkeypatch):
        monkeypatch.setattr(cal, "_fits_for", lambda _as_of: {})
        assert cal.record_state("m", 0.5) is None
        assert cal.record_state("m", None) is None

    def test_pav_bins_agree_with_pav(self):
        pairs = [(0.5, 1), (0.5, 0), (0.6, 1), (0.6, 1), (0.7, 0), (0.7, 1)]
        bins = cal._pav_bins(pairs)
        assert [(x, v) for x, v, _ in bins] == cal._pav(pairs)
        assert sum(n for _, _, n in bins) == len(pairs)


class TestLevels:
    def test_only_the_short_level_strings_travel(self):
        out = _levels_of({"key_level": "$159.70 resistance", "change_trigger": "x" * 500,
                          "triggers": {"move_to_sell": "close below $148.25",
                                       "reassess_to_buy": ""},
                          "raw_response": "huge", "feature_values": [1, 2, 3]})
        assert set(out) == {"key_level", "change_trigger", "move_to_sell"}
        assert len(out["change_trigger"]) == 160
        assert _levels_of(None) == {} and _levels_of("junk") == {}
