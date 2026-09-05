"""Storage is Eastern and fixed; display follows the reader's session.

Two different questions that were previously answered by the same constant.
A stamp is STORED in the zone this platform cuts its day boundaries on, so a
timestamp can sit next to a trade date without a conversion. It is SHOWN in
the viewer's own zone, because "18:04 ET" makes a reader in London do
arithmetic to find out whether the run they are watching is running now.
"""

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from services import persistence_service as ps
from services import progress_service as prog

# 2026-09-04 18:04 ET == 22:04 UTC == 23:04 in London.
INSTANT = datetime(2026, 9, 4, 22, 4, tzinfo=timezone.utc)


class TestStorage:
    def test_stored_stamps_are_aware_and_eastern(self):
        stamp = ps._stamp()
        assert stamp.tzinfo is not None, (
            "a naive stamp is read by Postgres in the server's own zone, so "
            "the same call records a different instant depending where it ran")
        assert stamp.utcoffset() == datetime.now(ps.STORAGE_TZ).utcoffset()

    def test_storage_zone_observes_dst_rather_than_a_fixed_offset(self):
        """A hard EST offset is wrong from March to November."""
        summer = datetime(2026, 7, 1, 12, tzinfo=timezone.utc).astimezone(ps.STORAGE_TZ)
        winter = datetime(2026, 1, 1, 12, tzinfo=timezone.utc).astimezone(ps.STORAGE_TZ)
        assert summer.utcoffset() != winter.utcoffset()


class TestDisplay:
    def test_no_request_context_is_eastern(self):
        assert prog.display_tz() is prog.DISPLAY_TZ
        assert prog.format_stamp(INSTANT) == "2026-09-04 18:04 ET"

    def _req(self, cookie=None):
        from flask import Flask

        app = Flask(__name__)
        headers = {"Cookie": f"{prog.TZ_COOKIE}={cookie}"} if cookie else {}
        return app.test_request_context("/", headers=headers)

    def test_a_reader_sees_their_own_zone(self):
        with self._req("Europe/London"):
            assert prog.display_tz().key == "Europe/London"
            assert prog.format_stamp(INSTANT) == "2026-09-04 23:04 BST"

    def test_no_cookie_falls_back_to_eastern(self):
        with self._req():
            assert prog.display_tz() is prog.DISPLAY_TZ
            assert prog.format_stamp(INSTANT) == "2026-09-04 18:04 ET"

    def test_a_junk_zone_falls_back_rather_than_raising(self):
        """The cookie is browser-supplied text; a box without that tzdata
        entry must render, not 500."""
        with self._req("Mars/Olympus_Mons"):
            assert prog.display_tz() is prog.DISPLAY_TZ
            assert prog.format_stamp(INSTANT) == "2026-09-04 18:04 ET"

    def test_the_eastern_label_stays_ET_not_EDT(self):
        """ET is what every surface has always printed; only a non-default
        zone gets its own abbreviation."""
        assert prog.display_tz_label(prog.DISPLAY_TZ) == "ET"
        assert prog.display_tz_label(ZoneInfo("Asia/Tokyo")) == "JST"

    def test_the_clock_follows_the_reader_too(self):
        with self._req("Europe/London"):
            assert prog.format_clock(INSTANT) == "23:04:00"
        assert prog.format_clock(INSTANT) == "18:04:00"
