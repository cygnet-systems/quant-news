"""Shared test isolation.

The research-answer reuse store (services.investigation_service) is a disk
cache keyed without the as-of, so an answer one test wrote would be read
back by the next as a free hit and silently change its budget arithmetic.
Every test gets its own empty store.
"""
import pytest


@pytest.fixture(autouse=True)
def _isolated_research_answer_store(tmp_path, monkeypatch):
    from services import investigation_service as inv
    monkeypatch.setattr(inv, "_REUSE_DIR", str(tmp_path / "research_answers"))
    monkeypatch.setattr(inv, "_reuse_disk", None)
    yield
