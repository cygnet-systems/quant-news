"""SQLAlchemy engine and session factory.

Usage:
    from db.session import get_session

    with get_session() as session:
        session.execute(...)
"""

import logging
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from config import DB
from db.models import Base

logger = logging.getLogger(__name__)

_engine = None
_SessionLocal = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(
            DB.URL,
            pool_size=DB.POOL_SIZE,
            max_overflow=DB.MAX_OVERFLOW,
            echo=DB.ECHO_SQL,
            pool_pre_ping=True,
        )
    return _engine


@contextmanager
def get_session():
    """Yield a transactional session that auto-commits on success, rolls back on error."""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), expire_on_commit=False)

    session: Session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db():
    """Create all tables (for testing / initial setup without Alembic)."""
    engine = get_engine()
    Base.metadata.create_all(engine)
    logger.info("Database tables created")
