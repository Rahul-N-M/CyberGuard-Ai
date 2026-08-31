"""
src/database/connection.py
Database engine factory and session management.
Provides init_db(), get_db(), and drop_db() utilities.
"""

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, scoped_session
from src.database.config import get_database_url
from src.database.models import Base

# ── Engine (created once, reused across the app) ──────────────
def _build_engine(echo: bool = False):
    url = get_database_url()
    connect_args = {}
    # SQLite needs check_same_thread=False when used across threads
    if url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
    return create_engine(url, echo=echo, connect_args=connect_args)


_engine = None
_Session = None


def get_engine(echo: bool = False):
    global _engine
    if _engine is None:
        _engine = _build_engine(echo=echo)
    return _engine


def get_session_factory():
    global _Session
    if _Session is None:
        _Session = scoped_session(sessionmaker(bind=get_engine()))
    return _Session


# ── Public helpers ─────────────────────────────────────────────
def init_db(echo: bool = False):
    """Create all tables defined in models.py. Safe to call multiple times."""
    engine = get_engine(echo=echo)
    Base.metadata.create_all(bind=engine)
    print(f"[CyberGuard] Database initialised -> {get_database_url()}")


def drop_db(echo: bool = False):
    """Drop all tables. USE WITH CAUTION — destroys all data."""
    engine = get_engine(echo=echo)
    Base.metadata.drop_all(bind=engine)
    print("[CyberGuard] All tables dropped.")


def get_db():
    """
    Yield a database session for use in a with-block or dependency injection.

    Usage:
        Session = get_session_factory()
        session = Session()
        try:
            # ... queries ...
            session.commit()
        finally:
            session.close()
    """
    Session = get_session_factory()
    session = Session()
    try:
        yield session
    finally:
        session.close()


def test_connection() -> bool:
    """Returns True if the database connection is healthy."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        print(f"[CyberGuard] Connection test failed: {e}")
        return False


if __name__ == "__main__":
    if test_connection():
        print("[CyberGuard] Connection OK")
        init_db(echo=True)
    else:
        print("[CyberGuard] Could not connect to database. Check .env settings.")
