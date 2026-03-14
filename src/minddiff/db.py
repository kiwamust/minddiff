"""Database setup — SQLite + SQLAlchemy."""

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session

from minddiff.config import settings
from minddiff.models import Base


def _enable_wal(dbapi_conn, connection_record):
    """Enable WAL mode for SQLite."""
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def get_engine(url: str | None = None):
    url = url or settings.database_url
    engine = create_engine(url, echo=settings.debug)
    if url.startswith("sqlite"):
        event.listen(engine, "connect", _enable_wal)
    return engine


def create_tables(engine=None):
    engine = engine or get_engine()
    Base.metadata.create_all(engine)
    return engine


def get_session_factory(engine=None) -> sessionmaker[Session]:
    engine = engine or get_engine()
    return sessionmaker(bind=engine)
