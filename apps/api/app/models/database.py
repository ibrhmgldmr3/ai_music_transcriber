from collections.abc import Iterator

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import settings

_connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
engine = create_engine(settings.database_url, connect_args=_connect_args, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


def get_db() -> Iterator[Session]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Columns added after the first release: (table, column, DDL type with default).
# create_all() only creates missing tables, so existing databases get these via ALTER.
_ADDED_COLUMNS = [
    ("projects", "separate_guitar", "BOOLEAN NOT NULL DEFAULT FALSE"),
    ("projects", "beats_per_measure", "INTEGER NOT NULL DEFAULT 4"),
    ("projects", "key_name", "VARCHAR(16)"),
    ("projects", "downbeat", "FLOAT"),
    ("projects", "model_version", "VARCHAR(96)"),
    ("projects", "edited", "BOOLEAN NOT NULL DEFAULT FALSE"),
]


def init_db() -> None:
    from app.models import project  # noqa: F401  (registers the tables)

    Base.metadata.create_all(bind=engine)
    _add_missing_columns()


def _add_missing_columns() -> None:
    inspector = inspect(engine)
    with engine.begin() as connection:
        for table, column, ddl in _ADDED_COLUMNS:
            existing = {c["name"] for c in inspector.get_columns(table)}
            if column not in existing:
                connection.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}"))
