from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from studio_runner.settings import settings


engine = create_engine(settings.db_dsn, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, expire_on_commit=False)
Base = declarative_base()
