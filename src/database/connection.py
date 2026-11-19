"""Explicit database configuration without embedded credentials or import I/O."""
from contextlib import contextmanager
from dataclasses import dataclass, field
import os
from typing import Generator, Optional

from omegaconf import DictConfig
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection, Engine, URL


class DatabaseConfigurationError(ValueError):
    """Sanitized application configuration failure, without credential values."""


class _RedactedURL(URL):
    """Keep SQLAlchemy 1.4 string formatting from revealing credentials."""
    def __str__(self) -> str:
        return self.render_as_string(hide_password=True)


@dataclass
class DatabaseConfig:
    host: str = "localhost"
    port: int = 5432
    database: str = "asset_price_forecasting"
    user: str = "postgres"
    password: Optional[str] = field(default=None, repr=False)
    allow_passwordless: bool = False

    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """Read APF_DB_* settings without connecting or logging their values."""
        return database_config_from_omegaconf({})

    def to_url(self) -> URL:
        """Build a correctly escaped URL with redacted str/repr diagnostics.

        The password remains available to SQLAlchemy for actual authentication.
        Explicit render_as_string(hide_password=False) discloses it and must
        never be used for logging, exceptions or tracking metadata.
        """
        if any(not isinstance(v, str) or not v for v in (self.host, self.database, self.user)):
            raise DatabaseConfigurationError("Database host, name and user are required.")
        if type(self.port) is not int or not 1 <= self.port <= 65535:
            raise DatabaseConfigurationError("Database port must be an integer in the valid range.")
        if type(self.allow_passwordless) is not bool:
            raise DatabaseConfigurationError("Passwordless authentication requires an explicit boolean.")
        if self.password is None:
            if not self.allow_passwordless:
                raise DatabaseConfigurationError("Database password is required; configure APF_DB_PASSWORD or explicitly enable passwordless authentication.")
        elif not isinstance(self.password, str) or not self.password:
            raise DatabaseConfigurationError("Database password must be a nonempty string.")
        return _RedactedURL.create("postgresql+psycopg2", username=self.user,
            password=self.password, host=self.host, port=self.port, database=self.database)


def database_config_from_omegaconf(cfg: DictConfig) -> DatabaseConfig:
    """Use explicit database fields, then APF_DB_* environment, then safe defaults.

    A null password is unset and can be supplied by APF_DB_PASSWORD. An empty
    password stays invalid. No implicit passwordless fallback is provided.
    """
    db_cfg = cfg.get("database", {})
    def setting(key, environment, default=None):
        value = db_cfg.get(key)
        return os.environ.get(environment, default) if value is None else value
    port = setting("port", "APF_DB_PORT", 5432)
    try:
        if isinstance(port, bool) or not isinstance(port, (int, str)):
            raise ValueError
        port = int(port)
    except (TypeError, ValueError, OverflowError):
        raise DatabaseConfigurationError("Database port must be an integer in the valid range.") from None
    passwordless = setting("allow_passwordless", "APF_DB_ALLOW_PASSWORDLESS", False)
    if isinstance(passwordless, str) and passwordless.lower() in {"true", "false"}:
        passwordless = passwordless.lower() == "true"
    if type(passwordless) is not bool:
        raise DatabaseConfigurationError("Passwordless authentication requires true or false.")
    return DatabaseConfig(host=setting("host", "APF_DB_HOST", "localhost"), port=port,
        database=setting("database", "APF_DB_NAME", "asset_price_forecasting"),
        user=setting("user", "APF_DB_USER", "postgres"),
        password=setting("password", "APF_DB_PASSWORD"), allow_passwordless=passwordless)


def create_database_engine(config: Optional[DatabaseConfig] = None) -> Engine:
    if config is None:
        config = DatabaseConfig.from_env()
    url = config.to_url()  # Validate before loading a driver or attempting I/O.
    try:
        return create_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=10, echo=False)
    except Exception:
        raise DatabaseConfigurationError("Database engine initialization failed; inspect configuration and driver availability securely.") from None


def create_engine_from_config(cfg: DictConfig) -> Engine:
    return create_database_engine(database_config_from_omegaconf(cfg))


@contextmanager
def get_connection(engine: Engine) -> Generator[Connection, None, None]:
    connection = engine.connect()
    try:
        yield connection
    finally:
        connection.close()


def test_connection(engine: Engine) -> bool:
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
