from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Model configuration
    SECUREMAIL_BUNDLE_PATH: str = "models/Model_XG_RF"
    MODEL_DIR: str = "models/Model_XG_RF"

    # Database configuration
    DATABASE_URL: str = "sqlite+aiosqlite:///./securemail.db"
    
    # API security & server
    API_PORT: int = 8000
    API_HOST: str = "0.0.0.0"
    ALLOWED_ORIGINS: str = "*"
    SECUREMAIL_API_KEY: Optional[str] = None

    # User authentication configuration
    AUTH_ALLOWED_EMAIL_DOMAINS: str = ""
    AUTH_JWT_SECRET: Optional[str] = None
    AUTH_ACCESS_TOKEN_MINUTES: int = 30

    # Optional read-only AI insights provider
    AGENT_PROVIDER: Optional[str] = None
    AGENT_MODEL: Optional[str] = None
    AGENT_API_KEY: Optional[str] = None
    AGENT_BASE_URL: Optional[str] = None
    AGENT_TIMEOUT_SECONDS: float = 20.0
    AGENT_MAX_RESPONSE_BYTES: int = 256 * 1024

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()

