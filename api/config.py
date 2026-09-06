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

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()

