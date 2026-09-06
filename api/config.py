from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./securemail.db"
    
    # API security & server
    API_PORT: int = 8000
    API_HOST: str = "0.0.0.0"
    ALLOWED_ORIGINS: str = "*"
    API_TOKEN: Optional[str] = None

    # Model
    MODEL_DIR: str = "models/Model_XG_RF"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
