"""
Config - all the settings in one place.

Reads from environment variables or .env file.
Change stuff here or set env vars to override.
"""

from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Loads config from env vars / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # api keys
    anthropic_api_key: str = ""
    tavily_api_key: str = ""

    # model stuff
    default_model: str = "claude-3-haiku-20240307"
    temperature: float = 0.0

    # research behavior
    # Priority: Tavily API (if TAVILY_API_KEY is set) > Mock data (fallback)
    # use_mock_data only applies when Tavily API key is NOT configured
    use_mock_data: bool = False
    max_research_attempts: int = 3
    confidence_threshold: float = 6.0  # below this triggers validation

    # where to store checkpoints
    checkpoint_backend: str = "memory"  # or "sqlite"
    sqlite_path: str = "data/checkpoints.db"
    postgres_url: Optional[str] = None

    # cache settings
    enable_cache: bool = True
    cache_ttl_seconds: int = 3600
    cache_max_size: int = 100

    # logging
    log_level: str = "INFO"
    log_to_file: bool = True
    log_file_path: str = "logs/research_assistant.log"
    log_max_bytes: int = 10485760  # 10MB
    log_backup_count: int = 5

    # misc
    enable_streaming: bool = True
    export_dir: str = "exports"

    # api server
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4

    def validate_api_key(self) -> bool:
        """Is Anthropic key set?"""
        return bool(self.anthropic_api_key and self.anthropic_api_key != "your_anthropic_api_key_here")

    def validate_tavily_key(self) -> bool:
        """Is Tavily key set?"""
        return bool(self.tavily_api_key and self.tavily_api_key != "your_tavily_api_key_here")


settings = Settings()
