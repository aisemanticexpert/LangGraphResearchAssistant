"""
Configuration management for the Research Assistant.

Uses pydantic-settings to load configuration from environment variables
and .env file with validation and type coercion.
"""

from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    All settings can be overridden via environment variables or .env file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # API Keys
    anthropic_api_key: str = ""
    tavily_api_key: str = ""  # Tavily Search API key (preferred search tool)

    # Model Configuration
    default_model: str = "claude-sonnet-4-20250514"
    temperature: float = 0.0

    # Research Settings
    use_mock_data: bool = True  # Set to False to use Tavily Search API
    max_research_attempts: int = 3
    confidence_threshold: float = 6.0

    # Persistence Settings
    checkpoint_backend: str = "memory"  # memory, sqlite, postgres
    sqlite_path: str = "data/checkpoints.db"
    postgres_url: Optional[str] = None

    # Caching Settings
    enable_cache: bool = True
    cache_ttl_seconds: int = 3600  # 1 hour default
    cache_max_size: int = 100  # Max number of cached queries

    # Logging Settings
    log_level: str = "INFO"
    log_to_file: bool = True
    log_file_path: str = "logs/research_assistant.log"
    log_max_bytes: int = 10485760  # 10MB
    log_backup_count: int = 5

    # Streaming Settings
    enable_streaming: bool = True

    # Export Settings
    export_dir: str = "exports"

    # API Server Settings (for FastAPI)
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4

    def validate_api_key(self) -> bool:
        """Check if Anthropic API key is configured."""
        return bool(self.anthropic_api_key and self.anthropic_api_key != "your_anthropic_api_key_here")

    def validate_tavily_key(self) -> bool:
        """Check if Tavily API key is configured."""
        return bool(self.tavily_api_key and self.tavily_api_key != "your_tavily_api_key_here")


# Global settings instance
settings = Settings()
