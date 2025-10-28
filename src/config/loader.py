"""
Configuration loader for local AI development.

Loads configuration from YAML files and environment variables with validation.
"""

import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv


class ModelConfig(BaseModel):
    """Model configuration."""

    model_name: str = Field(..., description="Name of the model")
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(4096, gt=0)
    timeout: int = Field(120, gt=0)


class OllamaConfig(BaseModel):
    """Ollama server configuration."""

    base_url: str = "http://localhost:11434"
    timeout: int = 120
    num_ctx: int = 8192
    num_gpu: int = 1
    num_thread: int = 8


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: Optional[str] = "logs/app.log"
    console: bool = True


class AppConfig(BaseModel):
    """Main application configuration."""

    default: ModelConfig
    ollama: OllamaConfig
    logging: LoggingConfig


# Global config instance
_config: Optional[AppConfig] = None


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """
    Load configuration from YAML file and environment variables.

    Args:
        config_path: Path to config YAML file. Defaults to config/models.yaml

    Returns:
        AppConfig: Validated configuration object

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValidationError: If config is invalid
    """
    global _config

    # Load environment variables
    load_dotenv()

    # Determine config path
    if config_path is None:
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "config" / "models.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load YAML
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    # Override with environment variables
    if "OLLAMA_BASE_URL" in os.environ:
        config_dict.setdefault("ollama", {})["base_url"] = os.environ["OLLAMA_BASE_URL"]

    if "DEFAULT_MODEL" in os.environ:
        config_dict.setdefault("default", {})["model_name"] = os.environ["DEFAULT_MODEL"]

    if "LOG_LEVEL" in os.environ:
        config_dict.setdefault("logging", {})["level"] = os.environ["LOG_LEVEL"]

    # Validate and create config
    try:
        _config = AppConfig(**config_dict)
        return _config
    except ValidationError as e:
        raise ValueError(f"Invalid configuration: {e}")


def get_config() -> AppConfig:
    """
    Get current configuration, loading it if not already loaded.

    Returns:
        AppConfig: Current configuration

    Raises:
        ValueError: If config hasn't been loaded and can't be loaded
    """
    global _config

    if _config is None:
        try:
            return load_config()
        except Exception as e:
            raise ValueError(f"Failed to load configuration: {e}")

    return _config


def reload_config(config_path: Optional[str] = None) -> AppConfig:
    """
    Reload configuration from file.

    Args:
        config_path: Path to config file

    Returns:
        AppConfig: Newly loaded configuration
    """
    global _config
    _config = None
    return load_config(config_path)


# Example usage
if __name__ == "__main__":
    # Load config
    config = load_config()

    print(f"Model: {config.default.model_name}")
    print(f"Temperature: {config.default.temperature}")
    print(f"Ollama URL: {config.ollama.base_url}")
    print(f"Log Level: {config.logging.level}")
