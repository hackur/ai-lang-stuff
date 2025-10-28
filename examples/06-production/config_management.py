"""
Configuration Management System

Purpose:
    Comprehensive configuration management demonstrating YAML configuration,
    environment variables, secret management, multi-environment support,
    and configuration validation.

Prerequisites:
    - Python 3.10+
    - pyyaml, pydantic, python-dotenv

Expected Output:
    Demonstrates loading configuration from multiple sources with validation,
    environment-specific settings, and secure secret handling.

Usage:
    # Default (development) environment
    uv run python examples/06-production/config_management.py

    # Production environment
    ENV=production uv run python examples/06-production/config_management.py

    # With custom config file
    CONFIG_FILE=/path/to/config.yaml uv run python examples/06-production/config_management.py

Features:
    - YAML configuration files
    - Environment variable overrides
    - Secure secret management
    - Multi-environment support (dev, staging, prod)
    - Configuration validation with Pydantic
    - Dynamic configuration reloading
    - Configuration merging and inheritance
"""

import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, SecretStr, validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ============================================================================
# Environment Enum
# ============================================================================


class Environment(str, Enum):
    """Supported deployment environments."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


# ============================================================================
# Configuration Models
# ============================================================================


class OllamaConfig(BaseModel):
    """Ollama server configuration."""

    base_url: str = Field(default="http://localhost:11434", description="Ollama server URL")
    timeout: int = Field(default=30, ge=1, le=300, description="Request timeout in seconds")
    max_retries: int = Field(default=3, ge=0, le=10, description="Maximum retry attempts")

    @validator("base_url")
    def validate_url(cls, v):
        """Ensure URL is properly formatted."""
        if not v.startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")
        return v.rstrip("/")


class ModelConfig(BaseModel):
    """Model-specific configuration."""

    name: str = Field(..., description="Model name")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Model temperature")
    max_tokens: Optional[int] = Field(
        default=None, ge=1, le=32768, description="Maximum tokens in response"
    )
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Top-p sampling")
    top_k: int = Field(default=40, ge=0, le=100, description="Top-k sampling")


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = Field(default="INFO", description="Log level")
    format: str = Field(default="json", description="Log format (json or text)")
    output_file: Optional[Path] = Field(default=None, description="Log output file")
    max_file_size_mb: int = Field(default=100, ge=1, le=1000, description="Max log file size")
    backup_count: int = Field(default=5, ge=1, le=100, description="Number of backup files")

    @validator("level")
    def validate_level(cls, v):
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"level must be one of {valid_levels}")
        return v.upper()

    @validator("format")
    def validate_format(cls, v):
        """Validate log format."""
        valid_formats = ["json", "text"]
        if v.lower() not in valid_formats:
            raise ValueError(f"format must be one of {valid_formats}")
        return v.lower()


class MonitoringConfig(BaseModel):
    """Monitoring and observability configuration."""

    enable_metrics: bool = Field(default=True, description="Enable metrics collection")
    enable_tracing: bool = Field(default=False, description="Enable distributed tracing")
    metrics_port: int = Field(default=9090, ge=1024, le=65535, description="Metrics server port")
    langsmith_project: Optional[str] = Field(default=None, description="LangSmith project name")


class SecurityConfig(BaseModel):
    """Security configuration."""

    api_key: Optional[SecretStr] = Field(default=None, description="API key for authentication")
    allowed_origins: List[str] = Field(
        default_factory=lambda: ["*"], description="CORS allowed origins"
    )
    rate_limit_per_minute: int = Field(
        default=60, ge=1, le=10000, description="Rate limit per minute"
    )
    enable_ssl: bool = Field(default=False, description="Enable SSL/TLS")


class DatabaseConfig(BaseModel):
    """Database configuration."""

    url: Optional[SecretStr] = Field(default=None, description="Database connection URL")
    pool_size: int = Field(default=5, ge=1, le=100, description="Connection pool size")
    max_overflow: int = Field(default=10, ge=0, le=100, description="Max pool overflow")
    echo: bool = Field(default=False, description="Echo SQL queries")


class PerformanceConfig(BaseModel):
    """Performance tuning configuration."""

    max_concurrent_requests: int = Field(
        default=10, ge=1, le=1000, description="Max concurrent requests"
    )
    request_timeout: int = Field(default=30, ge=1, le=300, description="Request timeout seconds")
    cache_enabled: bool = Field(default=True, description="Enable caching")
    cache_ttl_seconds: int = Field(default=3600, ge=0, le=86400, description="Cache TTL")


# ============================================================================
# Main Application Configuration
# ============================================================================


class AppConfig(BaseSettings):
    """Main application configuration with multi-source loading."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
    )

    # Environment
    environment: Environment = Field(
        default=Environment.DEVELOPMENT, description="Deployment environment"
    )

    # Application metadata
    app_name: str = Field(default="ai-lang-stuff", description="Application name")
    app_version: str = Field(default="0.1.0", description="Application version")
    debug: bool = Field(default=False, description="Debug mode")

    # Component configurations
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    model: ModelConfig = Field(
        default_factory=lambda: ModelConfig(name="qwen3:8b", temperature=0.7)
    )
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    database: Optional[DatabaseConfig] = Field(default=None)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)

    @validator("environment", pre=True)
    def parse_environment(cls, v):
        """Parse environment from string."""
        if isinstance(v, str):
            return Environment(v.lower())
        return v

    @classmethod
    def from_yaml(cls, config_file: Path, environment: Optional[Environment] = None) -> "AppConfig":
        """
        Load configuration from YAML file with environment-specific overrides.

        Args:
            config_file: Path to YAML configuration file
            environment: Target environment (overrides file setting)

        Returns:
            AppConfig instance
        """
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")

        # Load YAML
        with open(config_file, "r") as f:
            config_data = yaml.safe_load(f) or {}

        # Handle environment-specific configurations
        env = environment or Environment(
            config_data.get("environment", Environment.DEVELOPMENT.value)
        )

        # Merge environment-specific overrides
        if "environments" in config_data:
            env_overrides = config_data["environments"].get(env.value, {})
            # Deep merge env overrides into config_data
            cls._deep_merge(config_data, env_overrides)

        # Remove environments section before validation
        config_data.pop("environments", None)

        # Override environment if specified
        if environment:
            config_data["environment"] = environment.value

        return cls(**config_data)

    @staticmethod
    def _deep_merge(base: Dict, override: Dict) -> Dict:
        """Deep merge override dict into base dict."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                AppConfig._deep_merge(base[key], value)
            else:
                base[key] = value
        return base

    def save_yaml(self, output_file: Path):
        """Save configuration to YAML file."""
        config_dict = self.model_dump(mode="json", exclude_none=True)

        # Convert SecretStr to string for YAML export (with warning)
        self._expose_secrets(config_dict)

        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)

    def _expose_secrets(self, config_dict: Dict):
        """Convert SecretStr to strings (use carefully)."""
        for key, value in config_dict.items():
            if isinstance(value, dict):
                self._expose_secrets(value)
            elif hasattr(value, "get_secret_value"):
                config_dict[key] = "***REDACTED***"

    def get_summary(self) -> Dict[str, Any]:
        """Get configuration summary (safe for logging)."""
        return {
            "environment": self.environment.value,
            "app_name": self.app_name,
            "app_version": self.app_version,
            "debug": self.debug,
            "model": self.model.name,
            "logging_level": self.logging.level,
            "monitoring_enabled": self.monitoring.enable_metrics,
        }


# ============================================================================
# Configuration Factory
# ============================================================================


class ConfigFactory:
    """Factory for creating environment-specific configurations."""

    @staticmethod
    def create_development_config() -> AppConfig:
        """Create development configuration."""
        return AppConfig(
            environment=Environment.DEVELOPMENT,
            debug=True,
            logging=LoggingConfig(level="DEBUG", format="text"),
            model=ModelConfig(name="qwen3:8b", temperature=0.8),
            monitoring=MonitoringConfig(enable_metrics=True, enable_tracing=False),
            performance=PerformanceConfig(
                max_concurrent_requests=5, cache_enabled=True, cache_ttl_seconds=300
            ),
        )

    @staticmethod
    def create_staging_config() -> AppConfig:
        """Create staging configuration."""
        return AppConfig(
            environment=Environment.STAGING,
            debug=False,
            logging=LoggingConfig(level="INFO", format="json"),
            model=ModelConfig(name="qwen3:8b", temperature=0.7),
            monitoring=MonitoringConfig(enable_metrics=True, enable_tracing=True),
            performance=PerformanceConfig(
                max_concurrent_requests=20, cache_enabled=True, cache_ttl_seconds=3600
            ),
        )

    @staticmethod
    def create_production_config() -> AppConfig:
        """Create production configuration."""
        return AppConfig(
            environment=Environment.PRODUCTION,
            debug=False,
            logging=LoggingConfig(
                level="WARNING", format="json", output_file=Path("/var/log/agent/app.log")
            ),
            model=ModelConfig(name="qwen3:8b", temperature=0.5),
            monitoring=MonitoringConfig(
                enable_metrics=True, enable_tracing=True, langsmith_project="production"
            ),
            security=SecurityConfig(
                allowed_origins=["https://example.com"],
                rate_limit_per_minute=1000,
                enable_ssl=True,
            ),
            performance=PerformanceConfig(
                max_concurrent_requests=100, cache_enabled=True, cache_ttl_seconds=7200
            ),
        )


# ============================================================================
# Configuration Manager
# ============================================================================


class ConfigManager:
    """Manage application configuration lifecycle."""

    def __init__(self, config_file: Optional[Path] = None):
        self.config_file = config_file or self._find_config_file()
        self.config: Optional[AppConfig] = None

    @staticmethod
    def _find_config_file() -> Optional[Path]:
        """Auto-discover configuration file."""
        # Check environment variable
        if "CONFIG_FILE" in os.environ:
            return Path(os.environ["CONFIG_FILE"])

        # Check common locations
        search_paths = [
            Path("config.yaml"),
            Path("config/config.yaml"),
            Path(".config/config.yaml"),
            Path("/etc/ai-lang-stuff/config.yaml"),
        ]

        for path in search_paths:
            if path.exists():
                return path

        return None

    def load(self, environment: Optional[Environment] = None) -> AppConfig:
        """Load configuration with environment detection."""
        # Get environment from ENV variable if not specified
        if not environment:
            env_str = os.getenv("ENV", os.getenv("ENVIRONMENT", "development"))
            environment = Environment(env_str.lower())

        # Load from YAML if available, otherwise use factory
        if self.config_file and self.config_file.exists():
            print(f"Loading configuration from: {self.config_file}")
            self.config = AppConfig.from_yaml(self.config_file, environment)
        else:
            print(f"No config file found, using {environment.value} defaults")
            if environment == Environment.PRODUCTION:
                self.config = ConfigFactory.create_production_config()
            elif environment == Environment.STAGING:
                self.config = ConfigFactory.create_staging_config()
            else:
                self.config = ConfigFactory.create_development_config()

        return self.config

    def reload(self) -> AppConfig:
        """Reload configuration from file."""
        if not self.config:
            raise RuntimeError("Configuration not loaded yet")

        return self.load(self.config.environment)

    def validate(self) -> List[str]:
        """Validate current configuration."""
        if not self.config:
            return ["Configuration not loaded"]

        issues = []

        # Production-specific validations
        if self.config.environment == Environment.PRODUCTION:
            if self.config.debug:
                issues.append("Debug mode should be disabled in production")
            if self.config.logging.level == "DEBUG":
                issues.append("Debug logging should not be used in production")
            if not self.config.monitoring.enable_metrics:
                issues.append("Metrics should be enabled in production")
            if "*" in self.config.security.allowed_origins:
                issues.append("CORS should not allow all origins in production")

        return issues


# ============================================================================
# Demo Usage
# ============================================================================


def create_example_config_file():
    """Create an example configuration file."""
    example_config = {
        "environment": "development",
        "app_name": "ai-lang-stuff",
        "app_version": "0.1.0",
        "debug": True,
        "ollama": {"base_url": "http://localhost:11434", "timeout": 30, "max_retries": 3},
        "model": {"name": "qwen3:8b", "temperature": 0.7, "max_tokens": 2048},
        "logging": {"level": "INFO", "format": "json"},
        "monitoring": {
            "enable_metrics": True,
            "enable_tracing": False,
            "metrics_port": 9090,
        },
        "performance": {
            "max_concurrent_requests": 10,
            "request_timeout": 30,
            "cache_enabled": True,
        },
        # Environment-specific overrides
        "environments": {
            "staging": {
                "debug": False,
                "logging": {"level": "INFO"},
                "monitoring": {"enable_tracing": True},
            },
            "production": {
                "debug": False,
                "logging": {"level": "WARNING", "output_file": "/var/log/agent/app.log"},
                "monitoring": {"enable_tracing": True, "langsmith_project": "production"},
                "security": {
                    "allowed_origins": ["https://example.com"],
                    "rate_limit_per_minute": 1000,
                },
            },
        },
    }

    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    config_file = config_dir / "example_config.yaml"

    with open(config_file, "w") as f:
        yaml.dump(example_config, f, default_flow_style=False, indent=2)

    return config_file


def main():
    """Demonstrate configuration management."""
    print("=== Configuration Management Demo ===\n")

    # 1. Create example config file
    print("1. Creating example configuration file...")
    example_file = create_example_config_file()
    print(f"   Created: {example_file}\n")

    # 2. Load configuration for different environments
    print("2. Loading configurations for different environments:\n")

    environments = [Environment.DEVELOPMENT, Environment.STAGING, Environment.PRODUCTION]

    for env in environments:
        print(f"--- {env.value.upper()} Environment ---")
        manager = ConfigManager(config_file=example_file)
        config = manager.load(environment=env)

        print(f"Environment:     {config.environment.value}")
        print(f"Debug Mode:      {config.debug}")
        print(f"Model:           {config.model.name}")
        print(f"Temperature:     {config.model.temperature}")
        print(f"Log Level:       {config.logging.level}")
        print(f"Metrics:         {config.monitoring.enable_metrics}")
        print(f"Tracing:         {config.monitoring.enable_tracing}")
        print(f"Max Requests:    {config.performance.max_concurrent_requests}")

        # Validate configuration
        issues = manager.validate()
        if issues:
            print("\n⚠️  Validation Issues:")
            for issue in issues:
                print(f"   - {issue}")

        print()

    # 3. Demonstrate factory patterns
    print("3. Using configuration factory:\n")

    print("Development config:")
    dev_config = ConfigFactory.create_development_config()
    print(f"   Debug: {dev_config.debug}, Log Level: {dev_config.logging.level}")

    print("\nProduction config:")
    prod_config = ConfigFactory.create_production_config()
    print(f"   Debug: {prod_config.debug}, Log Level: {prod_config.logging.level}")
    print(f"   SSL: {prod_config.security.enable_ssl}")

    # 4. Export configuration
    print("\n4. Exporting configuration to file...")
    output_file = Path("config/exported_config.yaml")
    dev_config.save_yaml(output_file)
    print(f"   Exported to: {output_file}")

    # 5. Configuration summary
    print("\n5. Configuration Summary:")
    print(yaml.dump(dev_config.get_summary(), default_flow_style=False, indent=2))

    print("✓ Demo complete!")
    print("\nGenerated files:")
    print("  - config/example_config.yaml (multi-environment config)")
    print("  - config/exported_config.yaml (exported config)")


if __name__ == "__main__":
    main()
