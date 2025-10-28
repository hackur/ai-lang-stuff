"""
Production-Ready Agent Example

Purpose:
    Demonstrates a complete production-ready agent with robust error handling,
    comprehensive logging, configuration management, retry logic with exponential
    backoff, health checks, and graceful shutdown.

Prerequisites:
    - Ollama running with qwen3:8b model
    - Python 3.10+
    - Dependencies: langchain, langchain-ollama, tenacity, pyyaml

Expected Output:
    Production-quality agent that handles errors gracefully, logs comprehensively,
    and provides monitoring capabilities for deployment scenarios.

Usage:
    uv run python examples/06-production/production_agent.py

Features:
    - Comprehensive error handling with custom exceptions
    - Structured logging with multiple levels
    - Retry logic with exponential backoff
    - Configuration management via YAML
    - Health check endpoints
    - Graceful shutdown handling
    - Performance metrics tracking
    - Request/response validation
"""

import asyncio
import logging
import signal
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

import yaml
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field, ValidationError, validator
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


# ============================================================================
# Configuration Management
# ============================================================================


class AgentConfig(BaseModel):
    """Production agent configuration with validation."""

    model_name: str = Field(default="qwen3:8b", description="Ollama model name")
    base_url: str = Field(default="http://localhost:11434", description="Ollama server URL")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Model temperature")
    timeout: int = Field(default=30, gt=0, description="Request timeout in seconds")
    max_retries: int = Field(default=3, ge=0, description="Maximum retry attempts")
    log_level: str = Field(default="INFO", description="Logging level")
    enable_metrics: bool = Field(default=True, description="Enable performance metrics")
    max_concurrent_requests: int = Field(
        default=5, ge=1, description="Max concurrent requests"
    )

    @validator("log_level")
    def validate_log_level(cls, v):
        """Validate log level is valid."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return v.upper()

    @classmethod
    def from_yaml(cls, config_path: Path) -> "AgentConfig":
        """Load configuration from YAML file."""
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)

        return cls(**config_data)


# ============================================================================
# Custom Exceptions
# ============================================================================


class AgentException(Exception):
    """Base exception for agent errors."""

    pass


class ModelConnectionError(AgentException):
    """Raised when connection to model fails."""

    pass


class ValidationException(AgentException):
    """Raised when input/output validation fails."""

    pass


class TimeoutException(AgentException):
    """Raised when operation times out."""

    pass


class ResourceExhaustedError(AgentException):
    """Raised when resources are exhausted."""

    pass


# ============================================================================
# Structured Logging
# ============================================================================


class LogFormatter(logging.Formatter):
    """Custom formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with structured fields."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add extra fields if present
        if hasattr(record, "agent_id"):
            log_data["agent_id"] = record.agent_id
        if hasattr(record, "request_id"):
            log_data["request_id"] = record.request_id
        if hasattr(record, "duration"):
            log_data["duration_ms"] = record.duration

        return yaml.dump(log_data, default_flow_style=False)


def setup_logging(config: AgentConfig) -> logging.Logger:
    """Setup structured logging with configuration."""
    logger = logging.getLogger("production_agent")
    logger.setLevel(getattr(logging, config.log_level))

    # Console handler with structured formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(LogFormatter())
    logger.addHandler(console_handler)

    # File handler for production logs
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "agent.log")
    file_handler.setFormatter(LogFormatter())
    logger.addHandler(file_handler)

    return logger


# ============================================================================
# Performance Metrics
# ============================================================================


class AgentMetrics:
    """Track agent performance metrics."""

    def __init__(self):
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_duration_ms = 0.0
        self.errors_by_type: Dict[str, int] = {}
        self.start_time = time.time()

    def record_success(self, duration_ms: float):
        """Record successful request."""
        self.total_requests += 1
        self.successful_requests += 1
        self.total_duration_ms += duration_ms

    def record_failure(self, error_type: str, duration_ms: float):
        """Record failed request."""
        self.total_requests += 1
        self.failed_requests += 1
        self.total_duration_ms += duration_ms
        self.errors_by_type[error_type] = self.errors_by_type.get(error_type, 0) + 1

    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        uptime = time.time() - self.start_time
        avg_duration = (
            self.total_duration_ms / self.total_requests if self.total_requests > 0 else 0
        )

        return {
            "uptime_seconds": round(uptime, 2),
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": (
                round(self.successful_requests / self.total_requests * 100, 2)
                if self.total_requests > 0
                else 0
            ),
            "average_duration_ms": round(avg_duration, 2),
            "errors_by_type": self.errors_by_type,
        }


# ============================================================================
# Request/Response Models
# ============================================================================


class AgentRequest(BaseModel):
    """Validated agent request."""

    message: str = Field(..., min_length=1, max_length=10000)
    context: Optional[Dict[str, Any]] = Field(default=None)
    max_tokens: Optional[int] = Field(default=None, ge=1, le=4096)

    @validator("message")
    def validate_message(cls, v):
        """Ensure message is not just whitespace."""
        if not v.strip():
            raise ValueError("Message cannot be empty or whitespace")
        return v.strip()


class AgentResponse(BaseModel):
    """Validated agent response."""

    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    duration_ms: float
    model_name: str


# ============================================================================
# Health Check
# ============================================================================


class HealthStatus(str, Enum):
    """Health check status enum."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class HealthCheck:
    """Agent health check system."""

    def __init__(self, agent: "ProductionAgent"):
        self.agent = agent

    async def check_ollama_connection(self) -> bool:
        """Check if Ollama server is reachable."""
        try:
            # Simple test invocation
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self.agent.llm.invoke, [HumanMessage(content="health check")]
                ),
                timeout=5.0,
            )
            return bool(response)
        except Exception as e:
            self.agent.logger.error(f"Ollama health check failed: {e}")
            return False

    async def get_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        ollama_healthy = await self.check_ollama_connection()
        metrics = self.agent.metrics.get_summary()

        # Determine overall status
        if not ollama_healthy:
            status = HealthStatus.UNHEALTHY
        elif metrics["success_rate"] < 80:
            status = HealthStatus.DEGRADED
        else:
            status = HealthStatus.HEALTHY

        return {
            "status": status.value,
            "timestamp": datetime.utcnow().isoformat(),
            "checks": {
                "ollama_connection": ollama_healthy,
            },
            "metrics": metrics,
        }


# ============================================================================
# Production Agent
# ============================================================================


class ProductionAgent:
    """Production-ready agent with comprehensive features."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.logger = setup_logging(config)
        self.metrics = AgentMetrics() if config.enable_metrics else None
        self.is_shutdown = False
        self._semaphore = asyncio.Semaphore(config.max_concurrent_requests)

        # Initialize LLM with retry configuration
        self.llm = ChatOllama(
            model=config.model_name,
            base_url=config.base_url,
            temperature=config.temperature,
            timeout=config.timeout,
        )

        # Setup health check
        self.health = HealthCheck(self)

        # Setup graceful shutdown
        self._setup_signal_handlers()

        self.logger.info(
            "Production agent initialized",
            extra={
                "agent_id": id(self),
                "model": config.model_name,
                "max_retries": config.max_retries,
            },
        )

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""

        def shutdown_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, initiating graceful shutdown")
            self.is_shutdown = True
            sys.exit(0)

        signal.signal(signal.SIGINT, shutdown_handler)
        signal.signal(signal.SIGTERM, shutdown_handler)

    @retry(
        retry=retry_if_exception_type(ModelConnectionError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _invoke_with_retry(self, messages: List[BaseMessage]) -> str:
        """Invoke model with retry logic and exponential backoff."""
        if self.is_shutdown:
            raise RuntimeError("Agent is shutting down")

        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(self.llm.invoke, messages),
                timeout=self.config.timeout,
            )
            return response.content
        except asyncio.TimeoutError:
            raise TimeoutException(f"Request timed out after {self.config.timeout}s")
        except Exception as e:
            self.logger.error(f"Model invocation failed: {e}")
            raise ModelConnectionError(f"Failed to connect to model: {e}")

    async def process_request(
        self, request: AgentRequest, request_id: Optional[str] = None
    ) -> AgentResponse:
        """Process agent request with full production features."""
        start_time = time.time()
        request_id = request_id or f"req_{int(time.time() * 1000)}"

        self.logger.info(
            f"Processing request",
            extra={"request_id": request_id, "message_length": len(request.message)},
        )

        try:
            # Validate not shutting down
            if self.is_shutdown:
                raise RuntimeError("Agent is shutting down, rejecting new requests")

            # Acquire semaphore for concurrent request limiting
            async with self._semaphore:
                # Build messages with context
                messages = [
                    SystemMessage(
                        content="You are a helpful AI assistant. Provide clear, accurate responses."
                    ),
                ]

                if request.context:
                    context_str = yaml.dump(request.context)
                    messages.append(
                        SystemMessage(content=f"Context information:\n{context_str}")
                    )

                messages.append(HumanMessage(content=request.message))

                # Invoke with retry logic
                response_content = await self._invoke_with_retry(messages)

                # Calculate duration
                duration_ms = (time.time() - start_time) * 1000

                # Record metrics
                if self.metrics:
                    self.metrics.record_success(duration_ms)

                # Build response
                response = AgentResponse(
                    content=response_content,
                    metadata={
                        "request_id": request_id,
                        "timestamp": datetime.utcnow().isoformat(),
                    },
                    duration_ms=duration_ms,
                    model_name=self.config.model_name,
                )

                self.logger.info(
                    "Request completed successfully",
                    extra={"request_id": request_id, "duration": duration_ms},
                )

                return response

        except ValidationException as e:
            duration_ms = (time.time() - start_time) * 1000
            if self.metrics:
                self.metrics.record_failure("validation_error", duration_ms)

            self.logger.error(
                f"Validation error: {e}", extra={"request_id": request_id, "duration": duration_ms}
            )
            raise

        except TimeoutException as e:
            duration_ms = (time.time() - start_time) * 1000
            if self.metrics:
                self.metrics.record_failure("timeout", duration_ms)

            self.logger.error(
                f"Request timeout: {e}", extra={"request_id": request_id, "duration": duration_ms}
            )
            raise

        except ModelConnectionError as e:
            duration_ms = (time.time() - start_time) * 1000
            if self.metrics:
                self.metrics.record_failure("connection_error", duration_ms)

            self.logger.error(
                f"Model connection error: {e}",
                extra={"request_id": request_id, "duration": duration_ms},
            )
            raise

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            if self.metrics:
                self.metrics.record_failure("unexpected_error", duration_ms)

            self.logger.critical(
                f"Unexpected error: {e}", extra={"request_id": request_id, "duration": duration_ms}
            )
            raise AgentException(f"Unexpected error processing request: {e}")

    async def get_health(self) -> Dict[str, Any]:
        """Get agent health status."""
        return await self.health.get_status()

    async def shutdown(self):
        """Graceful shutdown procedure."""
        self.logger.info("Starting graceful shutdown")
        self.is_shutdown = True

        # Wait for in-flight requests to complete (max 30s)
        shutdown_timeout = 30
        start = time.time()

        while self._semaphore._value < self.config.max_concurrent_requests:
            if time.time() - start > shutdown_timeout:
                self.logger.warning("Shutdown timeout reached, forcing shutdown")
                break
            await asyncio.sleep(0.1)

        if self.metrics:
            self.logger.info("Final metrics", extra={"metrics": self.metrics.get_summary()})

        self.logger.info("Shutdown complete")


# ============================================================================
# Demo Usage
# ============================================================================


async def main():
    """Demonstrate production agent usage."""
    print("=== Production-Ready Agent Demo ===\n")

    # Create default config
    config = AgentConfig(
        model_name="qwen3:8b",
        log_level="INFO",
        max_retries=3,
        enable_metrics=True,
        max_concurrent_requests=5,
    )

    # Initialize agent
    agent = ProductionAgent(config)

    try:
        # Health check
        print("1. Running health check...")
        health_status = await agent.get_health()
        print(f"Health Status: {health_status['status']}")
        print(f"Metrics: {health_status['metrics']}\n")

        # Process single request
        print("2. Processing single request...")
        request = AgentRequest(
            message="Explain what makes a production-ready system in 2 sentences.",
            context={"domain": "software engineering"},
        )
        response = await agent.process_request(request, request_id="demo_001")
        print(f"Response: {response.content}")
        print(f"Duration: {response.duration_ms:.2f}ms\n")

        # Process multiple concurrent requests
        print("3. Processing concurrent requests...")
        requests = [
            AgentRequest(message=f"What is the capital of country #{i}?")
            for i in range(1, 4)
        ]

        results = await asyncio.gather(
            *[agent.process_request(req, f"demo_00{i+1}") for i, req in enumerate(requests)],
            return_exceptions=True,
        )

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Request {i+1} failed: {result}")
            else:
                print(f"Request {i+1} completed in {result.duration_ms:.2f}ms")

        print()

        # Final metrics
        print("4. Final metrics summary:")
        metrics = agent.metrics.get_summary()
        for key, value in metrics.items():
            print(f"  {key}: {value}")

    except Exception as e:
        print(f"\nError: {e}")

    finally:
        # Graceful shutdown
        print("\n5. Shutting down gracefully...")
        await agent.shutdown()
        print("Demo complete!")


if __name__ == "__main__":
    asyncio.run(main())
