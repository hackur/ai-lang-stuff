"""
Monitoring and Logging System

Purpose:
    Comprehensive monitoring and logging example demonstrating LangSmith integration,
    structured logging, performance metrics collection, error tracking, and usage analytics.

Prerequisites:
    - Ollama running with qwen3:8b model
    - Python 3.10+
    - Optional: LangSmith API key for tracing (can run without)

Expected Output:
    Detailed logs, performance metrics, error tracking, and optional LangSmith traces
    showing complete observability of agent operations.

Usage:
    # Without LangSmith
    uv run python examples/06-production/monitoring_logging.py

    # With LangSmith (optional)
    export LANGCHAIN_TRACING_V2=true
    export LANGCHAIN_PROJECT="production-monitoring"
    export LANGCHAIN_API_KEY="your-key"  # Optional for local
    uv run python examples/06-production/monitoring_logging.py

Features:
    - Structured JSON logging
    - LangSmith tracing integration
    - Performance metrics (latency, throughput, token usage)
    - Error tracking with categorization
    - Usage analytics and trends
    - Real-time monitoring dashboard
"""

import asyncio
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage
from langchain_core.outputs import LLMResult
from langchain_ollama import ChatOllama


# ============================================================================
# Structured Logging
# ============================================================================


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields
        for key in ["request_id", "user_id", "duration_ms", "tokens", "model"]:
            if hasattr(record, key):
                log_data[key] = getattr(record, key)

        return json.dumps(log_data)


def setup_structured_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup structured logging with JSON output."""
    logger = logging.getLogger("monitoring")
    logger.setLevel(getattr(logging, log_level))

    # Console handler with JSON formatting
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(JSONFormatter())
    logger.addHandler(console_handler)

    # File handler for persistent logs
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    file_handler = logging.FileHandler(log_dir / "monitoring.jsonl")
    file_handler.setFormatter(JSONFormatter())
    logger.addHandler(file_handler)

    return logger


# ============================================================================
# Performance Metrics
# ============================================================================


class MetricType(str, Enum):
    """Types of metrics tracked."""

    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    TOKEN_USAGE = "token_usage"
    COST = "cost"


@dataclass
class PerformanceMetrics:
    """Track performance metrics over time."""

    # Latency metrics (milliseconds)
    total_latency_ms: float = 0.0
    min_latency_ms: float = float("inf")
    max_latency_ms: float = 0.0
    request_count: int = 0

    # Throughput metrics
    requests_per_second: float = 0.0
    start_time: float = field(default_factory=time.time)

    # Token usage
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    # Error tracking
    error_count: int = 0
    errors_by_type: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # Success tracking
    success_count: int = 0

    def record_request(
        self,
        latency_ms: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
        error: Optional[str] = None,
    ):
        """Record a request with its metrics."""
        self.request_count += 1
        self.total_latency_ms += latency_ms
        self.min_latency_ms = min(self.min_latency_ms, latency_ms)
        self.max_latency_ms = max(self.max_latency_ms, latency_ms)

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

        if error:
            self.error_count += 1
            self.errors_by_type[error] += 1
        else:
            self.success_count += 1

        # Update throughput
        elapsed_time = time.time() - self.start_time
        if elapsed_time > 0:
            self.requests_per_second = self.request_count / elapsed_time

    def get_average_latency(self) -> float:
        """Get average latency in milliseconds."""
        if self.request_count == 0:
            return 0.0
        return self.total_latency_ms / self.request_count

    def get_error_rate(self) -> float:
        """Get error rate as percentage."""
        if self.request_count == 0:
            return 0.0
        return (self.error_count / self.request_count) * 100

    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        return {
            "total_requests": self.request_count,
            "successful_requests": self.success_count,
            "failed_requests": self.error_count,
            "error_rate_percent": round(self.get_error_rate(), 2),
            "latency_ms": {
                "average": round(self.get_average_latency(), 2),
                "min": round(self.min_latency_ms, 2) if self.min_latency_ms != float("inf") else 0,
                "max": round(self.max_latency_ms, 2),
            },
            "throughput": {
                "requests_per_second": round(self.requests_per_second, 2),
            },
            "tokens": {
                "total_input": self.total_input_tokens,
                "total_output": self.total_output_tokens,
                "total": self.total_input_tokens + self.total_output_tokens,
            },
            "errors_by_type": dict(self.errors_by_type),
        }


# ============================================================================
# Custom Callback Handler
# ============================================================================


class MonitoringCallbackHandler(BaseCallbackHandler):
    """Custom callback handler for detailed monitoring."""

    def __init__(self, logger: logging.Logger, metrics: PerformanceMetrics):
        self.logger = logger
        self.metrics = metrics
        self.request_start_time: Optional[float] = None

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        """Called when LLM starts."""
        self.request_start_time = time.time()
        self.logger.info(
            "LLM request started",
            extra={
                "prompts_count": len(prompts),
                "model": kwargs.get("invocation_params", {}).get("model"),
            },
        )

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Called when LLM ends."""
        if self.request_start_time:
            latency_ms = (time.time() - self.request_start_time) * 1000

            # Extract token usage if available
            input_tokens = 0
            output_tokens = 0
            if response.llm_output:
                usage = response.llm_output.get("usage", {})
                input_tokens = usage.get("prompt_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0)

            self.metrics.record_request(
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

            self.logger.info(
                "LLM request completed",
                extra={
                    "duration_ms": round(latency_ms, 2),
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                },
            )

    def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        """Called when LLM errors."""
        if self.request_start_time:
            latency_ms = (time.time() - self.request_start_time) * 1000
            error_type = type(error).__name__

            self.metrics.record_request(latency_ms=latency_ms, error=error_type)

            self.logger.error(
                f"LLM request failed: {error}",
                extra={"duration_ms": round(latency_ms, 2), "error_type": error_type},
            )


# ============================================================================
# Usage Analytics
# ============================================================================


@dataclass
class UsageAnalytics:
    """Track usage patterns and analytics."""

    requests_by_hour: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    requests_by_model: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    average_message_length: float = 0.0
    total_messages: int = 0

    def record_request(self, message: str, model: str):
        """Record a request for analytics."""
        # Track by hour
        current_hour = datetime.now().hour
        self.requests_by_hour[current_hour] += 1

        # Track by model
        self.requests_by_model[model] += 1

        # Update average message length
        message_length = len(message)
        self.average_message_length = (
            (self.average_message_length * self.total_messages) + message_length
        ) / (self.total_messages + 1)
        self.total_messages += 1

    def get_peak_hour(self) -> Optional[int]:
        """Get the hour with most requests."""
        if not self.requests_by_hour:
            return None
        return max(self.requests_by_hour.items(), key=lambda x: x[1])[0]

    def get_summary(self) -> Dict[str, Any]:
        """Get analytics summary."""
        return {
            "total_messages": self.total_messages,
            "average_message_length": round(self.average_message_length, 2),
            "peak_hour": self.get_peak_hour(),
            "requests_by_hour": dict(self.requests_by_hour),
            "requests_by_model": dict(self.requests_by_model),
        }


# ============================================================================
# Monitored Agent
# ============================================================================


class MonitoredAgent:
    """Agent with comprehensive monitoring and logging."""

    def __init__(
        self,
        model_name: str = "qwen3:8b",
        enable_langsmith: bool = False,
        log_level: str = "INFO",
    ):
        self.model_name = model_name
        self.logger = setup_structured_logging(log_level)
        self.metrics = PerformanceMetrics()
        self.analytics = UsageAnalytics()

        # Setup callback handler
        self.callback_handler = MonitoringCallbackHandler(self.logger, self.metrics)

        # Initialize LLM with callback
        self.llm = ChatOllama(
            model=model_name,
            callbacks=[self.callback_handler],
        )

        # Log LangSmith configuration
        if enable_langsmith:
            self.logger.info("LangSmith tracing enabled", extra={"model": model_name})
        else:
            self.logger.info(
                "Running without LangSmith (set LANGCHAIN_TRACING_V2=true to enable)",
                extra={"model": model_name},
            )

    async def process(self, message: str, request_id: Optional[str] = None) -> str:
        """Process message with full monitoring."""
        request_id = request_id or f"req_{int(time.time() * 1000)}"
        start_time = time.time()

        self.logger.info(
            "Processing request",
            extra={
                "request_id": request_id,
                "message_length": len(message),
                "model": self.model_name,
            },
        )

        try:
            # Record analytics
            self.analytics.record_request(message, self.model_name)

            # Process request
            messages = [HumanMessage(content=message)]
            response = await asyncio.to_thread(self.llm.invoke, messages)

            duration_ms = (time.time() - start_time) * 1000

            self.logger.info(
                "Request completed successfully",
                extra={
                    "request_id": request_id,
                    "duration_ms": round(duration_ms, 2),
                    "response_length": len(response.content),
                },
            )

            return response.content

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000

            self.logger.error(
                f"Request failed: {e}",
                extra={
                    "request_id": request_id,
                    "duration_ms": round(duration_ms, 2),
                    "error_type": type(e).__name__,
                },
                exc_info=True,
            )
            raise

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        return {
            "performance": self.metrics.get_summary(),
            "analytics": self.analytics.get_summary(),
            "timestamp": datetime.utcnow().isoformat(),
        }

    def print_dashboard(self):
        """Print a real-time monitoring dashboard."""
        summary = self.get_metrics_summary()

        print("\n" + "=" * 80)
        print("MONITORING DASHBOARD".center(80))
        print("=" * 80)

        print(f"\nTimestamp: {summary['timestamp']}")

        print("\n--- PERFORMANCE METRICS ---")
        perf = summary["performance"]
        print(f"Total Requests:       {perf['total_requests']}")
        print(f"Successful:           {perf['successful_requests']}")
        print(f"Failed:               {perf['failed_requests']}")
        print(f"Error Rate:           {perf['error_rate_percent']}%")

        print("\n--- LATENCY (ms) ---")
        latency = perf["latency_ms"]
        print(f"Average:              {latency['average']}")
        print(f"Min:                  {latency['min']}")
        print(f"Max:                  {latency['max']}")

        print("\n--- THROUGHPUT ---")
        print(f"Requests/second:      {perf['throughput']['requests_per_second']}")

        print("\n--- TOKEN USAGE ---")
        tokens = perf["tokens"]
        print(f"Total Input:          {tokens['total_input']}")
        print(f"Total Output:         {tokens['total_output']}")
        print(f"Total:                {tokens['total']}")

        print("\n--- ANALYTICS ---")
        analytics = summary["analytics"]
        print(f"Total Messages:       {analytics['total_messages']}")
        print(f"Avg Message Length:   {analytics['average_message_length']}")
        print(f"Peak Hour:            {analytics['peak_hour'] or 'N/A'}")

        if perf["errors_by_type"]:
            print("\n--- ERRORS BY TYPE ---")
            for error_type, count in perf["errors_by_type"].items():
                print(f"{error_type:<20}: {count}")

        print("\n" + "=" * 80 + "\n")


# ============================================================================
# Demo Usage
# ============================================================================


async def main():
    """Demonstrate monitoring and logging system."""
    print("=== Monitoring & Logging System Demo ===\n")

    # Check for LangSmith configuration
    import os

    langsmith_enabled = os.getenv("LANGCHAIN_TRACING_V2") == "true"

    # Initialize monitored agent
    agent = MonitoredAgent(
        model_name="qwen3:8b", enable_langsmith=langsmith_enabled, log_level="INFO"
    )

    print("1. Processing sample requests with full monitoring...\n")

    # Process multiple requests
    test_messages = [
        "What is machine learning?",
        "Explain neural networks in simple terms.",
        "What are the benefits of local AI models?",
        "How does gradient descent work?",
        "What is transfer learning?",
    ]

    for i, message in enumerate(test_messages, 1):
        print(f"Request {i}/{len(test_messages)}: {message[:50]}...")
        try:
            response = await agent.process(message, request_id=f"demo_{i:03d}")
            print(f"Response length: {len(response)} chars\n")
        except Exception as e:
            print(f"Error: {e}\n")

        # Small delay between requests
        await asyncio.sleep(0.5)

    # Display dashboard
    print("\n2. Real-time monitoring dashboard:")
    agent.print_dashboard()

    # Export metrics
    print("3. Exporting metrics to file...")
    metrics_file = Path("logs/metrics_export.json")
    with open(metrics_file, "w") as f:
        json.dump(agent.get_metrics_summary(), f, indent=2)
    print(f"Metrics exported to: {metrics_file}")

    print("\n✓ Demo complete!")
    print("\nCheck logs/ directory for:")
    print("  - monitoring.jsonl (structured logs)")
    print("  - metrics_export.json (performance metrics)")

    if langsmith_enabled:
        print("\nLangSmith traces available at: https://smith.langchain.com")
    else:
        print("\nTo enable LangSmith tracing:")
        print("  export LANGCHAIN_TRACING_V2=true")
        print("  export LANGCHAIN_PROJECT=production-monitoring")


if __name__ == "__main__":
    asyncio.run(main())
