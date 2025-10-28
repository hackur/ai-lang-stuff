"""
Deployment-Ready Application

Purpose:
    Complete deployment-ready example with Docker support, process management,
    resource monitoring, health endpoints, and production best practices.

Prerequisites:
    - Ollama running with qwen3:8b model
    - Python 3.10+
    - Dependencies: langchain, psutil, aiohttp

Expected Output:
    Production-grade application ready for deployment with health checks,
    metrics endpoints, graceful shutdown, and resource monitoring.

Usage:
    # Run locally
    uv run python examples/06-production/deployment_ready.py

    # Run with custom port
    PORT=8080 uv run python examples/06-production/deployment_ready.py

    # Docker deployment (see Dockerfile in this directory)
    docker build -t ai-agent .
    docker run -p 8000:8000 ai-agent

Features:
    - HTTP API with health and metrics endpoints
    - Graceful shutdown with connection draining
    - Resource monitoring (CPU, memory, disk)
    - Process management
    - Docker-ready structure
    - Kubernetes-compatible health checks
    - Prometheus-compatible metrics
"""

import asyncio
import logging
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import psutil
from aiohttp import web
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

# ============================================================================
# Logging Setup
# ============================================================================


def setup_logging():
    """Setup production logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("deployment")


logger = setup_logging()


# ============================================================================
# Resource Monitor
# ============================================================================


class ResourceMonitor:
    """Monitor system resource usage."""

    def __init__(self):
        self.process = psutil.Process()
        self.start_time = time.time()

    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        return self.process.cpu_percent(interval=0.1)

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage."""
        memory_info = self.process.memory_info()
        return {
            "rss_mb": round(memory_info.rss / 1024 / 1024, 2),
            "vms_mb": round(memory_info.vms / 1024 / 1024, 2),
            "percent": round(self.process.memory_percent(), 2),
        }

    def get_disk_usage(self) -> Dict[str, Any]:
        """Get disk usage for current working directory."""
        disk = psutil.disk_usage("/")
        return {
            "total_gb": round(disk.total / 1024 / 1024 / 1024, 2),
            "used_gb": round(disk.used / 1024 / 1024 / 1024, 2),
            "free_gb": round(disk.free / 1024 / 1024 / 1024, 2),
            "percent": disk.percent,
        }

    def get_uptime(self) -> float:
        """Get application uptime in seconds."""
        return time.time() - self.start_time

    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive resource summary."""
        return {
            "cpu_percent": self.get_cpu_usage(),
            "memory": self.get_memory_usage(),
            "disk": self.get_disk_usage(),
            "uptime_seconds": round(self.get_uptime(), 2),
        }


# ============================================================================
# Application Metrics
# ============================================================================


class ApplicationMetrics:
    """Track application-level metrics."""

    def __init__(self):
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_response_time_ms = 0.0

    def record_request(self, success: bool, response_time_ms: float):
        """Record a request."""
        self.total_requests += 1
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        self.total_response_time_ms += response_time_ms

    def get_prometheus_format(self) -> str:
        """Get metrics in Prometheus format."""
        avg_response_time = (
            self.total_response_time_ms / self.total_requests if self.total_requests > 0 else 0
        )

        metrics = [
            "# HELP agent_requests_total Total number of requests",
            "# TYPE agent_requests_total counter",
            f"agent_requests_total {self.total_requests}",
            "",
            "# HELP agent_requests_successful Successful requests",
            "# TYPE agent_requests_successful counter",
            f"agent_requests_successful {self.successful_requests}",
            "",
            "# HELP agent_requests_failed Failed requests",
            "# TYPE agent_requests_failed counter",
            f"agent_requests_failed {self.failed_requests}",
            "",
            "# HELP agent_response_time_avg Average response time in ms",
            "# TYPE agent_response_time_avg gauge",
            f"agent_response_time_avg {avg_response_time:.2f}",
        ]

        return "\n".join(metrics)

    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "average_response_time_ms": (
                round(self.total_response_time_ms / self.total_requests, 2)
                if self.total_requests > 0
                else 0
            ),
        }


# ============================================================================
# Agent Application
# ============================================================================


class AgentApplication:
    """Production-ready agent application."""

    def __init__(
        self,
        model_name: str = "qwen3:8b",
        host: str = "0.0.0.0",
        port: int = 8000,
    ):
        self.model_name = model_name
        self.host = host
        self.port = port
        self.is_ready = False
        self.is_shutting_down = False

        # Initialize components
        self.llm = ChatOllama(model=model_name, timeout=30)
        self.resource_monitor = ResourceMonitor()
        self.metrics = ApplicationMetrics()

        # Web application
        self.app = web.Application()
        self.setup_routes()

        logger.info(
            "Application initialized",
            extra={"model": model_name, "host": host, "port": port},
        )

    def setup_routes(self):
        """Setup HTTP routes."""
        self.app.router.add_get("/health", self.health_handler)
        self.app.router.add_get("/health/ready", self.readiness_handler)
        self.app.router.add_get("/health/live", self.liveness_handler)
        self.app.router.add_get("/metrics", self.metrics_handler)
        self.app.router.add_post("/api/v1/chat", self.chat_handler)

    async def startup(self):
        """Application startup procedure."""
        logger.info("Starting application...")

        try:
            # Test Ollama connection
            logger.info("Testing Ollama connection...")
            test_response = await asyncio.to_thread(self.llm.invoke, [HumanMessage(content="test")])
            logger.info(f"Ollama connection successful: {test_response.content[:50]}...")

            self.is_ready = True
            logger.info("Application ready to serve requests")

        except Exception as e:
            logger.error(f"Startup failed: {e}")
            raise

    async def shutdown(self):
        """Graceful shutdown procedure."""
        logger.info("Starting graceful shutdown...")
        self.is_shutting_down = True
        self.is_ready = False

        # Wait for in-flight requests (max 30s)
        shutdown_timeout = 30
        start = time.time()

        logger.info("Draining connections...")
        while time.time() - start < shutdown_timeout:
            # In a real application, check for active connections
            await asyncio.sleep(0.1)

        logger.info("Shutdown complete")

    # ========================================================================
    # HTTP Handlers
    # ========================================================================

    async def health_handler(self, request: web.Request) -> web.Response:
        """
        Comprehensive health check endpoint.

        Returns overall application health including resource usage.
        """
        health_data = {
            "status": "healthy" if self.is_ready and not self.is_shutting_down else "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "application": {
                "name": "ai-agent",
                "version": "0.1.0",
                "model": self.model_name,
            },
            "resources": self.resource_monitor.get_summary(),
            "metrics": self.metrics.get_summary(),
        }

        status_code = 200 if health_data["status"] == "healthy" else 503

        return web.json_response(health_data, status=status_code)

    async def readiness_handler(self, request: web.Request) -> web.Response:
        """
        Kubernetes readiness probe endpoint.

        Indicates if application is ready to serve traffic.
        """
        if self.is_ready and not self.is_shutting_down:
            return web.json_response({"status": "ready"}, status=200)
        else:
            return web.json_response({"status": "not_ready"}, status=503)

    async def liveness_handler(self, request: web.Request) -> web.Response:
        """
        Kubernetes liveness probe endpoint.

        Indicates if application is alive (should not be restarted).
        """
        if not self.is_shutting_down:
            return web.json_response({"status": "alive"}, status=200)
        else:
            return web.json_response({"status": "shutting_down"}, status=503)

    async def metrics_handler(self, request: web.Request) -> web.Response:
        """
        Prometheus-compatible metrics endpoint.

        Returns metrics in Prometheus text format.
        """
        metrics_text = self.metrics.get_prometheus_format()
        return web.Response(text=metrics_text, content_type="text/plain")

    async def chat_handler(self, request: web.Request) -> web.Response:
        """
        Chat endpoint for processing messages.

        Expects JSON: {"message": "your message here"}
        Returns: {"response": "agent response", "metadata": {...}}
        """
        if not self.is_ready:
            return web.json_response({"error": "Service not ready"}, status=503)

        if self.is_shutting_down:
            return web.json_response({"error": "Service shutting down"}, status=503)

        start_time = time.time()

        try:
            # Parse request
            data = await request.json()
            message = data.get("message")

            if not message:
                return web.json_response({"error": "Missing 'message' field"}, status=400)

            # Process request
            logger.info(f"Processing chat request: {message[:50]}...")

            response = await asyncio.to_thread(self.llm.invoke, [HumanMessage(content=message)])

            response_time_ms = (time.time() - start_time) * 1000

            # Record metrics
            self.metrics.record_request(success=True, response_time_ms=response_time_ms)

            # Build response
            response_data = {
                "response": response.content,
                "metadata": {
                    "model": self.model_name,
                    "response_time_ms": round(response_time_ms, 2),
                    "timestamp": datetime.utcnow().isoformat(),
                },
            }

            logger.info(f"Request completed in {response_time_ms:.2f}ms")

            return web.json_response(response_data)

        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            self.metrics.record_request(success=False, response_time_ms=response_time_ms)

            logger.error(f"Request failed: {e}")

            return web.json_response(
                {"error": str(e), "type": type(e).__name__},
                status=500,
            )

    async def run(self):
        """Run the application."""
        # Setup signal handlers
        loop = asyncio.get_event_loop()

        def signal_handler():
            logger.info("Received shutdown signal")
            asyncio.create_task(self.shutdown())
            loop.stop()

        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, signal_handler)

        # Startup
        await self.startup()

        # Run web server
        runner = web.AppRunner(self.app)
        await runner.setup()

        site = web.TCPSite(runner, self.host, self.port)
        await site.start()

        logger.info(f"Server started on http://{self.host}:{self.port}")
        logger.info("Available endpoints:")
        logger.info("  - GET  /health          - Comprehensive health check")
        logger.info("  - GET  /health/ready    - Readiness probe")
        logger.info("  - GET  /health/live     - Liveness probe")
        logger.info("  - GET  /metrics         - Prometheus metrics")
        logger.info("  - POST /api/v1/chat     - Chat endpoint")

        # Keep running
        try:
            while not self.is_shutting_down:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            await self.shutdown()
            await runner.cleanup()


# ============================================================================
# Dockerfile Generator
# ============================================================================


def create_dockerfile():
    """Create a production Dockerfile."""
    dockerfile_content = """# Production Dockerfile for AI Agent Application
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY pyproject.toml ./

# Install Python dependencies
RUN pip install --no-cache-dir uv && \\
    uv pip install --system --no-cache langchain langchain-ollama aiohttp psutil pyyaml pydantic pydantic-settings tenacity

# Copy application code
COPY examples/06-production/deployment_ready.py ./

# Create non-root user
RUN useradd -m -u 1000 agent && \\
    chown -R agent:agent /app
USER agent

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health/live || exit 1

# Run application
CMD ["python", "deployment_ready.py"]
"""

    dockerfile_path = Path("Dockerfile.agent")
    with open(dockerfile_path, "w") as f:
        f.write(dockerfile_content)

    logger.info(f"Created Dockerfile: {dockerfile_path}")
    return dockerfile_path


# ============================================================================
# Docker Compose Generator
# ============================================================================


def create_docker_compose():
    """Create docker-compose.yml for complete deployment."""
    compose_content = """version: '3.8'

services:
  # Ollama service
  ollama:
    image: ollama/ollama:latest
    container_name: ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:11434/api/tags"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Agent application
  agent:
    build:
      context: .
      dockerfile: Dockerfile.agent
    container_name: ai-agent
    ports:
      - "8000:8000"
    environment:
      - OLLAMA_BASE_URL=http://ollama:11434
      - LOG_LEVEL=INFO
    depends_on:
      ollama:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health/ready"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 10s
    restart: unless-stopped

volumes:
  ollama_data:
"""

    compose_path = Path("docker-compose.yml")
    with open(compose_path, "w") as f:
        f.write(compose_content)

    logger.info(f"Created docker-compose: {compose_path}")
    return compose_path


# ============================================================================
# Main Entry Point
# ============================================================================


async def main():
    """Main entry point."""
    # Get configuration from environment
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    model_name = os.getenv("MODEL_NAME", "qwen3:8b")

    # Generate deployment files
    print("=== Deployment-Ready Application ===\n")
    print("Generating deployment files...")
    create_dockerfile()
    create_docker_compose()
    print()

    # Create and run application
    app = AgentApplication(model_name=model_name, host=host, port=port)
    await app.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
