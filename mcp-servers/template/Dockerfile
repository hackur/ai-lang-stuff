# MCP Server Docker Template
#
# This Dockerfile creates a containerized MCP server.
# Customize for your specific server requirements.
#
# Build: docker build -t my-mcp-server .
# Run:   docker run -p 8000:8000 my-mcp-server

# ============================================================================
# Base Image
# ============================================================================

FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create app user (security best practice)
RUN useradd -m -u 1000 appuser

# Set working directory
WORKDIR /app

# ============================================================================
# Dependencies Stage
# ============================================================================

FROM base as dependencies

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# ============================================================================
# Production Stage
# ============================================================================

FROM base as production

# Copy Python dependencies from dependencies stage
COPY --from=dependencies /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=dependencies /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=appuser:appuser . /app

# Switch to non-root user
USER appuser

# Expose port (customize as needed)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Default command
CMD ["python", "server.py"]

# ============================================================================
# Development Stage
# ============================================================================

FROM dependencies as development

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    pytest-asyncio \
    black \
    ruff \
    mypy

# Copy application code
COPY --chown=appuser:appuser . /app

# Switch to non-root user
USER appuser

# Run in development mode
CMD ["python", "server.py"]

# ============================================================================
# Testing Stage
# ============================================================================

FROM development as testing

# Run tests
RUN pytest tests/ -v --cov=. --cov-report=term-missing

# ============================================================================
# Usage Examples
# ============================================================================

# Build production image:
#   docker build --target production -t my-mcp-server:latest .
#
# Build development image:
#   docker build --target development -t my-mcp-server:dev .
#
# Run production container:
#   docker run -d -p 8000:8000 --name mcp-server my-mcp-server:latest
#
# Run development container with volume mount:
#   docker run -d -p 8000:8000 -v $(pwd):/app --name mcp-server-dev my-mcp-server:dev
#
# Run tests:
#   docker build --target testing -t my-mcp-server:test .
#
# Run with custom config:
#   docker run -d -p 8000:8000 -v $(pwd)/custom-config.yaml:/app/config.yaml my-mcp-server:latest
#
# Run with environment variables:
#   docker run -d -p 8000:8000 -e SERVER_NAME="custom-server" my-mcp-server:latest
#
# View logs:
#   docker logs -f mcp-server
#
# Access container shell:
#   docker exec -it mcp-server /bin/bash

# ============================================================================
# Docker Compose Example
# ============================================================================

# Create docker-compose.yml:
#
# version: '3.8'
#
# services:
#   mcp-server:
#     build:
#       context: .
#       target: production
#     ports:
#       - "8000:8000"
#     volumes:
#       - ./config.yaml:/app/config.yaml:ro
#       - ./data:/app/data
#     environment:
#       - SERVER_NAME=my-mcp-server
#       - LOG_LEVEL=INFO
#     restart: unless-stopped
#     healthcheck:
#       test: ["CMD", "python", "-c", "import requests; requests.get('http://localhost:8000/health')"]
#       interval: 30s
#       timeout: 10s
#       retries: 3
#       start_period: 40s
#
# Then run:
#   docker-compose up -d
