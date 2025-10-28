# Production-Ready AI Agent Patterns

## Overview

This directory contains comprehensive production-ready patterns for deploying AI agents in real-world environments. Each example demonstrates enterprise-grade features including robust error handling, comprehensive logging, configuration management, monitoring, and deployment readiness.

These patterns are designed for **production deployments** where reliability, observability, and maintainability are critical.

---

## Table of Contents

1. [Production Principles](#production-principles)
2. [Available Examples](#available-examples)
3. [Prerequisites](#prerequisites)
4. [Quick Start](#quick-start)
5. [Production Best Practices](#production-best-practices)
6. [Deployment Strategies](#deployment-strategies)
7. [Monitoring & Observability](#monitoring--observability)
8. [Troubleshooting](#troubleshooting)
9. [Next Steps](#next-steps)

---

## Production Principles

### Core Production Requirements

Every production-ready AI system should implement:

1. **Reliability**
   - Graceful error handling
   - Retry logic with exponential backoff
   - Circuit breakers for external dependencies
   - Fallback mechanisms
   - Health checks and readiness probes

2. **Observability**
   - Structured logging (JSON format)
   - Performance metrics (latency, throughput, errors)
   - Distributed tracing
   - Resource monitoring (CPU, memory, disk)
   - Request/response logging

3. **Security**
   - Input validation
   - Secret management
   - Rate limiting
   - CORS configuration
   - SSL/TLS support

4. **Maintainability**
   - Clear configuration management
   - Environment-specific settings
   - Versioning
   - Documentation
   - Testing strategies

5. **Scalability**
   - Concurrent request handling
   - Resource pooling
   - Caching strategies
   - Horizontal scaling support

---

## Available Examples

### 1. `production_agent.py`

**Complete production-ready agent with all enterprise features.**

**Features:**
- Comprehensive error handling with custom exceptions
- Retry logic with exponential backoff (using tenacity)
- Structured logging with multiple levels
- YAML configuration management
- Health check system
- Performance metrics tracking
- Graceful shutdown handling
- Request/response validation (Pydantic)
- Concurrent request limiting

**Usage:**
```bash
uv run python examples/06-production/production_agent.py
```

**Key Classes:**
- `ProductionAgent`: Main agent with full production features
- `AgentConfig`: Configuration with validation
- `HealthCheck`: Health monitoring system
- `AgentMetrics`: Performance tracking

**When to use:**
- Deploying agents to production environments
- Need for robust error handling
- Require comprehensive monitoring
- Multi-environment deployments

---

### 2. `monitoring_logging.py`

**Comprehensive monitoring and logging system.**

**Features:**
- Structured JSON logging
- LangSmith integration (optional)
- Performance metrics collection
  - Latency tracking (min, max, average)
  - Throughput measurement
  - Token usage statistics
- Error tracking with categorization
- Usage analytics and trends
- Real-time monitoring dashboard
- Custom callback handlers

**Usage:**
```bash
# Without LangSmith
uv run python examples/06-production/monitoring_logging.py

# With LangSmith tracing
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_PROJECT="production-monitoring"
uv run python examples/06-production/monitoring_logging.py
```

**Key Classes:**
- `MonitoredAgent`: Agent with full monitoring
- `PerformanceMetrics`: Latency, throughput, token tracking
- `UsageAnalytics`: Usage patterns and trends
- `MonitoringCallbackHandler`: Custom LangChain callback

**Metrics Tracked:**
- Request count (total, success, failure)
- Latency (min, max, average)
- Throughput (requests/second)
- Token usage (input, output, total)
- Error rates by type
- Peak usage hours
- Average message length

**When to use:**
- Need detailed performance insights
- Debugging production issues
- Capacity planning
- SLA compliance tracking

---

### 3. `config_management.py`

**Enterprise configuration management system.**

**Features:**
- YAML configuration files
- Environment variable overrides
- Multi-environment support (dev, staging, prod)
- Configuration validation with Pydantic
- Secret management (SecretStr)
- Configuration merging and inheritance
- Environment-specific overrides
- Configuration factory patterns

**Usage:**
```bash
# Development environment (default)
uv run python examples/06-production/config_management.py

# Production environment
ENV=production uv run python examples/06-production/config_management.py

# Custom config file
CONFIG_FILE=/path/to/config.yaml uv run python examples/06-production/config_management.py
```

**Configuration Sections:**
- `OllamaConfig`: Ollama server settings
- `ModelConfig`: Model parameters
- `LoggingConfig`: Logging configuration
- `MonitoringConfig`: Observability settings
- `SecurityConfig`: Security parameters
- `PerformanceConfig`: Performance tuning

**Environment Support:**
```yaml
# Base configuration
environment: development
model:
  name: qwen3:8b
  temperature: 0.7

# Environment-specific overrides
environments:
  staging:
    debug: false
    monitoring:
      enable_tracing: true

  production:
    debug: false
    logging:
      level: WARNING
    security:
      enable_ssl: true
```

**When to use:**
- Managing multiple environments
- Need for configuration validation
- Secure secret handling
- Complex configuration requirements

---

### 4. `deployment_ready.py`

**Complete deployment-ready application with HTTP API.**

**Features:**
- HTTP API with health and metrics endpoints
- Kubernetes-compatible health probes
- Prometheus-compatible metrics
- Graceful shutdown with connection draining
- Resource monitoring (CPU, memory, disk)
- Process management
- Docker and Docker Compose support
- Production-grade error handling

**Endpoints:**
- `GET /health` - Comprehensive health check
- `GET /health/ready` - Readiness probe (Kubernetes)
- `GET /health/live` - Liveness probe (Kubernetes)
- `GET /metrics` - Prometheus metrics
- `POST /api/v1/chat` - Chat endpoint

**Usage:**
```bash
# Run locally
uv run python examples/06-production/deployment_ready.py

# Custom port
PORT=8080 uv run python examples/06-production/deployment_ready.py

# Docker deployment
docker build -f Dockerfile.agent -t ai-agent .
docker run -p 8000:8000 ai-agent

# Docker Compose (includes Ollama)
docker-compose up
```

**Health Check Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-10-26T12:00:00Z",
  "application": {
    "name": "ai-agent",
    "version": "0.1.0",
    "model": "qwen3:8b"
  },
  "resources": {
    "cpu_percent": 15.2,
    "memory": {
      "rss_mb": 256.5,
      "percent": 2.1
    },
    "uptime_seconds": 3600
  },
  "metrics": {
    "total_requests": 1500,
    "successful_requests": 1475,
    "average_response_time_ms": 234.5
  }
}
```

**When to use:**
- Deploying to Kubernetes/container environments
- Need for HTTP API interface
- Require health check endpoints
- Prometheus monitoring integration

---

## Prerequisites

### 1. Ollama Setup

Ensure Ollama is running with required models:

```bash
# Start Ollama
ollama serve

# Pull recommended models
ollama pull qwen3:8b          # Fast, efficient
ollama pull qwen3:30b-a3b     # Powerful MoE
```

Verify:
```bash
ollama list
curl http://localhost:11434/api/tags
```

### 2. Python Dependencies

Install production dependencies:

```bash
# Using uv (recommended)
uv add langchain langchain-ollama pyyaml pydantic pydantic-settings tenacity psutil aiohttp

# Or using pip
pip install langchain langchain-ollama pyyaml pydantic pydantic-settings tenacity psutil aiohttp
```

### 3. Optional: LangSmith

For distributed tracing (optional):

```bash
# Set environment variables
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_PROJECT="production"
export LANGCHAIN_API_KEY="your-key"  # Optional for local
```

### 4. Directory Structure

Create required directories:

```bash
mkdir -p logs config data
```

---

## Quick Start

### Running Examples

**1. Basic Production Agent:**
```bash
uv run python examples/06-production/production_agent.py
```

**2. Monitoring System:**
```bash
uv run python examples/06-production/monitoring_logging.py
```

**3. Configuration Management:**
```bash
uv run python examples/06-production/config_management.py
```

**4. Deployment-Ready API:**
```bash
uv run python examples/06-production/deployment_ready.py
```

Then test endpoints:
```bash
# Health check
curl http://localhost:8000/health

# Chat request
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, how are you?"}'

# Metrics
curl http://localhost:8000/metrics
```

---

## Production Best Practices

### 1. Error Handling

**Always use specific exceptions:**
```python
class AgentException(Exception):
    """Base exception for agent errors."""
    pass

class ModelConnectionError(AgentException):
    """Raised when connection to model fails."""
    pass

class ValidationException(AgentException):
    """Raised when validation fails."""
    pass
```

**Implement retry logic:**
```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10)
)
async def invoke_with_retry(self, messages):
    return await self.llm.invoke(messages)
```

**Graceful degradation:**
```python
try:
    response = await agent.process(request)
except ModelConnectionError:
    # Fall back to cached response or error message
    response = get_cached_response() or default_error_response()
```

---

### 2. Logging

**Use structured logging:**
```python
import logging
import json

logger = logging.getLogger(__name__)

# Log with structured data
logger.info(
    "Request processed",
    extra={
        "request_id": request_id,
        "duration_ms": duration,
        "status": "success"
    }
)
```

**Log levels:**
- `DEBUG`: Detailed diagnostic information
- `INFO`: General informational messages
- `WARNING`: Warning messages for unusual events
- `ERROR`: Error messages for failures
- `CRITICAL`: Critical issues requiring immediate attention

**What to log:**
- Request/response metadata (not full content in production)
- Error details with stack traces
- Performance metrics
- Security events
- Configuration changes
- Startup/shutdown events

**What NOT to log:**
- User secrets or passwords
- Full API keys
- Personal identifiable information (PII)
- Full request/response bodies in production

---

### 3. Configuration

**Environment-specific configuration:**
```yaml
# config/development.yaml
debug: true
logging:
  level: DEBUG
model:
  temperature: 0.8

# config/production.yaml
debug: false
logging:
  level: WARNING
  output_file: /var/log/agent/app.log
security:
  enable_ssl: true
  rate_limit_per_minute: 1000
```

**Secret management:**
```python
from pydantic import SecretStr

class SecurityConfig(BaseModel):
    api_key: SecretStr  # Never logged or exposed

    def get_api_key(self) -> str:
        return self.api_key.get_secret_value()
```

**Environment variables:**
```bash
# .env file (never commit to git!)
OLLAMA_API_KEY=secret123
DATABASE_URL=postgresql://user:pass@host/db
```

---

### 4. Health Checks

**Implement multiple health endpoints:**

**Liveness probe** - Is the application alive?
```python
async def liveness_handler(self, request):
    """Application is running (but may not be ready)"""
    return {"status": "alive"}
```

**Readiness probe** - Can it serve traffic?
```python
async def readiness_handler(self, request):
    """Application is ready to serve requests"""
    ollama_healthy = await check_ollama_connection()
    if not ollama_healthy:
        return {"status": "not_ready"}, 503
    return {"status": "ready"}
```

**Kubernetes configuration:**
```yaml
livenessProbe:
  httpGet:
    path: /health/live
    port: 8000
  initialDelaySeconds: 10
  periodSeconds: 30

readinessProbe:
  httpGet:
    path: /health/ready
    port: 8000
  initialDelaySeconds: 5
  periodSeconds: 10
```

---

### 5. Graceful Shutdown

**Implement proper shutdown:**
```python
import signal
import asyncio

class Application:
    def __init__(self):
        self.is_shutting_down = False
        self.setup_signal_handlers()

    def setup_signal_handlers(self):
        def shutdown_handler(signum, frame):
            self.logger.info("Received shutdown signal")
            self.is_shutting_down = True

        signal.signal(signal.SIGTERM, shutdown_handler)
        signal.signal(signal.SIGINT, shutdown_handler)

    async def shutdown(self):
        # Stop accepting new requests
        self.is_shutting_down = True

        # Wait for in-flight requests (max 30s)
        timeout = 30
        while has_active_requests() and timeout > 0:
            await asyncio.sleep(1)
            timeout -= 1

        # Cleanup resources
        await self.cleanup()
```

---

### 6. Monitoring

**Key metrics to track:**

**Latency metrics:**
- Average response time
- P50, P95, P99 percentiles
- Minimum/maximum latency

**Throughput metrics:**
- Requests per second
- Requests per minute
- Concurrent requests

**Error metrics:**
- Error rate (percentage)
- Errors by type
- Failed requests

**Resource metrics:**
- CPU usage
- Memory usage
- Disk usage
- Network I/O

**Business metrics:**
- Token usage
- Cost per request
- User sessions
- Peak usage times

**Prometheus metrics example:**
```python
def get_prometheus_metrics(self):
    return """
# HELP agent_requests_total Total requests
# TYPE agent_requests_total counter
agent_requests_total{status="success"} 1500
agent_requests_total{status="error"} 25

# HELP agent_latency_ms Request latency
# TYPE agent_latency_ms histogram
agent_latency_ms_bucket{le="100"} 800
agent_latency_ms_bucket{le="500"} 1450
agent_latency_ms_bucket{le="1000"} 1500
"""
```

---

### 7. Testing Production Code

**Unit tests:**
```python
import pytest

@pytest.mark.asyncio
async def test_agent_processes_request():
    agent = ProductionAgent(config)
    request = AgentRequest(message="test")
    response = await agent.process_request(request)
    assert response.content
    assert response.duration_ms > 0
```

**Integration tests:**
```python
@pytest.mark.asyncio
async def test_health_endpoint():
    async with aiohttp.ClientSession() as session:
        async with session.get("http://localhost:8000/health") as resp:
            assert resp.status == 200
            data = await resp.json()
            assert data["status"] == "healthy"
```

**Load tests:**
```python
import asyncio
import time

async def load_test(num_requests=100):
    start = time.time()
    tasks = [agent.process(f"request {i}") for i in range(num_requests)]
    await asyncio.gather(*tasks)
    duration = time.time() - start
    print(f"Processed {num_requests} requests in {duration:.2f}s")
    print(f"Throughput: {num_requests/duration:.2f} req/s")
```

---

## Deployment Strategies

### 1. Docker Deployment

**Build image:**
```bash
docker build -f Dockerfile.agent -t ai-agent:latest .
```

**Run container:**
```bash
docker run -d \
  --name ai-agent \
  -p 8000:8000 \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  -e LOG_LEVEL=INFO \
  ai-agent:latest
```

**Docker Compose:**
```bash
docker-compose up -d
```

---

### 2. Kubernetes Deployment

**Deployment manifest:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-agent
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ai-agent
  template:
    metadata:
      labels:
        app: ai-agent
    spec:
      containers:
      - name: ai-agent
        image: ai-agent:latest
        ports:
        - containerPort: 8000
        env:
        - name: LOG_LEVEL
          value: "INFO"
        - name: OLLAMA_BASE_URL
          value: "http://ollama-service:11434"
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 10
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
```

**Service manifest:**
```yaml
apiVersion: v1
kind: Service
metadata:
  name: ai-agent-service
spec:
  selector:
    app: ai-agent
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

**Deploy:**
```bash
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
```

---

### 3. Local Process Management

**Using systemd:**
```ini
# /etc/systemd/system/ai-agent.service
[Unit]
Description=AI Agent Application
After=network.target

[Service]
Type=simple
User=agent
WorkingDirectory=/opt/ai-agent
Environment="PATH=/opt/ai-agent/venv/bin"
ExecStart=/opt/ai-agent/venv/bin/python deployment_ready.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Manage service:**
```bash
sudo systemctl enable ai-agent
sudo systemctl start ai-agent
sudo systemctl status ai-agent
sudo journalctl -u ai-agent -f
```

---

## Monitoring & Observability

### LangSmith Integration

**Setup:**
```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_PROJECT="production"
export LANGCHAIN_API_KEY="your-key"
```

**Features:**
- Request/response tracing
- Latency tracking
- Token usage monitoring
- Error debugging
- Chain visualization

**Access:** https://smith.langchain.com

---

### Prometheus Metrics

**Scrape configuration:**
```yaml
scrape_configs:
  - job_name: 'ai-agent'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    scrape_interval: 15s
```

---

### Grafana Dashboards

**Key panels:**
- Request rate graph
- Latency percentiles (P50, P95, P99)
- Error rate percentage
- Resource usage (CPU, memory)
- Token usage over time

---

### Log Aggregation

**Structured logs to stdout:**
```python
# Logs are JSON formatted
{"timestamp": "2025-10-26T12:00:00Z", "level": "INFO", "message": "Request completed", "duration_ms": 234.5}
```

**Collect with:**
- **Fluentd**: Kubernetes log collector
- **Filebeat**: Log shipper to Elasticsearch
- **CloudWatch**: AWS log aggregation
- **Loki**: Grafana log aggregation

---

## Troubleshooting

### Issue: High Latency

**Symptoms:**
- Slow response times
- Timeouts
- User complaints

**Debug:**
```bash
# Check metrics endpoint
curl http://localhost:8000/metrics

# Review logs
tail -f logs/monitoring.jsonl | grep duration_ms

# Monitor resources
curl http://localhost:8000/health | jq .resources
```

**Solutions:**
- Use faster models (qwen3:30b-a3b for MoE speed)
- Implement caching
- Scale horizontally
- Optimize prompts

---

### Issue: Memory Leaks

**Symptoms:**
- Increasing memory usage over time
- OOM errors
- Container restarts

**Debug:**
```python
import tracemalloc

tracemalloc.start()
# ... run application
snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')
for stat in top_stats[:10]:
    print(stat)
```

**Solutions:**
- Check for circular references
- Clear caches periodically
- Use connection pooling
- Set memory limits

---

### Issue: Ollama Connection Errors

**Symptoms:**
- "Connection refused"
- "Model not found"
- Timeouts

**Debug:**
```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# Check model availability
ollama list

# Test model directly
ollama run qwen3:8b "test"
```

**Solutions:**
- Start Ollama: `ollama serve`
- Pull model: `ollama pull qwen3:8b`
- Check network connectivity
- Verify base_url configuration

---

### Issue: Rate Limiting

**Symptoms:**
- 429 Too Many Requests
- Degraded performance
- Request rejections

**Solutions:**
```python
class RateLimiter:
    def __init__(self, max_requests_per_minute=60):
        self.max_requests = max_requests_per_minute
        self.requests = []

    async def check_rate_limit(self):
        now = time.time()
        # Remove old requests
        self.requests = [r for r in self.requests if now - r < 60]

        if len(self.requests) >= self.max_requests:
            raise RateLimitError("Rate limit exceeded")

        self.requests.append(now)
```

---

## Next Steps

### From Development to Production

**Checklist:**
- [ ] Implement all examples in this directory
- [ ] Setup monitoring (metrics, logs, tracing)
- [ ] Configure multi-environment settings
- [ ] Write comprehensive tests
- [ ] Create deployment manifests
- [ ] Setup CI/CD pipeline
- [ ] Document runbooks for operations
- [ ] Perform load testing
- [ ] Setup alerting rules
- [ ] Create disaster recovery plan

### Advanced Topics

**Scaling:**
- Horizontal pod autoscaling (HPA)
- Load balancing strategies
- Connection pooling
- Caching layers (Redis)

**Security:**
- Authentication/Authorization
- API key management
- Network policies
- Secret rotation
- Audit logging

**Performance:**
- Model quantization
- Batch processing
- Async processing
- CDN integration

### Related Documentation

- [LangChain Production Guide](https://python.langchain.com/docs/guides/production)
- [LangSmith Documentation](https://docs.smith.langchain.com/)
- [Prometheus Best Practices](https://prometheus.io/docs/practices/)
- [Kubernetes Patterns](https://kubernetes.io/docs/concepts/)

---

## Contributing

Found issues or want to improve examples?

1. Add comprehensive docstrings
2. Include error handling
3. Add tests
4. Update this README
5. Submit PR

---

**Production deployment is complex - these examples provide a solid foundation. Customize for your specific requirements!**

For questions or issues, refer to the main project documentation in `/docs/` or create an issue on GitHub.
