# Python Agent Sandboxing & uv Package Management Plan (2025)

**Date**: 2025-10-28
**Status**: Research Complete - Ready for Implementation
**Based on**: Industry best practices, LangChain security guidelines, modern Python tooling

---

## Executive Summary

This document outlines a comprehensive plan for implementing secure Python agent execution and modern package management using uv. The approach follows 2025 industry best practices for:

1. **OS-level isolation** (containers, VM sandboxes)
2. **Resource management** (CPU, memory, time limits)
3. **Network segmentation** (isolated environments)
4. **Modern package management** (uv with lock files)
5. **Layered security** (defense in depth)

**Key Finding**: Pure Python sandboxing is fundamentally insecure. OS-level isolation is mandatory.

---

## Part 1: Security Architecture

### 1.1 Layered Security Model

```
┌─────────────────────────────────────────────┐
│  Layer 5: Monitoring & Audit Logging       │
├─────────────────────────────────────────────┤
│  Layer 4: Resource Limits (CPU/Memory/Time)│
├─────────────────────────────────────────────┤
│  Layer 3: Network Segmentation             │
├─────────────────────────────────────────────┤
│  Layer 2: Container/VM Isolation           │
├─────────────────────────────────────────────┤
│  Layer 1: Read-Only Filesystems            │
└─────────────────────────────────────────────┘
```

**Principle**: Never rely on a single security layer. Each layer provides defense if another is compromised.

### 1.2 Threat Model

#### Threats to Mitigate:
- **Code Injection**: Malicious code execution via agent inputs
- **Resource Exhaustion**: Infinite loops, memory leaks
- **Data Exfiltration**: Unauthorized access to sensitive files
- **Privilege Escalation**: Breaking out of sandbox
- **Network Attacks**: Unauthorized external connections

#### Not in Scope (macOS local-first):
- Multi-tenant isolation (single user)
- DDoS protection (local only)
- Cloud API attacks (no cloud dependencies)

---

## Part 2: Sandbox Technologies

### 2.1 Recommended: Docker Containers

**Why**: Best balance of security, portability, and ease of use.

#### Implementation:

```dockerfile
# Dockerfile.agent-sandbox
FROM python:3.12-slim

# Security: Run as non-root user
RUN useradd -m -u 1000 agent && \
    mkdir /workspace && \
    chown agent:agent /workspace

# Install uv (fast package manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/

# Copy only necessary files (read-only)
WORKDIR /workspace
COPY --chown=agent:agent pyproject.toml uv.lock ./
COPY --chown=agent:agent utils/ ./utils/
COPY --chown=agent:agent workflows/ ./workflows/

# Install dependencies with uv
RUN uv sync --frozen --no-dev

# Switch to non-root user
USER agent

# Set environment
ENV PATH="/workspace/.venv/bin:$PATH"
ENV PYTHONPATH="/workspace:$PYTHONPATH"

# Resource limits (set at runtime with --cpus, --memory)
# Security hardening
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

CMD ["python", "-c", "print('Sandbox ready. Mount examples/ and run agents.')"]
```

#### Usage:

```bash
# Build sandbox
docker build -t ai-agent-sandbox -f Dockerfile.agent-sandbox .

# Run agent with strict limits
docker run --rm \
  --cpus="2.0" \                    # Limit to 2 CPU cores
  --memory="4g" \                   # Limit to 4GB RAM
  --memory-swap="4g" \              # No swap usage
  --pids-limit=50 \                 # Limit number of processes
  --network=none \                  # No network (unless Ollama needed)
  --read-only \                     # Read-only filesystem
  --tmpfs /tmp:rw,noexec,nosuid \  # Writable temp, no execution
  -v $(pwd)/examples:/workspace/examples:ro \  # Mount examples read-only
  ai-agent-sandbox \
  python examples/error_handling_demo.py
```

### 2.2 Alternative: Firecracker MicroVMs (Advanced)

**Why**: Fastest VM-level isolation, used by AWS Lambda.

```bash
# Install Firecracker (macOS via Lima)
brew install lima

# Create VM config for agent execution
limactl start --name=agent-sandbox template://docker

# Run agent in VM
limactl shell agent-sandbox python /workspace/examples/error_handling_demo.py
```

### 2.3 Alternative: gVisor (Linux-only)

**Why**: Application kernel for enhanced container security.

```bash
# Install gVisor
curl -fsSL https://gvisor.dev/archive.key | sudo gpg --dearmor -o /usr/share/keyrings/gvisor-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/gvisor-archive-keyring.gpg] https://storage.googleapis.com/gvisor/releases release main" | sudo tee /etc/apt/sources.list.d/gvisor.list > /dev/null
sudo apt-get update && sudo apt-get install -y runsc

# Run Docker with gVisor runtime
docker run --runtime=runsc ai-agent-sandbox python examples/error_handling_demo.py
```

---

## Part 3: uv Package Management

### 3.1 uv Best Practices (2025)

#### Core Principles:
1. **Lock file for development** (`uv.lock` in version control)
2. **Requirements.txt for deployment** (generated from lock file)
3. **Reproducible builds** (exact versions, no surprises)
4. **Fast CI/CD** (10-100x faster than pip)

#### Project Structure:

```
ai-lang-stuff/
├── pyproject.toml          # Source of truth (dependencies)
├── uv.lock                 # Locked versions (commit to git)
├── requirements.txt        # Deployment fallback (generated)
├── .python-version         # Python version pinning
└── README.md
```

### 3.2 uv Commands Cheat Sheet

```bash
# Development Workflow
uv sync                     # Install deps from lock file
uv sync --frozen            # Fail if lock file out of date
uv sync --extra dev         # Install dev dependencies
uv sync --all-extras        # Install all optional deps

# Adding Dependencies
uv add langchain            # Add production dependency
uv add --dev pytest         # Add development dependency
uv add "numpy>=1.24,<2"     # Add with version constraints

# Running Code
uv run python script.py     # Run script with project venv
uv run pytest               # Run tests with project venv

# Deployment
uv export --no-dev > requirements.txt  # Generate requirements.txt
uv export --frozen > requirements-lock.txt  # With exact versions

# Python Version Management
uv python pin 3.12          # Pin Python version
uv venv --python 3.12       # Create venv with specific Python
uv python list              # List available Python versions

# Maintenance
uv lock --upgrade           # Upgrade dependencies
uv tree                     # Show dependency tree
uv pip list --outdated      # Check for updates
```

### 3.3 Deployment Strategies

#### Strategy 1: Docker with uv (Recommended)

```dockerfile
FROM python:3.12-slim

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/

# Copy lock file and sync
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --compile-bytecode

# Bytecode compilation for faster startup
ENV PYTHONDONTWRITEBYTECODE=1
```

#### Strategy 2: Traditional pip (Compatibility)

```dockerfile
FROM python:3.12-slim

# Generate and use requirements.txt
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
```

#### Strategy 3: Multi-stage build (Smallest image)

```dockerfile
# Stage 1: Build dependencies
FROM python:3.12-slim AS builder
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Stage 2: Runtime
FROM python:3.12-slim
COPY --from=builder /.venv /.venv
ENV PATH="/.venv/bin:$PATH"
COPY utils/ ./utils/
COPY workflows/ ./workflows/
```

---

## Part 4: Resource Management

### 4.1 CPU & Memory Limits

```bash
# Docker resource limits
docker run \
  --cpus="2.0" \              # 2 CPU cores max
  --cpus-shares=512 \         # CPU priority (relative)
  --memory="4g" \             # 4GB RAM hard limit
  --memory-reservation="2g" \ # 2GB RAM soft limit
  --memory-swap="4g" \        # No additional swap
  --kernel-memory="1g" \      # Kernel memory limit
  ai-agent-sandbox python agent.py
```

### 4.2 Time Limits

```python
# utils/execution_limits.py
import signal
from contextlib import contextmanager

@contextmanager
def timeout(seconds: int):
    """Context manager for execution timeout."""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Execution exceeded {seconds}s")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

# Usage
with timeout(30):  # 30 second limit
    result = agent.run("complex task")
```

### 4.3 Process & File Limits

```bash
# Limit number of processes
docker run --pids-limit=50 ai-agent-sandbox python agent.py

# Limit file descriptors
docker run --ulimit nofile=1024:1024 ai-agent-sandbox python agent.py

# Limit file size
docker run --ulimit fsize=104857600:104857600 ai-agent-sandbox python agent.py  # 100MB
```

---

## Part 5: Network Isolation

### 5.1 No Network (Default)

```bash
# Completely isolated (no Ollama access)
docker run --network=none ai-agent-sandbox python examples/error_handling_demo.py
```

### 5.2 Ollama Access Only

```bash
# Allow localhost access for Ollama
docker run --network=host ai-agent-sandbox python examples/01-foundation/basic_llm_interaction.py
```

### 5.3 Custom Bridge Network (Best Practice)

```bash
# Create isolated network
docker network create --driver bridge agent-network

# Run Ollama on same network
docker run -d --name ollama --network agent-network ollama/ollama

# Run agent with access to Ollama only
docker run --rm \
  --network agent-network \
  -e OLLAMA_HOST=http://ollama:11434 \
  ai-agent-sandbox python examples/01-foundation/basic_llm_interaction.py
```

---

## Part 6: Monitoring & Audit Logging

### 6.1 Execution Monitoring

```python
# utils/sandbox_monitor.py
import logging
import psutil
import time
from typing import Dict, Any

class SandboxMonitor:
    """Monitor sandbox execution for security and performance."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.start_time = time.time()
        self.metrics: Dict[str, Any] = {}

    def log_execution_start(self, agent_id: str, task: str):
        """Log when agent execution starts."""
        self.logger.info(
            "Agent execution started",
            extra={
                "agent_id": agent_id,
                "task": task,
                "timestamp": time.time(),
            }
        )

    def check_resource_usage(self) -> Dict[str, float]:
        """Check current resource usage."""
        process = psutil.Process()
        return {
            "cpu_percent": process.cpu_percent(interval=0.1),
            "memory_mb": process.memory_info().rss / 1024 / 1024,
            "num_threads": process.num_threads(),
            "num_fds": process.num_fds(),
        }

    def log_anomaly(self, anomaly_type: str, details: Dict[str, Any]):
        """Log security anomaly."""
        self.logger.warning(
            f"Security anomaly detected: {anomaly_type}",
            extra={"anomaly": anomaly_type, "details": details}
        )

    def log_execution_end(self, agent_id: str, success: bool):
        """Log when agent execution completes."""
        duration = time.time() - self.start_time
        self.logger.info(
            "Agent execution completed",
            extra={
                "agent_id": agent_id,
                "success": success,
                "duration_seconds": duration,
                "final_metrics": self.check_resource_usage(),
            }
        )
```

### 6.2 Structured Logging

```python
# config/logging_config.py
import logging.config

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {
            "class": "pythonjsonlogger.jsonlogger.JsonFormatter",
            "format": "%(asctime)s %(name)s %(levelname)s %(message)s",
        }
    },
    "handlers": {
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "logs/sandbox_execution.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "formatter": "json",
        },
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "json",
        }
    },
    "root": {
        "level": "INFO",
        "handlers": ["file", "console"]
    }
}

logging.config.dictConfig(LOGGING_CONFIG)
```

---

## Part 7: Implementation Roadmap

### Phase 1: Core Sandboxing (Week 1)

- [x] Dockerfile.sandbox created
- [ ] Test Docker sandbox with resource limits
- [ ] Implement timeout context manager
- [ ] Add resource monitoring
- [ ] Document Docker usage in README

### Phase 2: uv Integration (Week 1-2)

- [x] pyproject.toml with uv configuration
- [x] uv.lock committed to repository
- [ ] Generate requirements.txt for compatibility
- [ ] Update CI/CD to use uv
- [ ] Add uv deployment examples

### Phase 3: Security Hardening (Week 2-3)

- [ ] Implement SandboxMonitor class
- [ ] Add structured JSON logging
- [ ] Create network isolation examples
- [ ] Add seccomp profiles (Linux)
- [ ] AppArmor profiles (Linux)

### Phase 4: Production Patterns (Week 3-4)

- [ ] Multi-stage Docker builds
- [ ] Agent pool management
- [ ] Rate limiting implementation
- [ ] Anomaly detection system
- [ ] Security audit documentation

### Phase 5: Advanced Features (Week 4+)

- [ ] Firecracker microVM integration
- [ ] gVisor runtime support (Linux)
- [ ] Kubernetes deployment manifests
- [ ] Performance benchmarking
- [ ] Security penetration testing

---

## Part 8: Security Checklist

### Pre-Deployment Security Audit

- [ ] **Code Review**: All agent code reviewed for security
- [ ] **Dependencies**: All dependencies scanned (Snyk, Dependabot)
- [ ] **Secrets**: No hardcoded secrets or API keys
- [ ] **Permissions**: Minimal filesystem permissions
- [ ] **Network**: Network isolation configured
- [ ] **Resources**: CPU/memory limits set
- [ ] **Logging**: Audit logging enabled
- [ ] **Monitoring**: Resource monitoring active
- [ ] **Timeout**: Execution timeouts configured
- [ ] **User**: Running as non-root user
- [ ] **Filesystem**: Read-only where possible
- [ ] **Updates**: Base images up to date

### Runtime Security Monitoring

- [ ] Log all agent executions
- [ ] Monitor CPU/memory usage
- [ ] Track network connections
- [ ] Alert on anomalies
- [ ] Regular security scans
- [ ] Incident response plan

---

## Part 9: Testing Strategy

### 9.1 Security Tests

```python
# tests/security/test_sandbox_escape.py
import pytest
import docker

def test_sandbox_cannot_access_host_filesystem():
    """Verify sandbox cannot access host files."""
    client = docker.from_env()
    result = client.containers.run(
        "ai-agent-sandbox",
        "python -c 'import os; os.listdir(\"/etc\")'",
        remove=True,
        network_disabled=True,
    )
    # Should fail or only see container's /etc
    assert "/etc/passwd" not in result

def test_sandbox_cannot_escalate_privileges():
    """Verify sandbox cannot gain root access."""
    client = docker.from_env()
    with pytest.raises(docker.errors.APIError):
        client.containers.run(
            "ai-agent-sandbox",
            "sudo whoami",
            remove=True,
        )

def test_sandbox_respects_resource_limits():
    """Verify resource limits are enforced."""
    client = docker.from_env()
    container = client.containers.run(
        "ai-agent-sandbox",
        "python -c 'import time; time.sleep(10)'",
        mem_limit="100m",
        cpus="0.5",
        detach=True,
    )
    stats = container.stats(stream=False)
    memory_usage = stats["memory_stats"]["usage"]
    assert memory_usage < 100 * 1024 * 1024  # 100MB
```

### 9.2 Performance Tests

```python
# tests/performance/test_uv_speed.py
import time
import subprocess

def test_uv_sync_faster_than_pip():
    """Verify uv is faster than pip."""
    # Time uv sync
    start = time.time()
    subprocess.run(["uv", "sync", "--frozen"], check=True)
    uv_time = time.time() - start

    # Time pip install
    start = time.time()
    subprocess.run(["pip", "install", "-r", "requirements.txt"], check=True)
    pip_time = time.time() - start

    assert uv_time < pip_time, f"uv ({uv_time}s) should be faster than pip ({pip_time}s)"
```

---

## Part 10: Documentation

### 10.1 User-Facing Documentation

Create comprehensive guides:

1. **SANDBOX-GUIDE.md**: How to use sandboxes
2. **UV-GUIDE.md**: uv package management guide
3. **SECURITY.md**: Security best practices
4. **DEPLOYMENT.md**: Production deployment guide

### 10.2 Example Scripts

```bash
# scripts/run-sandbox.sh
#!/bin/bash
# Run agent in secure Docker sandbox

set -euo pipefail

AGENT_SCRIPT="${1:?Usage: $0 <agent_script>}"
CPU_LIMIT="${CPU_LIMIT:-2.0}"
MEM_LIMIT="${MEM_LIMIT:-4g}"
TIMEOUT="${TIMEOUT:-300}"  # 5 minutes

docker run --rm \
  --cpus="$CPU_LIMIT" \
  --memory="$MEM_LIMIT" \
  --network=none \
  --read-only \
  --tmpfs /tmp:rw,noexec,nosuid \
  -v "$(pwd)/examples:/workspace/examples:ro" \
  -v "$(pwd)/utils:/workspace/utils:ro" \
  ai-agent-sandbox \
  timeout "$TIMEOUT" python "$AGENT_SCRIPT"
```

---

## Conclusion

This plan provides a comprehensive, production-ready approach to Python agent sandboxing using:

- **Docker containers** for OS-level isolation
- **uv** for fast, reproducible package management
- **Resource limits** to prevent abuse
- **Network isolation** for security
- **Monitoring & logging** for observability
- **Layered security** for defense in depth

**Next Steps**:
1. Implement Phase 1 (Core Sandboxing)
2. Test with existing examples
3. Document usage patterns
4. Roll out to production

**Success Metrics**:
- Zero security incidents
- <5s agent startup time
- >99.9% sandbox uptime
- Full audit trail of executions

---

**References**:
- LangChain Security Guidelines: https://python.langchain.com/docs/security/
- uv Documentation: https://github.com/astral-sh/uv
- Docker Security Best Practices: https://docs.docker.com/engine/security/
- OWASP Sandboxing: https://cheatsheetseries.owasp.org/cheatsheets/Sandboxing_Cheat_Sheet.html

**Author**: Claude Code + User
**Last Updated**: 2025-10-28
**Version**: 1.0.0
