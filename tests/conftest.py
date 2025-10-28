"""
Pytest configuration and fixtures.
"""

import pytest


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests that require Ollama server running"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests that take a long time to run"
    )
    config.addinivalue_line(
        "markers", "benchmark: marks performance benchmark tests"
    )
