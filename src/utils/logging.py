"""
Logging configuration and utilities.

Provides consistent logging across the application with file and console output.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
    console: bool = True,
) -> None:
    """
    Configure logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file. If None, only console logging
        log_format: Custom log format string
        console: Whether to log to console

    Example:
        >>> setup_logging(level="DEBUG", log_file="app.log")
        >>> logger = get_logger(__name__)
        >>> logger.info("Application started")
    """
    # Convert level string to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Default format
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Create formatter
    formatter = logging.Formatter(log_format)

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove existing handlers
    root_logger.handlers = []

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started")
        >>> logger.error("An error occurred", exc_info=True)
    """
    return logging.getLogger(name)


# Convenience functions for common logging patterns
def log_function_call(logger: logging.Logger, func_name: str, **kwargs) -> None:
    """
    Log a function call with parameters.

    Args:
        logger: Logger instance
        func_name: Name of the function
        **kwargs: Function parameters to log
    """
    params = ", ".join(f"{k}={v}" for k, v in kwargs.items())
    logger.debug(f"Calling {func_name}({params})")


def log_execution_time(logger: logging.Logger, operation: str, elapsed: float) -> None:
    """
    Log execution time for an operation.

    Args:
        logger: Logger instance
        operation: Description of the operation
        elapsed: Time elapsed in seconds
    """
    logger.info(f"{operation} completed in {elapsed:.2f}s")


# Example usage
if __name__ == "__main__":
    # Setup logging
    setup_logging(level="DEBUG", log_file="logs/test.log")

    # Get logger
    logger = get_logger(__name__)

    # Test logging
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")

    # Test convenience functions
    log_function_call(logger, "process_data", input="test.txt", output="result.txt")
    log_execution_time(logger, "Data processing", 1.23)
