"""
Retry logic with exponential backoff.

Provides decorators and utilities for handling transient failures.
"""

import functools
import logging
import time
from typing import Callable, Optional, Tuple, Type

logger = logging.getLogger(__name__)


def retry_with_backoff(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None,
):
    """
    Decorator to retry a function with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries
        exponential_base: Base for exponential backoff calculation
        exceptions: Tuple of exception types to catch and retry
        on_retry: Optional callback function called on each retry

    Returns:
        Decorated function

    Example:
        >>> @retry_with_backoff(max_attempts=3, initial_delay=1.0)
        ... def fetch_data(url: str) -> dict:
        ...     response = requests.get(url)
        ...     response.raise_for_status()
        ...     return response.json()

        >>> @retry_with_backoff(
        ...     max_attempts=5,
        ...     exceptions=(ConnectionError, TimeoutError),
        ...     on_retry=lambda e, attempt: print(f"Retry {attempt}: {e}")
        ... )
        ... def connect_to_server():
        ...     # Connection code here
        ...     pass
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            delay = initial_delay

            while attempt < max_attempts:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    attempt += 1

                    if attempt >= max_attempts:
                        logger.error(f"{func.__name__} failed after {max_attempts} attempts: {e}")
                        raise

                    # Calculate next delay
                    current_delay = min(delay, max_delay)

                    logger.warning(
                        f"{func.__name__} failed (attempt {attempt}/{max_attempts}): {e}. "
                        f"Retrying in {current_delay:.1f}s..."
                    )

                    # Call retry callback if provided
                    if on_retry:
                        on_retry(e, attempt)

                    # Wait before retry
                    time.sleep(current_delay)

                    # Exponential backoff
                    delay *= exponential_base

        return wrapper

    return decorator


def retry_on_condition(
    condition: Callable[[any], bool],
    max_attempts: int = 3,
    delay: float = 1.0,
    on_retry: Optional[Callable[[any, int], None]] = None,
):
    """
    Decorator to retry based on return value condition.

    Args:
        condition: Function that takes return value and returns True if should retry
        max_attempts: Maximum number of attempts
        delay: Delay between attempts
        on_retry: Optional callback on retry

    Returns:
        Decorated function

    Example:
        >>> @retry_on_condition(
        ...     condition=lambda result: result is None,
        ...     max_attempts=3
        ... )
        ... def fetch_with_validation(url: str):
        ...     result = fetch(url)
        ...     return result if validate(result) else None
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0

            while attempt < max_attempts:
                result = func(*args, **kwargs)

                if not condition(result):
                    return result

                attempt += 1

                if attempt >= max_attempts:
                    logger.warning(
                        f"{func.__name__} condition not met after {max_attempts} attempts"
                    )
                    return result

                logger.debug(
                    f"{func.__name__} condition not met (attempt {attempt}/{max_attempts}). "
                    f"Retrying in {delay}s..."
                )

                if on_retry:
                    on_retry(result, attempt)

                time.sleep(delay)

            return result

        return wrapper

    return decorator


# Example usage and testing
if __name__ == "__main__":
    import random

    # Setup logging
    logging.basicConfig(level=logging.DEBUG)

    # Example 1: Retry on exception
    @retry_with_backoff(max_attempts=5, initial_delay=0.5)
    def unreliable_function():
        """Function that fails 70% of the time."""
        if random.random() < 0.7:
            raise ConnectionError("Simulated connection error")
        return "Success!"

    # Example 2: Retry on condition
    @retry_on_condition(
        condition=lambda x: x < 50,  # Retry if result < 50
        max_attempts=10,
        delay=0.1,
    )
    def get_random_number():
        """Returns random number, retries if below 50."""
        return random.randint(1, 100)

    # Test
    try:
        result = unreliable_function()
        print(f"Result: {result}")
    except ConnectionError as e:
        print(f"Failed: {e}")

    number = get_random_number()
    print(f"Random number (>= 50): {number}")
