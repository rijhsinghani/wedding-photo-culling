"""
Retry handler utility for API calls with exponential backoff
"""
import time
import logging
from functools import wraps
from typing import Any, Callable, Optional, Tuple, Type, Union

logger = logging.getLogger(__name__)


class RetryConfig:
    """Configuration for retry behavior"""
    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        exceptions: Tuple[Type[Exception], ...] = (Exception,)
    ):
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.exceptions = exceptions


def calculate_backoff_delay(
    attempt: int,
    initial_delay: float,
    exponential_base: float,
    max_delay: float,
    jitter: bool = True
) -> float:
    """Calculate exponential backoff delay with optional jitter"""
    delay = initial_delay * (exponential_base ** (attempt - 1))
    delay = min(delay, max_delay)
    
    if jitter:
        import random
        # Add up to 20% jitter
        jitter_amount = delay * 0.2 * random.random()
        delay = delay + jitter_amount
    
    return delay


def retry_on_exception(config: Optional[RetryConfig] = None):
    """
    Decorator to retry function calls on specified exceptions
    
    Args:
        config: RetryConfig object with retry parameters
        
    Returns:
        Decorated function that will retry on failure
    """
    if config is None:
        config = RetryConfig()
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(1, config.max_attempts + 1):
                try:
                    # Log attempt if not the first
                    if attempt > 1:
                        logger.info(
                            f"Retry attempt {attempt}/{config.max_attempts} "
                            f"for {func.__name__}"
                        )
                    
                    # Try to execute the function
                    result = func(*args, **kwargs)
                    
                    # Success - log if it was a retry
                    if attempt > 1:
                        logger.info(
                            f"Successfully completed {func.__name__} "
                            f"after {attempt} attempts"
                        )
                    
                    return result
                    
                except config.exceptions as e:
                    last_exception = e
                    
                    # Log the error
                    logger.warning(
                        f"Attempt {attempt}/{config.max_attempts} failed "
                        f"for {func.__name__}: {type(e).__name__}: {str(e)}"
                    )
                    
                    # Don't retry if this was the last attempt
                    if attempt >= config.max_attempts:
                        logger.error(
                            f"All {config.max_attempts} attempts failed "
                            f"for {func.__name__}"
                        )
                        raise
                    
                    # Calculate and apply backoff delay
                    delay = calculate_backoff_delay(
                        attempt,
                        config.initial_delay,
                        config.exponential_base,
                        config.max_delay,
                        config.jitter
                    )
                    
                    logger.info(
                        f"Waiting {delay:.2f} seconds before retry..."
                    )
                    time.sleep(delay)
                    
                except Exception as e:
                    # Unexpected exception - don't retry
                    logger.error(
                        f"Unexpected exception in {func.__name__}: "
                        f"{type(e).__name__}: {str(e)}"
                    )
                    raise
            
            # Should never reach here, but just in case
            if last_exception:
                raise last_exception
                
        return wrapper
    return decorator


class RetryableAPICall:
    """
    Context manager for retryable API calls with statistics tracking
    """
    def __init__(self, name: str, config: Optional[RetryConfig] = None):
        self.name = name
        self.config = config or RetryConfig()
        self.attempts = 0
        self.start_time = None
        self.success = False
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        
        if exc_type is None:
            self.success = True
            logger.debug(
                f"API call '{self.name}' completed successfully "
                f"in {elapsed:.2f}s"
            )
        else:
            logger.error(
                f"API call '{self.name}' failed after {elapsed:.2f}s: "
                f"{exc_type.__name__}: {exc_val}"
            )
        
        # Don't suppress the exception
        return False
    
    def track_attempt(self):
        """Track an attempt for statistics"""
        self.attempts += 1


# Convenience functions for common retry scenarios
def retry_gemini_api(func: Callable) -> Callable:
    """Retry decorator specifically for Gemini API calls"""
    config = RetryConfig(
        max_attempts=3,
        initial_delay=2.0,
        max_delay=30.0,
        exponential_base=2.0,
        jitter=True,
        exceptions=(
            TimeoutError,
            ConnectionError,
            OSError,
            # Add any Gemini-specific exceptions here
        )
    )
    return retry_on_exception(config)(func)