"""
Circuit Breaker implementation for external API resilience.

Prevents cascade failures when external APIs (Resemble, OpenAI, Pyannote) are unavailable.
When failure threshold is exceeded, the circuit "opens" and fails fast for a recovery period,
allowing the external service time to recover without overwhelming it with requests.

States:
- CLOSED: Normal operation, requests pass through
- OPEN: Service is down, requests fail immediately
- HALF_OPEN: Testing if service recovered, limited requests allowed
"""

import time
import logging
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from threading import Lock
from typing import Any, Callable, Dict, Optional, Type, Tuple

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing fast
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    failure_threshold: int = 3          # Failures before opening circuit
    recovery_timeout: float = 60.0      # Seconds before trying again
    half_open_max_calls: int = 1        # Calls allowed in half-open state
    excluded_exceptions: Tuple[Type[Exception], ...] = ()  # Don't count these as failures


@dataclass
class CircuitBreakerState:
    """Tracks the state of a circuit breaker."""
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    last_failure_time: float = 0.0
    half_open_calls: int = 0
    lock: Lock = field(default_factory=Lock)


class CircuitBreakerError(Exception):
    """Raised when circuit is open and request is rejected."""
    def __init__(self, service_name: str, time_until_retry: float):
        self.service_name = service_name
        self.time_until_retry = time_until_retry
        super().__init__(
            f"Circuit breaker open for '{service_name}'. "
            f"Retry in {time_until_retry:.1f}s"
        )


class CircuitBreakerRegistry:
    """
    Global registry for circuit breakers.
    Allows checking status and resetting circuits externally.
    """
    _breakers: Dict[str, "CircuitBreaker"] = {}
    _lock = Lock()

    @classmethod
    def register(cls, name: str, breaker: "CircuitBreaker") -> None:
        with cls._lock:
            cls._breakers[name] = breaker

    @classmethod
    def get(cls, name: str) -> Optional["CircuitBreaker"]:
        return cls._breakers.get(name)

    @classmethod
    def get_all_status(cls) -> Dict[str, Dict[str, Any]]:
        """Get status of all registered circuit breakers."""
        return {
            name: breaker.get_status()
            for name, breaker in cls._breakers.items()
        }

    @classmethod
    def reset_all(cls) -> None:
        """Reset all circuit breakers to closed state."""
        for breaker in cls._breakers.values():
            breaker.reset()


class CircuitBreaker:
    """
    Circuit breaker for external API calls.

    Usage:
        @circuit_breaker("resemble_api")
        def call_resemble_enhance(audio_path: Path) -> Path:
            ...

    Or manually:
        breaker = CircuitBreaker("openai_api", config=CircuitBreakerConfig(failure_threshold=5))
        with breaker:
            response = openai.transcribe(...)
    """

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitBreakerState()
        CircuitBreakerRegistry.register(name, self)

    def get_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status."""
        with self._state.lock:
            time_in_state = time.time() - self._state.last_failure_time if self._state.last_failure_time else 0
            return {
                "name": self.name,
                "state": self._state.state.value,
                "failure_count": self._state.failure_count,
                "time_since_last_failure": time_in_state,
                "config": {
                    "failure_threshold": self.config.failure_threshold,
                    "recovery_timeout": self.config.recovery_timeout,
                }
            }

    def reset(self) -> None:
        """Reset circuit to closed state."""
        with self._state.lock:
            self._state.state = CircuitState.CLOSED
            self._state.failure_count = 0
            self._state.half_open_calls = 0
            logger.info(f"Circuit breaker '{self.name}' reset to CLOSED")

    def _should_allow_request(self) -> bool:
        """Check if request should be allowed based on current state."""
        with self._state.lock:
            if self._state.state == CircuitState.CLOSED:
                return True

            if self._state.state == CircuitState.OPEN:
                # Check if recovery timeout has passed
                elapsed = time.time() - self._state.last_failure_time
                if elapsed >= self.config.recovery_timeout:
                    self._state.state = CircuitState.HALF_OPEN
                    self._state.half_open_calls = 0
                    logger.info(f"Circuit breaker '{self.name}' entering HALF_OPEN state")
                    return True
                return False

            if self._state.state == CircuitState.HALF_OPEN:
                if self._state.half_open_calls < self.config.half_open_max_calls:
                    self._state.half_open_calls += 1
                    return True
                return False

            return False

    def _record_success(self) -> None:
        """Record a successful call."""
        with self._state.lock:
            if self._state.state == CircuitState.HALF_OPEN:
                # Success in half-open state closes the circuit
                self._state.state = CircuitState.CLOSED
                self._state.failure_count = 0
                logger.info(f"Circuit breaker '{self.name}' recovered, now CLOSED")
            elif self._state.state == CircuitState.CLOSED:
                # Reset failure count on success
                self._state.failure_count = 0

    def _record_failure(self, exception: Exception) -> None:
        """Record a failed call."""
        # Check if this exception type should be excluded
        if isinstance(exception, self.config.excluded_exceptions):
            logger.debug(f"Circuit breaker '{self.name}' ignoring excluded exception: {type(exception).__name__}")
            return

        with self._state.lock:
            self._state.failure_count += 1
            self._state.last_failure_time = time.time()

            if self._state.state == CircuitState.HALF_OPEN:
                # Failure in half-open state opens the circuit again
                self._state.state = CircuitState.OPEN
                logger.warning(f"Circuit breaker '{self.name}' failed recovery, back to OPEN")
            elif self._state.state == CircuitState.CLOSED:
                if self._state.failure_count >= self.config.failure_threshold:
                    self._state.state = CircuitState.OPEN
                    logger.warning(
                        f"Circuit breaker '{self.name}' OPENED after "
                        f"{self._state.failure_count} failures"
                    )

    def _get_time_until_retry(self) -> float:
        """Get seconds until circuit will try again."""
        with self._state.lock:
            if self._state.state != CircuitState.OPEN:
                return 0.0
            elapsed = time.time() - self._state.last_failure_time
            return max(0.0, self.config.recovery_timeout - elapsed)

    def __enter__(self):
        """Context manager entry - check if request is allowed."""
        if not self._should_allow_request():
            raise CircuitBreakerError(self.name, self._get_time_until_retry())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - record success or failure."""
        if exc_type is None:
            self._record_success()
        else:
            self._record_failure(exc_val)
        return False  # Don't suppress exceptions

    def __call__(self, func: Callable) -> Callable:
        """Decorator usage."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper


# Pre-configured circuit breakers for audio pipeline APIs
def get_resemble_breaker() -> CircuitBreaker:
    """Get or create circuit breaker for Resemble API."""
    existing = CircuitBreakerRegistry.get("resemble_api")
    if existing:
        return existing
    return CircuitBreaker(
        "resemble_api",
        CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=120.0,  # 2 minutes - Resemble can have longer outages
        )
    )


def get_openai_breaker() -> CircuitBreaker:
    """Get or create circuit breaker for OpenAI API."""
    existing = CircuitBreakerRegistry.get("openai_api")
    if existing:
        return existing
    return CircuitBreaker(
        "openai_api",
        CircuitBreakerConfig(
            failure_threshold=5,
            recovery_timeout=60.0,
        )
    )


def get_pyannote_breaker() -> CircuitBreaker:
    """Get or create circuit breaker for Pyannote Precision-2 API."""
    existing = CircuitBreakerRegistry.get("pyannote_api")
    if existing:
        return existing
    return CircuitBreaker(
        "pyannote_api",
        CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout=180.0,  # 3 minutes - diarization is expensive
        )
    )


# Convenience decorator factories
def resemble_circuit(func: Callable) -> Callable:
    """Decorator to protect Resemble API calls."""
    return get_resemble_breaker()(func)


def openai_circuit(func: Callable) -> Callable:
    """Decorator to protect OpenAI API calls."""
    return get_openai_breaker()(func)


def pyannote_circuit(func: Callable) -> Callable:
    """Decorator to protect Pyannote API calls."""
    return get_pyannote_breaker()(func)
