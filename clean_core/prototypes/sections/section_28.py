"""Bounded software prototype for Section 28 rollback and annealing.

This models checkpoint selection and watchdog state only; it does not use
VRAM or guarantee prevention of livelock.
"""

import copy
import math
from collections import deque
from dataclasses import dataclass


def _finite_scalar(value, name):
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class WatchdogResult:
    """Outcome and control parameters after one loss observation."""

    rolled_back: bool
    checkpoint: object
    grounded_state: object
    slope: float | None
    temperature: float
    cooldown: float
    consecutive_resets: int


class Section28RollbackWatchdog:
    """Bounded checkpoint rollback with a trend watchdog and cooldown annealing.

    The latest saved checkpoint is restored when the mean loss slope across
    ``window_size`` observations exceeds ``impasse_threshold`` and the
    cooldown has elapsed.
    """

    def __init__(
        self,
        capacity=10_000,
        window_size=5,
        impasse_threshold=2.0,
        cooldown_initial=1.0,
        cooldown_max=60.0,
        temperature_max=1.0,
    ):
        self.capacity = _positive_integer(capacity, "capacity")
        self.window_size = _positive_integer(window_size, "window_size")
        if self.window_size < 2:
            raise ValueError("window_size must be at least 2")

        self.impasse_threshold = _finite_scalar(
            impasse_threshold, "impasse_threshold"
        )
        if self.impasse_threshold < 0.0:
            raise ValueError("impasse_threshold must be non-negative")

        self.cooldown_initial = _finite_scalar(cooldown_initial, "cooldown_initial")
        self.cooldown_max = _finite_scalar(cooldown_max, "cooldown_max")
        if self.cooldown_initial < 0.0 or self.cooldown_max < self.cooldown_initial:
            raise ValueError(
                "cooldown_max must be at least a non-negative cooldown_initial"
            )

        self.temperature_max = _finite_scalar(temperature_max, "temperature_max")
        if self.temperature_max <= 0.0:
            raise ValueError("temperature_max must be positive")

        self.ring_buffer = deque(maxlen=self.capacity)
        self._loss_history = deque(maxlen=self.window_size)
        self.cooldown = self.cooldown_initial
        self.temperature = self.temperature_max
        self.consecutive_resets = 0
        self.last_reset_time = None
        self._last_observation_time = None

    def push_checkpoint(self, state):
        """Save an independent copy of state, retaining at most capacity items."""
        self.ring_buffer.append(copy.deepcopy(state))

    def observe(self, loss, timestamp, grounded_state):
        """Observe one loss value and return a rollback/watchdog result.

        The timestamp must be non-decreasing. A rollback result contains
        independent copies of both the latest checkpoint and grounded state.
        """
        loss = _finite_scalar(loss, "loss")
        timestamp = _finite_scalar(timestamp, "timestamp")
        if timestamp < 0.0:
            raise ValueError("timestamp must be non-negative")
        if (
            self._last_observation_time is not None
            and timestamp < self._last_observation_time
        ):
            raise ValueError("timestamp must be non-decreasing")

        self._last_observation_time = timestamp
        self._loss_history.append(loss)
        if len(self._loss_history) < self.window_size:
            return self._result(False, None, None, None)

        losses = tuple(self._loss_history)
        slope = math.fsum(
            losses[index + 1] - losses[index]
            for index in range(len(losses) - 1)
        ) / (len(losses) - 1)

        if slope <= 0.0:
            self.consecutive_resets = 0
            self.cooldown = self.cooldown_initial
            self.temperature = self.temperature_max
            return self._result(False, None, None, slope)

        cooldown_elapsed = (
            self.last_reset_time is None
            or timestamp - self.last_reset_time >= self.cooldown
        )
        if (
            slope <= self.impasse_threshold
            or not cooldown_elapsed
            or not self.ring_buffer
        ):
            return self._result(False, None, None, slope)

        checkpoint = copy.deepcopy(self.ring_buffer[-1])
        grounded = copy.deepcopy(grounded_state)

        self.consecutive_resets += 1
        self.temperature = self.temperature_max * (0.8 ** self.consecutive_resets)
        self.cooldown = min(self.cooldown_max, 2.0 * self.cooldown)
        self.last_reset_time = timestamp
        return self._result(True, checkpoint, grounded, slope)

    def _result(self, rolled_back, checkpoint, grounded_state, slope):
        return WatchdogResult(
            rolled_back=rolled_back,
            checkpoint=checkpoint,
            grounded_state=grounded_state,
            slope=slope,
            temperature=self.temperature,
            cooldown=self.cooldown,
            consecutive_resets=self.consecutive_resets,
        )


__all__ = ["Section28RollbackWatchdog", "WatchdogResult"]
