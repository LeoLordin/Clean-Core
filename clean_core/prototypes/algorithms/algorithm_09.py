"""Bounded sleep replay and anti-livelock rollback for Algorithm 9.

This software prototype uses the Section 20 stratified reservoir helper and a
bounded in-memory checkpoint ring. It does not use or guarantee VRAM.
"""

import copy
import math
import random
from collections import deque
from collections.abc import Callable, Hashable, Iterable, MutableMapping
from dataclasses import dataclass
from typing import Any, TypeVar

from clean_core.prototypes.sections.section_20 import StratifiedReplayReservoir

T = TypeVar("T")
S = TypeVar("S", bound=Hashable)


def _finite_scalar(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


class CheckpointRingBuffer:
    """Keep independent copies of at most ``capacity`` complete states."""

    def __init__(self, capacity: int = 5) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        self.capacity = capacity
        self._checkpoints: deque[MutableMapping[str, Any]] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._checkpoints)

    def push(self, state: MutableMapping[str, Any]) -> None:
        """Save an independent snapshot, including nested state."""
        if not isinstance(state, MutableMapping):
            raise TypeError("checkpoint state must be a mutable mapping")
        self._checkpoints.append(copy.deepcopy(state))

    def latest(self) -> MutableMapping[str, Any]:
        """Return an independent copy of the newest checkpoint."""
        if not self._checkpoints:
            raise ValueError("checkpoint ring buffer is empty")
        return copy.deepcopy(self._checkpoints[-1])


@dataclass(frozen=True)
class WatchdogResult:
    """Watchdog state after a wake-phase derivative observation."""

    rolled_back: bool
    temperature: float
    cooldown: float
    consecutive_resets: int
    last_reset_time: float | None


class AntiLivelockWatchdog:
    """Apply derivative-triggered rollback with exponential cooldown backoff."""

    def __init__(
        self,
        checkpoints: CheckpointRingBuffer,
        impasse_threshold: float = 5.0,
        cooldown_initial: float = 1.0,
        cooldown_max: float = 60.0,
        temperature_max: float = 1.0,
    ) -> None:
        self.checkpoints = checkpoints
        self.impasse_threshold = _finite_scalar(impasse_threshold, "impasse_threshold")
        self.cooldown_initial = _finite_scalar(cooldown_initial, "cooldown_initial")
        self.cooldown_max = _finite_scalar(cooldown_max, "cooldown_max")
        self.temperature_max = _finite_scalar(temperature_max, "temperature_max")
        if self.impasse_threshold < 0.0:
            raise ValueError("impasse_threshold must be non-negative")
        if self.cooldown_initial < 0.0 or self.cooldown_max < self.cooldown_initial:
            raise ValueError(
                "cooldown_max must be at least a non-negative cooldown_initial"
            )
        if self.temperature_max < 0.0:
            raise ValueError("temperature_max must be non-negative")

        self.cooldown = self.cooldown_initial
        self.temperature = self.temperature_max
        self.consecutive_resets = 0
        self.last_reset_time: float | None = None
        self._last_observation_time: float | None = None

    def observe(
        self,
        free_energy_derivative: float,
        timestamp: float,
        current_state: MutableMapping[str, Any],
        environmental_stream: Any,
    ) -> WatchdogResult:
        """Restore a checkpoint in-place when an impasse is eligible.

        A successful rollback replaces all fields with the checkpoint, then
        grounds ``sensory_layer`` from the current stream and sets the
        broadcast ``temperatures`` value. Negative or zero derivatives clear
        the failure counter and restore the base cooldown.
        """
        derivative = _finite_scalar(free_energy_derivative, "free_energy_derivative")
        timestamp = _finite_scalar(timestamp, "timestamp")
        if timestamp < 0.0:
            raise ValueError("timestamp must be non-negative")
        if (
            self._last_observation_time is not None
            and timestamp < self._last_observation_time
        ):
            raise ValueError("timestamp must be non-decreasing")
        if not isinstance(current_state, MutableMapping):
            raise TypeError("current_state must be a mutable mapping")
        self._last_observation_time = timestamp

        if derivative <= 0.0:
            self.consecutive_resets = 0
            self.cooldown = self.cooldown_initial
            return self._result(False)

        cooldown_elapsed = (
            self.last_reset_time is None
            or timestamp - self.last_reset_time >= self.cooldown
        )
        if (
            derivative <= self.impasse_threshold
            or not cooldown_elapsed
            or not self.checkpoints
        ):
            return self._result(False)

        checkpoint = self.checkpoints.latest()
        current_state.clear()
        current_state.update(checkpoint)
        self.temperature = self.temperature_max * (0.8**self.consecutive_resets)
        current_state["sensory_layer"] = copy.deepcopy(environmental_stream)
        current_state["temperatures"] = self.temperature
        self.cooldown = min(self.cooldown_max, 2.0 * self.cooldown)
        self.consecutive_resets += 1
        self.last_reset_time = timestamp
        return self._result(True)

    def _result(self, rolled_back: bool) -> WatchdogResult:
        return WatchdogResult(
            rolled_back=rolled_back,
            temperature=self.temperature,
            cooldown=self.cooldown,
            consecutive_resets=self.consecutive_resets,
            last_reset_time=self.last_reset_time,
        )


class Algorithm09:
    """Compose bounded stratified replay with the wake-phase rollback policy.

    The projection callback reconstructs each sampled episode. The natural
    gradient callback receives all reconstructions and episodes in the batch,
    plus the signed replay step size, and is responsible for applying its
    model-specific update (including any K-FAC factors).
    """

    def __init__(
        self,
        capacity: int,
        clusters: Iterable[S],
        seed: int | None = None,
        checkpoint_capacity: int = 5,
        eta_replay: float = 0.01,
        impasse_threshold: float = 5.0,
        cooldown_initial: float = 1.0,
        cooldown_max: float = 60.0,
        temperature_max: float = 1.0,
    ) -> None:
        eta_replay = _finite_scalar(eta_replay, "eta_replay")
        if eta_replay <= 0.0:
            raise ValueError("eta_replay must be positive")
        self.eta_replay = eta_replay
        self.replay_buffer = StratifiedReplayReservoir(
            capacity=capacity,
            strata=clusters,
            rng=random.Random(seed),
        )
        self.checkpoints = CheckpointRingBuffer(checkpoint_capacity)
        self.watchdog = AntiLivelockWatchdog(
            self.checkpoints,
            impasse_threshold=impasse_threshold,
            cooldown_initial=cooldown_initial,
            cooldown_max=cooldown_max,
            temperature_max=temperature_max,
        )

    def remember(self, cluster: S, episode: T) -> None:
        """Add an episode to its bounded cluster reservoir."""
        self.replay_buffer.add(cluster, episode)

    def save_checkpoint(self, state: MutableMapping[str, Any]) -> None:
        """Save a full state snapshot for a possible wake-phase rollback."""
        self.checkpoints.push(state)

    def sleep(
        self,
        batch_size: int,
        projection_perceptron: Callable[[T], Any],
        natural_gradient_step: Callable[[tuple[Any, ...], tuple[T, ...], float], Any],
    ) -> Any:
        """Replay a balanced, deterministic-with-seed batch and consolidate it."""
        if not callable(projection_perceptron) or not callable(natural_gradient_step):
            raise TypeError("replay callbacks must be callable")
        episodes = tuple(self.replay_buffer.sample(batch_size))
        reconstructions = tuple(projection_perceptron(episode) for episode in episodes)
        return natural_gradient_step(
            reconstructions,
            episodes,
            -self.eta_replay,
        )

    def wake(
        self,
        free_energy_derivative: float,
        timestamp: float,
        current_state: MutableMapping[str, Any],
        environmental_stream: Any,
    ) -> WatchdogResult:
        """Check for an impasse and restore, ground, and cool if eligible."""
        return self.watchdog.observe(
            free_energy_derivative,
            timestamp,
            current_state,
            environmental_stream,
        )


__all__ = [
    "Algorithm09",
    "AntiLivelockWatchdog",
    "CheckpointRingBuffer",
    "WatchdogResult",
]
