"""Option abstract base class and implementations."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


def normalize_act_output(output: Any) -> tuple[Any, bool]:
    """Normalize option.act() output to (action, done) tuple.

    Allows option.act() to return either:
    - action (assumes done=False)
    - (action, done) tuple

    Args:
        output: Return value from option.act()

    Returns:
        Tuple of (action, done) where done is a boolean

    Examples:
        >>> normalize_act_output(5)
        (5, False)
        >>> normalize_act_output((5, True))
        (5, True)
        >>> normalize_act_output(([0.5, 0.3], False))
        ([0.5, 0.3], False)
    """
    if isinstance(output, tuple) and len(output) == 2:
        action, done = output
        return action, bool(done)
    else:
        # Assume single value is just the action, done=False
        return output, False


class Option(ABC):
    """Abstract base class for Options representing temporal abstractions.

    Options are the core abstraction in SMDPfier, representing sequences of
    actions that can span multiple time steps. Subclasses implement the
    stateful execution pattern.

    The execution lifecycle:
    1. begin(obs, info) - Initialize option state
    2. Loop:
       - act(obs, info) -> (action, done) - Select next action
       - env.step(action) -> (obs', reward, term, trunc, info')
       - on_step(obs', reward, term, trunc, info') - Process step result
       - If done or term or trunc: break
    3. Option completes with aggregated reward and duration=k_exec

    Subclasses must implement: begin, act, on_step, preview, identity
    """

    @abstractmethod
    def begin(self, obs: Any, info: dict[str, Any]) -> None:
        """Initialize option state before execution.

        Called once at the start of option execution with the initial
        observation and info dict.

        Args:
            obs: Initial observation when option begins
            info: Initial info dict when option begins
        """
        pass

    @abstractmethod
    def act(self, obs: Any, info: dict[str, Any]) -> Any | tuple[Any, bool]:
        """Select the next action to execute.

        Called each step with current observation and info. Can return:
        - action: Next action to execute (done=False assumed)
        - (action, done): Next action and whether option should terminate

        Args:
            obs: Current observation
            info: Current info dict

        Returns:
            Either action or (action, done) tuple. Use normalize_act_output
            to handle both cases uniformly.

        Examples:
            >>> def act(self, obs, info):
            ...     return 5  # Action 5, continue
            >>> def act(self, obs, info):
            ...     return (5, True)  # Action 5, then terminate
        """
        raise NotImplementedError

    @abstractmethod
    def on_step(
        self,
        obs: Any,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any]
    ) -> None:
        """Process the result of executing an action.

        Called after each env.step() with the resulting observation,
        reward, termination flags, and info dict. Use this to update
        internal state, collect data, or make decisions.

        Args:
            obs: Observation after action execution
            reward: Reward received from this step
            terminated: Whether episode terminated
            truncated: Whether episode was truncated
            info: Info dict from step
        """
        pass

    @abstractmethod
    def preview(self, obs: Any, info: dict[str, Any]) -> Any | None:
        """Preview the first action without executing.

        Used for action masking in discrete environments. Should return
        the first action that would be executed, or None if unavailable.

        Args:
            obs: Current observation
            info: Current info dict

        Returns:
            First action that would be executed, or None if option
            cannot determine or is unavailable

        Examples:
            >>> option.preview(obs, info)
            5  # Would execute action 5 first
        """
        pass

    @abstractmethod
    def identity(self) -> tuple[str, ...]:
        """Return stable identity tuple for hashing.

        Used to generate stable option IDs across equivalent instances.
        Should return a tuple of strings that uniquely identify this
        option's behavior.

        Returns:
            Tuple of strings representing option identity

        Examples:
            >>> option.identity()
            ('ListOption', '0,1,0', 'left-right-left')
        """
        pass

    @property
    def option_id(self) -> str:
        """Get the stable ID for this option.

        Returns:
            Stable hash-based ID derived from identity()
        """
        return make_option_id_from_identity(self.identity())

    @property
    def name(self) -> str:
        """Get the human-readable name for this option.

        Default implementation uses identity tuple. Subclasses can override.

        Returns:
            Human-readable name string
        """
        # Default: use last element of identity, or "unnamed"
        identity = self.identity()
        return identity[-1] if identity else "unnamed"

    @property
    def meta(self) -> dict[str, Any] | None:
        """Get optional metadata dictionary.

        Default implementation returns None. Subclasses can override
        to provide metadata.

        Returns:
            Optional metadata dict
        """
        return None


@dataclass(frozen=True)
class ListOption(Option):
    """Concrete Option that executes a fixed list of actions.

    Provides backward compatibility with the original Option dataclass
    while implementing the new stateful Option interface.

    Attributes:
        actions: Sequence of primitive actions to execute
        _name: Human-readable name (optional, defaults to auto-generated)
        _meta: Optional metadata dictionary

    Examples:
        >>> # Discrete actions
        >>> option = ListOption([0, 1, 0], name="left-right-left")

        >>> # Continuous actions
        >>> option = ListOption([[0.5, -0.3], [0.0, 1.0]], name="push-pull")

        >>> # With metadata
        >>> option = ListOption(
        ...     [2, 2, 1],
        ...     name="accelerate-brake",
        ...     meta={"category": "driving"}
        ... )
    """

    actions: Sequence[Any]
    _name: str | None = None
    _meta: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Validate option after initialization."""
        if not self.actions:
            raise ValueError("Option actions sequence cannot be empty")

        # Initialize internal state (mutable)
        object.__setattr__(self, '_step_index', 0)

    def begin(self, obs: Any, info: dict[str, Any]) -> None:
        """Initialize option state before execution."""
        object.__setattr__(self, '_step_index', 0)

    def act(self, obs: Any, info: dict[str, Any]) -> Any | tuple[Any, bool]:
        """Select the next action from the list."""
        if self._step_index >= len(self.actions):
            # Shouldn't happen in normal execution, but handle gracefully
            # Return last action and signal done
            return self.actions[-1], True

        action = self.actions[self._step_index]
        done = (self._step_index >= len(self.actions) - 1)
        return action, done

    def on_step(
        self,
        obs: Any,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any]
    ) -> None:
        """Advance to next action in the list."""
        object.__setattr__(self, '_step_index', self._step_index + 1)

    def preview(self, obs: Any, info: dict[str, Any]) -> Any | None:
        """Preview the first action in the list."""
        return self.actions[0] if self.actions else None

    def identity(self) -> tuple[str, ...]:
        """Return stable identity tuple."""
        # Serialize actions
        actions_str = _serialize_actions(self.actions)
        name_str = self._name if self._name else f"list_{actions_str[:8]}"
        return ("ListOption", actions_str, name_str)

    @property
    def name(self) -> str:
        """Get the human-readable name."""
        if self._name:
            return self._name
        # Auto-generate from actions
        return f"list_{len(self.actions)}act"

    @property
    def meta(self) -> dict[str, Any] | None:
        """Get optional metadata."""
        return self._meta

    def __len__(self) -> int:
        """Get the number of primitive actions in this option."""
        return len(self.actions)


def _serialize_action(action: Any) -> str:
    """Serialize a single action to a canonical string."""
    if hasattr(action, "__iter__") and not isinstance(action, (str, bytes)):
        # Handle sequences/arrays by recursively serializing elements
        return f"[{','.join(_serialize_action(item) for item in action)}]"
    else:
        # Handle scalars - convert to string with consistent precision
        return str(action)


def _serialize_actions(actions: Sequence[Any]) -> str:
    """Serialize a sequence of actions to a canonical string."""
    return f"[{','.join(_serialize_action(action) for action in actions)}]"


def make_option_id_from_identity(identity: tuple[str, ...]) -> str:
    """Generate a stable ID from an option identity tuple.

    Args:
        identity: Tuple of strings representing option identity

    Returns:
        Stable hash string (16-char hex digest)

    Examples:
        >>> make_option_id_from_identity(("ListOption", "[0,1]", "test"))
        'a1b2c3d4e5f6...'
    """
    content = "|".join(identity)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]



def make_option_id(actions: Sequence[Any], name: str) -> str:
    """Generate a stable ID for an option based on its actions and name.

    Legacy function for backward compatibility. New code should use
    ListOption and identity-based ID generation.

    Creates a deterministic hash that remains consistent across runs and
    sessions, enabling stable identification of options for logging,
    caching, and analysis purposes.

    Args:
        actions: Sequence of primitive actions
        name: Human-readable name for the option

    Returns:
        Stable hash string (hex digest) representing this option

    Examples:
        >>> option_id = make_option_id([0, 1, 0], "left-right-left")
        >>> print(option_id)  # e.g., "a1b2c3d4e5f6..."

        >>> # Same actions and name always produce same ID
        >>> id1 = make_option_id([0, 1], "test")
        >>> id2 = make_option_id([0, 1], "test")
        >>> assert id1 == id2
    """
    actions_str = _serialize_actions(actions)
    content = f"actions:{actions_str}|name:{name}"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


# Preserve reference to ABC before shadowing with factory function
OptionBase = Option

# Backward compatibility: Allow Option to be called as a factory function
# This mimics the old dataclass constructor behavior
def Option(actions: Sequence[Any], name: str, meta: dict[str, Any] | None = None) -> ListOption:
    """Factory function for creating ListOption instances (backward compatibility).

    This function provides backward compatibility with the old Option dataclass.
    New code should use ListOption directly.

    Args:
        actions: Sequence of primitive actions to execute
        name: Human-readable name for this option
        meta: Optional metadata dictionary

    Returns:
        ListOption instance

    Raises:
        ValueError: If name is empty
        TypeError: If name is not a string

    Examples:
        >>> option = Option([0, 1, 0], "left-right-left")
        >>> option = Option([0, 1], "test", meta={"category": "basic"})
    """
    # Backward compatibility validation (old Option dataclass enforced these)
    if not name:
        raise ValueError("Option name cannot be empty")
    if not isinstance(name, str):
        raise TypeError(f"Option name must be string, got {type(name)}")

    return ListOption(actions=actions, _name=name, _meta=meta)
