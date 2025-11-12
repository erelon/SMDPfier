"""Option dataclass and stable ID generation."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Option:
    """An Option represents a sequence of primitive actions to execute.

    Options are the core abstraction in SMDPfier, representing temporal
    abstractions that can span multiple time steps. Each option contains
    a sequence of actions to execute sequentially in the environment.

    Attributes:
        actions: Sequence of primitive actions to execute. Can be any type
            supported by the underlying environment (int, float, numpy arrays, etc.).
        name: Human-readable name for this option. Used for logging, debugging,
            and stable ID generation. Should be descriptive and unique.
        meta: Optional metadata dictionary for additional information about
            the option (e.g., expected rewards, difficulty, category).

    Examples:
        >>> # Discrete actions
        >>> option = Option(actions=[0, 1, 0], name="left-right-left")

        >>> # Continuous actions
        >>> option = Option(actions=[[0.5, -0.3], [0.0, 1.0]], name="push-pull")

        >>> # With metadata
        >>> option = Option(
        ...     actions=[2, 2, 1],
        ...     name="accelerate-brake",
        ...     meta={"category": "driving", "difficulty": 0.7}
        ... )
    """

    actions: Sequence[Any]
    name: str
    meta: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Validate option after initialization."""
        if not self.actions:
            raise ValueError("Option actions sequence cannot be empty")
        if not self.name:
            raise ValueError("Option name cannot be empty")
        if not isinstance(self.name, str):
            raise TypeError(f"Option name must be string, got {type(self.name)}")

    @property
    def option_id(self) -> str:
        """Get the stable ID for this option.

        Returns:
            Stable hash-based ID derived from actions content and name.
        """
        return make_option_id(self.actions, self.name)

    def __len__(self) -> int:
        """Get the number of primitive actions in this option."""
        return len(self.actions)


def make_option_id(actions: Sequence[Any], name: str) -> str:
    """Generate a stable ID for an option based on its actions and name.

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

    # Convert actions to a canonical string representation
    # Handle various action types (int, float, sequences, arrays)
    def _serialize_action(action: Any) -> str:
        if hasattr(action, "__iter__") and not isinstance(action, (str, bytes)):
            # Handle sequences/arrays by recursively serializing elements
            return f"[{','.join(_serialize_action(item) for item in action)}]"
        else:
            # Handle scalars - convert to string with consistent precision
            return str(action)

    actions_str = f"[{','.join(_serialize_action(action) for action in actions)}]"

    # Create content string for hashing
    content = f"actions:{actions_str}|name:{name}"

    # Generate stable hash
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
