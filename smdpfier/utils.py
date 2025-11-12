"""Utility functions for SMDPfier operations."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, Callable

import numpy as np

from .option import Option


def coerce_options_fn(
    options_provider: Callable[[Any, dict], list[Option]] | Sequence[Option],
) -> Callable[[Any, dict], list[Option]]:
    """Coerce options provider to a callable function.

    Converts static sequences of options into a callable that returns
    the same options regardless of observation and info. This allows
    uniform handling of both static and dynamic option providers.

    Args:
        options_provider: Either a callable that takes (obs, info) and returns
            a list of Options, or a static sequence of Options.

    Returns:
        A callable that takes (obs, info) and returns a list of Options.

    Examples:
        >>> # Static options
        >>> static_opts = [Option([0, 1], "test")]
        >>> fn = coerce_options_fn(static_opts)
        >>> opts = fn(obs={}, info={})  # Returns the static options

        >>> # Dynamic options (already callable)
        >>> def dynamic_opts(obs, info):
        ...     return [Option([obs['action']], "dynamic")]
        >>> fn = coerce_options_fn(dynamic_opts)  # Returns as-is
    """
    if callable(options_provider):
        # Already a callable, return as-is
        return options_provider
    else:
        # Convert static sequence to callable
        static_options = list(options_provider)  # Convert to list to ensure it's a sequence
        return lambda obs, info: static_options


def create_action_mask(
    available_actions: Iterable[int], max_actions: int
) -> np.ndarray:
    """Create a boolean mask for available discrete actions.

    Args:
        available_actions: Iterable of valid action indices
        max_actions: Total number of possible actions

    Returns:
        Boolean numpy array where True indicates available actions

    Examples:
        >>> mask = create_action_mask([0, 2, 4], max_actions=5)
        >>> print(mask)  # [True, False, True, False, True]
    """
    mask = np.full(max_actions, False, dtype=bool)
    for action_idx in available_actions:
        if 0 <= action_idx < max_actions:
            mask[action_idx] = True
    return mask


def truncate_options_with_mask(
    options: list[Option], max_options: int
) -> tuple[list[Option], int]:
    """Truncate options list and return number of dropped options.

    Args:
        options: List of available options
        max_options: Maximum number of options to keep

    Returns:
        Tuple of (truncated_options_list, num_dropped)

    Examples:
        >>> opts = [Option([0], f"opt_{i}") for i in range(10)]
        >>> truncated, dropped = truncate_options_with_mask(opts, 5)
        >>> len(truncated) == 5 and dropped == 5
        True
    """
    if len(options) <= max_options:
        return options, 0

    truncated = options[:max_options]
    num_dropped = len(options) - max_options
    return truncated, num_dropped


def summarize_observation(obs: Any, max_length: int = 100) -> str:
    """Create a brief string summary of an observation.

    Produces a concise representation of the observation for error messages
    and logging, handling various observation types (arrays, dicts, etc.).

    Args:
        obs: Environment observation of any type
        max_length: Maximum length of the summary string

    Returns:
        Brief string representation of the observation

    Examples:
        >>> obs = np.array([1, 2, 3, 4, 5])
        >>> summary = summarize_observation(obs, max_length=50)
        >>> print(summary)  # e.g., "array(5,) [1 2 3 4 5]"
    """
    try:
        if isinstance(obs, np.ndarray):
            shape_str = f"array{obs.shape}"
            if obs.size <= 10:
                content_str = f" {obs}"
            else:
                content_str = f" [{obs.flat[0]}...{obs.flat[-1]}]"
            summary = f"{shape_str}{content_str}"
        elif isinstance(obs, dict):
            keys = list(obs.keys())[:3]  # Show first 3 keys
            more = "..." if len(obs) > 3 else ""
            summary = f"dict({len(obs)}) {{{', '.join(str(k) for k in keys)}{more}}}"
        elif isinstance(obs, (list, tuple)):
            type_name = type(obs).__name__
            if len(obs) <= 5:
                summary = f"{type_name}({len(obs)}) {obs}"
            else:
                summary = f"{type_name}({len(obs)}) [{obs[0]}...{obs[-1]}]"
        else:
            summary = f"{type(obs).__name__}: {obs}"

        # Truncate to max_length
        if len(summary) > max_length:
            summary = summary[:max_length-3] + "..."

        return summary
    except Exception:
        # Fallback if summarization fails
        return f"{type(obs).__name__} (summary failed)"


def validate_duration_function_output(
    duration_output: Any, option: Option, context: str = "duration_fn"
) -> int | list[int]:
    """Validate and normalize duration function output.

    Ensures the duration function returns valid int or list[int] and
    that list lengths match option length when applicable.

    Args:
        duration_output: Raw output from duration function
        option: The option being processed
        context: Context string for error messages

    Returns:
        Validated duration as int or list[int]

    Raises:
        TypeError: If output type is invalid
        ValueError: If list length doesn't match option length
    """
    # Check for int (scalar duration)
    if isinstance(duration_output, int):
        if duration_output < 0:
            # Negative durations are technically allowed but not recommended
            pass
        return duration_output

    # Check for list of ints (per-action duration)
    if isinstance(duration_output, (list, tuple)):
        duration_list = list(duration_output)

        # Validate all elements are integers
        for i, d in enumerate(duration_list):
            if not isinstance(d, int):
                raise TypeError(
                    f"{context} returned list with non-integer element at index {i}: "
                    f"got {type(d).__name__}, expected int"
                )

        # Validate list length matches option length
        if len(duration_list) != len(option.actions):
            raise ValueError(
                f"{context} returned list of length {len(duration_list)} but option "
                f"'{option.name}' has {len(option.actions)} actions"
            )

        return duration_list

    # Invalid type
    raise TypeError(
        f"{context} must return int or list[int], got {type(duration_output).__name__}"
    )
