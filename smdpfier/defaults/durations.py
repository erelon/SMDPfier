"""Default duration providers for SMDPfier."""

from __future__ import annotations

from typing import Any, Callable

from ..option import Option


class ConstantOptionDuration:
    """Provide constant duration for all options (scalar output).

    Returns the same total duration value for every option, regardless
    of option length or content. Represents total planned duration.

    Examples:
        >>> duration_fn = ConstantOptionDuration(10)
        >>> duration = duration_fn(option, obs, info)  # Always returns 10
    """

    def __init__(self, duration: int) -> None:
        """Initialize constant option duration provider.

        Args:
            duration: Fixed duration value to return for all options (ticks)
        """
        self.duration = duration

    def __call__(self, option: Option, obs: Any, info: dict) -> int:
        """Return constant duration for any option.

        Args:
            option: The option (ignored)
            obs: Environment observation (ignored)
            info: Environment info (ignored)

        Returns:
            The constant duration value
        """
        return self.duration


class RandomOptionDuration:
    """Provide random duration for each option (scalar output).

    Samples a random total duration from a specified range for each option.
    Different calls with the same option may return different values unless
    seeded deterministically.

    Examples:
        >>> duration_fn = RandomOptionDuration(min_duration=5, max_duration=15)
        >>> duration = duration_fn(option, obs, info)  # Returns 5-15 randomly
    """

    def __init__(
        self, min_duration: int, max_duration: int, rng_seed: int | None = None
    ) -> None:
        """Initialize random option duration provider.

        Args:
            min_duration: Minimum duration value (inclusive)
            max_duration: Maximum duration value (inclusive)
            rng_seed: Random seed for reproducible sampling
        """
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.rng_seed = rng_seed

    def __call__(self, option: Option, obs: Any, info: dict) -> int:
        """Return random duration for the option.

        Args:
            option: The option (may be used for seeding)
            obs: Environment observation (ignored)
            info: Environment info (ignored)

        Returns:
            Randomly sampled duration value
        """
        import random
        if self.rng_seed is not None:
            # Use option ID for deterministic seeding
            seed = hash((self.rng_seed, option.option_id)) % (2**32)
            rng = random.Random(seed)
            return rng.randint(self.min_duration, self.max_duration)
        else:
            return random.randint(self.min_duration, self.max_duration)


class ConstantActionDuration:
    """Provide constant per-action durations (list output).

    Returns a list where each action in the option has the same duration.
    List length equals option length, enabling per-step duration tracking.

    Examples:
        >>> duration_fn = ConstantActionDuration(5)
        >>> duration = duration_fn(option_len_3, obs, info)  # Returns [5, 5, 5]
    """

    def __init__(self, duration_per_action: int) -> None:
        """Initialize constant per-action duration provider.

        Args:
            duration_per_action: Fixed duration value per action (ticks)
        """
        self.duration_per_action = duration_per_action

    def __call__(self, option: Option, obs: Any, info: dict) -> list[int]:
        """Return constant duration for each action in the option.

        Args:
            option: The option containing the action sequence
            obs: Environment observation (ignored)
            info: Environment info (ignored)

        Returns:
            List of duration values, one per action
        """
        return [self.duration_per_action] * len(option.actions)


class RandomActionDuration:
    """Provide random per-action durations (list output).

    Returns a list where each action gets a random duration sampled
    from the specified range. List length equals option length.

    Examples:
        >>> duration_fn = RandomActionDuration(min_duration=3, max_duration=7)
        >>> duration = duration_fn(option_len_3, obs, info)  # Returns [5, 3, 7] etc.
    """

    def __init__(
        self, min_duration: int, max_duration: int, rng_seed: int | None = None
    ) -> None:
        """Initialize random per-action duration provider.

        Args:
            min_duration: Minimum duration value per action (inclusive)
            max_duration: Maximum duration value per action (inclusive)
            rng_seed: Random seed for reproducible sampling
        """
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.rng_seed = rng_seed

    def __call__(self, option: Option, obs: Any, info: dict) -> list[int]:
        """Return random duration for each action in the option.

        Args:
            option: The option containing the action sequence
            obs: Environment observation (ignored)
            info: Environment info (ignored)

        Returns:
            List of randomly sampled duration values, one per action
        """
        import random

        if self.rng_seed is not None:
            # Use option ID for deterministic seeding
            seed = hash((self.rng_seed, option.option_id)) % (2**32)
            rng = random.Random(seed)
        else:
            rng = random

        return [
            rng.randint(self.min_duration, self.max_duration)
            for _ in range(len(option.actions))
        ]


class MapActionDuration:
    """Provide duration by mapping actions to durations using a function.

    Applies a user-provided function to each action to determine its duration.
    Useful for domain-specific duration logic.

    Examples:
        >>> def custom_duration(action):
        ...     return action * 2 + 1  # Duration depends on action value
        >>> duration_fn = MapActionDuration(custom_duration)
        >>> duration = duration_fn(option, obs, info)  # Returns mapped durations
    """

    def __init__(self, mapping_fn: Callable[[Any], int]) -> None:
        """Initialize function-based action duration provider.

        Args:
            mapping_fn: Function taking an action and returning its duration (ticks)
        """
        self.mapping_fn = mapping_fn

    def __call__(self, option: Option, obs: Any, info: dict) -> list[int]:
        """Return duration for each action by applying the mapping function.

        Args:
            option: The option containing the action sequence
            obs: Environment observation (ignored)
            info: Environment info (ignored)

        Returns:
            List of duration values computed by mapping function
        """
        return [self.mapping_fn(action) for action in option.actions]
        """Return mapped duration list for each action in option.

        Args:
            option: The option (actions are mapped)
            obs: Environment observation (passed to mapping_fn if needed)
            info: Environment info (passed to mapping_fn if needed)

        Returns:
            List of mapped durations with length = len(option.actions)
        """
        return [self.mapping_fn(action) for action in option.actions]
