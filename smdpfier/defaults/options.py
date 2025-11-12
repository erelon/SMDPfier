"""Default option generators for SMDPfier."""

from __future__ import annotations

from typing import Any

from ..option import Option


class RandomStaticLen:
    """Generate random options with fixed length.

    Creates options by randomly sampling actions from the environment's
    action space, with all generated options having the same length.

    Examples:
        >>> generator = RandomStaticLen(length=3, action_space_size=2)
        >>> options = generator(obs, info)  # Returns options like [0,1,0], [1,1,0], etc.
    """

    def __init__(
        self,
        length: int,
        action_space_size: int | None = None,
        num_options: int = 10,
        rng_seed: int | None = None,
    ) -> None:
        """Initialize random static length option generator.

        Args:
            length: Fixed length for all generated options
            action_space_size: Size of discrete action space (auto-detected if None)
            num_options: Number of options to generate
            rng_seed: Random seed for reproducible generation
        """
        self.length = length
        self.action_space_size = action_space_size
        self.num_options = num_options
        self.rng_seed = rng_seed

    def __call__(self, obs: Any, info: dict) -> list[Option]:
        """Generate random options with fixed length.

        Args:
            obs: Environment observation (used for action space detection)
            info: Environment info dict

        Returns:
            List of randomly generated Option objects
        """
        import random

        import gymnasium as gym

        # Set up random state
        rng = random.Random(self.rng_seed)

        # Get action space size from info if available, fallback to auto-detection
        action_space_size = self.action_space_size
        if action_space_size is None:
            # Try to get from env action space (passed in info)
            if "action_space" in info:
                action_space = info["action_space"]
                if isinstance(action_space, gym.spaces.Discrete):
                    action_space_size = action_space.n
                else:
                    raise ValueError(f"Cannot auto-detect action space size for {type(action_space)}")
            else:
                raise ValueError("action_space_size must be provided or action_space must be in info")

        # Get availability mask if provided
        availability_mask = info.get("action_mask")
        if availability_mask is not None:
            available_actions = [i for i, available in enumerate(availability_mask) if available]
            if not available_actions:
                # No actions available, return empty list
                return []
        else:
            available_actions = list(range(action_space_size))

        options = []
        for i in range(self.num_options):
            # Generate random action sequence
            actions = [rng.choice(available_actions) for _ in range(self.length)]
            option_name = f"random_static_{self.length}_{i}"
            options.append(Option(actions, option_name))

        return options


class RandomVarLen:
    """Generate random options with variable length.

    Creates options by randomly sampling actions from the environment's
    action space, with option lengths varying between min_length and max_length.

    Examples:
        >>> generator = RandomVarLen(min_length=2, max_length=5, action_space_size=4)
        >>> options = generator(obs, info)  # Returns options of varying lengths 2-5
    """

    def __init__(
        self,
        min_length: int,
        max_length: int,
        action_space_size: int | None = None,
        num_options: int = 10,
        rng_seed: int | None = None,
    ) -> None:
        """Initialize random variable length option generator.

        Args:
            min_length: Minimum length for generated options
            max_length: Maximum length for generated options (inclusive)
            action_space_size: Size of discrete action space (auto-detected if None)
            num_options: Number of options to generate
            rng_seed: Random seed for reproducible generation
        """
        if min_length < 1:
            raise ValueError("min_length must be at least 1")
        if max_length < min_length:
            raise ValueError("max_length must be >= min_length")

        self.min_length = min_length
        self.max_length = max_length
        self.action_space_size = action_space_size
        self.num_options = num_options
        self.rng_seed = rng_seed

    def __call__(self, obs: Any, info: dict) -> list[Option]:
        """Generate random options with variable length.

        Args:
            obs: Environment observation (used for action space detection)
            info: Environment info dict

        Returns:
            List of randomly generated Option objects with varying lengths
        """
        import random

        import gymnasium as gym

        # Set up random state
        rng = random.Random(self.rng_seed)

        # Get action space size from info if available, fallback to auto-detection
        action_space_size = self.action_space_size
        if action_space_size is None:
            # Try to get from env action space (passed in info)
            if "action_space" in info:
                action_space = info["action_space"]
                if isinstance(action_space, gym.spaces.Discrete):
                    action_space_size = action_space.n
                else:
                    raise ValueError(f"Cannot auto-detect action space size for {type(action_space)}")
            else:
                raise ValueError("action_space_size must be provided or action_space must be in info")

        # Get availability mask if provided
        availability_mask = info.get("action_mask")
        if availability_mask is not None:
            available_actions = [i for i, available in enumerate(availability_mask) if available]
            if not available_actions:
                # No actions available, return empty list
                return []
        else:
            available_actions = list(range(action_space_size))

        options = []
        for i in range(self.num_options):
            # Generate random length
            length = rng.randint(self.min_length, self.max_length)
            # Generate random action sequence
            actions = [rng.choice(available_actions) for _ in range(length)]
            option_name = f"random_var_{length}_{i}"
            options.append(Option(actions, option_name))

        return options
