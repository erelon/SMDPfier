"""SMDPfier: Main wrapper class for SMDP-level behavior in Gymnasium environments."""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence
from typing import Any, Callable, Literal, SupportsFloat, TypeVar

import gymnasium as gym
import numpy as np

from .errors import SMDPOptionExecutionError, SMDPOptionValidationError
from .option import Option
from .utils import (
    coerce_options_fn,
    create_action_mask,
    summarize_observation,
    truncate_options_with_mask,
)

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class SMDPfier(gym.Wrapper[ObsType, ActType, ObsType, ActType]):
    """Gymnasium wrapper that adds SMDP-level behavior with options and durations.

    SMDPfier enables Semi-Markov Decision Process (SMDP) behavior by allowing users
    to choose Options (chains of primitive actions) while attaching wall-time
    durations (in ticks) per option or per action. The wrapper executes one
    env.step per primitive action and reports planned and executed durations
    so learners can apply true SMDP discounting (e.g., γ^{ticks}).

    The wrapper provides two action interfaces:
    - "index": Exposes Discrete(max_options) with info["action_mask"]
    - "direct": Accepts Option objects directly for scripted control

    Duration metadata is provided in integer ticks and does not control how many
    primitive steps are executed - it's purely for SMDP discounting calculations.

    Attributes:
        action_interface: Current interface mode ("index" or "direct")
        max_options: Maximum number of options for index interface
        time_units: Time units for durations (always "ticks")
    """

    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        *,
        options_provider: Callable[[Any, dict], list[Option]] | Sequence[Option],
        duration_fn: Callable[[Option, Any, dict], int | list[int]],
        reward_agg: Callable[[list[float]], float] = lambda rewards: sum(rewards),
        action_interface: Literal["index", "direct"] = "index",
        max_options: int | None = None,
        availability_fn: Callable[[Any], Iterable[int]] | None = None,
        precheck: bool = False,
        partial_duration_policy: Literal[
            "proportional", "full", "zero"
        ] = "proportional",
        time_units: str = "ticks",
        rng_seed: int | None = None,
    ) -> None:
        """Initialize SMDPfier wrapper.

        Args:
            env: The Gymnasium environment to wrap
            options_provider: **REQUIRED** Either:
                - Callable taking (obs, info) -> list[Option] for dynamic discovery
                - Static sequence of Option objects for fixed catalog
                Internally coerced to a callable for uniform handling.
            duration_fn: **REQUIRED** Function taking (option, obs0, info0) and
                returning either:
                - int: Total planned duration for the whole option (ticks)
                - list[int]: Per-action durations (length must match option length)
                Duration is metadata only and doesn't control execution steps.
            reward_agg: Function to aggregate per-primitive rewards into option reward.
                Default: sum_rewards (simple summation).
            action_interface: Interface for action selection:
                - "index": Expose Discrete(max_options) with action masking
                - "direct": Accept Option objects directly in step()
            max_options: Maximum number of options for index interface. Required
                when action_interface="index" and using dynamic options_provider.
                Overflow handled by truncation policy with num_dropped reporting.
            availability_fn: Optional function taking obs and returning iterable
                of available discrete action indices. Used for action masking in
                discrete environments. Ignored for continuous action spaces.
            precheck: Enable precheck validation before option execution:
                - Discrete: mask-based validation using availability_fn
                - With snapshot support: dry-run validation
                - Graceful fallback: skip validation if not supported
            partial_duration_policy: How to handle duration when option execution
                is interrupted (by termination/truncation):
                - "proportional": Scale duration by executed fraction (default)
                - "full": Report full planned duration regardless of execution
                - "zero": Report zero duration for interrupted execution
            time_units: String metadata for duration units. Always "ticks".
            rng_seed: Random seed for reproducible behavior in stochastic components.
                Used by default option/duration generators and internal sampling.

        Raises:
            ValueError: If required arguments are missing or invalid
            TypeError: If argument types are incorrect

        Examples:
            >>> # Static options with constant durations
            >>> options = [
            ...     Option([0, 0, 1], "left-left-right"),
            ...     Option([1, 1, 0], "right-right-left")
            ... ]
            >>> env = SMDPfier(
            ...     gym.make("CartPole-v1"),
            ...     options_provider=options,
            ...     duration_fn=ConstantOptionDuration(10)
            ... )

            >>> # Dynamic options with per-action durations
            >>> def dynamic_options(obs, info):
            ...     return [Option([act], f"single_{act}") for act in range(4)]
            >>> env = SMDPfier(
            ...     gym.make("Taxi-v3"),
            ...     options_provider=dynamic_options,
            ...     duration_fn=ConstantActionDuration(5),
            ...     max_options=4,
            ...     precheck=True
            ... )
        """
        super().__init__(env)

        # Store configuration
        self._options_provider_raw = options_provider
        self._duration_fn = duration_fn
        self._reward_agg = reward_agg
        self.action_interface = action_interface
        self.max_options = max_options
        self._availability_fn = availability_fn
        self._precheck = precheck
        self._partial_duration_policy = partial_duration_policy
        self.time_units = time_units
        self._rng_seed = rng_seed

        # Validate arguments
        self._validate_init_args()

        # Initialize internal state (implementation pending)
        self._setup_internal_state()

        # Setup action space for index interface
        if action_interface == "index":
            self._setup_index_action_space()

    def _validate_init_args(self) -> None:
        """Validate constructor arguments."""
        # Validate required arguments
        if self._options_provider_raw is None:
            raise ValueError("options_provider is required")
        if self._duration_fn is None:
            raise ValueError("duration_fn is required")

        # Validate action_interface
        if self.action_interface not in ("index", "direct"):
            raise ValueError(f"action_interface must be 'index' or 'direct', got {self.action_interface}")

        # For index interface with non-callable options_provider, max_options not strictly required
        # but for dynamic provider, it is required
        if (self.action_interface == "index" and
            callable(self._options_provider_raw) and
            self.max_options is None):
            raise ValueError("max_options is required for index interface with dynamic options_provider")

        # Validate partial_duration_policy
        if self._partial_duration_policy not in ("proportional", "full", "zero"):
            raise ValueError(f"partial_duration_policy must be 'proportional', 'full', or 'zero', got {self._partial_duration_policy}")

        # Validate time_units
        if self.time_units != "ticks":
            raise ValueError(f"time_units must be 'ticks', got {self.time_units}")

    def _setup_internal_state(self) -> None:
        """Initialize internal state and coerce options provider."""
        # Coerce options provider to callable
        self._options_fn = coerce_options_fn(self._options_provider_raw)

        # Detect if we're in static or dynamic mode
        self._is_static_provider = not callable(self._options_provider_raw)

        # Setup RNG if needed
        if self._rng_seed is not None:
            random.seed(self._rng_seed)
            np.random.seed(self._rng_seed)

        # For static providers, validate max_options
        if self._is_static_provider and self.action_interface == "index":
            static_options = list(self._options_provider_raw)
            if self.max_options is None:
                self.max_options = len(static_options)
            elif self.max_options < len(static_options):
                # Will be truncated, that's fine
                pass

    def _setup_index_action_space(self) -> None:
        """Setup Discrete action space for index interface."""
        if self.max_options is None:
            raise ValueError("max_options must be set for index interface")

        # Replace action space with Discrete for index interface
        self.action_space = gym.spaces.Discrete(self.max_options)

    def get_action_space(self) -> gym.Space[ActType]:
        """Get the action space for the current interface.

        Returns:
            - Discrete(max_options) for index interface
            - Original env action space for direct interface
        """
        if self.action_interface == "index":
            return self.action_space
        else:
            return self.env.action_space

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        """Reset the environment and initialize SMDP state.

        Args:
            seed: Random seed for environment reset
            options: Additional options for environment reset

        Returns:
            Tuple of (observation, info) where info contains initial SMDP metadata
        """
        # Reset underlying environment
        obs, info = self.env.reset(seed=seed, options=options)

        # Track current state
        self._last_obs = obs
        self._last_info = info

        # Add SMDP metadata to info (initial state, no option executed yet)
        if "smdp" not in info:
            info["smdp"] = {}

        # Add action mask for index interface
        if self.action_interface == "index":
            available_options = self.get_available_options(obs, info)
            num_available = len(available_options)

            # Create action mask
            if self.max_options is not None:
                action_mask = np.full(self.max_options, False, dtype=bool)
                for i in range(min(num_available, self.max_options)):
                    action_mask[i] = True
                info["smdp"]["action_mask"] = action_mask

                # Count dropped options - enhance info with action space
                enhanced_info_reset = dict(info)
                enhanced_info_reset["action_space"] = self.env.action_space
                if self._availability_fn is not None:
                    try:
                        available_actions = list(self._availability_fn(obs))
                        enhanced_info_reset["action_mask"] = create_action_mask(available_actions,
                                                                               self.env.action_space.n if hasattr(self.env.action_space, 'n') else len(available_actions))
                    except Exception:
                        pass
                all_options = self._options_fn(obs, enhanced_info_reset)
                _, num_dropped = truncate_options_with_mask(all_options, self.max_options)
                info["smdp"]["num_dropped"] = num_dropped
            else:
                info["smdp"]["action_mask"] = None
                info["smdp"]["num_dropped"] = 0
        else:
            info["smdp"]["action_mask"] = None
            info["smdp"]["num_dropped"] = 0

        return obs, info

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute a single SMDP step (one complete option).

        For index interface, action is an integer index into available options.
        For direct interface, action is an Option object.

        Executes all primitive actions in the selected option sequentially,
        aggregating rewards and tracking execution metadata.

        Args:
            action: Either option index (index interface) or Option object (direct)

        Returns:
            Tuple of (obs, reward, terminated, truncated, info) where:
            - obs: Final observation after option execution
            - reward: Aggregated reward from all primitive steps
            - terminated/truncated: Episode end status
            - info: Contains rich SMDP metadata in info["smdp"]

        Info Structure:
            info["smdp"] contains:
            {
                "option": {
                    "id": str,           # Stable option ID
                    "name": str,         # Human-readable name
                    "len": int,          # Number of primitive actions
                    "meta": dict | None  # Option metadata
                },
                "k_exec": int,              # Primitive steps actually executed
                "rewards": list[float],     # Per-primitive rewards
                "duration_planned": int,    # Planned duration in ticks
                "duration_exec": int,       # Actual duration in ticks
                "terminated_early": bool,   # Whether execution was interrupted
                "action_mask": array | None,  # Available options (index interface only)
                "num_dropped": int,         # Options dropped due to overflow (>=0)
                "time_units": str           # Always "ticks"
            }

        Raises:
            SMDPOptionExecutionError: If option execution fails at runtime
            SMDPOptionValidationError: If precheck validation fails
            ValueError: If action is invalid for current interface
        """
        # Use tracked state for initial observation and info
        obs_for_duration = getattr(self, '_last_obs', None)
        info_for_duration = getattr(self, '_last_info', {})

        # Resolve the option from action
        if self.action_interface == "index":
            # action is an index into available options
            if not isinstance(action, (int, np.integer)):
                raise ValueError(f"For index interface, action must be integer, got {type(action)}")

            # Get current available options
            available_options = self.get_available_options(obs_for_duration, info_for_duration)

            if action < 0 or action >= len(available_options):
                raise ValueError(f"Action index {action} out of range [0, {len(available_options)})")

            option = available_options[action]
        else:
            # Direct interface - action is already an Option
            if not isinstance(action, Option):
                raise ValueError(f"For direct interface, action must be Option, got {type(action)}")
            option = action

        # Precheck validation if enabled
        if self._precheck:
            if not self.validate_option(option, obs_for_duration, info_for_duration):
                raise SMDPOptionValidationError(
                    option_name=option.name,
                    option_id=option.option_id,
                    validation_type="precheck",
                    short_obs_summary=summarize_observation(obs_for_duration)
                )

        # Calculate planned duration
        duration_result = self._duration_fn(option, obs_for_duration, info_for_duration)
        if isinstance(duration_result, int):
            duration_planned = duration_result
            duration_per_action = None
        else:
            # List of per-action durations
            duration_per_action = list(duration_result)
            if len(duration_per_action) != len(option.actions):
                raise ValueError(
                    f"Duration list length {len(duration_per_action)} does not match "
                    f"option actions length {len(option.actions)}"
                )
            duration_planned = sum(duration_per_action)

        # Execute the option
        primitive_rewards = []
        k_exec = 0
        terminated = False
        truncated = False
        final_obs = None
        final_info = None
        current_primitive_action = None

        try:
            for _step_idx, primitive_action in enumerate(option.actions):
                current_primitive_action = primitive_action
                # Execute primitive action
                obs, reward, terminated, truncated, info = self.env.step(primitive_action)
                primitive_rewards.append(float(reward))
                k_exec += 1
                final_obs = obs
                final_info = info

                # Stop if episode ended
                if terminated or truncated:
                    break

        except Exception as e:
            # Wrap in SMDPOptionExecutionError
            obs_summary = summarize_observation(final_obs) if final_obs is not None else "unknown"
            raise SMDPOptionExecutionError(
                option_name=option.name,
                option_id=option.option_id,
                failing_step_index=k_exec,
                action_repr=str(current_primitive_action) if current_primitive_action is not None else "unknown",
                short_obs_summary=obs_summary,
                base_error=e
            )

        # Calculate actual duration executed
        terminated_early = k_exec < len(option.actions)

        if duration_per_action is not None:
            # List-based duration: sum up executed portions
            duration_exec = sum(duration_per_action[:k_exec])
        else:
            # Scalar duration: apply partial policy
            if not terminated_early:
                duration_exec = duration_planned
            else:
                if self._partial_duration_policy == "proportional":
                    fraction = k_exec / len(option.actions)
                    duration_exec = int(duration_planned * fraction)
                elif self._partial_duration_policy == "full":
                    duration_exec = duration_planned
                else:  # "zero"
                    duration_exec = 0

        # Aggregate rewards
        aggregated_reward = self._reward_agg(primitive_rewards)

        # Prepare SMDP info
        smdp_info = {
            "option": {
                "id": option.option_id,
                "name": option.name,
                "len": len(option.actions),
                "meta": option.meta,
            },
            "k_exec": k_exec,
            "rewards": primitive_rewards,
            "duration_planned": duration_planned,
            "duration_exec": duration_exec,
            "terminated_early": terminated_early,
            "time_units": self.time_units,
        }

        # Add action mask for index interface
        if self.action_interface == "index":
            # Get available options for next step
            next_available_options = self.get_available_options(final_obs, final_info)
            num_available = len(next_available_options)

            if self.max_options is not None:
                action_mask = np.full(self.max_options, False, dtype=bool)
                for i in range(min(num_available, self.max_options)):
                    action_mask[i] = True
                smdp_info["action_mask"] = action_mask

                # Count dropped options - enhance info with action space
                enhanced_final_info = dict(final_info)
                enhanced_final_info["action_space"] = self.env.action_space
                if self._availability_fn is not None:
                    try:
                        available_actions = list(self._availability_fn(final_obs))
                        enhanced_final_info["action_mask"] = create_action_mask(available_actions,
                                                                               self.env.action_space.n if hasattr(self.env.action_space, 'n') else len(available_actions))
                    except Exception:
                        pass
                all_next_options = self._options_fn(final_obs, enhanced_final_info)
                _, num_dropped = truncate_options_with_mask(all_next_options, self.max_options)
                smdp_info["num_dropped"] = num_dropped
            else:
                smdp_info["action_mask"] = None
                smdp_info["num_dropped"] = 0
        else:
            smdp_info["action_mask"] = None
            smdp_info["num_dropped"] = 0

        # Add to final info
        final_info["smdp"] = smdp_info

        # Track final state for next step
        self._last_obs = final_obs
        self._last_info = final_info

        return final_obs, aggregated_reward, terminated, truncated, final_info

    def get_available_options(self, obs: ObsType, info: dict[str, Any]) -> list[Option]:
        """Get list of currently available options for given state.

        Args:
            obs: Current environment observation
            info: Current environment info dict

        Returns:
            List of available Option objects (may be truncated if > max_options)
        """
        # Get options from provider - add action space info
        enhanced_info = dict(info)
        enhanced_info["action_space"] = self.env.action_space
        if self._availability_fn is not None:
            try:
                available_actions = list(self._availability_fn(obs))
                enhanced_info["action_mask"] = create_action_mask(available_actions,
                                                                   self.env.action_space.n if hasattr(self.env.action_space, 'n') else len(available_actions))
            except Exception:
                # If availability function fails, don't crash - just continue without mask
                pass

        options = self._options_fn(obs, enhanced_info)

        # Truncate if needed for index interface
        if self.action_interface == "index" and self.max_options is not None:
            options, _ = truncate_options_with_mask(options, self.max_options)

        return options

    def validate_option(
        self, option: Option, obs: ObsType, info: dict[str, Any]
    ) -> bool:
        """Validate whether an option can be executed in the current state.

        Performs precheck validation if enabled, otherwise returns True.

        Args:
            option: Option to validate
            obs: Current environment observation
            info: Current environment info dict

        Returns:
            True if option appears valid, False otherwise

        Raises:
            SMDPOptionValidationError: If validation fails with detailed context
        """
        if not self._precheck:
            return True

        try:
            # Light precheck implementation - just basic validation
            # More sophisticated snapshotting logic would be added by later agents

            # Check if availability_fn is provided for discrete action spaces
            if (self._availability_fn is not None and
                isinstance(self.env.action_space, gym.spaces.Discrete)):

                available_actions = set(self._availability_fn(obs))

                # Check if all actions in the option are available
                for i, action in enumerate(option.actions):
                    if isinstance(action, (int, np.integer)) and action not in available_actions:
                        raise SMDPOptionValidationError(
                            option_name=option.name,
                            option_id=option.option_id,
                            validation_type="mask",
                            failing_step_index=i,
                            action_repr=str(action),
                            short_obs_summary=summarize_observation(obs)
                        )

            # If no availability function or not discrete, assume valid
            return True

        except SMDPOptionValidationError:
            # Re-raise validation errors
            raise
        except Exception as e:
            # Wrap other errors
            raise SMDPOptionValidationError(
                option_name=option.name,
                option_id=option.option_id,
                validation_type="precheck",
                short_obs_summary=summarize_observation(obs),
                base_error=e
            )

    def get_option_duration(
        self, option: Option, obs: ObsType, info: dict[str, Any]
    ) -> int | list[int]:
        """Get planned duration for an option in the current state.

        Args:
            option: Option to get duration for
            obs: Current environment observation
            info: Current environment info dict

        Returns:
            Planned duration as int (total) or list[int] (per-action)
        """
        from .utils import validate_duration_function_output

        # Call duration function
        duration_output = self._duration_fn(option, obs, info)

        # Validate and return
        return validate_duration_function_output(duration_output, option, "duration_fn")

    def _execute_option(
        self, option: Option, obs: ObsType, info: dict[str, Any]
    ) -> tuple[ObsType, list[float], bool, bool, dict[str, Any]]:
        """Execute a single option by running all its primitive actions.

        Internal method that performs the actual option execution, handling
        early termination, reward aggregation, and metadata collection.

        Args:
            option: Option to execute
            obs: Starting observation
            info: Starting info dict

        Returns:
            Tuple of (final_obs, rewards_list, terminated, truncated, smdp_info)
        """
        primitive_rewards = []
        k_exec = 0
        terminated = False
        truncated = False
        final_obs = obs
        final_info = info
        current_primitive_action = None

        try:
            for _step_idx, primitive_action in enumerate(option.actions):
                current_primitive_action = primitive_action
                # Execute primitive action
                obs, reward, terminated, truncated, info = self.env.step(primitive_action)
                primitive_rewards.append(float(reward))
                k_exec += 1
                final_obs = obs
                final_info = info

                # Stop if episode ended
                if terminated or truncated:
                    break

        except Exception as e:
            # Wrap in SMDPOptionExecutionError
            obs_summary = summarize_observation(final_obs) if final_obs is not None else "unknown"
            raise SMDPOptionExecutionError(
                option_name=option.name,
                option_id=option.option_id,
                failing_step_index=k_exec,
                action_repr=str(current_primitive_action) if current_primitive_action is not None else "unknown",
                short_obs_summary=obs_summary,
                base_error=e
            )

        return final_obs, primitive_rewards, terminated, truncated, final_info

    def _compute_executed_duration(
        self,
        planned_duration: int | list[int],
        k_executed: int,
        total_actions: int,
        terminated_early: bool,
    ) -> int:
        """Compute actual duration based on execution and partial policy.

        Args:
            planned_duration: Original planned duration (int or list[int])
            k_executed: Number of actions actually executed
            total_actions: Total number of actions in option
            terminated_early: Whether execution was interrupted

        Returns:
            Executed duration in ticks according to partial_duration_policy
        """
        if isinstance(planned_duration, list):
            # List-based duration: sum executed portions
            return sum(planned_duration[:k_executed])
        else:
            # Scalar duration: apply partial policy
            if not terminated_early and k_executed == total_actions:
                # Full execution completed
                return planned_duration
            else:
                # Partial execution - apply policy
                if self._partial_duration_policy == "proportional":
                    # Proportional to fraction executed
                    fraction = k_executed / max(1, total_actions)
                    return round(planned_duration * fraction)
                elif self._partial_duration_policy == "full":
                    # Always return full duration
                    return planned_duration
                else:  # "zero"
                    # Return zero duration for partial execution
                    return 0

    # Helper methods for state tracking
    def _get_current_available_options(self) -> list[Option]:
        """Get available options for current state."""
        obs = getattr(self, '_last_obs', None)
        info = getattr(self, '_last_info', {})

        if obs is not None:
            return self.get_available_options(obs, info)
        else:
            # Fallback for cases where state is not available
            if self._is_static_provider:
                static_options = list(self._options_provider_raw)
                if self.action_interface == "index" and self.max_options is not None:
                    static_options, _ = truncate_options_with_mask(static_options, self.max_options)
                return static_options
            else:
                # For dynamic providers, we need state - this is a limitation
                return []

    def _get_last_observation(self):
        """Get last observation."""
        return getattr(self, '_last_obs', None)

    def _get_last_info(self):
        """Get last info dict."""
        return getattr(self, '_last_info', {})

