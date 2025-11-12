"""Core wrapper functionality tests."""

import gymnasium as gym
import pytest

from smdpfier import Option, SMDPfier
from smdpfier.defaults import ConstantOptionDuration


class TestSMDPfierCore:
    """Test core SMDPfier wrapper functionality."""

    def test_wrapper_initialization(self) -> None:
        """Test that SMDPfier can be initialized with valid arguments."""
        env = gym.make("CartPole-v1")
        options = [Option([0, 1], "test-option")]

        smdp_env = SMDPfier(
            env, options_provider=options, duration_fn=ConstantOptionDuration(5)
        )

        assert smdp_env is not None
        assert smdp_env.action_interface == "index"
        assert smdp_env.time_units == "ticks"

    def test_static_options_provider(self) -> None:
        """Test wrapper with static sequence of options."""
        env = gym.make("CartPole-v1")
        options = [
            Option([0, 1], "left-right"),
            Option([1, 0], "right-left"),
            Option([0, 0, 1], "left-left-right")
        ]

        smdp_env = SMDPfier(
            env,
            options_provider=options,
            duration_fn=ConstantOptionDuration(10)
        )

        assert smdp_env is not None
        assert len(smdp_env._options_provider_raw) == 3
        assert smdp_env._is_static_provider is True

        # Test that get_available_options returns the static options
        obs, info = smdp_env.reset()
        available_options = smdp_env.get_available_options(obs, info)
        assert len(available_options) == 3
        assert available_options[0].name == "left-right"

    def test_dynamic_options_provider(self) -> None:
        """Test wrapper with dynamic callable options provider."""
        env = gym.make("CartPole-v1")

        def dynamic_options(obs, info):
            """Return different options based on observation."""
            return [
                Option([0], "left"),
                Option([1], "right"),
                Option([0, 1], "left-right")
            ]

        smdp_env = SMDPfier(
            env,
            options_provider=dynamic_options,
            duration_fn=ConstantOptionDuration(3),
            max_options=3
        )

        assert smdp_env is not None
        assert callable(smdp_env._options_provider_raw)
        assert smdp_env._is_static_provider is False

        # Test that get_available_options calls the dynamic function
        obs, info = smdp_env.reset()
        available_options = smdp_env.get_available_options(obs, info)
        assert len(available_options) == 3
        assert available_options[0].name == "left"

    def test_index_interface(self) -> None:
        """Test index-based action interface."""
        env = gym.make("CartPole-v1")
        options = [Option([0], "left"), Option([1], "right")]

        smdp_env = SMDPfier(
            env,
            options_provider=options,
            duration_fn=ConstantOptionDuration(5),
            action_interface="index"
        )

        assert smdp_env.action_interface == "index"
        assert isinstance(smdp_env.action_space, gym.spaces.Discrete)
        assert smdp_env.action_space.n == 2

        # Test reset provides action mask
        obs, info = smdp_env.reset()
        assert "smdp" in info
        assert "action_mask" in info["smdp"]
        action_mask = info["smdp"]["action_mask"]
        assert len(action_mask) == 2
        assert action_mask[0]  # First option available
        assert action_mask[1]  # Second option available

        # Test step with index
        obs, reward, terminated, truncated, info = smdp_env.step(0)
        assert "smdp" in info
        assert info["smdp"]["option"]["name"] == "left"

    def test_direct_interface(self) -> None:
        """Test direct Option passing interface."""
        env = gym.make("CartPole-v1")
        options = [Option([0], "left"), Option([1], "right")]

        smdp_env = SMDPfier(
            env,
            options_provider=options,
            duration_fn=ConstantOptionDuration(3),
            action_interface="direct"
        )

        assert smdp_env.action_interface == "direct"
        # For direct interface, action space should be original
        assert smdp_env.get_action_space() == env.action_space

        # Test reset - no action mask for direct interface
        obs, info = smdp_env.reset()
        assert "smdp" in info
        assert info["smdp"]["action_mask"] is None

        # Test step with Option object
        option = Option([1, 1], "double-right")
        obs, reward, terminated, truncated, info = smdp_env.step(option)
        assert "smdp" in info
        assert info["smdp"]["option"]["name"] == "double-right"
        assert info["smdp"]["k_exec"] == 2

    def test_info_payload_structure(self) -> None:
        """Test that info['smdp'] contains all required fields."""
        env = gym.make("CartPole-v1")
        options = [Option([0, 1, 0], "left-right-left")]

        smdp_env = SMDPfier(
            env,
            options_provider=options,
            duration_fn=ConstantOptionDuration(7)
        )

        obs, info = smdp_env.reset()
        obs, reward, terminated, truncated, info = smdp_env.step(0)

        # Check required SMDP info structure
        assert "smdp" in info
        smdp_info = info["smdp"]

        # Check option metadata
        assert "option" in smdp_info
        option_info = smdp_info["option"]
        assert "id" in option_info
        assert "name" in option_info
        assert "len" in option_info
        assert "meta" in option_info
        assert option_info["name"] == "left-right-left"
        assert option_info["len"] == 3

        # Check execution metadata
        assert "k_exec" in smdp_info
        assert "rewards" in smdp_info
        assert "duration_planned" in smdp_info
        assert "duration_exec" in smdp_info
        assert "terminated_early" in smdp_info
        assert "time_units" in smdp_info

        # Check values
        assert smdp_info["k_exec"] == 3
        assert len(smdp_info["rewards"]) == 3
        assert smdp_info["duration_planned"] == 7
        assert smdp_info["duration_exec"] == 7
        assert smdp_info["terminated_early"] is False
        assert smdp_info["time_units"] == "ticks"

        # Check interface-specific fields
        assert "action_mask" in smdp_info  # For index interface
        assert "num_dropped" in smdp_info

    def test_option_execution_metadata(self) -> None:
        """Test that execution metadata is correctly tracked."""
        env = gym.make("CartPole-v1")
        options = [Option([0, 1, 0, 1], "alternating")]

        smdp_env = SMDPfier(
            env,
            options_provider=options,
            duration_fn=ConstantOptionDuration(12)
        )

        obs, info = smdp_env.reset()
        obs, reward, terminated, truncated, info = smdp_env.step(0)

        smdp_info = info["smdp"]

        # Check option identification
        assert smdp_info["option"]["name"] == "alternating"
        assert smdp_info["option"]["len"] == 4
        assert len(smdp_info["option"]["id"]) > 0  # Should have stable ID

        # Check execution counts
        assert smdp_info["k_exec"] == 4  # All 4 actions executed
        assert len(smdp_info["rewards"]) == 4  # One reward per action

        # Check duration metadata
        assert smdp_info["duration_planned"] == 12
        assert smdp_info["duration_exec"] == 12  # Full duration since no early termination
        assert smdp_info["terminated_early"] is False

        # Verify rewards are floats
        for r in smdp_info["rewards"]:
            assert isinstance(r, float)

    def test_reward_aggregation(self) -> None:
        """Test different reward aggregation functions."""
        from smdpfier.defaults.rewards import mean_rewards, sum_rewards

        env = gym.make("CartPole-v1")
        options = [Option([0, 1], "left-right")]

        # Test with sum_rewards (default)
        smdp_env_sum = SMDPfier(
            env,
            options_provider=options,
            duration_fn=ConstantOptionDuration(5),
            reward_agg=sum_rewards
        )

        obs, info = smdp_env_sum.reset()
        obs, reward_sum, terminated, truncated, info = smdp_env_sum.step(0)

        # Should get sum of primitive rewards
        primitive_rewards = info["smdp"]["rewards"]
        expected_sum = sum(primitive_rewards)
        assert abs(reward_sum - expected_sum) < 1e-6

        # Test with mean_rewards
        env2 = gym.make("CartPole-v1")
        smdp_env_mean = SMDPfier(
            env2,
            options_provider=options,
            duration_fn=ConstantOptionDuration(5),
            reward_agg=mean_rewards
        )

        obs, info = smdp_env_mean.reset()
        obs, reward_mean, terminated, truncated, info = smdp_env_mean.step(0)

        # Should get mean of primitive rewards
        primitive_rewards = info["smdp"]["rewards"]
        expected_mean = sum(primitive_rewards) / len(primitive_rewards)
        assert abs(reward_mean - expected_mean) < 1e-6

    def test_early_termination_handling(self) -> None:
        """Test behavior when episode terminates during option execution."""
        # This test is tricky to implement reliably because it depends on
        # the environment terminating mid-option, which is hard to control
        # We'll test the partial duration policy logic instead

        env = gym.make("CartPole-v1")
        long_option = Option([0] * 100, "very-long-left")  # Very long option

        # Test proportional partial duration policy
        smdp_env = SMDPfier(
            env,
            options_provider=[long_option],
            duration_fn=ConstantOptionDuration(100),
            partial_duration_policy="proportional"
        )

        obs, info = smdp_env.reset()

        # Execute the long option - it will likely terminate early
        obs, reward, terminated, truncated, info = smdp_env.step(0)

        smdp_info = info["smdp"]
        k_exec = smdp_info["k_exec"]
        duration_planned = smdp_info["duration_planned"]
        duration_exec = smdp_info["duration_exec"]
        terminated_early = smdp_info["terminated_early"]

        if terminated_early:
            # If terminated early, duration_exec should be proportional
            expected_duration = int(duration_planned * (k_exec / len(long_option.actions)))
            assert duration_exec == expected_duration
            assert k_exec < len(long_option.actions)
        else:
            # If completed, should have full duration
            assert duration_exec == duration_planned
            assert k_exec == len(long_option.actions)

        # The test passes regardless of whether early termination occurred
        # because we're testing the logic in both cases


class TestOptionClass:
    """Test Option dataclass functionality."""

    def test_option_creation(self) -> None:
        """Test Option creation with various action types."""
        # Discrete actions
        opt1 = Option([0, 1, 2], "discrete-test")
        assert len(opt1) == 3
        assert opt1.name == "discrete-test"

        # Continuous actions
        opt2 = Option([[0.5, 0.2], [1.0, -0.3]], "continuous-test")
        assert len(opt2) == 2

        # With metadata
        opt3 = Option([0], "meta-test", meta={"type": "test"})
        assert opt3.meta == {"type": "test"}

    def test_option_id_generation(self) -> None:
        """Test that option IDs are stable and unique."""
        opt1 = Option([0, 1], "test")
        opt2 = Option([0, 1], "test")  # Same content
        opt3 = Option([0, 1], "different")  # Different name
        opt4 = Option([1, 0], "test")  # Different actions

        # Same content should produce same ID
        assert opt1.option_id == opt2.option_id

        # Different content should produce different IDs
        assert opt1.option_id != opt3.option_id
        assert opt1.option_id != opt4.option_id

    def test_option_validation(self) -> None:
        """Test Option validation on creation."""
        # Valid option
        Option([0], "valid")

        # Empty actions should raise error
        with pytest.raises(ValueError, match="actions sequence cannot be empty"):
            Option([], "empty-actions")

        # Empty name should raise error
        with pytest.raises(ValueError, match="name cannot be empty"):
            Option([0], "")

        # Non-string name should raise error
        with pytest.raises(TypeError, match="name must be string"):
            Option([0], 123)
