"""Duration function tests."""

import gymnasium as gym

from smdpfier import SMDPfier, Option
from smdpfier.defaults.durations import (
    ConstantOptionDuration,
    RandomOptionDuration,
    ConstantActionDuration,
    RandomActionDuration,
    MapActionDuration,
)


class TestDurationFunctions:
    """Test default duration provider functions."""

    def test_constant_option_duration(self) -> None:
        """Test ConstantOptionDuration returns same value for all options."""
        duration_fn = ConstantOptionDuration(10)

        # Test with different options and observations
        option1 = Option([0, 1], "test1")
        option2 = Option([1, 0, 1, 0], "test2")
        obs1 = [0.1, 0.2, 0.3, 0.4]
        obs2 = [-0.5, 1.0, -0.2, 0.8]

        assert duration_fn(option1, obs1, {}) == 10
        assert duration_fn(option1, obs2, {}) == 10
        assert duration_fn(option2, obs1, {}) == 10
        assert duration_fn(option2, obs2, {}) == 10

    def test_random_option_duration(self) -> None:
        """Test RandomOptionDuration returns values in specified range."""
        # Test seeded version for deterministic behavior
        duration_fn = RandomOptionDuration(min_duration=5, max_duration=15, rng_seed=42)
        option = Option([0, 1, 0], "test")
        obs = [0.0, 0.0, 0.0, 0.0]

        # Call multiple times with same option - should get same result due to seeding
        result1 = duration_fn(option, obs, {})
        result2 = duration_fn(option, obs, {})
        assert result1 == result2
        assert 5 <= result1 <= 15

        # Test with different option - should get different result
        option2 = Option([1, 0, 1], "test2")
        result3 = duration_fn(option2, obs, {})
        assert 5 <= result3 <= 15
        # Due to different option IDs, results may differ

        # Test unseeded version gives values in range
        unseeded_fn = RandomOptionDuration(min_duration=3, max_duration=7)
        results = [unseeded_fn(option, obs, {}) for _ in range(20)]
        assert all(3 <= r <= 7 for r in results)
        # Should have some variation
        assert len(set(results)) > 1

    def test_constant_action_duration(self) -> None:
        """Test ConstantActionDuration returns list matching option length."""
        duration_fn = ConstantActionDuration(5)

        # Test with different option lengths
        option1 = Option([0], "single")
        option2 = Option([0, 1, 0], "triple")
        option3 = Option([1, 0, 1, 0, 1], "five")
        obs = [0.0, 0.0, 0.0, 0.0]

        result1 = duration_fn(option1, obs, {})
        result2 = duration_fn(option2, obs, {})
        result3 = duration_fn(option3, obs, {})

        assert result1 == [5]
        assert result2 == [5, 5, 5]
        assert result3 == [5, 5, 5, 5, 5]
        assert len(result1) == 1
        assert len(result2) == 3
        assert len(result3) == 5

    def test_random_action_duration(self) -> None:
        """Test RandomActionDuration returns valid list of durations."""
        # Test seeded version
        duration_fn = RandomActionDuration(min_duration=2, max_duration=8, rng_seed=42)
        option = Option([0, 1, 0], "test")
        obs = [0.0, 0.0, 0.0, 0.0]

        result = duration_fn(option, obs, {})
        assert len(result) == 3  # Same as option length
        assert all(2 <= d <= 8 for d in result)
        assert all(isinstance(d, int) for d in result)

        # Test deterministic behavior with same inputs
        result2 = duration_fn(option, obs, {})
        assert result == result2

        # Test with different option length
        option2 = Option([1, 0, 1, 0, 1, 0], "longer")
        result3 = duration_fn(option2, obs, {})
        assert len(result3) == 6
        assert all(2 <= d <= 8 for d in result3)

    def test_map_action_duration(self) -> None:
        """Test MapActionDuration applies mapping function correctly."""
        def custom_mapping(action):
            return action * 2 + 1

        duration_fn = MapActionDuration(custom_mapping)
        option = Option([0, 1, 2], "test")
        obs = [0.0, 0.0, 0.0, 0.0]

        result = duration_fn(option, obs, {})
        expected = [0*2+1, 1*2+1, 2*2+1]  # [1, 3, 5]
        assert result == expected

        # Test with different mapping
        def action_cost_mapping(action):
            costs = {0: 2, 1: 5, 2: 3}
            return costs.get(action, 4)

        duration_fn2 = MapActionDuration(action_cost_mapping)
        result2 = duration_fn2(option, obs, {})
        assert result2 == [2, 5, 3]

    def test_duration_output_validation(self) -> None:
        """Test that duration functions return valid int or list[int]."""
        option = Option([0, 1], "test")
        obs = [0.0, 0.0, 0.0, 0.0]

        # Scalar duration functions
        constant_opt = ConstantOptionDuration(10)
        random_opt = RandomOptionDuration(5, 15, rng_seed=42)

        scalar_result1 = constant_opt(option, obs, {})
        scalar_result2 = random_opt(option, obs, {})

        assert isinstance(scalar_result1, int)
        assert isinstance(scalar_result2, int)

        # List duration functions
        constant_act = ConstantActionDuration(3)
        random_act = RandomActionDuration(1, 5, rng_seed=42)
        map_act = MapActionDuration(lambda x: x + 2)

        list_result1 = constant_act(option, obs, {})
        list_result2 = random_act(option, obs, {})
        list_result3 = map_act(option, obs, {})

        assert isinstance(list_result1, list)
        assert isinstance(list_result2, list)
        assert isinstance(list_result3, list)
        assert all(isinstance(d, int) for d in list_result1)
        assert all(isinstance(d, int) for d in list_result2)
        assert all(isinstance(d, int) for d in list_result3)


class TestPartialDurationPolicies:
    """Test partial duration policy calculations."""

    def test_proportional_policy(self) -> None:
        """Test proportional partial duration calculation."""
        # Use CartPole environment that can terminate early
        env = gym.make("CartPole-v1")

        # Create a long option that will likely cause termination
        long_option = Option([0] * 20, "long-left-sequence")

        smdp_env = SMDPfier(
            env,
            options_provider=[long_option],
            duration_fn=ConstantOptionDuration(30),
            partial_duration_policy="proportional",
            action_interface="index",
            max_options=1
        )

        obs, info = smdp_env.reset(seed=42)
        obs, reward, term, trunc, info = smdp_env.step(0)

        smdp_info = info["smdp"]
        if smdp_info["terminated_early"]:
            # Proportional: duration_exec should be (k_exec / total_actions) * planned_duration
            expected_duration = int((smdp_info["k_exec"] / len(long_option.actions)) * 30)
            assert smdp_info["duration_exec"] == expected_duration
            assert smdp_info["duration_planned"] == 30

    def test_full_policy(self) -> None:
        """Test full partial duration calculation."""
        env = gym.make("CartPole-v1")

        long_option = Option([0] * 15, "long-left-sequence")

        smdp_env = SMDPfier(
            env,
            options_provider=[long_option],
            duration_fn=ConstantOptionDuration(25),
            partial_duration_policy="full",
            action_interface="index",
            max_options=1
        )

        obs, info = smdp_env.reset(seed=42)
        obs, reward, term, trunc, info = smdp_env.step(0)

        smdp_info = info["smdp"]
        if smdp_info["terminated_early"]:
            # Full policy: should always return planned duration
            assert smdp_info["duration_exec"] == 25
            assert smdp_info["duration_planned"] == 25

    def test_zero_policy(self) -> None:
        """Test zero partial duration calculation."""
        env = gym.make("CartPole-v1")

        long_option = Option([0] * 15, "long-left-sequence")

        smdp_env = SMDPfier(
            env,
            options_provider=[long_option],
            duration_fn=ConstantOptionDuration(20),
            partial_duration_policy="zero",
            action_interface="index",
            max_options=1
        )

        obs, info = smdp_env.reset(seed=42)
        obs, reward, term, trunc, info = smdp_env.step(0)

        smdp_info = info["smdp"]
        if smdp_info["terminated_early"]:
            # Zero policy: should return 0 duration when terminated early
            assert smdp_info["duration_exec"] == 0
            assert smdp_info["duration_planned"] == 20

    def test_partial_policy_with_list_durations(self) -> None:
        """Test partial policies work with per-action duration lists."""
        env = gym.make("CartPole-v1")

        long_option = Option([0] * 10, "long-sequence")

        smdp_env = SMDPfier(
            env,
            options_provider=[long_option],
            duration_fn=ConstantActionDuration(3),  # Each action takes 3 ticks
            partial_duration_policy="proportional",
            action_interface="index",
            max_options=1
        )

        obs, info = smdp_env.reset(seed=42)
        obs, reward, term, trunc, info = smdp_env.step(0)

        smdp_info = info["smdp"]
        if smdp_info["terminated_early"]:
            # With list durations, proportional should sum up executed portions
            expected_duration = smdp_info["k_exec"] * 3  # Each executed action contributes 3 ticks
            assert smdp_info["duration_exec"] == expected_duration
            assert smdp_info["duration_planned"] == 10 * 3  # 10 actions * 3 ticks each

    def test_integration_with_wrapper(self) -> None:
        """Test duration integration with SMDPfier wrapper for CartPole."""
        env = gym.make("CartPole-v1")

        # Static options for CartPole
        options = [
            Option([0, 1], "left-right"),
            Option([1, 0], "right-left"),
            Option([0, 0, 1], "left-left-right")
        ]

        smdp_env = SMDPfier(
            env,
            options_provider=options,
            duration_fn=ConstantOptionDuration(5),
            action_interface="index",
            max_options=len(options)
        )

        obs, info = smdp_env.reset(seed=42)
        obs, reward, term, trunc, info = smdp_env.step(0)  # Execute "left-right"

        smdp_info = info["smdp"]
        assert smdp_info["duration_planned"] == 5
        if not smdp_info["terminated_early"]:
            assert smdp_info["duration_exec"] == 5
            assert smdp_info["k_exec"] == 2  # Two actions executed

        # Test with per-action durations
        smdp_env2 = SMDPfier(
            env,
            options_provider=options,
            duration_fn=ConstantActionDuration(4),
            action_interface="index",
            max_options=len(options)
        )

        obs, info = smdp_env2.reset(seed=42)
        obs, reward, term, trunc, info = smdp_env2.step(2)  # Execute "left-left-right"

        smdp_info = info["smdp"]
        assert smdp_info["duration_planned"] == 3 * 4  # 3 actions * 4 ticks each
        if not smdp_info["terminated_early"]:
            assert smdp_info["duration_exec"] == 3 * 4
            assert smdp_info["k_exec"] == 3
