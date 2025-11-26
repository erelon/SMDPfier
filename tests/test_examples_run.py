"""Test that examples can run without errors."""

import subprocess
import sys
from pathlib import Path


class TestExamplesRun:
    """Test that example scripts execute without errors."""

    def test_cartpole_example_runs(self) -> None:
        """Test that CartPole example executes without errors and produces expected info."""
        examples_dir = Path(__file__).parent.parent / "examples"
        cartpole_script = examples_dir / "cartpole_index_static.py"

        # Run example with limited steps
        result = subprocess.run(
            [sys.executable, str(cartpole_script), "--max-steps", "2"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 0, f"CartPole example failed: {result.stderr}"

        # Check key output content
        output = result.stdout
        assert "CartPole with SMDPfier" in output
        assert "Original action space: Discrete(2)" in output
        assert "SMDP action space: Discrete(6)" in output
        assert "Option:" in output
        assert "Duration:" in output
        assert "Executed" in output
        assert "ticks" in output
        assert "Demo completed!" in output

    def test_taxi_example_runs(self) -> None:
        """Test that Taxi example executes without errors and shows masking."""
        examples_dir = Path(__file__).parent.parent / "examples"
        taxi_script = examples_dir / "taxi_index_dynamic_mask.py"

        # Run example with limited steps
        result = subprocess.run(
            [sys.executable, str(taxi_script), "--max-steps", "2"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 0, f"Taxi example failed: {result.stderr}"

        # Check key output content
        output = result.stdout
        assert "Taxi with SMDPfier" in output
        assert "Original action space: Discrete(6)" in output
        assert "SMDP action space: Discrete(12)" in output
        assert "Precheck enabled: True" in output
        assert "Available options mask:" in output
        assert "Duration:" in output
        assert "Mean reward:" in output
        assert "Demo completed!" in output

    def test_pendulum_example_runs(self) -> None:
        """Test that Pendulum example executes without errors and shows continuous actions."""
        examples_dir = Path(__file__).parent.parent / "examples"
        pendulum_script = examples_dir / "pendulum_direct_continuous.py"

        # Run example with limited steps
        result = subprocess.run(
            [sys.executable, str(pendulum_script), "--max-steps", "2"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 0, f"Pendulum example failed: {result.stderr}"

        # Check key output content
        output = result.stdout
        assert "Pendulum with SMDPfier" in output
        assert "Original action space: Box" in output
        assert "SMDP action space: Box" in output
        assert "Direct Interface" in output
        assert "Continuous Actions" in output
        assert "Actions:" in output
        assert "Discounted reward:" in output
        assert "Duration:" in output
        assert "Demo completed!" in output

    def test_examples_have_correct_imports(self) -> None:
        """Test that example files can import SMDPfier components."""
        # This basic test can work even without full implementation
        examples_dir = Path(__file__).parent.parent / "examples"

        for example_file in examples_dir.glob("*.py"):
            # Check that file contains expected imports
            content = example_file.read_text()
            assert "from smdpfier import" in content
            assert "SMDPfier" in content
            assert "Option" in content

    def test_examples_have_main_function(self) -> None:
        """Test that example files have main() function and __main__ guard."""
        examples_dir = Path(__file__).parent.parent / "examples"

        for example_file in examples_dir.glob("*.py"):
            content = example_file.read_text()
            assert "def main(" in content
            assert 'if __name__ == "__main__"' in content

    def test_default_rate_reward_calculation(self) -> None:
        """Test that default sum-based reward matches expected calculation."""
        import gymnasium as gym
        import pytest

        from smdpfier import Option, SMDPfier

        env = gym.make("CartPole-v1")
        options = [Option([0, 1], "test-option")]

        # Use default reward_agg (should be sum_rewards)
        smdp_env = SMDPfier(
            env,
            options_provider=options,
            action_interface="index",
        )

        obs, info = smdp_env.reset(seed=42)
        obs, reward, terminated, truncated, info = smdp_env.step(0)

        # Verify info contains duration and rewards
        smdp_info = info["smdp"]
        assert "duration" in smdp_info
        assert "rewards" in smdp_info

        duration = smdp_info["duration"]
        primitive_rewards = smdp_info["rewards"]
        k_exec = smdp_info["k_exec"]

        # Verify duration equals k_exec
        assert duration == k_exec

        # Verify reward equals sum of primitive rewards (default)
        expected_sum = sum(primitive_rewards)
        assert reward == pytest.approx(expected_sum), (
            f"Reward should equal sum: {reward} vs {expected_sum}"
        )

        env.close()

    def test_pendulum_continuous_actions_and_durations(self) -> None:
        """Test Pendulum continuous actions and duration handling."""
        import gymnasium as gym

        from smdpfier import Option, SMDPfier
        from smdpfier.defaults import discounted_sum

        env = gym.make("Pendulum-v1")

        continuous_options = [
            Option([[1.0], [-1.0]], "oscillate", meta={"pattern": "simple"}),
            Option([[0.5], [0.0], [-0.5]], "gradual", meta={"pattern": "gradual"}),
        ]

        smdp_env = SMDPfier(
            env,
            options_provider=continuous_options,
            reward_agg=discounted_sum(gamma=0.95),
            action_interface="direct",
            rng_seed=42,
        )

        obs, info = smdp_env.reset(seed=789)

        # Execute first option (2 actions)
        obs, reward, terminated, truncated, info = smdp_env.step(continuous_options[0])

        # Validate SMDP info for continuous actions
        smdp_info = info["smdp"]
        assert smdp_info["k_exec"] == 2  # Executed 2 actions
        assert smdp_info["duration"] == 2  # duration = k_exec
        assert len(smdp_info["rewards"]) == 2  # One reward per action

        # Execute second option (3 actions)
        obs, reward, terminated, truncated, info = smdp_env.step(continuous_options[1])

        smdp_info = info["smdp"]
        assert smdp_info["k_exec"] == 3  # Executed 3 actions
        assert smdp_info["duration"] == 3  # duration = k_exec
        assert len(smdp_info["rewards"]) == 3  # One reward per action

        env.close()

    def test_partial_execution_duration_policy(self) -> None:
        """Test partial execution duration handling."""
        import gymnasium as gym

        from smdpfier import Option, SMDPfier
        from smdpfier.defaults import sum_rewards

        # Create a short-lived environment that terminates quickly
        env = gym.make("CartPole-v1")

        # Long option that may not complete
        long_option = Option([0] * 10, "long-left", meta={})

        smdp_env = SMDPfier(
            env,
            options_provider=[long_option],
            reward_agg=sum_rewards,
            action_interface="index",
            max_options=1,
            rng_seed=42,
        )

        obs, info = smdp_env.reset(seed=999)

        # Execute the long option
        obs, reward, terminated, truncated, info = smdp_env.step(0)

        smdp_info = info["smdp"]

        # Check duration handling - duration always equals k_exec
        assert smdp_info["duration"] == smdp_info["k_exec"]

        if smdp_info["terminated_early"]:
            # If terminated early, k_exec < option length
            assert smdp_info["k_exec"] < 10
        else:
            # If completed fully, k_exec == option length
            assert smdp_info["k_exec"] == 10

        env.close()
