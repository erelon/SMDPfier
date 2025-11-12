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


class TestSMDPInfoPayload:
    """Test SMDP info payload structure and content."""

    def test_cartpole_smdp_info_structure(self) -> None:
        """Test that CartPole example produces correct SMDP info structure."""
        import gymnasium as gym

        from smdpfier import Option, SMDPfier
        from smdpfier.defaults import ConstantOptionDuration, sum_rewards

        env = gym.make("CartPole-v1")

        static_options = [
            Option([0, 1], "left-right", meta={"test": "option"}),
            Option([1, 0], "right-left", meta={"test": "option2"}),
        ]

        smdp_env = SMDPfier(
            env,
            options_provider=static_options,
            duration_fn=ConstantOptionDuration(5),
            reward_agg=sum_rewards,
            action_interface="index",
            max_options=len(static_options),
            rng_seed=42,
        )

        obs, info = smdp_env.reset(seed=123)
        obs, reward, terminated, truncated, info = smdp_env.step(0)

        # Validate SMDP info structure
        assert "smdp" in info
        smdp_info = info["smdp"]

        # Check required fields
        assert "option" in smdp_info
        assert "k_exec" in smdp_info
        assert "rewards" in smdp_info
        assert "duration_planned" in smdp_info
        assert "duration_exec" in smdp_info
        assert "terminated_early" in smdp_info
        assert "time_units" in smdp_info

        # Check option sub-structure
        option_info = smdp_info["option"]
        assert "id" in option_info
        assert "name" in option_info
        assert "len" in option_info
        assert "meta" in option_info

        # Check values
        assert smdp_info["k_exec"] == 2  # Should execute 2 actions
        assert len(smdp_info["rewards"]) == 2  # Should have 2 reward values
        assert smdp_info["duration_planned"] == 5  # ConstantOptionDuration(5)
        assert smdp_info["duration_exec"] == 5  # Full execution
        assert smdp_info["time_units"] == "ticks"
        assert option_info["name"] == "left-right"
        assert option_info["len"] == 2
        assert option_info["meta"]["test"] == "option"

        env.close()

    def test_taxi_dynamic_options_and_masking(self) -> None:
        """Test Taxi dynamic options and action masking functionality."""
        # Import the functions from the taxi example
        import sys
        from pathlib import Path

        import gymnasium as gym

        from smdpfier import SMDPfier
        from smdpfier.defaults import RandomActionDuration, mean_rewards

        examples_dir = Path(__file__).parent.parent / "examples"
        sys.path.insert(0, str(examples_dir))

        from taxi_index_dynamic_mask import (
            create_taxi_options,
            taxi_availability_function,
        )

        env = gym.make("Taxi-v3")

        smdp_env = SMDPfier(
            env,
            options_provider=create_taxi_options,
            duration_fn=RandomActionDuration(min_duration=2, max_duration=4),
            reward_agg=mean_rewards,
            action_interface="index",
            max_options=12,
            availability_fn=taxi_availability_function,
            precheck=True,
            rng_seed=42,
        )

        obs, info = smdp_env.reset(seed=456)

        # Check action mask is present
        assert "action_mask" in info
        action_mask = info["action_mask"]
        assert len(action_mask) <= 12  # Should not exceed max_options
        assert len(action_mask) >= 6   # Should have at least the basic options
        assert any(action_mask)  # At least some options should be available

        # Find available option and execute
        available_indices = [i for i, avail in enumerate(action_mask) if avail]
        assert len(available_indices) > 0

        obs, reward, terminated, truncated, info = smdp_env.step(available_indices[0])

        # Validate SMDP info
        smdp_info = info["smdp"]
        assert smdp_info["k_exec"] >= 1
        assert isinstance(smdp_info["rewards"], list)
        assert 2 <= smdp_info["duration_exec"] <= 4  # RandomActionDuration range

        env.close()

    def test_pendulum_continuous_actions_and_durations(self) -> None:
        """Test Pendulum continuous actions and duration handling."""
        import gymnasium as gym

        from smdpfier import Option, SMDPfier
        from smdpfier.defaults import ConstantActionDuration, discounted_sum

        env = gym.make("Pendulum-v1")

        continuous_options = [
            Option([[1.0], [-1.0]], "oscillate", meta={"pattern": "simple"}),
            Option([[0.5], [0.0], [-0.5]], "gradual", meta={"pattern": "gradual"}),
        ]

        smdp_env = SMDPfier(
            env,
            options_provider=continuous_options,
            duration_fn=ConstantActionDuration(3),  # 3 ticks per action
            reward_agg=discounted_sum(gamma=0.95),
            action_interface="direct",
            rng_seed=42,
        )

        obs, info = smdp_env.reset(seed=789)

        # Execute first option (2 actions, 3 ticks each = 6 total ticks)
        obs, reward, terminated, truncated, info = smdp_env.step(continuous_options[0])

        # Validate SMDP info for continuous actions
        smdp_info = info["smdp"]
        assert smdp_info["k_exec"] == 2  # Executed 2 actions
        assert smdp_info["duration_planned"] == 6  # 2 actions * 3 ticks each
        assert smdp_info["duration_exec"] == 6  # Full execution
        assert len(smdp_info["rewards"]) == 2  # One reward per action

        # Execute second option (3 actions, 3 ticks each = 9 total ticks)
        obs, reward, terminated, truncated, info = smdp_env.step(continuous_options[1])

        smdp_info = info["smdp"]
        assert smdp_info["k_exec"] == 3  # Executed 3 actions
        assert smdp_info["duration_planned"] == 9  # 3 actions * 3 ticks each
        assert smdp_info["duration_exec"] == 9  # Full execution
        assert len(smdp_info["rewards"]) == 3  # One reward per action

        env.close()

    def test_partial_execution_duration_policy(self) -> None:
        """Test partial execution duration policies."""
        import gymnasium as gym

        from smdpfier import Option, SMDPfier
        from smdpfier.defaults import ConstantOptionDuration, sum_rewards

        # Create a short-lived environment that terminates quickly
        env = gym.make("CartPole-v1")

        # Long option that may not complete
        long_option = Option([0] * 10, "long-left", meta={})

        smdp_env = SMDPfier(
            env,
            options_provider=[long_option],
            duration_fn=ConstantOptionDuration(20),  # 20 ticks for full option
            reward_agg=sum_rewards,
            action_interface="index",
            max_options=1,
            partial_duration_policy="proportional",
            rng_seed=42,
        )

        obs, info = smdp_env.reset(seed=999)

        # Execute the long option
        obs, reward, terminated, truncated, info = smdp_env.step(0)

        smdp_info = info["smdp"]

        # Check duration handling
        assert smdp_info["duration_planned"] == 20
        if smdp_info["terminated_early"]:
            # If terminated early, duration_exec should be proportional
            expected_duration = (smdp_info["k_exec"] / 10) * 20  # proportional policy
            assert smdp_info["duration_exec"] == int(expected_duration)
        else:
            # If completed fully, duration should match planned
            assert smdp_info["duration_exec"] == 20

        env.close()
