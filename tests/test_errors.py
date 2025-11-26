"""Error handling and exception tests."""

import pytest

from smdpfier.errors import SMDPOptionExecutionError, SMDPOptionValidationError


class TestSMDPOptionExecutionError:
    """Test SMDPOptionExecutionError functionality."""

    def test_execution_error_creation(self) -> None:
        """Test creating execution error with all context fields."""
        error = SMDPOptionExecutionError(
            option_name="test-option",
            option_id="abc123",
            failing_step_index=2,
            action_repr="[1.0, 0.5]",
            short_obs_summary="state: [0.1, 0.2, 0.3]",
            base_error=ValueError("Invalid action"),
        )

        assert error.option_name == "test-option"
        assert error.option_id == "abc123"
        assert error.failing_step_index == 2
        assert error.action_repr == "[1.0, 0.5]"
        assert error.short_obs_summary == "state: [0.1, 0.2, 0.3]"
        assert isinstance(error.base_error, ValueError)
        assert "test-option" in str(error)
        assert "abc123" in str(error)

    def test_execution_error_auto_message(self) -> None:
        """Test that error message is auto-generated when not provided."""
        error = SMDPOptionExecutionError(
            option_name="auto-msg",
            option_id="xyz789",
            failing_step_index=0,
            action_repr="42",
            short_obs_summary="simple state",
        )

        message = str(error)
        assert "auto-msg" in message
        assert "xyz789" in message
        assert "step 0" in message
        assert "42" in message
        assert "simple state" in message

    def test_execution_error_custom_message(self) -> None:
        """Test execution error with custom message."""
        custom_msg = "Custom error message for testing"
        error = SMDPOptionExecutionError(
            option_name="custom",
            option_id="custom123",
            failing_step_index=1,
            action_repr="test",
            short_obs_summary="test state",
            message=custom_msg,
        )

        assert str(error) == custom_msg
        assert error.message == custom_msg


class TestSMDPOptionValidationError:
    """Test SMDPOptionValidationError functionality."""

    def test_validation_error_creation(self) -> None:
        """Test creating validation error with all context fields."""
        error = SMDPOptionValidationError(
            option_name="validation-test",
            option_id="val456",
            validation_type="mask",
            failing_step_index=1,
            action_repr="3",
            short_obs_summary="masked state",
            base_error=RuntimeError("Mask check failed"),
        )

        assert error.option_name == "validation-test"
        assert error.option_id == "val456"
        assert error.validation_type == "mask"
        assert error.failing_step_index == 1
        assert error.action_repr == "3"
        assert error.short_obs_summary == "masked state"
        assert isinstance(error.base_error, RuntimeError)

    def test_validation_error_minimal_fields(self) -> None:
        """Test validation error with minimal required fields."""
        error = SMDPOptionValidationError(
            option_name="minimal", option_id="min123", validation_type="dry_run"
        )

        assert error.option_name == "minimal"
        assert error.option_id == "min123"
        assert error.validation_type == "dry_run"
        assert error.failing_step_index is None
        assert error.action_repr is None
        assert error.short_obs_summary is None
        assert error.base_error is None

    def test_validation_error_auto_message(self) -> None:
        """Test auto-generated validation error message."""
        error = SMDPOptionValidationError(
            option_name="auto-validation",
            option_id="auto789",
            validation_type="precheck",
        )

        message = str(error)
        assert "auto-validation" in message
        assert "auto789" in message
        assert "precheck validation" in message


class TestErrorIntegration:
    """Test error integration with SMDPfier wrapper."""

    def test_execution_error_raised_on_runtime_failure(self) -> None:
        """Test that execution errors are raised when options fail at runtime."""
        import gymnasium as gym
        from smdpfier import SMDPfier, Option

        # Create CartPole environment (discrete action space 0,1)
        env = gym.make("CartPole-v1")

        # Create option with invalid action
        invalid_option = Option([0, 1, 99], "invalid-action-option")  # Action 99 is invalid

        smdp_env = SMDPfier(
            env,
            options_provider=[invalid_option],
            action_interface="direct"
        )

        obs, info = smdp_env.reset()

        # This should raise SMDPOptionExecutionError
        try:
            obs, reward, term, trunc, info = smdp_env.step(invalid_option)
            assert False, "Expected SMDPOptionExecutionError was not raised"
        except SMDPOptionExecutionError as e:
            assert e.option_name == "invalid-action-option"
            assert e.failing_step_index == 2  # Third action (index 2) fails
            assert "99" in e.action_repr
            assert e.base_error is not None

    def test_validation_error_raised_on_precheck_failure(self) -> None:
        """Test that validation errors are properly raised during precheck."""
        import gymnasium as gym

        from smdpfier import SMDPfier
        from smdpfier.errors import SMDPOptionValidationError
        from smdpfier.option import Option

        # Use Taxi-v3 which has action constraints
        env = gym.make("Taxi-v3")

        def restrictive_availability(obs):
            return [0, 1]  # Only first two actions available

        # Create option that uses unavailable action
        invalid_option = Option([2, 3], "invalid_actions")

        smdp_env = SMDPfier(
            env,
            options_provider=[invalid_option],
            action_interface="index",
            availability_fn=restrictive_availability,
            precheck=True
        )

        obs, info = smdp_env.reset(seed=42)

        # Should raise validation error
        with pytest.raises(SMDPOptionValidationError) as exc_info:
            smdp_env.step(0)

        error = exc_info.value
        assert error.option_name == "invalid_actions"
        assert error.option_id == invalid_option.option_id
        assert error.validation_type == "mask"
        assert error.failing_step_index == 0  # First invalid action
        assert error.action_repr == "2"

        env.close()

    def test_error_context_accuracy(self) -> None:
        """Test that error context accurately reflects the failure state."""
        import gymnasium as gym

        from smdpfier import SMDPfier
        from smdpfier.errors import SMDPOptionValidationError
        from smdpfier.option import Option

        env = gym.make("Taxi-v3")

        def availability_fn(obs):
            return [0, 1]  # Limit available actions

        # Test validation error context
        invalid_option = Option([0, 1, 2, 3], "test_invalid")

        smdp_env = SMDPfier(
            env,
            options_provider=[invalid_option],
            action_interface="index",
            availability_fn=availability_fn,
            precheck=True
        )

        obs, info = smdp_env.reset(seed=42)

        try:
            smdp_env.step(0)
            raise AssertionError("Should have raised validation error")
        except SMDPOptionValidationError as e:
            # Check all context fields are populated
            assert e.option_name == "test_invalid"
            assert e.option_id == invalid_option.option_id
            assert e.validation_type == "mask"
            assert e.failing_step_index == 2  # Should fail on action 2 (first invalid)
            assert e.action_repr == "2"
            assert e.short_obs_summary is not None
            assert len(e.short_obs_summary) > 0

            # Check error message contains key information
            error_msg = str(e)
            assert "test_invalid" in error_msg
            assert invalid_option.option_id in error_msg

        env.close()

    def test_error_message_clarity(self) -> None:
        """Test that error messages provide clear debugging information."""
        # Test both execution and validation error messages

        # Test execution error message
        exec_error = SMDPOptionExecutionError(
            option_name="clarity_test_exec",
            option_id="clear123",
            failing_step_index=1,
            action_repr="action_42",
            short_obs_summary="obs_summary_here",
            base_error=RuntimeError("Base cause")
        )

        exec_msg = str(exec_error)
        assert "clarity_test_exec" in exec_msg
        assert "clear123" in exec_msg
        assert "step 1" in exec_msg
        assert "action_42" in exec_msg

        # Test validation error message
        val_error = SMDPOptionValidationError(
            option_name="clarity_test_val",
            option_id="clear456",
            validation_type="mask",
            failing_step_index=0,
            action_repr="invalid_action_99"
        )

        val_msg = str(val_error)
        assert "clarity_test_val" in val_msg
        assert "clear456" in val_msg
        assert "mask validation" in val_msg
        assert "invalid_action_99" in val_msg
