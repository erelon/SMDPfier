"""Comprehensive error classes for SMDPfier operations."""

from __future__ import annotations


class SMDPOptionExecutionError(Exception):
    """Raised when an option fails during execution in the environment.

    This error provides rich context about what went wrong during option
    execution, including the failing action, environment state, and
    execution progress.

    Attributes:
        option_name: Human-readable name of the failing option
        option_id: Stable ID of the failing option
        failing_step_index: Zero-based index of the action that failed
        action_repr: String representation of the failing action
        short_obs_summary: Brief summary of the environment observation
        base_error: Original exception that caused the failure
        message: Detailed error message combining all context
    """

    def __init__(
        self,
        option_name: str,
        option_id: str,
        failing_step_index: int,
        action_repr: str,
        short_obs_summary: str,
        base_error: Exception | None = None,
        message: str | None = None,
    ) -> None:
        """Initialize execution error with rich context.

        Args:
            option_name: Human-readable name of the failing option
            option_id: Stable ID of the failing option
            failing_step_index: Zero-based index of the action that failed
            action_repr: String representation of the failing action
            short_obs_summary: Brief summary of the environment observation
            base_error: Original exception that caused the failure
            message: Custom error message (auto-generated if None)
        """
        self.option_name = option_name
        self.option_id = option_id
        self.failing_step_index = failing_step_index
        self.action_repr = action_repr
        self.short_obs_summary = short_obs_summary
        self.base_error = base_error

        if message is None:
            base_msg = f" (caused by: {base_error})" if base_error else ""
            message = (
                f"Option '{option_name}' (id: {option_id}) failed at step "
                f"{failing_step_index} with action {action_repr}. "
                f"Environment state: {short_obs_summary}{base_msg}"
            )

        self.message = message
        super().__init__(message)


class SMDPOptionValidationError(Exception):
    """Raised when an option fails validation before execution.

    This error is thrown during precheck validation when an option
    is determined to be invalid for the current environment state,
    before any execution attempts.

    Attributes:
        option_name: Human-readable name of the invalid option
        option_id: Stable ID of the invalid option
        validation_type: Type of validation that failed (e.g., "mask", "dry_run")
        failing_step_index: Zero-based index where validation failed (if applicable)
        action_repr: String representation of the problematic action (if applicable)
        short_obs_summary: Brief summary of the environment observation
        base_error: Original exception that caused the validation failure
        message: Detailed error message combining all context
    """

    def __init__(
        self,
        option_name: str,
        option_id: str,
        validation_type: str,
        failing_step_index: int | None = None,
        action_repr: str | None = None,
        short_obs_summary: str | None = None,
        base_error: Exception | None = None,
        message: str | None = None,
    ) -> None:
        """Initialize validation error with rich context.

        Args:
            option_name: Human-readable name of the invalid option
            option_id: Stable ID of the invalid option
            validation_type: Type of validation that failed
            failing_step_index: Zero-based index where validation failed
            action_repr: String representation of the problematic action
            short_obs_summary: Brief summary of the environment observation
            base_error: Original exception that caused the validation failure
            message: Custom error message (auto-generated if None)
        """
        self.option_name = option_name
        self.option_id = option_id
        self.validation_type = validation_type
        self.failing_step_index = failing_step_index
        self.action_repr = action_repr
        self.short_obs_summary = short_obs_summary
        self.base_error = base_error

        if message is None:
            step_info = (
                f" at step {failing_step_index} with action {action_repr}"
                if failing_step_index is not None and action_repr is not None
                else ""
            )
            obs_info = (
                f". Environment state: {short_obs_summary}" if short_obs_summary else ""
            )
            base_msg = f" (caused by: {base_error})" if base_error else ""

            message = (
                f"Option '{option_name}' (id: {option_id}) failed "
                f"{validation_type} validation{step_info}{obs_info}{base_msg}"
            )

        self.message = message
        super().__init__(message)
