"""Smoke test for masking and precheck functionality."""

import smdpfier


class TestSmokeTest:
    """Basic smoke test to verify package imports work."""

    def test_package_import(self) -> None:
        """Test that the package imports correctly and has basic attributes."""
        # Test that main components are importable
        from smdpfier import SMDPfier, Option
        from smdpfier.defaults import ConstantOptionDuration

        # Verify basic functionality exists
        option = Option([0, 1], "test")
        duration_fn = ConstantOptionDuration(5)

        assert option.name == "test"
        assert len(option.actions) == 2
        assert hasattr(smdpfier, "__version__") or hasattr(smdpfier, "SMDPfier")
