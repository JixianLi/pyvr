"""Tests for RenderConfig class."""

import pytest

from pyvr.config import RenderConfig, RenderConfigError


class TestRenderConfigPresets:
    """Test RenderConfig preset factory methods."""

    def test_fast_preset(self):
        """Fast preset should have performance-oriented settings."""
        config = RenderConfig.fast()

        assert config.step_size > 0.01  # Larger step size
        assert config.max_steps < 200  # Fewer steps
        assert config.early_ray_termination is True

    def test_balanced_preset(self):
        """Balanced preset should have default settings."""
        config = RenderConfig.balanced()

        assert config.step_size == 0.01
        assert config.max_steps == 500
        assert config.early_ray_termination is True

    def test_high_quality_preset(self):
        """High quality preset should have quality-oriented settings."""
        config = RenderConfig.high_quality()

        assert config.step_size < 0.01  # Smaller step size
        assert config.max_steps > 500  # More steps

    def test_ultra_quality_preset(self):
        """Ultra quality preset should have maximum quality settings."""
        config = RenderConfig.ultra_quality()

        assert config.step_size <= 0.001
        assert config.max_steps >= 1000

    def test_preview_preset(self):
        """Preview preset should have minimal quality settings."""
        config = RenderConfig.preview()

        assert config.step_size >= 0.05
        assert config.max_steps <= 100

    def test_custom_preset(self):
        """Custom preset should accept arbitrary parameters."""
        config = RenderConfig.custom(
            step_size=0.015, max_steps=300, early_ray_termination=False
        )

        assert config.step_size == 0.015
        assert config.max_steps == 300
        assert config.early_ray_termination is False


class TestRenderConfigValidation:
    """Test RenderConfig validation."""

    def test_invalid_step_size_zero(self):
        """Zero step_size should raise error."""
        with pytest.raises(ValueError, match="step_size"):
            RenderConfig(step_size=0.0)

    def test_invalid_step_size_negative(self):
        """Negative step_size should raise error."""
        with pytest.raises(ValueError, match="step_size"):
            RenderConfig(step_size=-0.01)

    def test_invalid_max_steps_zero(self):
        """max_steps = 0 should raise error."""
        with pytest.raises(ValueError, match="max_steps"):
            RenderConfig(max_steps=0)

    def test_invalid_max_steps_negative(self):
        """Negative max_steps should raise error."""
        with pytest.raises(ValueError, match="max_steps"):
            RenderConfig(max_steps=-10)

    def test_large_step_size_warning(self):
        """Very large step_size should emit warning."""
        with pytest.warns(UserWarning, match="step_size"):
            RenderConfig(step_size=2.0)

    def test_large_max_steps_warning(self):
        """Very large max_steps should emit warning."""
        with pytest.warns(UserWarning, match="max_steps"):
            RenderConfig(max_steps=50000)

    def test_invalid_opacity_threshold_too_high(self):
        """opacity_threshold > 1.0 should raise error."""
        with pytest.raises(ValueError, match="opacity_threshold"):
            RenderConfig(opacity_threshold=1.5)

    def test_invalid_opacity_threshold_negative(self):
        """Negative opacity_threshold should raise error."""
        with pytest.raises(ValueError, match="opacity_threshold"):
            RenderConfig(opacity_threshold=-0.1)

    def test_valid_opacity_threshold_boundaries(self):
        """Opacity threshold at boundaries should be valid."""
        config1 = RenderConfig(opacity_threshold=0.0)
        assert config1.opacity_threshold == 0.0

        config2 = RenderConfig(opacity_threshold=1.0)
        assert config2.opacity_threshold == 1.0


class TestRenderConfigMethods:
    """Test RenderConfig methods."""

    def test_copy(self):
        """copy should create independent instance."""
        original = RenderConfig.balanced()
        copy = original.copy()

        # Modify copy
        copy.step_size = 0.005

        # Original should be unchanged
        assert original.step_size == 0.01

    def test_with_step_size(self):
        """with_step_size should create new config."""
        original = RenderConfig.balanced()
        modified = original.with_step_size(0.005)

        assert modified.step_size == 0.005
        assert original.step_size == 0.01  # Unchanged

    def test_with_step_size_validates(self):
        """with_step_size should validate the new value."""
        config = RenderConfig.balanced()
        with pytest.raises(ValueError, match="step_size"):
            config.with_step_size(-0.01)

    def test_with_max_steps(self):
        """with_max_steps should create new config."""
        original = RenderConfig.balanced()
        modified = original.with_max_steps(800)

        assert modified.max_steps == 800
        assert original.max_steps == 500  # Unchanged

    def test_with_max_steps_validates(self):
        """with_max_steps should validate the new value."""
        config = RenderConfig.balanced()
        with pytest.raises(ValueError, match="max_steps"):
            config.with_max_steps(0)

    def test_estimate_samples_per_ray(self):
        """estimate_samples_per_ray should return reasonable value."""
        config = RenderConfig.balanced()
        samples = config.estimate_samples_per_ray()

        assert samples > 0
        assert samples <= config.max_steps

    def test_estimate_samples_per_ray_capped_by_max_steps(self):
        """Estimated samples should not exceed max_steps."""
        config = RenderConfig(step_size=0.0001, max_steps=100)
        samples = config.estimate_samples_per_ray()

        assert samples == 100  # Capped by max_steps

    def test_estimate_render_time_relative(self):
        """estimate_render_time_relative should return reasonable values."""
        fast = RenderConfig.fast()
        balanced = RenderConfig.balanced()
        hq = RenderConfig.high_quality()

        # Fast should be faster than balanced
        assert fast.estimate_render_time_relative() < 1.0

        # Balanced should be 1.0 (reference)
        assert balanced.estimate_render_time_relative() == 1.0

        # High quality should be slower than balanced
        assert hq.estimate_render_time_relative() > 1.0

    def test_repr(self):
        """__repr__ should return informative string."""
        config = RenderConfig.balanced()
        repr_str = repr(config)

        assert "RenderConfig" in repr_str
        assert "step_size" in repr_str
        assert "samples/ray" in repr_str


class TestRenderConfigInitialization:
    """Test RenderConfig initialization."""

    def test_default_initialization(self):
        """Default initialization should use balanced settings."""
        config = RenderConfig()

        assert config.step_size == 0.01
        assert config.max_steps == 500
        assert config.early_ray_termination is True
        assert config.opacity_threshold == 0.95

    def test_custom_initialization(self):
        """Custom initialization should accept all parameters."""
        config = RenderConfig(
            step_size=0.015,
            max_steps=300,
            early_ray_termination=False,
            opacity_threshold=0.9,
        )

        assert config.step_size == 0.015
        assert config.max_steps == 300
        assert config.early_ray_termination is False
        assert config.opacity_threshold == 0.9

    def test_post_init_validation(self):
        """__post_init__ should call validate()."""
        with pytest.raises(ValueError, match="step_size"):
            RenderConfig(step_size=-1.0)


class TestRenderConfigEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_small_step_size(self):
        """Very small step size should be valid."""
        config = RenderConfig(step_size=0.0001, max_steps=50000)
        # Should emit warning about max_steps but not fail
        assert config.step_size == 0.0001

    def test_step_size_boundary(self):
        """Step size just above zero should be valid."""
        config = RenderConfig(step_size=0.0001)
        assert config.step_size == 0.0001

    def test_max_steps_boundary(self):
        """max_steps = 1 should be valid."""
        config = RenderConfig(max_steps=1)
        assert config.max_steps == 1

    def test_all_presets_are_valid(self):
        """All presets should pass validation."""
        presets = [
            RenderConfig.preview(),
            RenderConfig.fast(),
            RenderConfig.balanced(),
            RenderConfig.high_quality(),
            RenderConfig.ultra_quality(),
        ]

        for preset in presets:
            # Should not raise any errors
            preset.validate()
            assert preset.step_size > 0
            assert preset.max_steps > 0

    def test_copy_preserves_all_attributes(self):
        """Copy should preserve all configuration attributes."""
        original = RenderConfig(
            step_size=0.015,
            max_steps=300,
            early_ray_termination=False,
            opacity_threshold=0.85,
        )
        copy = original.copy()

        assert copy.step_size == original.step_size
        assert copy.max_steps == original.max_steps
        assert copy.early_ray_termination == original.early_ray_termination
        assert copy.opacity_threshold == original.opacity_threshold

    def test_preset_ordering(self):
        """Presets should have expected performance ordering."""
        preview = RenderConfig.preview()
        fast = RenderConfig.fast()
        balanced = RenderConfig.balanced()
        hq = RenderConfig.high_quality()
        ultra = RenderConfig.ultra_quality()

        # Step sizes should decrease (more quality)
        assert preview.step_size > fast.step_size
        assert fast.step_size > balanced.step_size
        assert balanced.step_size > hq.step_size
        assert hq.step_size > ultra.step_size

        # Max steps should generally increase
        assert preview.max_steps < balanced.max_steps
        assert balanced.max_steps < hq.max_steps
        assert hq.max_steps < ultra.max_steps
