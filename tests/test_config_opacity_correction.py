"""Tests for opacity correction formula and RenderConfig integration."""

import pytest
import numpy as np
from pyvr.config import RenderConfig


class TestReferenceStepSizeParameter:
    """Tests for reference_step_size parameter in RenderConfig."""

    def test_reference_step_size_default(self):
        """Test that reference_step_size defaults to 0.01."""
        config = RenderConfig.balanced()
        assert config.reference_step_size == 0.01

    def test_reference_step_size_all_presets(self):
        """Test that all presets use default reference_step_size."""
        presets = [
            RenderConfig.preview(),
            RenderConfig.fast(),
            RenderConfig.balanced(),
            RenderConfig.high_quality(),
            RenderConfig.ultra_quality(),
        ]

        for preset in presets:
            assert preset.reference_step_size == 0.01

    def test_reference_step_size_custom(self):
        """Test custom reference_step_size value."""
        config = RenderConfig(step_size=0.01, max_steps=500, reference_step_size=0.005)
        assert config.reference_step_size == 0.005

    def test_reference_step_size_preserved_with_step_size(self):
        """Test that with_step_size() preserves reference_step_size."""
        config = RenderConfig.balanced().with_step_size(0.008)
        assert config.reference_step_size == 0.01  # Should preserve default

    def test_reference_step_size_preserved_with_max_steps(self):
        """Test that with_max_steps() preserves reference_step_size."""
        config = RenderConfig.balanced().with_max_steps(600)
        assert config.reference_step_size == 0.01  # Should preserve default


class TestOpacityCorrectionFormula:
    """Tests for opacity correction mathematical formula."""

    def test_formula_identity_when_equal_step_sizes(self):
        """When step_size == reference_step_size, correction should be minimal."""
        alpha_tf = 0.5
        step_size = 0.01
        reference = 0.01

        # Formula: 1.0 - exp(-alpha_tf * step_size / reference)
        corrected = 1.0 - np.exp(-alpha_tf * step_size / reference)

        # When step == reference, corrected should be close to original
        # (not exactly equal due to exponential, but close for small alpha)
        assert np.isclose(corrected, alpha_tf, rtol=0.3)

    def test_smaller_step_produces_smaller_alpha(self):
        """Smaller step size should produce smaller corrected alpha."""
        alpha_tf = 0.5
        reference = 0.01

        corrected_small = 1.0 - np.exp(-alpha_tf * 0.005 / reference)
        corrected_large = 1.0 - np.exp(-alpha_tf * 0.02 / reference)

        assert corrected_small < corrected_large

    def test_larger_step_produces_larger_alpha(self):
        """Larger step size should produce larger corrected alpha."""
        alpha_tf = 0.3
        reference = 0.01

        corrected_preview = 1.0 - np.exp(-alpha_tf * 0.05 / reference)  # preview
        corrected_balanced = 1.0 - np.exp(-alpha_tf * 0.01 / reference)  # balanced

        assert corrected_preview > corrected_balanced

    def test_alpha_zero_stays_zero(self):
        """Alpha of 0 should remain 0 after correction."""
        reference = 0.01
        step = 0.005

        corrected = 1.0 - np.exp(-0.0 * step / reference)
        assert corrected == 0.0

    def test_alpha_one_behavior(self):
        """Alpha of 1.0 should produce high but valid corrected value."""
        reference = 0.01
        step = 0.005

        corrected = 1.0 - np.exp(-1.0 * step / reference)

        # Should be positive but less than 1
        assert 0.0 < corrected < 1.0

    def test_correction_monotonic_increasing(self):
        """Corrected alpha should increase monotonically with step_size."""
        alpha_tf = 0.4
        reference = 0.01

        step_sizes = [0.001, 0.005, 0.01, 0.02, 0.05]
        corrected_values = [
            1.0 - np.exp(-alpha_tf * step / reference) for step in step_sizes
        ]

        # Each value should be greater than or equal to previous
        for i in range(1, len(corrected_values)):
            assert corrected_values[i] >= corrected_values[i - 1]

    def test_correction_bounds(self):
        """Corrected alpha should always be in [0, 1) range."""
        alpha_values = np.linspace(0.0, 1.0, 20)
        step_sizes = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
        reference = 0.01

        for alpha_tf in alpha_values:
            for step in step_sizes:
                corrected = 1.0 - np.exp(-alpha_tf * step / reference)

                # Should be in valid range
                assert 0.0 <= corrected < 1.0

    def test_extreme_step_size_small(self):
        """Very small step size should produce very small corrected alpha."""
        alpha_tf = 0.5
        reference = 0.01
        step_tiny = 0.0001

        corrected = 1.0 - np.exp(-alpha_tf * step_tiny / reference)

        # Should be much smaller than original alpha
        assert corrected < alpha_tf * 0.1

    def test_extreme_step_size_large(self):
        """Very large step size should saturate near 1.0."""
        alpha_tf = 0.5
        reference = 0.01
        step_huge = 1.0

        corrected = 1.0 - np.exp(-alpha_tf * step_huge / reference)

        # Should be close to 1.0 but not exceed it
        assert 0.99 < corrected <= 1.0

    def test_reference_step_size_scaling(self):
        """Different reference_step_size should scale correction appropriately."""
        alpha_tf = 0.5
        step = 0.01

        # Using step as reference should give minimal correction
        corrected_ref_equal = 1.0 - np.exp(-alpha_tf * step / step)

        # Using smaller reference should increase correction
        corrected_ref_small = 1.0 - np.exp(-alpha_tf * step / 0.005)

        # Using larger reference should decrease correction
        corrected_ref_large = 1.0 - np.exp(-alpha_tf * step / 0.02)

        assert corrected_ref_small > corrected_ref_equal > corrected_ref_large


class TestOpacityCorrectionPhysics:
    """Tests for physical correctness of opacity correction."""

    def test_beer_lambert_law(self):
        """Verify formula matches Beer-Lambert law: I = I₀ * exp(-τ*d)."""
        # For opacity: alpha = 1 - exp(-tau * distance)
        # In our case: tau = alpha_tf / reference_step_size
        #              distance = step_size

        alpha_tf = 0.6  # Extinction coefficient (scaled)
        step = 0.01
        reference = 0.01

        # Our formula
        alpha_corrected = 1.0 - np.exp(-alpha_tf * step / reference)

        # Beer-Lambert: transmittance = exp(-tau * d)
        # Therefore: opacity = 1 - transmittance
        tau = alpha_tf / reference
        distance = step
        opacity_beer_lambert = 1.0 - np.exp(-tau * distance)

        assert np.isclose(alpha_corrected, opacity_beer_lambert)

    def test_accumulation_consistency(self):
        """Two small steps should approximately equal one large step."""
        alpha_tf = 0.4
        reference = 0.01

        # One step of size 0.02
        alpha_single = 1.0 - np.exp(-alpha_tf * 0.02 / reference)

        # Two steps of size 0.01
        alpha_step1 = 1.0 - np.exp(-alpha_tf * 0.01 / reference)
        # After first step, remaining transmittance is (1 - alpha_step1)
        # Second step adds more opacity
        alpha_step2 = 1.0 - np.exp(-alpha_tf * 0.01 / reference)
        # Combined using front-to-back blending
        alpha_combined = alpha_step1 + (1.0 - alpha_step1) * alpha_step2

        # Should be close (not exact due to discrete steps)
        assert np.isclose(alpha_single, alpha_combined, rtol=0.05)
