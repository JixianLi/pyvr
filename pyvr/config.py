"""
Rendering configuration for PyVR.

This module provides the RenderConfig class for managing rendering
quality and performance parameters.
"""

from dataclasses import dataclass


@dataclass
class RenderConfig:
    """
    Rendering quality and performance configuration.

    This class encapsulates ray marching parameters and other rendering
    settings that affect quality vs. performance tradeoffs.

    Attributes:
        step_size: Ray marching step size in normalized volume space.
                   Smaller values = higher quality but slower.
        max_steps: Maximum number of ray marching steps per ray.
                   Higher values = more detail but slower.
        early_ray_termination: Enable early ray termination optimization.
                              Stops ray when accumulated opacity reaches threshold.
        opacity_threshold: Opacity threshold for early ray termination (0.0-1.0).
                          Only used if early_ray_termination is True.
        reference_step_size: Reference step size for opacity correction.
                            Transfer functions define opacity at this reference step size.
                            When rendering at different step sizes, opacity is corrected
                            using Beer-Lambert law to maintain physical consistency.

                            Guidelines:
                            - Feature-dense volumes (medical, turbulence): 0.005-0.008
                            - Simple volumes (synthetic, smooth fields): 0.015-0.02
                            - Default (0.01): Matches balanced preset, good for most cases

                            The same transfer function will produce identical appearance
                            across all quality presets when reference_step_size is set correctly.

                            Physical basis:
                                alpha_corrected = 1.0 - exp(-alpha_tf * step_size / reference_step_size)

                            This implements Beer-Lambert law for light absorption through a medium.

    Example:
        >>> # Use a preset
        >>> config = RenderConfig.balanced()
        >>>
        >>> # Create custom config
        >>> config = RenderConfig(step_size=0.01, max_steps=500)
    """

    step_size: float = 0.01
    max_steps: int = 500
    early_ray_termination: bool = True
    opacity_threshold: float = 0.95
    reference_step_size: float = 0.01

    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()

    def validate(self) -> None:
        """
        Validate rendering configuration parameters.

        Raises:
            ValueError: If any parameter is invalid
        """
        if self.step_size <= 0:
            raise ValueError("step_size must be positive")

        if self.step_size > 1.0:
            import warnings

            warnings.warn(
                f"step_size {self.step_size} is very large and may produce poor quality",
                UserWarning,
            )

        if self.max_steps < 1:
            raise ValueError("max_steps must be at least 1")

        if self.max_steps > 10000:
            import warnings

            warnings.warn(
                f"max_steps {self.max_steps} is very large and may be slow",
                UserWarning,
            )

        if not (0.0 <= self.opacity_threshold <= 1.0):
            raise ValueError("opacity_threshold must be between 0.0 and 1.0")

    @classmethod
    def fast(cls) -> "RenderConfig":
        """
        Fast rendering preset - prioritizes speed over quality.

        Best for: Interactive exploration, real-time manipulation, previews

        Settings:
        - step_size: 0.02 (larger steps, fewer samples)
        - max_steps: 100 (limit maximum samples)
        - early_ray_termination: True (stop early when possible)

        Returns:
            RenderConfig configured for fast rendering

        Example:
            >>> config = RenderConfig.fast()
            >>> renderer = VolumeRenderer(width=512, height=512, config=config)
        """
        return cls(
            step_size=0.02,
            max_steps=100,
            early_ray_termination=True,
            opacity_threshold=0.90,
        )

    @classmethod
    def balanced(cls) -> "RenderConfig":
        """
        Balanced rendering preset - good quality and reasonable speed.

        Best for: General use, most visualization tasks, development

        Settings:
        - step_size: 0.01 (standard sampling density)
        - max_steps: 500 (sufficient for most volumes)
        - early_ray_termination: True (maintain performance)

        Returns:
            RenderConfig configured for balanced rendering (default)

        Example:
            >>> config = RenderConfig.balanced()
            >>> renderer = VolumeRenderer(width=512, height=512, config=config)
        """
        return cls(
            step_size=0.01,
            max_steps=500,
            early_ray_termination=True,
            opacity_threshold=0.95,
        )

    @classmethod
    def high_quality(cls) -> "RenderConfig":
        """
        High quality rendering preset - prioritizes quality over speed.

        Best for: Final renders, publications, detailed analysis, screenshots

        Settings:
        - step_size: 0.005 (small steps, dense sampling)
        - max_steps: 1000 (allow more detail)
        - early_ray_termination: True (but with higher threshold)

        Returns:
            RenderConfig configured for high quality rendering

        Example:
            >>> config = RenderConfig.high_quality()
            >>> renderer = VolumeRenderer(width=1024, height=1024, config=config)
        """
        return cls(
            step_size=0.005,
            max_steps=1000,
            early_ray_termination=True,
            opacity_threshold=0.98,
        )

    @classmethod
    def ultra_quality(cls) -> "RenderConfig":
        """
        Ultra quality rendering preset - maximum quality, very slow.

        Best for: Final publication renders, extreme detail requirements

        Settings:
        - step_size: 0.001 (very small steps)
        - max_steps: 2000 (maximum detail)
        - early_ray_termination: False (sample entire ray)

        Returns:
            RenderConfig configured for ultra quality rendering

        Warning:
            This preset can be very slow. Use sparingly.

        Example:
            >>> config = RenderConfig.ultra_quality()
            >>> renderer = VolumeRenderer(width=2048, height=2048, config=config)
        """
        return cls(
            step_size=0.001,
            max_steps=2000,
            early_ray_termination=False,
            opacity_threshold=1.0,
        )

    @classmethod
    def preview(cls) -> "RenderConfig":
        """
        Preview quality preset - extremely fast, low quality.

        Best for: Quick previews, testing, rapid iteration

        Settings:
        - step_size: 0.05 (very large steps)
        - max_steps: 50 (minimal samples)
        - early_ray_termination: True (aggressive early stopping)

        Returns:
            RenderConfig configured for preview rendering

        Example:
            >>> config = RenderConfig.preview()
            >>> renderer = VolumeRenderer(width=256, height=256, config=config)
        """
        return cls(
            step_size=0.05,
            max_steps=50,
            early_ray_termination=True,
            opacity_threshold=0.80,
        )

    @classmethod
    def custom(
        cls,
        step_size: float,
        max_steps: int,
        early_ray_termination: bool = True,
        opacity_threshold: float = 0.95,
    ) -> "RenderConfig":
        """
        Create custom rendering configuration.

        Args:
            step_size: Custom step size
            max_steps: Custom maximum steps
            early_ray_termination: Enable early termination
            opacity_threshold: Opacity threshold for early termination

        Returns:
            RenderConfig with custom parameters

        Example:
            >>> config = RenderConfig.custom(
            ...     step_size=0.015,
            ...     max_steps=300,
            ...     early_ray_termination=True
            ... )
        """
        return cls(
            step_size=step_size,
            max_steps=max_steps,
            early_ray_termination=early_ray_termination,
            opacity_threshold=opacity_threshold,
        )

    def copy(self) -> "RenderConfig":
        """
        Create a copy of this configuration.

        Returns:
            New RenderConfig instance with same parameters

        Example:
            >>> original = RenderConfig.balanced()
            >>> modified = original.copy()
            >>> modified.step_size = 0.005  # Doesn't affect original
        """
        return RenderConfig(
            step_size=self.step_size,
            max_steps=self.max_steps,
            early_ray_termination=self.early_ray_termination,
            opacity_threshold=self.opacity_threshold,
            reference_step_size=self.reference_step_size,
        )

    def with_step_size(self, step_size: float) -> "RenderConfig":
        """
        Create new config with modified step size.

        Args:
            step_size: New step size value

        Returns:
            New RenderConfig with updated step_size

        Example:
            >>> config = RenderConfig.balanced().with_step_size(0.005)
        """
        config = self.copy()
        config.step_size = step_size
        config.validate()
        return config

    def with_max_steps(self, max_steps: int) -> "RenderConfig":
        """
        Create new config with modified max steps.

        Args:
            max_steps: New max steps value

        Returns:
            New RenderConfig with updated max_steps

        Example:
            >>> config = RenderConfig.balanced().with_max_steps(800)
        """
        config = self.copy()
        config.max_steps = max_steps
        config.validate()
        return config

    def estimate_samples_per_ray(self) -> int:
        """
        Estimate average number of samples per ray.

        This is a rough estimate assuming rays traverse the entire volume diagonal.

        Returns:
            Estimated number of samples per ray

        Example:
            >>> config = RenderConfig.balanced()
            >>> config.estimate_samples_per_ray()
            173
        """
        # Assume unit cube diagonal (~1.732)
        diagonal_length = 1.732
        estimated_samples = int(diagonal_length / self.step_size)
        return min(estimated_samples, self.max_steps)

    def estimate_render_time_relative(self) -> float:
        """
        Estimate relative render time compared to balanced preset.

        Returns:
            Multiplier relative to balanced preset (1.0 = balanced speed)

        Example:
            >>> fast = RenderConfig.fast()
            >>> fast.estimate_render_time_relative()
            0.2
            >>> hq = RenderConfig.high_quality()
            >>> hq.estimate_render_time_relative()
            5.0
        """
        balanced = RenderConfig.balanced()
        balanced_samples = balanced.estimate_samples_per_ray()
        our_samples = self.estimate_samples_per_ray()

        relative_time = our_samples / balanced_samples
        return relative_time

    def __repr__(self) -> str:
        """String representation of rendering configuration."""
        return (
            f"RenderConfig("
            f"step_size={self.step_size}, "
            f"max_steps={self.max_steps}, "
            f"early_termination={self.early_ray_termination}, "
            f"~{self.estimate_samples_per_ray()} samples/ray)"
        )


class RenderConfigError(Exception):
    """Exception raised for rendering configuration errors."""

    pass
