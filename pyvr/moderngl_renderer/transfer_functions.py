import numpy as np


class OpacityTransferFunction:
    """
    Maps scalar values (e.g., density) to opacity (alpha) values for volume rendering.
    Supports piecewise linear transfer functions and can output a 1D texture.
    """

    def __init__(self, control_points=None, lut_size=256):
        """
        control_points: list of (scalar, opacity) tuples, sorted by scalar.
        lut_size: size of the lookup table and texture (instance variable).
        Example: [(0.0, 0.0), (0.2, 0.1), (0.5, 0.8), (1.0, 1.0)]
        """
        if control_points is None:
            # Default: linear ramp from 0 to 1
            control_points = [(0.0, 0.0), (1.0, 1.0)]
        self.control_points = sorted(control_points)
        self.lut_size = lut_size

    @classmethod
    def linear(cls, low=0.0, high=1.0, lut_size=256):
        """Linear ramp from low to high."""
        return cls([(0.0, low), (1.0, high)], lut_size=lut_size)

    @classmethod
    def one_step(cls, step=0.5, low=0.0, high=1.0, lut_size=256):
        """
        Step function: low opacity up to 'step', then high opacity.
        Example: step=0.5, low=0.0, high=1.0 gives [(0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (1.0, 1.0)]
        """
        return cls(
            [(0.0, low), (step, low), (step + 1e-12, high), (1.0, high)],
            lut_size=lut_size,
        )

    @classmethod
    def peaks(cls, peaks, opacity=1.0, eps=0.02, lut_size=256, base=0.0):
        """
        Create an OTF with one or more narrow peaks at specified positions.

        Args:
            peaks (list of float): Positions of peaks (between 0 and 1).
            opacity (float): Opacity value at the peak(s).
            eps (float): Half-width of each peak (controls sharpness).
            lut_size (int): LUT size.
            base (float): Base opacity outside peaks.

        Returns:
            OpacityTransferFunction instance.
        """
        control_points = [(0.0, base)]
        for peak in sorted(peaks):
            left = max(0.0, peak - eps)
            right = min(1.0, peak + eps)
            if left > control_points[-1][0]:
                control_points.append((left, base))
            control_points.append((peak, opacity))
            if right > peak:
                control_points.append((right, base))
        if control_points[-1][0] < 1.0:
            control_points.append((1.0, base))
        return cls(control_points, lut_size=lut_size)

    def __call__(self, size=None):
        """alias for to_lut(size=None)"""
        self.to_lut(size)
        return self.to_lut(size)

    def to_lut(self, size=None):
        """
        Generate a lookup table (numpy array) for fast mapping.
        Vectorized version.
        """
        if size is None:
            size = self.lut_size
        x = np.linspace(0, 1, size)
        xp, fp = zip(*self.control_points)
        lut = np.interp(x, xp, fp).astype(np.float32)
        return lut

    def to_texture(self, ctx=None, size=None, moderngl_manager=None):
        """
        Generate a 2D moderngl texture (height=1) from the LUT.
        The texture has a single channel (R) for opacity.
        ctx: moderngl context (for backward compatibility)
        moderngl_manager: ModernGLManager instance (preferred)
        Returns: moderngl.Texture or texture unit (int)
        """
        if moderngl_manager is not None:
            # New way: use ModernGLManager
            self.lut_size = size or self.lut_size
            lut = self.to_lut(self.lut_size)
            return moderngl_manager.create_lut_texture(lut, channels=1)
        elif ctx is not None:
            # Legacy way: direct ModernGL context (for backward compatibility)
            import moderngl

            if ctx is None:
                raise ValueError(
                    "A moderngl context must be provided to create a texture."
                )
            self.lut_size = size or self.lut_size
            lut = self.to_lut(self.lut_size)
            data = lut.reshape((self.lut_size, 1)).astype(np.float32)
            tex = ctx.texture((self.lut_size, 1), 1, data.tobytes(), dtype="f4")
            tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            tex.repeat_x = False
            tex.repeat_y = False
            return tex
        else:
            raise ValueError("Either moderngl_manager or ctx must be provided.")


class ColorTransferFunction:
    """
    Maps scalar values (e.g., density) to RGB color values for volume rendering.
    Supports piecewise linear color transfer functions and can output a 1D LUT as a 2D texture.
    """

    def __init__(self, control_points=None, lut_size=256):
        """
        control_points: list of (scalar, (r, g, b)) tuples, sorted by scalar.
        lut_size: size of the lookup table and texture (instance variable).
        Example: [(0.0, (0,0,0)), (0.5, (1,0,0)), (1.0, (1,1,1))]
        """
        if control_points is None:
            control_points = [(0.0, (0.0, 0.0, 0.0)), (1.0, (1.0, 1.0, 1.0))]
        self.control_points = sorted(control_points)
        self.lut_size = lut_size

    @classmethod
    def from_matplotlib_colormap(cls, cmap, lut_size=256):
        """
        Create a ColorTransferFunction from a matplotlib colormap.
        Args:
            cmap: a matplotlib colormap instance (e.g., plt.get_cmap('viridis'))
            lut_size: number of control points to sample
        Returns:
            ColorTransferFunction instance
        """
        x = np.linspace(0, 1, lut_size)
        colors = cmap(x)
        # Use only RGB, ignore alpha if present
        if colors.shape[1] == 4:
            colors = colors[:, :3]
        control_points = [
            (float(xi), tuple(map(float, rgb))) for xi, rgb in zip(x, colors)
        ]
        return cls(control_points, lut_size=lut_size)

    def to_lut(self, size=None):
        """
        Generate a lookup table (numpy array) for fast mapping.
        Output shape: (size, 3) for RGB.
        """
        if size is None:
            size = self.lut_size
        x = np.linspace(0, 1, size)
        xp, fp = zip(*self.control_points)
        fp = np.array(fp)
        lut = np.empty((size, 3), dtype=np.float32)
        for c in range(3):
            lut[:, c] = np.interp(x, xp, fp[:, c])
        return lut

    def to_texture(self, ctx=None, size=None, moderngl_manager=None):
        """
        Generate a 2D moderngl texture (height=1) from the LUT.
        The texture has 3 channels (RGB).
        ctx: moderngl context (for backward compatibility)
        moderngl_manager: ModernGLManager instance (preferred)
        Returns: moderngl.Texture or texture unit (int)
        """
        if moderngl_manager is not None:
            # New way: use ModernGLManager
            self.lut_size = size or self.lut_size
            lut = self.to_lut(self.lut_size)
            return moderngl_manager.create_lut_texture(lut, channels=3)
        elif ctx is not None:
            # Legacy way: direct ModernGL context (for backward compatibility)
            import moderngl

            if ctx is None:
                raise ValueError(
                    "A moderngl context must be provided to create a texture."
                )
            self.lut_size = size or self.lut_size
            lut = self.to_lut(self.lut_size)
            data = lut.reshape((self.lut_size, 1, 3)).astype(np.float32)
            tex = ctx.texture((self.lut_size, 1), 3, data.tobytes(), dtype="f4")
            tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            tex.repeat_x = False
            tex.repeat_y = False
            return tex
        else:
            raise ValueError("Either moderngl_manager or ctx must be provided.")
