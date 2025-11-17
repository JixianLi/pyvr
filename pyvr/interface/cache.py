"""
Caching utilities for interface components.

This module provides caching for computationally expensive operations
like histogram calculation.
"""

import os
import pickle
import hashlib
from pathlib import Path
from typing import Optional, Tuple
import numpy as np


# Cache directory (in tmp_dev, which is in .gitignore)
CACHE_DIR = Path(__file__).parent.parent.parent / "tmp_dev" / "histogram_cache"


def _ensure_cache_dir() -> None:
    """Ensure cache directory exists."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def compute_volume_hash(volume_data: np.ndarray) -> str:
    """
    Compute hash for volume data for cache key generation.

    Uses shape + sample of data for efficiency on large volumes.

    Args:
        volume_data: 3D numpy array

    Returns:
        Hexadecimal hash string

    Example:
        >>> data = np.random.rand(128, 128, 128)
        >>> hash_key = compute_volume_hash(data)
        >>> len(hash_key)
        64
    """
    hasher = hashlib.sha256()

    # Hash shape
    hasher.update(str(volume_data.shape).encode())

    # Hash dtype
    hasher.update(str(volume_data.dtype).encode())

    # Sample data for hash (don't hash entire array for performance)
    # Use 1000 evenly spaced samples
    total_elements = volume_data.size
    if total_elements > 1000:
        indices = np.linspace(0, total_elements - 1, 1000, dtype=int)
        sample = volume_data.flat[indices]
    else:
        sample = volume_data.ravel()

    hasher.update(sample.tobytes())

    return hasher.hexdigest()


def get_cached_histogram(
    volume_data: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Retrieve cached histogram for volume if available.

    Args:
        volume_data: 3D numpy array

    Returns:
        Tuple of (bin_edges, counts) if cached, None otherwise

    Example:
        >>> data = np.random.rand(128, 128, 128)
        >>> histogram = get_cached_histogram(data)
        >>> if histogram is None:
        ...     # Compute histogram
    """
    _ensure_cache_dir()

    cache_key = compute_volume_hash(volume_data)
    cache_file = CACHE_DIR / f"{cache_key}.pkl"

    if cache_file.exists():
        try:
            with open(cache_file, "rb") as f:
                cached_data = pickle.load(f)
                return cached_data["bin_edges"], cached_data["counts"]
        except (pickle.UnpicklingError, KeyError, EOFError):
            # Corrupted cache file, remove it
            cache_file.unlink()
            return None

    return None


def cache_histogram(
    volume_data: np.ndarray, bin_edges: np.ndarray, counts: np.ndarray
) -> None:
    """
    Save histogram to cache.

    Args:
        volume_data: 3D numpy array (used for cache key)
        bin_edges: Histogram bin edges
        counts: Histogram counts per bin

    Example:
        >>> data = np.random.rand(128, 128, 128)
        >>> edges, counts = compute_log_histogram(data)
        >>> cache_histogram(data, edges, counts)
    """
    _ensure_cache_dir()

    cache_key = compute_volume_hash(volume_data)
    cache_file = CACHE_DIR / f"{cache_key}.pkl"

    cached_data = {
        "bin_edges": bin_edges,
        "counts": counts,
    }

    try:
        with open(cache_file, "wb") as f:
            pickle.dump(cached_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    except (OSError, pickle.PicklingError) as e:
        # Cache write failed, not critical - just log
        print(f"Warning: Failed to cache histogram: {e}")


def compute_log_histogram(
    volume_data: np.ndarray, num_bins: int = 256
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute log-scale histogram of volume data.

    Args:
        volume_data: 3D numpy array
        num_bins: Number of histogram bins (default: 256)

    Returns:
        Tuple of (bin_edges, log_counts) where log_counts are log10(counts + 1)

    Example:
        >>> data = np.random.rand(128, 128, 128)
        >>> edges, log_counts = compute_log_histogram(data)
        >>> assert len(log_counts) == 256
    """
    # Compute histogram in [0, 1] range
    counts, bin_edges = np.histogram(
        volume_data.ravel(), bins=num_bins, range=(0.0, 1.0)
    )

    # Apply log scale for better visibility (log10(count + 1))
    log_counts = np.log10(counts + 1)

    return bin_edges, log_counts


def get_or_compute_histogram(
    volume_data: np.ndarray, num_bins: int = 256
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get histogram from cache or compute if not cached.

    This is the main entry point for histogram access.

    Args:
        volume_data: 3D numpy array
        num_bins: Number of histogram bins (default: 256)

    Returns:
        Tuple of (bin_edges, log_counts)

    Example:
        >>> from pyvr.volume import Volume
        >>> volume = Volume(data=np.random.rand(128, 128, 128))
        >>> edges, counts = get_or_compute_histogram(volume.data)
    """
    # Try cache first
    cached = get_cached_histogram(volume_data)
    if cached is not None:
        return cached

    # Compute if not cached
    bin_edges, log_counts = compute_log_histogram(volume_data, num_bins)

    # Cache for future use
    cache_histogram(volume_data, bin_edges, log_counts)

    return bin_edges, log_counts


def clear_histogram_cache() -> int:
    """
    Clear all cached histograms.

    Returns:
        Number of cache files deleted

    Example:
        >>> num_deleted = clear_histogram_cache()
        >>> print(f"Cleared {num_deleted} cached histograms")
    """
    if not CACHE_DIR.exists():
        return 0

    deleted = 0
    for cache_file in CACHE_DIR.glob("*.pkl"):
        try:
            cache_file.unlink()
            deleted += 1
        except OSError:
            pass

    return deleted
