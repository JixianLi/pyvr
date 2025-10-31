"""Tests for histogram caching functionality."""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from pyvr.interface import cache


@pytest.fixture
def temp_cache_dir(monkeypatch):
    """Create temporary cache directory for testing."""
    temp_dir = Path(tempfile.mkdtemp())
    monkeypatch.setattr(cache, 'CACHE_DIR', temp_dir)

    yield temp_dir

    # Cleanup
    shutil.rmtree(temp_dir)


class TestVolumeHash:
    """Tests for volume hash computation."""

    def test_compute_volume_hash_consistent(self):
        """Test hash is consistent for same data."""
        data = np.random.rand(64, 64, 64)

        hash1 = cache.compute_volume_hash(data)
        hash2 = cache.compute_volume_hash(data)

        assert hash1 == hash2

    def test_compute_volume_hash_different_data(self):
        """Test different data produces different hash."""
        data1 = np.random.rand(64, 64, 64)
        data2 = np.random.rand(64, 64, 64)

        hash1 = cache.compute_volume_hash(data1)
        hash2 = cache.compute_volume_hash(data2)

        assert hash1 != hash2

    def test_compute_volume_hash_different_shape(self):
        """Test different shape produces different hash."""
        data1 = np.ones((64, 64, 64))
        data2 = np.ones((128, 128, 128))

        hash1 = cache.compute_volume_hash(data1)
        hash2 = cache.compute_volume_hash(data2)

        assert hash1 != hash2

    def test_compute_volume_hash_length(self):
        """Test hash is SHA256 hex (64 characters)."""
        data = np.random.rand(64, 64, 64)
        hash_str = cache.compute_volume_hash(data)

        assert len(hash_str) == 64
        assert all(c in '0123456789abcdef' for c in hash_str)


class TestHistogramCaching:
    """Tests for histogram caching."""

    def test_cache_and_retrieve_histogram(self, temp_cache_dir):
        """Test caching and retrieving histogram."""
        data = np.random.rand(64, 64, 64)
        bin_edges = np.linspace(0, 1, 257)
        counts = np.random.rand(256)

        # Cache histogram
        cache.cache_histogram(data, bin_edges, counts)

        # Retrieve from cache
        cached_edges, cached_counts = cache.get_cached_histogram(data)

        assert np.allclose(cached_edges, bin_edges)
        assert np.allclose(cached_counts, counts)

    def test_get_cached_histogram_not_cached(self, temp_cache_dir):
        """Test returns None for uncached volume."""
        data = np.random.rand(64, 64, 64)

        result = cache.get_cached_histogram(data)

        assert result is None

    def test_cache_creates_file(self, temp_cache_dir):
        """Test cache file is created."""
        data = np.random.rand(64, 64, 64)
        bin_edges = np.linspace(0, 1, 257)
        counts = np.random.rand(256)

        cache.cache_histogram(data, bin_edges, counts)

        # Check cache file exists
        cache_files = list(temp_cache_dir.glob("*.pkl"))
        assert len(cache_files) == 1

    def test_clear_histogram_cache(self, temp_cache_dir):
        """Test clearing cache."""
        # Cache multiple histograms
        for _ in range(3):
            data = np.random.rand(64, 64, 64)
            bin_edges = np.linspace(0, 1, 257)
            counts = np.random.rand(256)
            cache.cache_histogram(data, bin_edges, counts)

        # Clear cache
        deleted = cache.clear_histogram_cache()

        assert deleted == 3
        assert len(list(temp_cache_dir.glob("*.pkl"))) == 0


class TestLogHistogram:
    """Tests for log-scale histogram computation."""

    def test_compute_log_histogram_shape(self):
        """Test histogram has correct shape."""
        data = np.random.rand(64, 64, 64)

        bin_edges, log_counts = cache.compute_log_histogram(data, num_bins=256)

        assert len(bin_edges) == 257  # n_bins + 1
        assert len(log_counts) == 256

    def test_compute_log_histogram_range(self):
        """Test histogram covers [0, 1] range."""
        data = np.random.rand(64, 64, 64)

        bin_edges, log_counts = cache.compute_log_histogram(data)

        assert bin_edges[0] == pytest.approx(0.0)
        assert bin_edges[-1] == pytest.approx(1.0)

    def test_compute_log_histogram_log_scale(self):
        """Test counts are in log scale."""
        # Create data with known distribution
        data = np.zeros((64, 64, 64))
        data[32, 32, 32] = 1.0  # Single spike at 1.0

        bin_edges, log_counts = cache.compute_log_histogram(data, num_bins=10)

        # All log counts should be >= 0 (log10(n + 1))
        assert np.all(log_counts >= 0)

        # First bin should have highest count (contains all the zeros)
        assert log_counts[0] > log_counts[-1]

    def test_compute_log_histogram_custom_bins(self):
        """Test custom number of bins."""
        data = np.random.rand(64, 64, 64)

        bin_edges, log_counts = cache.compute_log_histogram(data, num_bins=128)

        assert len(log_counts) == 128


class TestGetOrComputeHistogram:
    """Tests for get_or_compute_histogram convenience function."""

    def test_get_or_compute_computes_first_time(self, temp_cache_dir):
        """Test histogram is computed on first call."""
        data = np.random.rand(64, 64, 64)

        bin_edges, log_counts = cache.get_or_compute_histogram(data)

        # Should have computed and cached
        assert bin_edges is not None
        assert log_counts is not None

        # Cache file should exist
        assert len(list(temp_cache_dir.glob("*.pkl"))) == 1

    def test_get_or_compute_uses_cache_second_time(self, temp_cache_dir, monkeypatch):
        """Test histogram is retrieved from cache on second call."""
        data = np.random.rand(64, 64, 64)

        # First call computes
        bin_edges1, log_counts1 = cache.get_or_compute_histogram(data)

        # Mock compute to detect if called
        compute_called = []
        original_compute = cache.compute_log_histogram
        def mock_compute(*args, **kwargs):
            compute_called.append(True)
            return original_compute(*args, **kwargs)

        monkeypatch.setattr(cache, 'compute_log_histogram', mock_compute)

        # Second call should use cache
        bin_edges2, log_counts2 = cache.get_or_compute_histogram(data)

        # Should not have called compute
        assert len(compute_called) == 0

        # Results should match
        assert np.allclose(bin_edges1, bin_edges2)
        assert np.allclose(log_counts1, log_counts2)
