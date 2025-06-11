import pytest

pytestmark = pytest.mark.unit
import numpy as np
from utility_test_functions import assert_attr_changed

from mbtrack2 import TransverseSpaceCharge


class TestTransverseSpaceCharge:

    # Track a bunch with valid x, y coordinates through the space charge element
    def test_track_with_track_alive(self, demo_ring, large_bunch):
        tsc = TransverseSpaceCharge(demo_ring, 1000.0)
        large_bunch.alive[5:700] = False
        assert_attr_changed(tsc, large_bunch, attrs_changed=["xp", "yp"])

    # Track a bunch with valid x, y coordinates through the space charge element
    def test_track_without_track_alive(self, demo_ring, large_bunch):
        tsc = TransverseSpaceCharge(demo_ring, 1000.0)
        large_bunch.track_alive = False
        assert_attr_changed(tsc, large_bunch, attrs_changed=["xp", "yp"])

    # Initialize with zero interaction length and verify no kicks are applied
    def test_zero_interaction_length_no_kicks(self, demo_ring, small_bunch):
        tsc = TransverseSpaceCharge(demo_ring, 0.0)
        assert_attr_changed(tsc,
                            small_bunch,
                            attrs_changed=["xp", "yp"],
                            change=False)

    # Handle a bunch with bins having zero particles without errors
    def test_handle_bins_with_zero_particles(self, demo_ring, small_bunch):
        small_bunch.alive[:] = False
        tsc = TransverseSpaceCharge(demo_ring, 1.0)
        tsc.track(small_bunch)
        assert True

    # Test with different n_bins
    @pytest.mark.parametrize("n_bins", [(1), (2), (100)])
    def test_n_bins_zero_behavior(self, demo_ring, small_bunch, n_bins):
        tsc = TransverseSpaceCharge(demo_ring, 1.0, n_bins=n_bins)
        tsc.track(small_bunch)
        assert True
