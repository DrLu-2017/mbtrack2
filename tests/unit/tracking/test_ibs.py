import numpy as np
import pytest

from mbtrack2 import IntrabeamScattering

pytestmark = pytest.mark.unit
from utility_test_functions import assert_attr_changed


@pytest.fixture
def generate_ibs(ring_with_at_lattice):

    def generate(ring=ring_with_at_lattice,
                 model="CIMP",
                 n_points=100,
                 n_bin=100):
        ibs = IntrabeamScattering(ring=ring,
                                  model=model,
                                  n_points=n_points,
                                  n_bin=n_bin)
        return ibs

    return generate


class TestIntrabeamScattering:

    # Track a bunch and validate the momentum changes
    @pytest.mark.parametrize("model", [("Bane"), ("CIMP"), ("PM"), ("PS")])
    def test_track_model_change_attr(self, small_bunch, generate_ibs, model):
        ibs = generate_ibs(model=model)
        if model == "Bane":
            assert_attr_changed(ibs,
                                small_bunch,
                                attrs_changed=["xp", "delta"])
        else:
            assert_attr_changed(ibs, small_bunch)

    # Initialize with an invalid model name and expect a ValueError
    def test_initialize_invalid_model(self, demo_ring):
        with pytest.warns(UserWarning):
            with pytest.raises(ValueError):
                IntrabeamScattering(ring=demo_ring, model="InvalidModel")

    # Track a bunch with n_bin set to 1 and verify uniform distribution handling
    def test_track_with_n_bin_one(self, small_bunch, generate_ibs):
        ibs = generate_ibs(n_bin=1)
        assert_attr_changed(ibs, small_bunch)

    # Handle a case where the bunch is empty and ensure no operations are performed
    def test_empty_bunch_handling(self, small_bunch, generate_ibs):
        small_bunch.alive[:] = False
        ibs = generate_ibs()
        assert_attr_changed(ibs, small_bunch, change=False)

    # Handle a case where the bunch has lost some macro-particles
    def test_partial_empty_bunch_handling(self, small_bunch, generate_ibs):
        small_bunch.alive[0:5] = False
        ibs = generate_ibs()
        assert_attr_changed(ibs, small_bunch)

    # Test the warning when lattice file is not loaded and optics are approximated
    def test_warning_for_approximated_optics(self, demo_ring):
        demo_ring.optics.use_local_values = True
        with pytest.warns(UserWarning):
            IntrabeamScattering(ring=demo_ring, model="CIMP")

    # Validate the computation of growth rates for each model
    def test_growth_rate_computation(self, demo_ring, small_bunch,
                                     generate_ibs):
        for model in ["PS", "PM", "Bane", "CIMP"]:
            ibs = generate_ibs(model=model)
            ibs.initialize(small_bunch)
            if model in ["PS", "PM"]:
                vabq, v1aq, v1bq = ibs.scatter()
                T_x, T_y, T_p = ibs.get_scatter_T(vabq=vabq,
                                                  v1aq=v1aq,
                                                  v1bq=v1bq)
            elif model == "Bane":
                gval = ibs.scatter()
                T_x, T_y, T_p = ibs.get_scatter_T(gval=gval)
            elif model == "CIMP":
                g_ab, g_ba = ibs.scatter()
                T_x, T_y, T_p = ibs.get_scatter_T(g_ab=g_ab, g_ba=g_ba)
            assert T_x >= 0 and T_y >= 0 and T_p >= 0
