import pytest

pytestmark = pytest.mark.unit
import numpy as np
import pandas as pd
from utility_test_functions import assert_attr_changed

from mbtrack2 import LongRangeResistiveWall, Resonator, WakePotential


@pytest.fixture
def generate_resonator():

    def generate(time=np.linspace(-1000e-12, 1000e-12, int(1e5)),
                 frequency=np.linspace(0.1e6, 100e9, int(1e5)),
                 Rs=100,
                 fr=1e9,
                 Q=20,
                 plane="long"):
        res = Resonator(time, frequency, Rs, fr, Q, plane)
        return res

    return generate


@pytest.fixture
def generate_wakepotential(demo_ring, generate_resonator):

    def generate(ring=demo_ring,
                 wakefield=None,
                 n_bin=80,
                 interp_on_postion=True,
                 **kwargs):
        if wakefield is None:
            wakefield = generate_resonator(**kwargs)
        wp = WakePotential(ring, wakefield, n_bin, interp_on_postion)
        return wp

    return generate


class TestWakePotential:

    # Compute charge density profile for a given Bunch object
    def test_charge_density_profile(self, generate_wakepotential, small_bunch):
        wp = generate_wakepotential()
        wp.charge_density(small_bunch)
        assert len(wp.rho) == len(wp.tau)
        assert wp.dtau > 0

    # Calculate dipole moment for a Bunch object on specified plane
    @pytest.mark.parametrize("plane", [("x"), ("y")])
    def test_dipole_moment_calculation(self, generate_wakepotential,
                                       small_bunch, plane):
        wp = generate_wakepotential()
        wp.charge_density(small_bunch)
        dipole = wp.dipole_moment(small_bunch, plane, wp.tau)
        assert len(dipole) == len(wp.tau)

    # Prepare wake function for a specified wake type
    @pytest.mark.parametrize("plane", [("x"), ("y"), ("long")])
    def test_prepare_wakefunction_smaller_window(self, generate_wakepotential,
                                                 plane):
        wp = generate_wakepotential(plane=plane)
        tau = np.linspace(-100e-12, 100e-12, int(1e5))
        if plane == "x" or plane == "y":
            comp = f"W{plane}dip"
        else:
            comp = "Wlong"
        tau0, dtau0, W0 = wp.prepare_wakefunction(comp, tau)
        assert len(tau0) > 0
        assert len(W0) > 0

    # Prepare wake function for a specified wake type
    @pytest.mark.parametrize("plane", [("x"), ("y"), ("long")])
    def test_prepare_wakefunction_larger_window(self, generate_wakepotential,
                                                plane):
        wp = generate_wakepotential(plane=plane)
        tau = np.linspace(-2000e-12, 2000e-12, int(1e5))
        if plane == "x" or plane == "y":
            comp = f"W{plane}dip"
        else:
            comp = "Wlong"
        tau0, dtau0, W0 = wp.prepare_wakefunction(comp, tau)
        assert len(tau0) > 0
        assert len(W0) > 0

    # Compute wake potential for a Bunch object and wake type
    @pytest.mark.parametrize("plane", [("x"), ("y"), ("long")])
    def test_compute_wakepotential(self, generate_wakepotential, small_bunch,
                                   plane):
        wp = generate_wakepotential(plane=plane)
        wp.charge_density(small_bunch)
        if plane == "x" or plane == "y":
            comp = f"W{plane}dip"
        else:
            comp = "Wlong"
        tau0, Wp = wp.get_wakepotential(small_bunch, comp)
        assert len(Wp) == len(tau0)

    # Track a Bunch object through the WakePotential element
    @pytest.mark.parametrize("plane, attr", [("x", "xp"), ("y", "yp"),
                                             ("long", "delta")])
    def test_track_bunch(self, generate_wakepotential, small_bunch, plane,
                         attr):
        wp = generate_wakepotential(plane=plane)
        assert_attr_changed(wp, small_bunch, attrs_changed=[attr])

    # Track a Bunch object through the WakePotential element
    @pytest.mark.parametrize("plane, attr", [("x", "xp"), ("y", "yp"),
                                             ("long", "delta")])
    def test_track_bunch_no_exact_interp(self, generate_wakepotential,
                                         small_bunch, plane, attr):
        wp = generate_wakepotential(plane=plane, interp_on_postion=False)
        assert_attr_changed(wp, small_bunch, attrs_changed=[attr])

    # Handle empty Bunch object in track method
    @pytest.mark.parametrize("plane, attr", [("x", "xp"), ("y", "yp"),
                                             ("long", "delta")])
    def test_track_empty_bunch(self, generate_wakepotential, small_bunch,
                               plane, attr):
        wp = generate_wakepotential(plane=plane)
        small_bunch.alive[:] = False
        assert_attr_changed(wp,
                            small_bunch,
                            attrs_changed=[attr],
                            change=False)

    # Manage non-uniformly sampled wake functions
    def test_non_uniform_sampling_error(self, generate_wakepotential):
        with pytest.raises(ValueError):
            wp = generate_wakepotential(time=np.logspace(1, 5, 50))
            wp.check_sampling()

    # Calculate reference loss factor and compare to Gaussian bunch
    @pytest.mark.parametrize("plane", [("x"), ("y"), ("long")])
    def test_reference_loss_factor_comparison(self, generate_wakepotential,
                                              large_bunch, plane):
        # Generate a WakePotential instance
        wp = generate_wakepotential(plane=plane)

        # Calculate the reference loss factor using the small bunch
        large_bunch["x"] += 1e-3
        large_bunch["y"] += 1e-3
        wp.track(large_bunch)
        loss_data = wp.reference_loss(large_bunch)

        # Check if the loss data is a DataFrame and contains expected columns
        assert isinstance(loss_data,
                          pd.DataFrame), "Loss data should be a DataFrame"
        assert 'TD factor' in loss_data.columns, "DataFrame should contain 'TD factor' column"
        assert 'FD factor' in loss_data.columns, "DataFrame should contain 'FD factor' column"
        assert 'Relative error [%]' in loss_data.columns, "DataFrame should contain 'Relative error [%]' column"

        # Verify that the calculated loss factors are within a reasonable range
        for index, row in loss_data.iterrows():
            assert abs(row['Relative error [%]']
                       ) < 1, f"Relative error for {index} is too high"

    # Reduce wake function sampling by a specified factor
    def test_reduce_sampling(self, generate_wakepotential):
        # Create a WakePotential instance with a mock wakefield
        wp = generate_wakepotential()

        # Original length of the wake function data
        original_length = len(wp.wakefield.Wlong.data.index)

        # Define a reduction factor
        reduction_factor = 2

        # Reduce the sampling of the wake function
        wp.reduce_sampling(reduction_factor)

        # New length of the wake function data after reduction
        new_length = len(wp.wakefield.Wlong.data.index)

        # Assert that the new length is approximately half of the original length
        assert new_length == original_length // reduction_factor

    # Test plotting functions with different plot options
    def test_plot_last_wake_with_various_options(self, generate_wakepotential,
                                                 small_bunch):
        wp = generate_wakepotential(plane="x")
        wp.track(small_bunch)

        # Test with default options
        fig1 = wp.plot_last_wake('Wxdip')
        assert fig1 is not None

        # Test with plot_rho=False
        fig2 = wp.plot_last_wake('Wxdip', plot_rho=False)
        assert fig2 is not None

        # Test with plot_dipole=True
        fig3 = wp.plot_last_wake('Wxdip', plot_dipole=True)
        assert fig3 is not None

        # Test with plot_wake_function=False
        fig4 = wp.plot_last_wake('Wxdip', plot_wake_function=False)
        assert fig4 is not None

    @pytest.mark.parametrize("plane", [("x"), ("y"), ("long")])
    def test_get_gaussian_wakepotential(self, generate_wakepotential, plane):
        wp = generate_wakepotential(plane=plane)
        if plane == "x" or plane == "y":
            comp = f"W{plane}dip"
        else:
            comp = "Wlong"

        tau0, W0, Wp, profile0, dipole0 = wp.get_gaussian_wakepotential(
            1e-12, wake_type=comp)

        assert tau0 is not None
        assert W0 is not None
        assert Wp is not None
        assert profile0 is not None
        assert dipole0 is not None

    @pytest.mark.parametrize("plane", [("x"), ("y"), ("long")])
    def test_plot_gaussian_wake(self, generate_wakepotential, plane):
        wp = generate_wakepotential(plane=plane)
        if plane == "x" or plane == "y":
            comp = f"W{plane}dip"
        else:
            comp = "Wlong"
        fig = wp.plot_gaussian_wake(sigma=10e-12, wake_type=comp)
        assert fig is not None


@pytest.fixture
def generate_lrrw(demo_ring, beam_uniform):

    def generate(ring=demo_ring,
                 beam=beam_uniform,
                 length=demo_ring.L,
                 rho=1e-6,
                 radius=6e-12,
                 types=["Wlong", "Wxdip", "Wydip"],
                 nt=50,
                 x3=None,
                 y3=None,
                 average_beta=None):
        lrrw = LongRangeResistiveWall(ring=ring,
                                      beam=beam,
                                      length=length,
                                      rho=rho,
                                      radius=radius,
                                      types=types,
                                      nt=nt,
                                      x3=x3,
                                      y3=y3,
                                      average_beta=average_beta)
        return lrrw

    return generate


class TestLongRangeResistiveWall:

    # Test tracking w/ uniform beam
    @pytest.mark.parametrize("types, attr", [("Wxdip", "xp"), ("Wydip", "yp"),
                                             ("Wlong", "delta")])
    def test_track(self, generate_lrrw, beam_uniform, types, attr):
        lrrw = generate_lrrw(types=types)
        for i, bunch in enumerate(beam_uniform.not_empty):
            setattr(self, "initial_attr_{i}", bunch[attr].copy())
        with np.errstate(over='ignore'):
            lrrw.track(beam_uniform)
        for i, bunch in enumerate(beam_uniform.not_empty):
            assert not np.allclose(getattr(self, "initial_attr_{i}"),
                                   bunch[attr])

    # Test tracking w/ non-uniform beam
    @pytest.mark.parametrize("types, attr", [("Wxdip", "xp"), ("Wydip", "yp"),
                                             ("Wlong", "delta")])
    def test_track_non_uniform(self, generate_lrrw, beam_non_uniform, types,
                               attr):
        lrrw = generate_lrrw(types=types, beam=beam_non_uniform)
        for i, bunch in enumerate(beam_non_uniform.not_empty):
            setattr(self, "initial_attr_{i}", bunch[attr].copy())
        with np.errstate(over='ignore'):
            lrrw.track(beam_non_uniform)
        for i, bunch in enumerate(beam_non_uniform.not_empty):
            assert not np.allclose(getattr(self, "initial_attr_{i}"),
                                   bunch[attr])

    # Test tracking w/ mpi beam
    @pytest.mark.parametrize("types, attr", [("Wxdip", "xp"), ("Wydip", "yp"),
                                             ("Wlong", "delta")])
    def test_track_mpi(self, generate_lrrw, beam_1bunch_mpi, types, attr):
        lrrw = generate_lrrw(types=types, beam=beam_1bunch_mpi)
        bunch = beam_1bunch_mpi[beam_1bunch_mpi.mpi.bunch_num]
        initial_attr = bunch[attr].copy()
        with np.errstate(over='ignore'):
            lrrw.track(beam_1bunch_mpi)
            lrrw.track(
                beam_1bunch_mpi
            )  # two turns are needed to see the effect of the bunch on itself
        assert not np.allclose(initial_attr, bunch[attr])

    # test that two turns are needed to see the effect of the bunch on itself
    def test_kick_signle_bunch(self, generate_lrrw, beam_1bunch_mpi):
        lrrw = generate_lrrw(beam=beam_1bunch_mpi)
        with np.errstate(over='ignore'):
            lrrw.track(beam_1bunch_mpi)
            kick = lrrw.get_kick(0, "Wlong")
        assert kick == 0

    @pytest.mark.parametrize("plane", [("long"), ("x"), ("y")])
    def test_wake(self, generate_lrrw, plane):
        lrrw = generate_lrrw()
        t = 1e-9
        if plane == "long":
            wake = lrrw.Wlong(t)
        else:
            wake = lrrw.Wdip(t, plane)
        assert isinstance(wake, float)

    # Update tables with a Beam object and verify the correct table updates
    @pytest.mark.parametrize("beam", [("beam_uniform"), ("beam_1bunch_mpi")])
    def test_update_tables_with_beam(self, generate_lrrw, request, beam):
        beam = request.getfixturevalue(beam)
        lrrw = generate_lrrw()
        initial_tau = np.copy(lrrw.tau)
        initial_x = np.copy(lrrw.x)
        initial_y = np.copy(lrrw.y)
        initial_charge = np.copy(lrrw.charge)
        lrrw.update_tables(beam)
        assert not np.array_equal(lrrw.tau, initial_tau)
        assert not np.array_equal(lrrw.x, initial_x)
        assert not np.array_equal(lrrw.y, initial_y)
        assert not np.array_equal(lrrw.charge, initial_charge)

    # Verify the behavior when the approximated wake functions are not valid
    def test_invalid_approximated_wake_functions(self, demo_ring,
                                                 beam_uniform):
        demo_ring.T1 = 1e-18  # Set T1 to a very small value to trigger the error
        with pytest.raises(ValueError):
            LongRangeResistiveWall(demo_ring, beam_uniform, 100, 1e-6, 6e-12)

    # Test track_bunch method with different wake types and verify correct kick application
    def test_track_bunch_kick_application(self, generate_lrrw, beam_uniform):
        # Arrange
        lrrw = generate_lrrw()
        bunch = beam_uniform[0]
        initial_delta = bunch["delta"].copy()
        initial_xp = bunch["xp"].copy()
        initial_yp = bunch["yp"].copy()

        # Act
        lrrw.track_bunch(bunch, rank=0)

        # Assert
        if "Wlong" in lrrw.types:
            assert not np.array_equal(
                bunch["delta"],
                initial_delta), "Longitudinal kick not applied correctly"
        if "Wxdip" in lrrw.types:
            assert not np.array_equal(
                bunch["xp"],
                initial_xp), "Horizontal dipole kick not applied correctly"
        if "Wydip" in lrrw.types:
            assert not np.array_equal(
                bunch["yp"],
                initial_yp), "Vertical dipole kick not applied correctly"

    def test_average_beta_local(self, generate_lrrw):
        lrrw = generate_lrrw()
        assert lrrw.norm_x == 1
        assert lrrw.norm_y == 1

    def test_average_beta_mean(self, generate_lrrw,
                               generate_ring_with_at_lattice):
        ring = generate_ring_with_at_lattice(tracking_loc=5)
        lrrw = generate_lrrw(ring=ring)
        assert lrrw.norm_x != 1
        assert lrrw.norm_y != 1
