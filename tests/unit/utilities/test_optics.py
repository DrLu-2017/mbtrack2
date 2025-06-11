import numpy as np
import pytest

pytestmark = pytest.mark.unit
from pathlib import Path

import at
import matplotlib.pyplot as plt

from mbtrack2 import Optics, PhysicalModel


@pytest.fixture
def local_optics():
    beta = np.array([1, 1])
    alpha = np.array([0, 0])
    dispersion = np.array([0, 0, 0, 0])
    local_optics = Optics(local_beta=beta,
                          local_alpha=alpha,
                          local_dispersion=dispersion)
    return local_optics


@pytest.fixture
def generate_at_optics():

    def generate(lattice_file=None,
                 tracking_loc=0,
                 local_beta=None,
                 local_alpha=None,
                 local_dispersion=None,
                 **kwargs):
        if lattice_file is None:
            path_to_file = Path(__file__).parent
            lattice_file = path_to_file / ".." / ".." / ".." / "examples" / "SOLEIL_OLD.mat"
        optics = Optics(lattice_file=lattice_file,
                        tracking_loc=tracking_loc,
                        local_beta=local_beta,
                        local_alpha=local_alpha,
                        local_dispersion=local_dispersion,
                        **kwargs)
        return optics

    return generate


@pytest.fixture
def at_optics(generate_at_optics):
    return generate_at_optics()


class TestOptics:

    # Initialize Optics with a valid lattice file and verify attributes are set correctly
    def test_initialize_with_at_lattice(self, at_optics):
        assert not at_optics.use_local_values
        assert at_optics.lattice is not None

    # Initialize Optics with local parameters and verify attributes are set correctly
    def test_initialize_with_local_parameters(self, local_optics):
        gamma = (1 + local_optics.local_alpha**2) / local_optics.local_beta
        assert local_optics.use_local_values
        np.testing.assert_allclose(gamma, local_optics.local_gamma)

    # Load a lattice using load_from_AT and verify optic functions are interpolated
    def test_interpolation(self, at_optics):
        lattice = at_optics.lattice

        refpts = np.arange(0, len(lattice))
        twiss0, tune, chrom, twiss = at.linopt(lattice,
                                               refpts=refpts,
                                               get_chrom=True)
        randnum = np.random.randint(0, len(lattice))
        pos = twiss.s_pos[randnum]

        np.testing.assert_allclose(at_optics.beta(pos),
                                   twiss.beta[randnum, :],
                                   rtol=1e-3,
                                   atol=1e-4)
        np.testing.assert_allclose(at_optics.alpha(pos),
                                   twiss.alpha[randnum, :],
                                   rtol=1e-3,
                                   atol=1e-4)
        np.testing.assert_allclose(at_optics.mu(pos),
                                   twiss.mu[randnum, :],
                                   rtol=1e-3,
                                   atol=1e-4)
        np.testing.assert_allclose(at_optics.dispersion(pos),
                                   twiss.dispersion[randnum, :],
                                   rtol=1e-3,
                                   atol=1e-4)

    # Retrieve beta, alpha, gamma, dispersion, and mu functions at specified positions
    def test_local_optic_functions(self, local_optics):
        position = np.array([0.5])
        beta = np.squeeze(local_optics.beta(position))
        alpha = np.squeeze(local_optics.alpha(position))
        gamma = np.squeeze(local_optics.gamma(position))
        dispersion = np.squeeze(local_optics.dispersion(position))
        mu = np.squeeze(local_optics.mu(position))

        np.testing.assert_allclose(beta, local_optics.local_beta)
        np.testing.assert_allclose(alpha, local_optics.local_alpha)
        np.testing.assert_allclose(gamma, local_optics.local_gamma)
        np.testing.assert_allclose(dispersion, local_optics.local_dispersion)
        np.testing.assert_allclose(mu, np.array([0, 0]))

    # Plot optical variables and verify the plot is generated correctly
    def test_optics_plot(self, local_optics):
        fig = local_optics.plot('beta', 'x')
        assert fig is not None

    @pytest.mark.parametrize('tracking_loc', [(0), (5), (100)])
    def test_optics_tracking_loc(self, generate_at_optics, tracking_loc):
        at_optics = generate_at_optics(tracking_loc=tracking_loc, n_points=1e4)

        np.testing.assert_allclose(at_optics.beta(tracking_loc),
                                   at_optics.local_beta)
        np.testing.assert_allclose(at_optics.alpha(tracking_loc),
                                   at_optics.local_alpha)
        np.testing.assert_allclose(at_optics.gamma(tracking_loc),
                                   at_optics.local_gamma,
                                   rtol=1e-3)  # high interpol. error
        np.testing.assert_allclose(at_optics.dispersion(tracking_loc),
                                   at_optics.local_dispersion)


class TestPhysicalModel:

    @pytest.fixture
    def generate_pm(self, ring_with_at_lattice):

        def generate(ring=ring_with_at_lattice,
                     x_right=0.01,
                     y_top=0.02,
                     shape="rect",
                     rho=1e-7,
                     x_left=None,
                     y_bottom=None,
                     n_points=1e4):
            pm = PhysicalModel(ring,
                               x_right=x_right,
                               y_top=y_top,
                               shape=shape,
                               rho=rho,
                               x_left=x_left,
                               y_bottom=y_bottom,
                               n_points=n_points)
            return pm

        return generate

    # Initialize PhysicalModel with default symmetric aperture values and verify array shapes
    def test_init_default_symmetric_aperture(self, generate_pm):
        n_points = 1e3
        pm = generate_pm(n_points=n_points)
        assert pm.x_right.shape == (int(n_points), )
        assert pm.x_left.shape == (int(n_points), )
        assert np.allclose(pm.x_left, -pm.x_right)
        assert np.allclose(pm.y_bottom, -pm.y_top)

    # Change aperture values in a specific section using change_values() with symmetric settings
    def test_change_values_symmetric(self, generate_pm):
        x_right_init = 0.01
        y_top_init = 0.02
        shape_init = 'rect'
        rho_init = 1e-7
        model = generate_pm(x_right=x_right_init,
                            y_top=y_top_init,
                            shape=shape_init,
                            rho=rho_init)
        x_right_modif = 0.015
        y_top_modif = 0.025
        shape_modif = 'elli'
        rho_modif = 1e-6
        start_position = 10
        end_position = 20
        model.change_values(start_position,
                            end_position,
                            x_right=x_right_modif,
                            y_top=y_top_modif,
                            shape=shape_modif,
                            rho=rho_modif)
        idx = (model.position > start_position) & (model.position
                                                   < end_position)
        idx2 = ((model.position[:-1] > start_position) &
                (model.position[1:] < end_position))

        # Assert modif
        assert np.allclose(model.x_right[idx], x_right_modif)
        assert np.allclose(model.x_left[idx], -x_right_modif)
        assert np.allclose(model.y_top[idx], y_top_modif)
        assert np.allclose(model.y_bottom[idx], -y_top_modif)
        assert np.all(model.shape[idx2] == shape_modif)
        assert np.allclose(model.rho[idx2], rho_modif)
        # Assert init values
        assert np.allclose(model.x_right[~idx], x_right_init)
        assert np.allclose(model.x_left[~idx], -x_right_init)
        assert np.allclose(model.y_top[~idx], y_top_init)
        assert np.allclose(model.y_bottom[~idx], -y_top_init)
        assert np.all(model.shape[~idx2] == shape_init)
        assert np.allclose(model.rho[~idx2], rho_init)

    # Change values with sym=False and verify asymmetric updates
    def test_change_values_asymmetric(self, generate_pm):
        x_right_init = 0.01
        y_top_init = 0.02
        model = generate_pm(x_right=x_right_init, y_top=y_top_init)
        model.change_values(10, 20, x_right=0.02, x_left=-0.015, sym=False)
        idx = (model.position > 10) & (model.position < 20)
        assert np.allclose(model.x_right[idx], 0.02)
        assert np.allclose(model.x_left[idx], -0.015)

    # Create tapered transition between two positions with different aperture values
    def test_taper_transition(self, generate_pm):
        x_right_init = 0.01
        y_top_init = 0.02
        shape_init = 'rect'
        rho_init = 1e-7
        model = generate_pm(x_right=x_right_init,
                            y_top=y_top_init,
                            shape=shape_init,
                            rho=rho_init)
        x_right_start = 0.01
        x_right_end = 0.025
        shape_modif = 'elli'
        rho_modif = 1e-6
        start_position = 10
        end_position = 20
        model.taper(start_position,
                    end_position,
                    x_right_start=x_right_start,
                    x_right_end=x_right_end,
                    shape=shape_modif,
                    rho=rho_modif)
        idx = (model.position > start_position) & (model.position
                                                   < end_position)
        idx2 = ((model.position[:-1] > start_position) &
                (model.position[1:] < end_position))

        # Assert edge points
        assert model.x_right[idx][0] == pytest.approx(x_right_start)
        assert model.x_right[idx][-1] == pytest.approx(x_right_end)
        assert model.x_left[idx][0] == pytest.approx(-x_right_start)
        assert model.x_left[idx][-1] == pytest.approx(-x_right_end)
        # Assert modif
        assert np.all(model.x_right[idx][1:] > x_right_start)
        assert np.all(model.x_left[idx][1:] < -x_right_start)
        assert np.all(model.x_right[idx][:-1] < x_right_end)
        assert np.all(model.x_left[idx][:-1] > -x_right_end)
        assert np.all(model.shape[idx2] == shape_modif)
        assert np.allclose(model.rho[idx2], rho_modif)
        # Assert init values
        assert np.allclose(model.x_right[~idx], x_right_init)
        assert np.allclose(model.x_left[~idx], -x_right_init)
        assert np.allclose(model.y_top, y_top_init)
        assert np.allclose(model.y_bottom, -y_top_init)
        assert np.all(model.shape[~idx2] == shape_init)
        assert np.allclose(model.rho[~idx2], rho_init)

    # Create tapered transition between two positions with different aperture values
    def test_taper_nonsym_transition(self, generate_pm):
        x_right_init = 0.01
        y_top_init = 0.02
        shape_init = 'rect'
        rho_init = 1e-7
        model = generate_pm(x_right=x_right_init,
                            y_top=y_top_init,
                            shape=shape_init,
                            rho=rho_init)
        x_right_start = 0.01
        x_right_end = 0.025
        shape_modif = 'elli'
        rho_modif = 1e-6
        start_position = 10
        end_position = 20
        model.taper(start_position,
                    end_position,
                    x_right_start=x_right_start,
                    x_right_end=x_right_end,
                    shape=shape_modif,
                    rho=rho_modif,
                    sym=False)
        idx = (model.position > start_position) & (model.position
                                                   < end_position)
        idx2 = ((model.position[:-1] > start_position) &
                (model.position[1:] < end_position))

        # Assert edge points
        assert model.x_right[idx][0] == pytest.approx(x_right_start)
        assert model.x_right[idx][-1] == pytest.approx(x_right_end)
        # Assert modif
        assert np.all(model.x_right[idx][1:] > x_right_start)
        assert np.all(model.x_right[idx][:-1] < x_right_end)
        assert np.all(model.shape[idx2] == shape_modif)
        assert np.allclose(model.rho[idx2], rho_modif)
        # Assert init values
        assert np.allclose(model.x_right[~idx], x_right_init)
        assert np.allclose(model.x_left, -x_right_init)
        assert np.allclose(model.y_top, y_top_init)
        assert np.allclose(model.y_bottom, -y_top_init)
        assert np.all(model.shape[~idx2] == shape_init)
        assert np.allclose(model.rho[~idx2], rho_init)

    # Calculate effective radius for resistive wall with default right/top settings
    def test_resistive_wall_effective_radius_default(self, generate_pm,
                                                     ring_with_at_lattice):
        model = generate_pm()
        out = model.resistive_wall_effective_radius(
            ring_with_at_lattice.optics)
        for val in out:
            assert val > 0

    # Plot aperture values and verify figure/axes objects are returned
    def test_plot_aperture_returns(self, generate_pm):
        model = generate_pm()
        fig, axs = model.plot_aperture()
        assert fig is not None
        assert axs is not None
        plt.close(fig)

    # Interpolate aperture values at arbitrary position using get_aperture()
    def test_get_aperture_interpolation(self, generate_pm):
        model = generate_pm()
        aperture = model.get_aperture(15.5)
        assert len(aperture) == 4
        assert all(isinstance(x, float) for x in aperture)

    # Handle zero resistivity sections in resistive_wall_effective_radius calculations
    def test_zero_resistivity_handling(self, generate_pm,
                                       ring_with_at_lattice):
        model = generate_pm()
        model.change_values(10, 20, rho=0)
        out = model.resistive_wall_effective_radius(
            ring_with_at_lattice.optics)
        out[0] < ring_with_at_lattice.L
        for val in out:
            assert val > 0

    # Handle start_position > end_position in change_values and taper methods
    def test_invalid_position_order(self, generate_pm):
        model = generate_pm(x_right=0.01)
        with pytest.raises(ValueError):
            model.change_values(20, 10, x_right=0.02)
        with pytest.raises(ValueError):
            model.taper(20, 10, x_right_start=0.01, x_right_end=0.03)

    # Handle positions outside ring length range
    def test_positions_outside_range(self, generate_pm):
        model = generate_pm(x_right=0.01)
        with pytest.raises(ValueError):
            model.change_values(-10, 10, x_right=0.02)
        with pytest.raises(ValueError):
            model.change_values(0, model.position[-1] + 10, x_right=0.02)
        with pytest.raises(ValueError):
            model.taper(-10, 10, x_right_start=0.01, x_right_end=0.03)
        with pytest.raises(ValueError):
            model.taper(0,
                        model.position[-1] + 10,
                        x_right_start=0.01,
                        x_right_end=0.03)
