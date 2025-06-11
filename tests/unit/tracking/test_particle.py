import matplotlib.pyplot as plt
import numpy as np
import pytest

pytestmark = pytest.mark.unit
from scipy.constants import c, e, m_p

from mbtrack2 import Beam, Bunch, Particle


class TestParticle:

    # Accessing the E_rest property to calculate rest energy
    def test_access_E_rest_property(self):
        particle = Particle(mass=m_p, charge=e)
        expected_E_rest = m_p * c**2 / e
        assert particle.E_rest == expected_E_rest


@pytest.fixture
def generate_bunch(demo_ring):

    def generate(
        ring=demo_ring,
        mp_number=1e3,
        current=1e-3,
        track_alive=True,
        alive=True,
        load_from_file=None,
        load_suffix=None,
        init_gaussian=True,
    ):
        bunch = Bunch(ring=ring,
                      mp_number=mp_number,
                      current=current,
                      track_alive=track_alive,
                      alive=alive,
                      load_from_file=load_from_file,
                      load_suffix=load_suffix)
        if init_gaussian:
            bunch.init_gaussian()
        return bunch

    return generate


@pytest.fixture
def small_bunch(generate_bunch):
    return generate_bunch(mp_number=10)


@pytest.fixture
def large_bunch(generate_bunch):
    return generate_bunch(mp_number=1e5)


class TestBunch:

    # Calculate and verify the mean, std, skew, and kurtosis of particle positions
    def test_statistics_single_mp(self, generate_bunch):
        bunch = generate_bunch(mp_number=1)
        assert len(bunch.mean) == 6
        assert len(bunch.std) == 6
        assert len(bunch.skew) == 6
        assert len(bunch.kurtosis) == 6
        assert len(bunch.emit) == 3
        assert len(bunch.cs_invariant) == 3
        assert np.any(np.isnan(bunch.mean)) == False
        assert np.any(np.isnan(bunch.std)) == False
        assert np.any(np.isnan(bunch.skew)) == False
        assert np.any(np.isnan(bunch.kurtosis)) == False
        assert np.any(np.isnan(bunch.emit)) == False
        assert np.any(np.isnan(bunch.cs_invariant)) == False

    # Initialize a Bunch with zero macro-particles and verify behavior
    def test_initialize_zero_macro_particles(self, generate_bunch):
        bunch = generate_bunch(mp_number=0)
        assert bunch.mp_number == 0
        assert bunch.is_empty == True

    def test_drop_zero_macro_particles(self, generate_bunch):
        bunch = generate_bunch(mp_number=1)
        bunch.alive[0] = False
        assert len(bunch) == 0
        assert bunch.is_empty == True
        assert pytest.approx(bunch.charge) == 0

    # Verify the behavior of init_gaussian with custom covariance and mean
    def test_init_gaussian_custom_cov_mean(self, large_bunch):

        custom_cov = np.eye(6) * 0.5
        custom_mean = np.ones(6) * 2.0

        large_bunch.init_gaussian(cov=custom_cov, mean=custom_mean)

        assert np.allclose(large_bunch.mean, custom_mean, atol=1e-1)
        assert np.allclose(np.cov(large_bunch.particles['x'],
                                  large_bunch.particles['xp']),
                           custom_cov[:2, :2],
                           atol=1e-1)

    def test_bunch_values(self, small_bunch, demo_ring):
        mp_number = small_bunch.mp_number
        current = small_bunch.current

        assert len(small_bunch) == mp_number
        np.testing.assert_allclose(small_bunch.alive,
                                   np.ones((mp_number, ), dtype=bool))
        assert pytest.approx(small_bunch.charge) == current * demo_ring.T0
        assert pytest.approx(
            small_bunch.charge_per_mp) == current * demo_ring.T0 / mp_number
        assert pytest.approx(
            small_bunch.particle_number) == current * demo_ring.T0 / e
        assert small_bunch.is_empty == False

    def test_bunch_magic(self, generate_bunch):
        mybunch = generate_bunch(init_gaussian=False)
        for label in mybunch:
            np.testing.assert_allclose(mybunch[label], np.zeros(len(mybunch)))
            mybunch[label] = np.ones(len(mybunch))
            np.testing.assert_allclose(mybunch[label], np.ones(len(mybunch)))

    def test_bunch_losses(self, small_bunch):
        charge_init = small_bunch.charge
        small_bunch.alive[0] = False
        assert len(small_bunch) == small_bunch.mp_number - 1
        assert pytest.approx(
            small_bunch.charge
        ) == charge_init * len(small_bunch) / small_bunch.mp_number

    def test_bunch_init_gauss(self, large_bunch):
        large_bunch.init_gaussian(mean=np.ones((6, )))
        np.testing.assert_allclose(large_bunch.mean, np.ones((6, )), rtol=1e-2)

    def test_bunch_save_load(self, small_bunch, generate_bunch, tmp_path):
        small_bunch["x"] += 1
        small_bunch.save(str(tmp_path / "test"))

        mybunch2 = generate_bunch(mp_number=1, current=1e-5)
        mybunch2.load(str(tmp_path / "test.hdf5"))

        assert small_bunch.mp_number == mybunch2.mp_number
        assert pytest.approx(small_bunch.charge) == mybunch2.charge
        for label in small_bunch:
            np.testing.assert_allclose(small_bunch[label], mybunch2[label])

    def test_bunch_stats(self, demo_ring, large_bunch):
        np.testing.assert_array_almost_equal(large_bunch.mean,
                                             np.zeros((6, )),
                                             decimal=5)
        sig = np.concatenate(
            (demo_ring.sigma(), [demo_ring.sigma_0, demo_ring.sigma_delta]))
        np.testing.assert_allclose(large_bunch.std, sig, rtol=1e-2)
        np.testing.assert_allclose(large_bunch.emit[:2],
                                   demo_ring.emit,
                                   rtol=1e-2)
        np.testing.assert_allclose(large_bunch.cs_invariant[:2],
                                   demo_ring.emit * 2,
                                   rtol=1e-2)

    @pytest.mark.parametrize('n_bin', [(75), (1), (2)])
    def test_bunch_binning(self, small_bunch, n_bin):
        (bins, sorted_index, profile,
         center) = small_bunch.binning(n_bin=n_bin)
        profile0 = np.zeros((len(bins) - 1, ))
        for i, val in enumerate(sorted_index):
            assert bins[val] <= small_bunch["tau"][i] <= bins[val + 1]
            profile0[val] += 1
        np.testing.assert_allclose(profile0, profile)

    def test_bunch_plots(self, small_bunch):
        small_bunch.plot_phasespace()
        small_bunch.plot_profile()
        assert True

    def test_bunch_emittance(self, generate_bunch, demo_ring):
        mp_number = 1_000_000
        mybunch = generate_bunch(mp_number=mp_number, track_alive=False)

        np.testing.assert_allclose(
            mybunch.emit[0],
            demo_ring.emit[0],
            rtol=1e-2,
            atol=0,
            err_msg=
            f'Emittances do not match. {demo_ring.emit[0]} initialised, {mybunch.emit[0]:} calculated'
        )
        np.testing.assert_allclose(
            mybunch.emit[1],
            demo_ring.emit[1],
            rtol=1e-2,
            atol=0,
            err_msg=
            f'Emittances do not match. {demo_ring.emit[1]} initialised, {mybunch.emit[1]:} calculated'
        )

        np.testing.assert_allclose(
            mybunch.emit[0],
            mybunch.cs_invariant[0] / 2,
            rtol=1e-2,
            atol=0,
            err_msg=
            f'Emittances do not match. {mybunch.cs_invariant[0]/2} calculated with optics functions, {mybunch.emit[0]:} calculated with coordinates only'
        )
        np.testing.assert_allclose(
            mybunch.emit[1],
            mybunch.cs_invariant[1] / 2,
            rtol=1e-2,
            atol=0,
            err_msg=
            f'Emittances do not match. {mybunch.cs_invariant[1]/2} calculated with optics functions, {mybunch.emit[1]:} calculated with coordinates only'
        )

    def test_bunch_emittance_with_dispersion(self, generate_bunch, demo_ring):
        mp_number = 1_000_000
        mybunch = generate_bunch(mp_number=mp_number, track_alive=False)

        np.testing.assert_allclose(
            mybunch.emit[0],
            demo_ring.emit[0],
            rtol=1e-2,
            atol=0,
            err_msg=
            f'Emittances do not match. {demo_ring.emit[0]} initialised, {mybunch.emit[0]:} calculated'
        )
        np.testing.assert_allclose(
            mybunch.emit[1],
            demo_ring.emit[1],
            rtol=1e-2,
            atol=0,
            err_msg=
            f'Emittances do not match. {demo_ring.emit[1]} initialised, {mybunch.emit[1]:} calculated'
        )

        np.testing.assert_allclose(
            mybunch.emit[0],
            mybunch.cs_invariant[0] / 2,
            rtol=1e-2,
            atol=0,
            err_msg=
            f'Emittances do not match. {mybunch.cs_invariant[0]/2} calculated with optics functions, {mybunch.emit[0]:} calculated with coordinates only'
        )
        np.testing.assert_allclose(
            mybunch.emit[1],
            mybunch.cs_invariant[1] / 2,
            rtol=1e-2,
            atol=0,
            err_msg=
            f'Emittances do not match. {mybunch.cs_invariant[1]/2} calculated with optics functions, {mybunch.emit[1]:} calculated with coordinates only'
        )


@pytest.fixture
def generate_beam(demo_ring, generate_bunch):

    def generate(ring=demo_ring,
                 filling_pattern=None,
                 current_per_bunch=1e-3,
                 mp_per_bunch=10,
                 track_alive=True,
                 mpi=False):

        beam = Beam(ring=ring)
        if filling_pattern is None:
            filling_pattern = np.ones((ring.h, ), dtype=bool)
        beam.init_beam(filling_pattern=filling_pattern,
                       current_per_bunch=current_per_bunch,
                       mp_per_bunch=mp_per_bunch,
                       track_alive=track_alive,
                       mpi=mpi)

        return beam

    return generate


@pytest.fixture
def beam_uniform(generate_beam):
    return generate_beam()


@pytest.fixture
def beam_non_uniform(generate_beam, demo_ring):
    filling_pattern = np.ones((demo_ring.h, ), dtype=bool)
    filling_pattern[4] = False
    filling_pattern[-3:] = False
    return generate_beam(filling_pattern=filling_pattern)


@pytest.fixture
def beam_1bunch_mpi(generate_beam, demo_ring):
    filling_pattern = np.zeros((demo_ring.h, ), dtype=bool)
    filling_pattern[0] = True
    return generate_beam(filling_pattern=filling_pattern, mpi=True)


class TestBeam:

    @pytest.mark.parametrize("n_bunch", [(1), (5), (10), (20)])
    def test_initialize_beam(self, generate_beam, demo_ring, n_bunch):
        filling_pattern = np.zeros((demo_ring.h, ), dtype=bool)
        filling_pattern[0:n_bunch] = True
        beam = generate_beam(filling_pattern=filling_pattern)
        assert len(beam) == n_bunch

    @pytest.mark.parametrize("n_bunch", [(1), (5), (10), (20)])
    def test_calculate_total_beam_properties(self, generate_beam, demo_ring,
                                             n_bunch):
        filling_pattern = np.zeros((demo_ring.h, ), dtype=bool)
        filling_pattern[0:n_bunch] = True
        beam = generate_beam(filling_pattern=filling_pattern,
                             current_per_bunch=0.002)
        assert beam.current == pytest.approx(0.002 * n_bunch)
        assert beam.charge == pytest.approx(
            np.sum([bunch.charge for bunch in beam]))
        assert beam.particle_number == pytest.approx(
            np.sum([bunch.particle_number for bunch in beam]))

    def test_save_and_load_beam_data(self, tmp_path, beam_uniform, demo_ring):
        file_path = tmp_path / "beam_data"
        beam_uniform.save(str(file_path))
        loaded_beam = Beam(demo_ring)
        loaded_beam.load(str(file_path) + ".hdf5", mpi=False)
        assert np.array_equal(beam_uniform.bunch_current,
                              loaded_beam.bunch_current)
        assert np.array_equal(beam_uniform.bunch_mean, loaded_beam.bunch_mean)
        assert np.array_equal(beam_uniform.bunch_std, loaded_beam.bunch_std)

    @pytest.mark.parametrize("var,option", [("bunch_current", None),
                                            ("bunch_charge", None),
                                            ("bunch_particle", None),
                                            ("bunch_mean", "x"),
                                            ("bunch_std", "x"),
                                            ("bunch_emit", "x")])
    def test_plot_variables_with_respect_to_bunch_number(
            self, beam_uniform, var, option):
        fig = beam_uniform.plot(var, option)
        assert fig is not None
        plt.close("all")

    def test_initialize_beam_mismatched_bunch_list_length(
            self, demo_ring, generate_bunch):
        mismatched_bunch_list = [
            generate_bunch() for _ in range(demo_ring.h - 1)
        ]
        with pytest.raises(ValueError):
            Beam(demo_ring, mismatched_bunch_list)

    def test_filling_pattern_and_distance_between_bunches(
            self, generate_beam, demo_ring):
        filling_pattern = np.ones((demo_ring.h, ), dtype=bool)
        filling_pattern[5] = False
        filling_pattern[8:10] = False
        beam = generate_beam(filling_pattern=filling_pattern)
        np.testing.assert_array_equal(beam.filling_pattern, filling_pattern)
        expected_distances = np.ones((demo_ring.h, ))
        expected_distances[4] = 2
        expected_distances[5] = 0
        expected_distances[7] = 3
        expected_distances[8:10] = 0
        np.testing.assert_array_equal(beam.distance_between_bunches,
                                      expected_distances)

    def test_update_filling_pattern_and_distance_between_bunches(
            self, beam_uniform, demo_ring):
        for i in [5, 8, 9]:
            beam_uniform[i].charge = 0
        beam_uniform[5].charge = 0
        beam_uniform.update_filling_pattern()
        beam_uniform.update_distance_between_bunches()

        expected_filling_pattern = np.ones((demo_ring.h, ), dtype=bool)
        expected_filling_pattern[5] = False
        expected_filling_pattern[8:10] = False
        np.testing.assert_array_equal(beam_uniform.filling_pattern,
                                      expected_filling_pattern)

        expected_distances = np.ones((demo_ring.h, ))
        expected_distances[4] = 2
        expected_distances[5] = 0
        expected_distances[7] = 3
        expected_distances[8:10] = 0
        np.testing.assert_array_equal(beam_uniform.distance_between_bunches,
                                      expected_distances)

    def test_mpi_gather_and_close_consistency(self, mocker, demo_ring,
                                              generate_bunch):
        mock_mpi = mocker.patch('mbtrack2.tracking.parallel.Mpi')
        mock_mpi_instance = mock_mpi.return_value
        mock_mpi_instance.comm.allgather.return_value = [
            generate_bunch() for _ in range(demo_ring.h)
        ]
        beam = Beam(ring=demo_ring)
        beam.mpi_init()
        beam.mpi_gather()
        assert mock_mpi_instance.comm.allgather.called
        beam.mpi_close()
        assert not beam.mpi_switch
        assert beam.mpi is None

    def test_init_bunch_list(self, demo_ring):
        filling_pattern = np.ones((demo_ring.h, ), dtype=bool)
        filling_pattern[5] = False
        filling_pattern[8:10] = False
        bunch_list = [
            Bunch(demo_ring, mp_number=1, alive=filling_pattern[i])
            for i in range(demo_ring.h)
        ]
        beam = Beam(demo_ring, bunch_list)
        assert len(beam) == filling_pattern.sum()
        np.testing.assert_array_equal(beam.filling_pattern, filling_pattern)
        assert beam.distance_between_bunches is not None

    def test_init_bunch_list_mpi(self, demo_ring, generate_bunch):
        filling_pattern = np.zeros((demo_ring.h, ), dtype=bool)
        filling_pattern[0] = True
        beam = Beam(demo_ring)
        beam.init_bunch_list_mpi(generate_bunch(), filling_pattern)

        assert len(beam) == 1
        np.testing.assert_array_equal(beam.filling_pattern, filling_pattern)
        assert beam.distance_between_bunches[0] == demo_ring.h
