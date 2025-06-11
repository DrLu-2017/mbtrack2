import numpy as np
import pytest

pytestmark = pytest.mark.unit
from scipy.special import factorial
from utility_test_functions import assert_attr_changed

from mbtrack2 import (
    Element,
    LongitudinalMap,
    SkewQuadrupole,
    SynchrotronRadiation,
    TransverseMap,
    TransverseMapSector,
    transverse_map_sector_generator,
)


class TestElement:

    def test_parallel_decorator_with_mpi_beam(self, beam_1bunch_mpi):

        class SubElement(Element):

            @Element.parallel
            def track(self, bunch):
                bunch.charge = 1

        element = SubElement()
        element.track(beam_1bunch_mpi)
        assert beam_1bunch_mpi[
            beam_1bunch_mpi.mpi.bunch_num].charge == pytest.approx(1)

    def test_parallel_decorator_with_beam(self, beam_non_uniform):

        class SubElement(Element):

            @Element.parallel
            def track(self, bunch):
                bunch.charge = 1

        element = SubElement()
        element.track(beam_non_uniform)
        for i, bunch in enumerate(beam_non_uniform):
            if beam_non_uniform.filling_pattern[i] == True:
                assert bunch.charge == pytest.approx(1)
            else:
                assert bunch.charge == pytest.approx(0)

    def test_parallel_decorator_with_bunch(self, small_bunch):

        class SubElement(Element):

            @Element.parallel
            def track(self, bunch):
                bunch.charge = 1

        element = SubElement()
        element.track(small_bunch)
        assert small_bunch.charge == pytest.approx(1)

    # Decorator correctly skips track method if bunch is empty
    def test_skip_track_if_bunch_empty(self, mocker, generate_bunch):
        mock_track = mocker.Mock()
        decorated_track = Element.track_bunch_if_non_empty(mock_track)
        empty_bunch = generate_bunch(mp_number=0)
        decorated_track(None, empty_bunch)
        mock_track.assert_not_called()

    # Decorator calls track method if bunch is not empty
    def test_call_track_if_bunch_not_empty(self, mocker, small_bunch):
        mock_track = mocker.Mock()
        decorated_track = Element.track_bunch_if_non_empty(mock_track)
        decorated_track(None, small_bunch)
        mock_track.assert_called_once()

    # Decorator respects track_alive flag and calls track method
    def test_respect_track_alive_flag(self, mocker, generate_bunch):
        mock_track = mocker.Mock()
        decorated_track = Element.track_bunch_if_non_empty(mock_track)
        bunch = generate_bunch(track_alive=False)
        decorated_track(None, bunch)
        mock_track.assert_called_once()

    # Track method executes when bunch is not empty and track_alive is True
    def test_executes_with_nonempty_bunch(self, small_bunch):
        called = []

        @Element.track_bunch_if_non_empty
        def track_method(self, bunch):
            called.append(True)

        small_bunch.track_alive = True
        track_method(self, small_bunch)
        assert called == [True]

    # Track method executes when track_alive is False regardless of bunch size
    def test_executes_when_track_alive_false(self, small_bunch):
        called = []

        @Element.track_bunch_if_non_empty
        def track_method(self, bunch):
            called.append(True)

        small_bunch.track_alive = False
        track_method(self, small_bunch)
        assert called == [True]

    # Empty bunch with track_alive=True skips track method execution
    def test_skips_empty_bunch(self, generate_bunch):
        called = []

        @Element.track_bunch_if_non_empty
        def track_method(self, bunch):
            called.append(True)

        empty_bunch = generate_bunch(alive=False)
        empty_bunch.track_alive = True
        track_method(self, empty_bunch)
        assert not called


class TestLongitudinalMap:

    # Track a Bunch object using the track method
    def test_track_bunch(self, small_bunch, demo_ring):
        long_map = LongitudinalMap(demo_ring)
        assert_attr_changed(long_map,
                            small_bunch,
                            attrs_changed=["tau", "delta"])


class TestSynchrotronRadiation:

    # SynchrotronRadiation initializes correctly with default switch values
    def test_initialization_with_default_switch(self, demo_ring):
        sr = SynchrotronRadiation(demo_ring)
        assert np.array_equal(sr.switch, np.ones((3, ), dtype=bool))

    # SynchrotronRadiation modifies 'delta', 'xp', and 'yp' attributes of Bunch
    def test_modifies_bunch_attributes(self, small_bunch, demo_ring):
        sr = SynchrotronRadiation(demo_ring)
        assert_attr_changed(sr, small_bunch)

    # switch array has all False values, ensuring no changes to Bunch
    def test_no_changes_with_all_false_switch(self, small_bunch, demo_ring):
        sr = SynchrotronRadiation(demo_ring,
                                  switch=np.zeros((3, ), dtype=bool))
        assert_attr_changed(sr, small_bunch, change=False)


class TestSkewQuadrupole:

    # Initialize SkewQuadrupole with a positive strength and track a Bunch object
    def test_modifies_bunch_attributes(self, small_bunch):
        skew_quad = SkewQuadrupole(strength=0.1)
        assert_attr_changed(skew_quad, small_bunch, attrs_changed=["xp", "yp"])


class TestTransverseMapSector:

    @pytest.fixture
    def generate_trans_map_sector(demo_ring):

        def generate(phase_diff=np.array([np.pi, np.pi]),
                     chro_diff=np.array([0.01, 0.01]),
                     adts=None):
            alpha0 = np.array([1.0, 1.0])
            beta0 = np.array([1.0, 1.0])
            dispersion0 = np.array([0.0, 0.0, 0.0, 0.0])
            alpha1 = np.array([2.0, 2.0])
            beta1 = np.array([2.0, 2.0])
            dispersion1 = np.array([0.1, 0.1, 0.1, 0.1])
            sector = TransverseMapSector(demo_ring,
                                         alpha0,
                                         beta0,
                                         dispersion0,
                                         alpha1,
                                         beta1,
                                         dispersion1,
                                         phase_diff,
                                         chro_diff,
                                         adts=adts)
            return sector

        return generate

    # Track a Bunch object through TransverseMapSector and ensure coordinates are updated
    def test_track_bunch_coordinates_update(self, generate_trans_map_sector,
                                            small_bunch):
        sector = generate_trans_map_sector()
        assert_attr_changed(sector,
                            small_bunch,
                            attrs_changed=["x", "xp", "y", "yp"])

    # Compute chromatic tune advances for a Bunch with non-zero chromaticity differences
    @pytest.mark.parametrize("chro_diff", [
        (np.array([0.02, 0.03])),
        (np.array([0.02, 0.03, 0.05, 0.06])),
        (np.array([
            0.02,
            0.03,
            0.05,
            0.06,
            0.02,
            0.03,
        ])),
        (np.array([0.02, 0.03, 0.05, 0.06, 0.02, 0.03, 0.05, 0.06])),
        (np.array([0.02, 0.03, 0.05, 0.06, 0.02, 0.03, 0.05, 0.06, 0.05, 0.06
                   ])),
    ])
    def test_chromatic_tune_advances(self, generate_trans_map_sector,
                                     small_bunch, chro_diff):
        # chro_diff = np.array([0.02, 0.03])
        sector = generate_trans_map_sector(chro_diff=chro_diff)
        tune_advance_x_chro, tune_advance_y_chro = sector._compute_chromatic_tune_advances(
            small_bunch)

        order = len(chro_diff) // 2
        coefs = np.array([1 / factorial(i) for i in range(order + 1)])
        coefs[0] = 0
        chro_diff = np.concatenate(([0, 0], chro_diff))
        tune_advance_x = np.polynomial.polynomial.Polynomial(
            chro_diff[::2] * coefs)(small_bunch['delta'])
        tune_advance_y = np.polynomial.polynomial.Polynomial(
            chro_diff[1::2] * coefs)(small_bunch['delta'])

        assert np.allclose(tune_advance_x, tune_advance_x_chro)
        assert np.allclose(tune_advance_y, tune_advance_y_chro)

    # Check that adts are taken into account in calculation
    def test_amplitude_dependent_tune_shifts(self, generate_trans_map_sector,
                                             small_bunch):

        sector_no_adts = generate_trans_map_sector()
        adts = [
            np.array([1e10, 1e10, 1e10]),
            np.array([1e10, 1e10, 1e10]),
            np.array([1e10, 1e10, 1e10]),
            np.array([1e10, 1e10, 1e10])
        ]
        sector_adts = generate_trans_map_sector(adts=adts)

        attrs = ["x", "xp", "y", "yp"]
        initial_values = {attr: small_bunch[attr].copy() for attr in attrs}

        sector_no_adts.track(small_bunch)
        no_adts = {attr: small_bunch[attr].copy() for attr in attrs}

        for attr in attrs:
            small_bunch[attr] = initial_values[attr]

        sector_adts.track(small_bunch)
        adts = {attr: small_bunch[attr].copy() for attr in attrs}

        for attr in attrs:
            assert not np.array_equal(initial_values[attr], no_adts[attr])
            assert not np.array_equal(initial_values[attr], adts[attr])
            assert not np.array_equal(adts[attr], no_adts[attr])


class TestTransverseMap:

    class Old_TransverseMap(Element):
        """
        Transverse map from mbtrack2 0.7.0.
    
        Parameters
        ----------
        ring : Synchrotron object
        """

        def __init__(self, ring):
            self.ring = ring
            self.alpha = self.ring.optics.local_alpha
            self.beta = self.ring.optics.local_beta
            self.gamma = self.ring.optics.local_gamma
            self.dispersion = self.ring.optics.local_dispersion
            if self.ring.adts is not None:
                self.adts_poly = [
                    np.poly1d(self.ring.adts[0]),
                    np.poly1d(self.ring.adts[1]),
                    np.poly1d(self.ring.adts[2]),
                    np.poly1d(self.ring.adts[3]),
                ]

        @Element.parallel
        def track(self, bunch):
            """
            Tracking method for the element.
            No bunch to bunch interaction, so written for Bunch objects and
            @Element.parallel is used to handle Beam objects.
    
            Parameters
            ----------
            bunch : Bunch or Beam object
            """
            print("x 0", bunch['x'][0])
            # Compute phase advance which depends on energy via chromaticity and ADTS
            if self.ring.adts is None:
                phase_advance_x = (
                    2 * np.pi *
                    (self.ring.tune[0] + self.ring.chro[0] * bunch["delta"]))
                phase_advance_y = (
                    2 * np.pi *
                    (self.ring.tune[1] + self.ring.chro[1] * bunch["delta"]))
                print("old phase advance ", phase_advance_x[0] / (2 * np.pi),
                      phase_advance_y[0] / (2 * np.pi))
            else:
                Jx = ((self.ring.optics.local_gamma[0] * bunch["x"]**2) +
                      (2 * self.ring.optics.local_alpha[0] * bunch["x"] *
                       bunch["xp"]) +
                      (self.ring.optics.local_beta[0] * bunch["xp"]**2))
                Jy = ((self.ring.optics.local_gamma[1] * bunch["y"]**2) +
                      (2 * self.ring.optics.local_alpha[1] * bunch["y"] *
                       bunch["yp"]) +
                      (self.ring.optics.local_beta[1] * bunch["yp"]**2))
                phase_advance_x = (
                    2 * np.pi *
                    (self.ring.tune[0] + self.ring.chro[0] * bunch["delta"] +
                     self.adts_poly[0](Jx) + self.adts_poly[2](Jy)))
                phase_advance_y = (
                    2 * np.pi *
                    (self.ring.tune[1] + self.ring.chro[1] * bunch["delta"] +
                     self.adts_poly[1](Jx) + self.adts_poly[3](Jy)))

            # 6x6 matrix corresponding to (x, xp, delta, y, yp, delta)
            matrix = np.zeros((6, 6, len(bunch)), dtype=np.float64)

            # Horizontal
            c_x = np.cos(phase_advance_x)
            s_x = np.sin(phase_advance_x)

            matrix[0, 0, :] = c_x + self.alpha[0] * s_x
            matrix[0, 1, :] = self.beta[0] * s_x
            matrix[0, 2, :] = self.dispersion[0]
            matrix[1, 0, :] = -1 * self.gamma[0] * s_x
            matrix[1, 1, :] = c_x - self.alpha[0] * s_x
            matrix[1, 2, :] = self.dispersion[1]
            matrix[2, 2, :] = 1

            print(matrix[0, 0, :], matrix[0, 1, :], matrix[0, 2, :])

            # Vertical
            c_y = np.cos(phase_advance_y)
            s_y = np.sin(phase_advance_y)

            matrix[3, 3, :] = c_y + self.alpha[1] * s_y
            matrix[3, 4, :] = self.beta[1] * s_y
            matrix[3, 5, :] = self.dispersion[2]
            matrix[4, 3, :] = -1 * self.gamma[1] * s_y
            matrix[4, 4, :] = c_y - self.alpha[1] * s_y
            matrix[4, 5, :] = self.dispersion[3]
            matrix[5, 5, :] = 1

            x = (matrix[0, 0] * bunch["x"] + matrix[0, 1] * bunch["xp"] +
                 matrix[0, 2] * bunch["delta"])
            xp = (matrix[1, 0] * bunch["x"] + matrix[1, 1] * bunch["xp"] +
                  matrix[1, 2] * bunch["delta"])
            y = (matrix[3, 3] * bunch["y"] + matrix[3, 4] * bunch["yp"] +
                 matrix[3, 5] * bunch["delta"])
            yp = (matrix[4, 3] * bunch["y"] + matrix[4, 4] * bunch["yp"] +
                  matrix[4, 5] * bunch["delta"])

            bunch["x"] = x
            bunch["xp"] = xp
            bunch["y"] = y
            bunch["yp"] = yp

    def test_trans_map_base(self, demo_ring, small_bunch):
        old_map = self.Old_TransverseMap(demo_ring)
        current_map = TransverseMap(demo_ring)

        attrs = ["x", "xp", "y", "yp"]
        initial_values = {attr: small_bunch[attr].copy() for attr in attrs}

        old_map.track(small_bunch)
        old = {attr: small_bunch[attr].copy() for attr in attrs}

        for attr in attrs:
            small_bunch[attr] = initial_values[attr]

        current_map.track(small_bunch)
        current = {attr: small_bunch[attr].copy() for attr in attrs}

        for attr in attrs:
            assert not np.array_equal(initial_values[attr], current[attr])
            assert not np.array_equal(initial_values[attr], old[attr])
            assert np.allclose(current[attr], old[attr])

    def test_trans_map_adts(self, ring_with_at_lattice, small_bunch):
        ring_with_at_lattice.get_adts()

        # Needed as old map was not handling dispersion properly!
        ring_with_at_lattice.optics.local_dispersion = np.array([0, 0, 0, 0])

        old_map = self.Old_TransverseMap(ring_with_at_lattice)
        current_map = TransverseMap(ring_with_at_lattice)

        assert np.array_equal(old_map.gamma, current_map.gamma0)
        assert np.array_equal(old_map.gamma, current_map.gamma1)
        assert np.array_equal(old_map.beta, current_map.beta0)
        assert np.array_equal(old_map.beta, current_map.beta1)
        assert np.array_equal(old_map.alpha, current_map.alpha0)
        assert np.array_equal(old_map.alpha, current_map.alpha1)
        assert np.array_equal(old_map.dispersion, current_map.dispersion0)
        assert np.array_equal(old_map.dispersion, current_map.dispersion1)

        attrs = ["x", "xp", "y", "yp"]
        initial_values = {attr: small_bunch[attr].copy() for attr in attrs}

        old_map.track(small_bunch)
        old = {attr: small_bunch[attr].copy() for attr in attrs}

        for attr in attrs:
            small_bunch[attr] = initial_values[attr]

        current_map.track(small_bunch)
        current = {attr: small_bunch[attr].copy() for attr in attrs}

        for attr in attrs:
            assert not np.array_equal(initial_values[attr], current[attr])
            assert not np.array_equal(initial_values[attr], old[attr])
            np.testing.assert_allclose(current[attr], old[attr])


class TestTransverseMapSectorGenerator:

    # Generate sectors with local optics values when use_local_values is True
    def test_local_optics_values(self, demo_ring):
        positions = np.array([0, 25, 50, 75])
        sectors = transverse_map_sector_generator(demo_ring, positions)

        assert len(sectors) == len(positions)
        for sector in sectors:
            assert np.array_equal(sector.alpha0, demo_ring.optics.local_alpha)
            assert np.array_equal(sector.beta0, demo_ring.optics.local_beta)
            assert np.array_equal(sector.dispersion0,
                                  demo_ring.optics.local_dispersion)

    # Generate sectors with AT lattice optics when use_local_values is False
    def test_at_lattice_optics(self, ring_with_at_lattice):
        positions = np.array([0, 25, 50, 75])
        sectors = transverse_map_sector_generator(ring_with_at_lattice,
                                                  positions)

        assert len(sectors) == len(positions)
        for i, sector in enumerate(sectors):
            assert np.array_equal(
                sector.alpha0, ring_with_at_lattice.optics.alpha(positions[i]))
            assert np.array_equal(
                sector.beta0, ring_with_at_lattice.optics.beta(positions[i]))

    # Calculate phase differences between consecutive positions
    def test_phase_differences(self, ring_with_at_lattice):
        positions = np.array([0, 25, 50])
        ring_with_at_lattice.optics.use_local_values = False
        sectors = transverse_map_sector_generator(ring_with_at_lattice,
                                                  positions)

        for i, sector in enumerate(sectors):
            if i < len(positions) - 1:
                expected_phase = ring_with_at_lattice.optics.mu(
                    positions[i + 1]) - ring_with_at_lattice.optics.mu(
                        positions[i])
                assert np.allclose(sector.tune_diff * (2 * np.pi),
                                   expected_phase)

    # Compute chromaticity differences between sectors
    def test_chromaticity_differences(self, demo_ring):
        positions = np.array([0, 50])
        sectors = transverse_map_sector_generator(demo_ring, positions)

        expected_chro = np.asarray(demo_ring.chro) / len(positions)
        assert np.allclose(sectors[0].chro_diff, expected_chro)

    # Handle ADTS parameters correctly when provided
    def test_adts_handling(self, demo_ring):
        demo_ring.adts = np.array([1.0, 1.0, 1.0, 1.0])
        positions = np.array([0, 50])
        sectors = transverse_map_sector_generator(demo_ring, positions)

        expected_adts = demo_ring.adts / len(positions)
        assert np.allclose(sectors[0].adts_poly[0].coef, expected_adts[0])
