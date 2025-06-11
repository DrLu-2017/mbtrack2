import pytest

pytestmark = pytest.mark.unit
from mbtrack2 import (
    CircularAperture,
    ElipticalAperture,
    LongitudinalAperture,
    RectangularAperture,
)


class TestCircularAperture:

    # Initialize CircularAperture with a positive radius and track a bunch with particles inside the radius
    def test_track_particles_inside_radius(self, small_bunch):
        aperture = CircularAperture(radius=1.0)
        small_bunch["x"] = 0.5
        small_bunch["y"] = -0.5
        aperture.track(small_bunch)
        assert all(small_bunch.alive)

    # Track a bunch with particles exactly on the radius boundary
    @pytest.mark.parametrize("x, y", [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0),
                                      (0.0, -1.0)])
    def test_track_particles_on_boundary(self, small_bunch, x, y):
        aperture = CircularAperture(radius=1.0)
        small_bunch["x"] = x
        small_bunch["y"] = y
        aperture.track(small_bunch)
        assert all(small_bunch.alive)

    # Track a bunch with all particles outside the radius
    @pytest.mark.parametrize("x, y", [(0.75, 0.75), (0.75, -0.75), (1.1, 0.0),
                                      (0.0, 1.1)])
    def test_track_all_particles_outside_radius(self, small_bunch, x, y):
        aperture = CircularAperture(radius=1.0)
        small_bunch["x"] = x
        small_bunch["y"] = y
        aperture.track(small_bunch)
        assert not any(small_bunch.alive)

    # Track a bunch with no particles
    def test_track_no_particles(self, small_bunch):
        aperture = CircularAperture(radius=1.0)
        small_bunch.alive[:] = False
        aperture.track(small_bunch)
        assert len(small_bunch) == 0


class TestElipticalAperture:

    # Track a bunch where all particles are within the elliptical aperture
    def test_track_particles_within_aperture(self, small_bunch):
        aperture = ElipticalAperture(X_radius=1.0, Y_radius=2.0)
        small_bunch["x"] = 0.5
        small_bunch["y"] = 1.0
        aperture.track(small_bunch)
        assert all(small_bunch.alive)

    # Track a bunch where some particles are outside the elliptical aperture
    @pytest.mark.parametrize("x, y", [(0.5, 1.75), (0.5, -1.75), (1.1, 0.0),
                                      (0.0, 2.1)])
    def test_track_particles_outside_aperture(self, small_bunch, x, y):
        aperture = ElipticalAperture(X_radius=1.0, Y_radius=2.0)
        small_bunch["x"] = x
        small_bunch["y"] = y
        aperture.track(small_bunch)
        assert not all(small_bunch.alive)

    # Track a bunch with no particles
    def test_track_no_particles(self, small_bunch):
        aperture = ElipticalAperture(X_radius=1.0, Y_radius=2.0)
        small_bunch.alive[:] = False
        aperture.track(small_bunch)
        assert len(small_bunch) == 0

    # Track a bunch with particles exactly on the radius boundary
    @pytest.mark.parametrize("x, y", [(1.0, 0.0), (0.0, 2.0), (-1.0, 0.0),
                                      (0.0, -2.0)])
    def test_track_particles_on_boundary(self, small_bunch, x, y):
        aperture = ElipticalAperture(X_radius=1.0, Y_radius=2.0)
        small_bunch["x"] = x
        small_bunch["y"] = y
        aperture.track(small_bunch)
        assert all(small_bunch.alive)


class TestRectangularAperture:

    # Track method correctly identifies particles within the rectangular aperture
    def test_track_particles_within_aperture(self, small_bunch):
        aperture = RectangularAperture(X_right=1.0, Y_top=1.0)
        small_bunch["x"] = 0.5
        small_bunch["y"] = 0.5
        aperture.track(small_bunch)
        assert all(small_bunch.alive)

    # Particles outside the defined apertures are marked as 'lost'
    @pytest.mark.parametrize("x, y", [(1.1, 1.1), (-1.1, -1.1), (1.1, 0.0),
                                      (0.0, 1.1)])
    def test_particles_outside_aperture_lost(self, small_bunch, x, y):
        aperture = RectangularAperture(X_right=1.0, Y_top=1.0)
        small_bunch["x"] = x
        small_bunch["y"] = y
        aperture.track(small_bunch)
        assert not any(small_bunch.alive)

    # Track a bunch with no particles
    def test_track_no_particles(self, small_bunch):
        aperture = RectangularAperture(X_right=1.0, Y_top=1.0)
        small_bunch.alive[:] = False
        aperture.track(small_bunch)
        assert len(small_bunch) == 0

    # Manages particles exactly on the aperture boundaries
    @pytest.mark.parametrize("x, y", [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0),
                                      (0.0, -1.0), (1.0, 1.0)])
    def test_particles_on_boundary(self, small_bunch, x, y):
        aperture = RectangularAperture(X_right=1.0, Y_top=1.0)
        small_bunch.particles["x"] = x
        small_bunch.particles["y"] = y
        aperture.track(small_bunch)
        assert all(small_bunch.alive)

    @pytest.mark.parametrize("x, y", [(1.1, 1.1), (-1.1, -1.1), (0.0, 0.0),
                                      (0.5, -0.6)])
    def test_non_default_rect_loss(self, small_bunch, x, y):
        aperture = RectangularAperture(X_right=1.0,
                                       Y_top=1.0,
                                       X_left=0.1,
                                       Y_bottom=-0.5)
        small_bunch["x"] = x
        small_bunch["y"] = y
        aperture.track(small_bunch)
        assert not any(small_bunch.alive)

    @pytest.mark.parametrize("x, y", [(0.5, 0.0), (0.2, -0.3)])
    def test_non_default_rect(self, small_bunch, x, y):
        aperture = RectangularAperture(X_right=1.0,
                                       Y_top=1.0,
                                       X_left=0.1,
                                       Y_bottom=-0.5)
        small_bunch["x"] = x
        small_bunch["y"] = y
        aperture.track(small_bunch)
        assert all(small_bunch.alive)


class TestLongitudinalAperture:

    # Track a Bunch object with particles within the longitudinal bounds
    def test_track_bunch_within_bounds(self, small_bunch):
        aperture = LongitudinalAperture(tau_up=1.0)
        small_bunch["tau"] = -0.5
        aperture.track(small_bunch)
        assert all(small_bunch.alive)

    # Track a Bunch object with particles exactly on the boundary values
    @pytest.mark.parametrize("tau", [(-1.0), (1.0)])
    def test_track_bunch_on_boundary(self, small_bunch, tau):
        aperture = LongitudinalAperture(tau_up=1.0)
        small_bunch["tau"] = tau
        aperture.track(small_bunch)
        assert all(small_bunch.alive)

    # Track a bunch with no particles
    def test_track_no_particles(self, small_bunch):
        aperture = LongitudinalAperture(tau_up=1.0)
        small_bunch.alive[:] = False
        aperture.track(small_bunch)
        assert len(small_bunch) == 0

    # Track a Bunch object with all particles outside the longitudinal bounds
    @pytest.mark.parametrize("tau", [(1.1), (-0.6)])
    def test_track_bunch_outside_bounds(self, small_bunch, tau):
        aperture = LongitudinalAperture(tau_up=1.0, tau_low=-0.5)
        small_bunch["tau"] = tau
        aperture.track(small_bunch)
        assert not any(small_bunch.alive)
