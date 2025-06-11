import os

import h5py as hp
import numpy as np
import pytest
from utility_test_functions import assert_attr_changed

from mbtrack2 import BeamIonElement, IonAperture, IonMonitor, IonParticles

pytestmark = pytest.mark.unit


@pytest.fixture
def generate_ion_particles(demo_ring):

    def generate(mp_number=100,
                 ion_element_length=1.0,
                 ring=demo_ring,
                 track_alive=False,
                 alive=True):
        ions = IonParticles(mp_number=mp_number,
                            ion_element_length=ion_element_length,
                            ring=ring,
                            track_alive=track_alive,
                            alive=alive)
        return ions

    return generate


class TestIonParticles:

    # Generate particle distribution using electron bunch parameters via generate_as_a_distribution()
    def test_generate_as_distribution(self, generate_ion_particles,
                                      large_bunch):
        mp_number = 1e5
        charge = 1e-9
        ions = generate_ion_particles(mp_number)
        ions.generate_as_a_distribution(large_bunch, charge)

        assert np.isclose(ions["x"].mean(),
                          large_bunch["x"].mean(),
                          rtol=0.1,
                          atol=1e-5)
        assert np.isclose(ions["x"].std(),
                          large_bunch["x"].std(),
                          rtol=0.1,
                          atol=1e-5)
        assert np.isclose(ions["y"].mean(),
                          large_bunch["y"].mean(),
                          rtol=0.1,
                          atol=1e-5)
        assert np.isclose(ions["y"].std(),
                          large_bunch["y"].std(),
                          rtol=0.1,
                          atol=1e-5)
        assert np.all(ions["xp"] == 0)
        assert np.all(ions["yp"] == 0)
        assert np.all(ions["delta"] == 0)
        assert np.all(ions["tau"] >= -ions.ion_element_length)
        assert np.all(ions["tau"] <= ions.ion_element_length)
        assert np.isclose(ions.charge, charge)
        assert np.isclose(ions["charge"].mean(),
                          charge / mp_number,
                          rtol=0.01,
                          atol=1e-9)

    # Generate random particle samples from electron bunch via generate_from_random_samples()
    def test_generate_from_random_samples(self, generate_ion_particles,
                                          large_bunch):
        mp_number = 1e5
        charge = 1e-9
        ions = generate_ion_particles(mp_number)
        ions.generate_from_random_samples(large_bunch, charge)

        assert np.all(np.isin(ions["x"], large_bunch["x"]))
        assert np.all(np.isin(ions["y"], large_bunch["y"]))
        assert np.all(ions["xp"] == 0)
        assert np.all(ions["yp"] == 0)
        assert np.all(ions["delta"] == 0)
        assert np.all(ions["tau"] >= -ions.ion_element_length)
        assert np.all(ions["tau"] <= ions.ion_element_length)
        assert np.isclose(ions.charge, charge)
        assert np.isclose(ions["charge"].mean(),
                          charge / mp_number,
                          rtol=0.01,
                          atol=1e-9)

    # Add two IonParticles instances together and verify combined particle arrays
    def test_add_ion_particles(self, generate_ion_particles):
        ions1 = generate_ion_particles(mp_number=100)
        ions2 = generate_ion_particles(mp_number=50)

        combined = ions1 + ions2
        assert combined.mp_number == 150
        assert all(
            combined[coord].shape == (150, )
            for coord in ["x", "xp", "y", "yp", "tau", "delta", "charge"])
        assert np.all(combined.alive == True)

    # Initialize with alive=False and verify all particles marked as dead
    def test_init_dead_particles(self, generate_ion_particles):
        ions = generate_ion_particles(alive=False)
        assert np.all(ions.alive == False)
        assert all(
            ions[coord].shape == (1, )
            for coord in ["x", "xp", "y", "yp", "tau", "delta", "charge"])

    # Generate distributions with electron bunch containing no particles
    def test_generate_from_empty_bunch(self, generate_ion_particles,
                                       small_bunch):
        small_bunch.alive[:] = False
        ions = generate_ion_particles()

        with pytest.raises(ValueError):
            ions.generate_as_a_distribution(small_bunch, 1e-9)
        with pytest.raises(ValueError):
            ions.generate_from_random_samples(small_bunch, 1e-9)


@pytest.fixture
def generate_ion_monitor(tmp_path):

    def generate(save_every=1,
                 buffer_size=10,
                 total_size=10,
                 file_name=tmp_path / "test_monitor.hdf5"):
        monitor = IonMonitor(save_every=save_every,
                             buffer_size=buffer_size,
                             total_size=total_size,
                             file_name=file_name)
        return monitor

    return generate


class TestIonMonitor:

    # Monitor initialization with valid parameters creates HDF5 file and sets up data structures
    def test_monitor_init_creates_valid_structures(self, generate_ion_monitor,
                                                   tmp_path):
        monitor = generate_ion_monitor()
        assert monitor.file is not None
        assert monitor.buffer_size == 10
        assert monitor.total_size == 10
        assert monitor.save_every == 1
        assert monitor.buffer_count == 0
        assert monitor.write_count == 0
        assert monitor.track_count == 0

    # Buffer writes to file when full and resets counter
    def test_buffer_writes_when_full(self, generate_ion_monitor,
                                     generate_ion_particles):
        monitor = generate_ion_monitor(buffer_size=2, total_size=4)
        ions = generate_ion_particles()

        for _ in range(2):
            monitor.track(ions)

        assert monitor.buffer_count == 0
        assert monitor.write_count == 1

    # Data structures are properly initialized with correct shapes and types
    def test_data_structures_initialization(self, generate_ion_monitor):
        monitor = generate_ion_monitor()

        assert monitor.mean.shape == (4, 10)
        assert monitor.std.shape == (4, 10)
        assert monitor.charge.shape == (10, )
        assert monitor.time.shape == (10, )
        assert monitor.time.dtype == int

    # Initialize monitor with total_size not divisible by buffer_size
    def test_invalid_total_size_raises_error(self, generate_ion_monitor):
        with pytest.raises(
                ValueError,
                match="total_size must be divisible by buffer_size"):
            generate_ion_monitor(buffer_size=3, total_size=10)

    # Initialize with invalid/missing file_name
    def test_invalid_filename_handling(self, generate_ion_monitor):
        with pytest.raises(OSError):
            generate_ion_monitor(file_name="/invalid/path/file.hdf5")


class TestElipticalAperture:

    # Track a bunch where all particles are within the elliptical aperture
    def test_track_particles_within_aperture(self, generate_ion_particles):
        ions = generate_ion_particles()
        mp = len(ions)
        aperture = IonAperture(X_radius=1.0, Y_radius=2.0)
        ions["x"] = np.ones_like(ions["x"]) * 0.5
        ions["y"] = np.ones_like(ions["y"]) * 1.0
        aperture.track(ions)
        assert all(ions.alive)
        assert mp == len(ions)

    # Track a bunch where some particles are outside the elliptical aperture
    @pytest.mark.parametrize("x, y", [(0.5, 1.75), (0.5, -1.75), (1.1, 0.0),
                                      (0.0, 2.1)])
    def test_track_particles_outside_aperture(self, generate_ion_particles, x,
                                              y):
        ions = generate_ion_particles()
        mp = len(ions)
        aperture = IonAperture(X_radius=1.0, Y_radius=2.0)
        ions["x"] = np.ones_like(ions["x"]) * x
        ions["y"] = np.ones_like(ions["y"]) * y
        aperture.track(ions)
        assert mp != len(ions)

    # Track a bunch with no particles
    def test_track_no_particles(self, generate_ion_particles):
        ions = generate_ion_particles()
        ions["x"] = np.ones_like(ions["x"]) * 100
        ions["y"] = np.ones_like(ions["y"]) * 100
        aperture = IonAperture(X_radius=1.0, Y_radius=2.0)
        aperture.track(ions)
        assert len(ions) == 0

        aperture.track(ions)
        assert True

    # Track a bunch with particles exactly on the radius boundary
    @pytest.mark.parametrize("x, y", [(1.0, 0.0), (0.0, 2.0), (-1.0, 0.0),
                                      (0.0, -2.0)])
    def test_track_particles_on_boundary(self, generate_ion_particles, x, y):
        ions = generate_ion_particles()
        mp = len(ions)
        aperture = IonAperture(X_radius=1.0, Y_radius=2.0)
        ions["x"] = np.ones_like(ions["x"]) * x
        ions["y"] = np.ones_like(ions["y"]) * y
        aperture.track(ions)
        assert all(ions.alive)
        assert mp == len(ions)


@pytest.fixture
def generate_beam_ion(demo_ring):

    def generate(ion_mass=1.67e-27,
                 ion_charge=1.6e-19,
                 ionization_cross_section=1e-22,
                 residual_gas_density=1e50,
                 ring=demo_ring,
                 ion_field_model="strong",
                 electron_field_model="strong",
                 ion_element_length=demo_ring.L,
                 n_ion_macroparticles_per_bunch=30,
                 generate_method='samples'):

        beam_ion = BeamIonElement(
            ion_mass=ion_mass,
            ion_charge=ion_charge,
            ionization_cross_section=ionization_cross_section,
            residual_gas_density=residual_gas_density,
            ring=ring,
            ion_field_model=ion_field_model,
            electron_field_model=electron_field_model,
            ion_element_length=ion_element_length,
            n_ion_macroparticles_per_bunch=n_ion_macroparticles_per_bunch,
            generate_method=generate_method)
        return beam_ion

    return generate


class TestBeamIonElement:

    @pytest.mark.parametrize('ion_field_model, electron_field_model',
                             [('weak', 'weak'), ('weak', 'strong'),
                              ('strong', 'weak'), ('strong', 'strong')])
    def test_track_bunch(self, generate_beam_ion, small_bunch, ion_field_model,
                         electron_field_model):
        beam_ion = generate_beam_ion(ion_field_model=ion_field_model,
                                     electron_field_model=electron_field_model)
        assert_attr_changed(beam_ion, small_bunch, attrs_changed=["xp", "yp"])

    @pytest.mark.parametrize('ion_field_model, electron_field_model',
                             [('weak', 'weak'), ('weak', 'strong'),
                              ('strong', 'weak'), ('strong', 'strong')])
    def test_track_bunch_partially_lost(self, generate_beam_ion, small_bunch,
                                        ion_field_model, electron_field_model):
        beam_ion = generate_beam_ion(ion_field_model=ion_field_model,
                                     electron_field_model=electron_field_model)
        assert_attr_changed(beam_ion, small_bunch, attrs_changed=["xp", "yp"])
        charge_gen = beam_ion.ion_beam.charge

        # loose half of electron bunch
        small_bunch.alive[0:5] = False
        assert_attr_changed(beam_ion, small_bunch, attrs_changed=["xp", "yp"])

        assert np.isclose(beam_ion.ion_beam.charge, charge_gen * 1.5)

    @pytest.mark.parametrize('ion_field_model, electron_field_model',
                             [('weak', 'weak'), ('weak', 'strong'),
                              ('strong', 'weak'), ('strong', 'strong')])
    def test_track_beam(self, generate_beam_ion, beam_uniform, ion_field_model,
                        electron_field_model):
        beam_ion = generate_beam_ion(ion_field_model=ion_field_model,
                                     electron_field_model=electron_field_model)
        assert_attr_changed(beam_ion, beam_uniform, attrs_changed=["xp", "yp"])

    @pytest.mark.parametrize('ion_field_model, electron_field_model',
                             [('weak', 'weak'), ('weak', 'strong'),
                              ('strong', 'weak'), ('strong', 'strong')])
    def test_track_beam_non_uniform(self, generate_beam_ion, beam_non_uniform,
                                    ion_field_model, electron_field_model):
        beam_ion = generate_beam_ion(ion_field_model=ion_field_model,
                                     electron_field_model=electron_field_model)
        assert_attr_changed(beam_ion,
                            beam_non_uniform,
                            attrs_changed=["xp", "yp"])

    # Ion generation creates expected number of macroparticles with proper distribution
    @pytest.mark.parametrize('generate_method', [('samples'),
                                                 ('distribution')])
    def test_ion_generation(self, generate_beam_ion, large_bunch,
                            generate_method):
        n_ions = 1e5
        large_bunch["x"] += 1
        large_bunch["y"] += 1
        beam_ion = generate_beam_ion(n_ion_macroparticles_per_bunch=n_ions,
                                     generate_method=generate_method)
        beam_ion.generate_new_ions(large_bunch)

        assert len(beam_ion.ion_beam["x"]) == n_ions + 1
        assert np.isclose(beam_ion.ion_beam["x"].mean(),
                          large_bunch["x"].mean(),
                          rtol=0.1)
        assert np.isclose(beam_ion.ion_beam["y"].mean(),
                          large_bunch["y"].mean(),
                          rtol=0.1)

    # Ion drift tracking properly updates ion positions based on momentum
    def test_ion_drift_tracking(self, generate_beam_ion, small_bunch):
        beam_ion = generate_beam_ion()
        beam_ion.generate_new_ions(small_bunch)
        beam_ion.ion_beam["xp"] = np.ones_like(beam_ion.ion_beam["x"])
        beam_ion.ion_beam["yp"] = np.ones_like(beam_ion.ion_beam["y"])

        drift_length = 2.0
        initial_x = beam_ion.ion_beam["x"].copy()
        initial_y = beam_ion.ion_beam["y"].copy()

        beam_ion.track_ions_in_a_drift(drift_length)

        assert np.allclose(beam_ion.ion_beam["x"], initial_x + drift_length)
        assert np.allclose(beam_ion.ion_beam["y"], initial_y + drift_length)

    # Monitor records ion beam data at specified intervals when enabled
    def test_monitor_recording(self, generate_beam_ion, small_bunch,
                               generate_ion_monitor, tmp_path):
        file_name = tmp_path / "test_monitor.hdf5"
        monitor = generate_ion_monitor(file_name=file_name)
        beam_ion = generate_beam_ion()
        beam_ion.monitors.append(monitor)

        beam_ion.track(small_bunch)

        assert os.path.exists(file_name)
        with hp.File(file_name, 'r') as f:
            cond = False
            for key in f.keys():
                if key.startswith('IonData'):
                    cond = True
        assert cond

    # Empty electron bunch handling during ion generation
    def test_empty_bunch_handling(self, generate_beam_ion, generate_bunch):
        beam_ion = generate_beam_ion()
        empty_bunch = generate_bunch(mp_number=0, init_gaussian=False)

        with pytest.raises(ValueError):
            beam_ion.generate_new_ions(empty_bunch)

    # Boundary conditions at aperture edges
    def test_aperture_boundary(self, generate_beam_ion, small_bunch):
        x_radius = 0.001
        aperture = IonAperture(X_radius=x_radius, Y_radius=x_radius)
        beam_ion = generate_beam_ion()
        beam_ion.apertures.append(aperture)

        beam_ion.generate_new_ions(small_bunch)

        beam_ion.ion_beam["x"] = np.ones_like(
            beam_ion.ion_beam["x"]) * (x_radius*1.1)
        beam_ion.track(beam_ion.ion_beam)

        assert len(beam_ion.ion_beam["x"]) == 0

    # Ion clearing removes all particles as expected
    def test_ion_clearing(self, generate_beam_ion, small_bunch):
        beam_ion = generate_beam_ion()
        beam_ion.generate_new_ions(small_bunch)
        assert len(beam_ion.ion_beam["x"]) > 0

        beam_ion.clear_ions()
        assert len(beam_ion.ion_beam["x"]) == 1
        assert beam_ion.ion_beam["x"][0] == 0

    # Tracking with a pre-set cloud
    def test_track_preset_cloud(self, generate_beam_ion, small_bunch,
                                generate_ion_particles):
        beam_ion = generate_beam_ion()
        assert beam_ion.ion_beam.charge == 0

        preset_ion_particles = generate_ion_particles(mp_number=1000)
        preset_charge = 1e-9
        preset_ion_particles.generate_as_a_distribution(
            small_bunch, preset_charge)
        beam_ion.ion_beam += preset_ion_particles
        assert np.isclose(beam_ion.ion_beam.charge, preset_charge)

        beam_ion.generate_ions = False
        assert_attr_changed(beam_ion, small_bunch, attrs_changed=["xp", "yp"])
        assert np.isclose(beam_ion.ion_beam.charge, preset_charge)

        beam_ion.generate_ions = True
        assert_attr_changed(beam_ion, small_bunch, attrs_changed=["xp", "yp"])
        assert beam_ion.ion_beam.charge > preset_charge
