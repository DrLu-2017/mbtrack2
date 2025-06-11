import numpy as np
import pytest

pytestmark = pytest.mark.unit
from utility_test_functions import assert_attr_changed

from mbtrack2 import Sweep


@pytest.fixture
def generate_sweep(demo_ring):

    def generate(ring=demo_ring,
                 f0=1e3,
                 f1=2e3,
                 t1=1e-3,
                 level=1e3,
                 plane="tau",
                 bunch_to_sweep=None):
        sweep = Sweep(ring=ring,
                      f0=f0,
                      f1=f1,
                      t1=t1,
                      level=level,
                      plane=plane,
                      bunch_to_sweep=bunch_to_sweep)
        return sweep

    return generate


class TestSweep:

    # Sweep applies frequency chirp correctly to a single bunch
    @pytest.mark.parametrize("plane, attr", [("x", "xp"), ("y", "yp"),
                                             ("tau", "delta")])
    def test_single_bunch_chirp(self, generate_sweep, small_bunch, plane,
                                attr):
        sweep = generate_sweep(plane=plane)
        assert_attr_changed(sweep, small_bunch, attrs_changed=[attr])

    # Sweep applies frequency chirp correctly to all bunches in a beam
    @pytest.mark.parametrize("plane, attr", [("x", "xp"), ("y", "yp"),
                                             ("tau", "delta")])
    def test_all_bunches_chirp(self, generate_sweep, beam_uniform, plane,
                               attr):
        sweep = generate_sweep(plane=plane)
        assert_attr_changed(sweep, beam_uniform, attrs_changed=[attr])

    # Sweep applies frequency chirp correctly to all bunches in a beam
    @pytest.mark.parametrize("plane, attr", [("x", "xp"), ("y", "yp"),
                                             ("tau", "delta")])
    def test_beam_chirp_single_bunch(self, generate_sweep, beam_uniform, plane,
                                     attr):
        sweep = generate_sweep(plane=plane, bunch_to_sweep=5)
        initial_attr = [bunch[attr].copy() for bunch in beam_uniform]
        sweep.track(beam_uniform)
        for i, bunch in enumerate(beam_uniform):
            if i == 5:
                assert not np.allclose(
                    initial_attr[i],
                    bunch[attr]), f"Chirp not applied correctly to bunch {i}"
            else:
                assert np.allclose(initial_attr[i],
                                   bunch[attr]), f"Chirp applied to bunch {i}"

    # Plot method generates a correct sweep voltage plot
    def test_plot_sweep_voltage(self, generate_sweep):
        sweep = generate_sweep()
        fig = sweep.plot()
        assert fig is not None, "Plot method did not return a figure"

    # Sweep tracks correctly when MPI is enabled
    @pytest.mark.parametrize("plane, attr", [("x", "xp"), ("y", "yp"),
                                             ("tau", "delta")])
    def test_mpi_tracking(self, generate_sweep, beam_1bunch_mpi, plane, attr):
        sweep = generate_sweep(plane=plane)
        bunch = beam_1bunch_mpi[beam_1bunch_mpi.mpi.bunch_num]
        initial_attr = bunch[attr].copy()
        sweep.track(beam_1bunch_mpi)
        assert not np.allclose(
            initial_attr, bunch[attr]), "Chirp not applied correctly with MPI"

    # Sweep correctly resets count after completing a full sweep
    def test_count_reset_after_full_sweep(self, generate_sweep, small_bunch):
        sweep = generate_sweep()
        for _ in range(sweep.N):  # Complete one full sweep cycle
            sweep.track(small_bunch)
        assert sweep.count == 0, "Count did not reset after completing a full sweep"
