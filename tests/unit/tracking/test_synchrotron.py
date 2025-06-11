import numpy as np
import pytest

pytestmark = pytest.mark.unit
import at
from scipy.constants import c, e

from mbtrack2 import Electron, Synchrotron


@pytest.fixture
def demo_ring(local_optics):
    h = 20
    L = 100
    E0 = 1e9
    particle = Electron()
    ac = 1e-3
    U0 = 250e3
    tau = np.array([10e-3, 10e-3, 5e-3])
    tune = np.array([18.2, 10.3])
    emit = np.array([50e-9, 50e-9 * 0.01])
    sigma_0 = 30e-12
    sigma_delta = 1e-3
    chro = [1.0, 1.0]

    ring = Synchrotron(h,
                       local_optics,
                       particle,
                       L=L,
                       E0=E0,
                       ac=ac,
                       U0=U0,
                       tau=tau,
                       emit=emit,
                       tune=tune,
                       sigma_delta=sigma_delta,
                       sigma_0=sigma_0,
                       chro=chro)

    return ring


@pytest.fixture
def demo_ring_h1(demo_ring):
    demo_ring.h = 1
    return demo_ring


@pytest.fixture
def generate_ring_with_at_lattice(generate_at_optics):

    def generate(**kwargs):
        optics = generate_at_optics(**kwargs)
        h = 416
        tau = np.array([6.56e-3, 6.56e-3, 3.27e-3])
        emit = np.array([3.9e-9, 3.9e-9 * 0.01])
        sigma_0 = 15e-12
        sigma_delta = 1.025e-3
        particle = Electron()
        ring = Synchrotron(h,
                           optics,
                           particle,
                           tau=tau,
                           emit=emit,
                           sigma_0=sigma_0,
                           sigma_delta=sigma_delta,
                           **kwargs)
        return ring

    return generate


@pytest.fixture
def ring_with_at_lattice(generate_ring_with_at_lattice):
    ring = generate_ring_with_at_lattice()
    return ring


class TestSynchrotron:

    def test_synchrotron_values(self, demo_ring):
        h = 20
        L = 100
        E0 = 1e9
        particle = Electron()
        ac = 1e-3
        U0 = 250e3
        tau = np.array([10e-3, 10e-3, 5e-3])
        tune = np.array([18.2, 10.3])
        emit = np.array([50e-9, 50e-9 * 0.01])
        sigma_0 = 30e-12
        sigma_delta = 1e-3
        chro = [1.0, 1.0]

        assert pytest.approx(demo_ring.h) == h
        assert pytest.approx(demo_ring.L) == L
        assert pytest.approx(demo_ring.E0) == E0
        assert pytest.approx(demo_ring.U0) == U0
        assert pytest.approx(demo_ring.ac) == ac
        np.testing.assert_allclose(demo_ring.tau, tau)
        np.testing.assert_allclose(demo_ring.tune, tune)
        np.testing.assert_allclose(demo_ring.emit, emit)
        assert pytest.approx(demo_ring.sigma_0) == sigma_0
        assert pytest.approx(demo_ring.sigma_delta) == sigma_delta
        np.testing.assert_allclose(demo_ring.chro, chro)
        assert pytest.approx(demo_ring.T0) == L / c
        assert pytest.approx(demo_ring.T1) == L / c / h
        assert pytest.approx(demo_ring.f0) == c / L
        assert pytest.approx(demo_ring.f1) == 1 / (L/c/h)
        assert pytest.approx(demo_ring.omega0) == 2 * np.pi * c / L
        assert pytest.approx(demo_ring.omega1) == 2 * np.pi * 1 / (L/c/h)
        assert pytest.approx(demo_ring.k1) == 2 * np.pi * 1 / (L/c/h) / c
        assert pytest.approx(
            demo_ring.gamma) == E0 / (particle.mass * c**2 / e)
        assert pytest.approx(
            demo_ring.beta) == np.sqrt(1 - (E0 /
                                            (particle.mass * c**2 / e))**-2)

    def test_synchrotron_mcf(self, demo_ring):
        demo_ring.mcf_order = [5e-4, 1e-4, 1e-3]
        assert pytest.approx(
            demo_ring.mcf(0.5)) == 5e-4 * (0.5**2) + 1e-4*0.5 + 1e-3
        assert pytest.approx(demo_ring.eta(
            0.5)) == demo_ring.mcf(0.5) - 1 / (demo_ring.gamma**2)

    def test_synchrotron_tune(self, demo_ring):
        tuneS = demo_ring.synchrotron_tune(1e6)
        assert pytest.approx(tuneS, rel=1e-4) == 0.0017553

    def test_synchrotron_sigma(self, demo_ring):
        np.testing.assert_allclose(
            demo_ring.sigma(),
            np.array([
                2.23606798e-04, 2.23606798e-04, 2.23606798e-05, 2.23606798e-05
            ]))

    def test_synchrotron_sigma_position(self, ring_with_at_lattice):
        pos = np.linspace(0, ring_with_at_lattice.L, 100)
        sig = ring_with_at_lattice.sigma(pos)
        assert sig.shape == (4, 100)

    def test_get_adts(self, ring_with_at_lattice):
        ring_with_at_lattice.get_adts()
        assert ring_with_at_lattice.adts is not None

    def test_get_adts_plot(self, ring_with_at_lattice):
        axes = ring_with_at_lattice.get_adts(plot=True)
        assert axes is not None

    def test_get_chroma(self, ring_with_at_lattice):
        ring_with_at_lattice.get_chroma()
        assert len(ring_with_at_lattice.chro) == 8

    def test_get_mcf_order(self, ring_with_at_lattice):
        ring_with_at_lattice.get_mcf_order()
        assert len(ring_with_at_lattice.mcf_order) == 3

    def test_synchrotron_long_twiss(self, demo_ring):
        tuneS, long_alpha, long_beta, long_gamma = demo_ring.get_longitudinal_twiss(
            1e6, add=False)
        assert pytest.approx(tuneS,
                             rel=1e-4) == demo_ring.synchrotron_tune(1e6)
        assert pytest.approx(long_alpha, rel=1e-4) == -0.0055146
        assert pytest.approx(long_beta, rel=1e-4) == 3.0236e-08
        assert pytest.approx(long_gamma, rel=1e-4) == 3.30736e7

    def test_to_pyat(self, demo_ring):
        pyat_simple_ring = demo_ring.to_pyat(1e6)
        assert isinstance(pyat_simple_ring, at.lattice.lattice_object.Lattice)
