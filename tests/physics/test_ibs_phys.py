import numpy as np
import pytest

pytestmark = pytest.mark.phys
from mbtrack2 import Electron, Synchrotron
from mbtrack2.tracking import (
    Bunch,
    IntrabeamScattering,
    LongitudinalMap,
    RFCavity,
    SynchrotronRadiation,
    TransverseMap,
)
from mbtrack2.utilities import Optics


@pytest.fixture
def model_ring():
    h = 416  # Harmonic number of the accelerator.
    L = 353.97  # Ring circumference in [m].
    E0 = 2.75e9  # Nominal (total) energy of the ring in [eV].
    particle = Electron()  # Particle considered.
    ac = 1.0695e-5  #1.0695e-4
    U0 = 452.6e3  # Energy loss per turn in [eV].
    tau = np.array([
        7.68e-3, 14.14e-3, 12.18e-3
    ])  #horizontal, vertical and longitudinal damping times in [s].
    tune = np.array([54.2, 18.3])
    emit = np.array([84.4e-12, 84.4e-13])
    sigma_0 = 9e-12
    sigma_delta = 9.07649e-4
    chro = np.array([1.8, 1.3])
    beta = np.array([3.288, 4.003])
    alpha = np.array([0.004, 0.02])
    dispersion = np.array([0.008, 0.01, 0.003, 0.001])
    optics = Optics(local_beta=beta,
                    local_alpha=alpha,
                    local_dispersion=dispersion)
    ring = Synchrotron(h=h,
                       optics=optics,
                       particle=particle,
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


def test_ibs_PS(model_ring):
    modelname = "PS"
    bunch = Bunch(model_ring,
                  mp_number=10000,
                  current=1.2e-3,
                  track_alive=True)
    np.random.seed(42)
    bunch.init_gaussian()

    long_map = LongitudinalMap(model_ring)
    trans_map = TransverseMap(model_ring)
    rf = RFCavity(model_ring,
                  m=1,
                  Vc=1.8e6,
                  theta=np.arccos(model_ring.U0 / 1.8e6))
    sr = SynchrotronRadiation(model_ring, switch=[1, 1, 1])
    with pytest.warns(UserWarning):
        ibs = IntrabeamScattering(model_ring,
                                  model=modelname,
                                  n_points=1000,
                                  n_bin=1)
    long_map.track(bunch)
    trans_map.track(bunch)
    rf.track(bunch)
    sr.track(bunch)
    ibs.initialize(bunch)
    vabq, v1aq, v1bq = ibs.scatter()
    T_x, T_y, T_p = ibs.get_scatter_T(vabq=vabq, v1aq=v1aq, v1bq=v1bq)
    ibs.kick(bunch, T_x, T_y, T_p)
    assert pytest.approx(T_x, rel=0.1) == 17.37129660972345
    assert pytest.approx(T_y, rel=0.1) == 11.436684189250942
    assert pytest.approx(T_p, rel=0.1) == 94.03135908935525
    assert pytest.approx(vabq, rel=0.1) == 10320.6063359
    assert pytest.approx(v1aq, rel=0.1) == -4.13639174
    assert pytest.approx(v1bq, rel=0.1) == -7.9521672


def test_ibs_PM(model_ring):
    modelname = "PM"
    bunch = Bunch(model_ring,
                  mp_number=10000,
                  current=1.2e-3,
                  track_alive=True)
    np.random.seed(42)
    bunch.init_gaussian()

    long_map = LongitudinalMap(model_ring)
    trans_map = TransverseMap(model_ring)
    rf = RFCavity(model_ring,
                  m=1,
                  Vc=1.8e6,
                  theta=np.arccos(model_ring.U0 / 1.8e6))
    sr = SynchrotronRadiation(model_ring, switch=[1, 1, 1])
    with pytest.warns(UserWarning):
        ibs = IntrabeamScattering(model_ring,
                                  model=modelname,
                                  n_points=1000,
                                  n_bin=1)
    long_map.track(bunch)
    trans_map.track(bunch)
    rf.track(bunch)
    sr.track(bunch)
    ibs.initialize(bunch)
    vabq, v1aq, v1bq = ibs.scatter()
    T_x, T_y, T_p = ibs.get_scatter_T(vabq=vabq, v1aq=v1aq, v1bq=v1bq)
    ibs.kick(bunch, T_x, T_y, T_p)
    assert pytest.approx(T_x, rel=0.1) == 9.61398035949779
    assert pytest.approx(T_y, rel=0.1) == 6.3217250733412085
    assert pytest.approx(T_p, rel=0.1) == 52.035206316407645
    assert pytest.approx(vabq, rel=0.1) == 20417.58389059
    assert pytest.approx(v1aq, rel=0.1) == -2.88678694
    assert pytest.approx(v1bq, rel=0.1) == -4.31433794


def test_ibs_Bane(model_ring):
    modelname = "Bane"
    bunch = Bunch(model_ring,
                  mp_number=10000,
                  current=1.2e-3,
                  track_alive=True)
    np.random.seed(42)
    bunch.init_gaussian()

    long_map = LongitudinalMap(model_ring)
    trans_map = TransverseMap(model_ring)
    rf = RFCavity(model_ring,
                  m=1,
                  Vc=1.8e6,
                  theta=np.arccos(model_ring.U0 / 1.8e6))
    sr = SynchrotronRadiation(model_ring, switch=[1, 1, 1])
    with pytest.warns(UserWarning):
        ibs = IntrabeamScattering(model_ring,
                                  model=modelname,
                                  n_points=1000,
                                  n_bin=1)
    long_map.track(bunch)
    trans_map.track(bunch)
    rf.track(bunch)
    sr.track(bunch)
    ibs.initialize(bunch)
    gval = ibs.scatter()
    T_x, T_y, T_p = ibs.get_scatter_T(gval=gval)
    ibs.kick(bunch, T_x, T_y, T_p)
    assert pytest.approx(T_x, rel=0.1) == 218.0367680074528
    assert pytest.approx(T_y, rel=0.1) == 22.671331141800803
    assert pytest.approx(T_p, rel=0.1) == 65.48600681585525
    assert pytest.approx(gval, rel=0.1) == 0.94439633


def test_ibs_CIMP(model_ring):
    modelname = "CIMP"
    bunch = Bunch(model_ring,
                  mp_number=10000,
                  current=1.2e-3,
                  track_alive=True)
    np.random.seed(42)
    bunch.init_gaussian()

    long_map = LongitudinalMap(model_ring)
    trans_map = TransverseMap(model_ring)
    rf = RFCavity(model_ring,
                  m=1,
                  Vc=1.8e6,
                  theta=np.arccos(model_ring.U0 / 1.8e6))
    sr = SynchrotronRadiation(model_ring, switch=[1, 1, 1])
    with pytest.warns(UserWarning):
        ibs = IntrabeamScattering(model_ring,
                                  model=modelname,
                                  n_points=1000,
                                  n_bin=1)
    long_map.track(bunch)
    trans_map.track(bunch)
    rf.track(bunch)
    sr.track(bunch)
    ibs.initialize(bunch)
    g_ab, g_ba = ibs.scatter()
    T_x, T_y, T_p = ibs.get_scatter_T(g_ab=g_ab, g_ba=g_ba)
    assert pytest.approx(T_x, rel=0.1) == 214.1723224950628
    assert pytest.approx(T_y, rel=0.1) == 22.208894776125746
    assert pytest.approx(T_p, rel=0.1) == 64.34338366871349
    assert pytest.approx(g_ab, rel=0.1) == 0.83332409
    assert pytest.approx(g_ba, rel=0.1) == 1.74674087
