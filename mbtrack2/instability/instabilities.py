# -*- coding: utf-8 -*-
"""
General calculations about instability thresholds.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c, e, epsilon_0, m_e, mu_0, pi


def mbi_threshold(ring, sigma, R, b):
    """
    Compute the microbunching instability (MBI) threshold for a bunched beam
    considering the steady-state parallel plate model [1][2].
    
    Parameters
    ----------
    ring : Synchrotron object
    sigma : float
        RMS bunch length in [s]
    R : float
        dipole bending radius in [m]
    b : float
        vertical distance between the conducting parallel plates in [m]
        
    Returns
    -------
    I : float
        MBI current threshold in [A]
        
    [1] : Y. Cai, "Theory of microwave instability and coherent synchrotron 
    radiation in electron storage rings", SLAC-PUB-14561
    [2] : D. Zhou, "Coherent synchrotron radiation and microwave instability in 
    electron storage rings", PhD thesis, p112
    """

    sigma = sigma * c
    Ia = 4 * pi * epsilon_0 * m_e * c**3 / e  # Alfven current
    chi = sigma * (R / b**3)**(1 / 2)  # Shielding paramter
    xi = 0.5 + 0.34*chi
    N = (ring.L * Ia * ring.ac * ring.gamma * ring.sigma_delta**2 * xi *
         sigma**(1 / 3) / (c * e * R**(1 / 3)))
    I = N * e / ring.T0

    return I


def cbi_threshold(ring, I, Vrf, f, beta, Ncav=1):
    """
    Compute the longitudinal and transverse coupled bunch instability 
    thresolds driven by HOMs [1].
    
    Approximate formula, does not take into account variation with Q.
    For better estimate use lcbi_growth_rate.

    Parameters
    ----------
    ring : Synchrotron object
    I : float
        Total beam current in [A].
    Vrf : float
        Total RF voltage in [V].
    f : float
        Frequency of the HOM in [Hz].
    beta : array-like of shape (2,)
        Horizontal and vertical beta function at the HOM position in [m].
    Ncav : int, optional
        Number of RF cavity.

    Returns
    -------
    Zlong : float
        Maximum longitudinal impedance of the HOM in [ohm].
    Zxdip : float
        Maximum horizontal dipolar impedance of the HOM in [ohm/m].
    Zydip : float
        Maximum vertical dipolar impedance of the HOM in [ohm/m].
        
    References
    ----------
    [1] : Ruprecht, Martin, et al. "Calculation of Transverse Coupled Bunch 
    Instabilities in Electron Storage Rings Driven By Quadrupole Higher Order 
    Modes." 7th Int. Particle Accelerator Conf.(IPAC'16), Busan, Korea. 
    """

    fs = ring.synchrotron_tune(Vrf) * ring.f0
    Zlong = fs / (f * ring.ac * ring.tau[2]) * (2 * ring.E0) / (ring.f0 * I *
                                                                Ncav)
    Zxdip = 1 / (ring.tau[0] * beta[0]) * (2 * ring.E0) / (ring.f0 * I * Ncav)
    Zydip = 1 / (ring.tau[1] * beta[1]) * (2 * ring.E0) / (ring.f0 * I * Ncav)

    return (Zlong, Zxdip, Zydip)


def lcbi_growth_rate_mode(ring,
                          I,
                          M,
                          mu,
                          synchrotron_tune=None,
                          Vrf=None,
                          fr=None,
                          RL=None,
                          QL=None,
                          Z=None,
                          bunch_length=None):
    """
    Compute the longitudinal coupled bunch instability growth rate driven by
    an impedance for a given coupled bunch mode mu [1-2].
    
    Use either a list of resonators (fr, RL, QL) or an Impedance object (Z).

    Parameters
    ----------
    ring : Synchrotron object
    I : float
        Total beam current in [A].
    M : int
        Nomber of bunches in the beam.
    mu : int
        Coupled bunch mode number (= 0, ..., M-1).
    synchrotron_tune : float, optional
        Synchrotron tune.
    Vrf : float, optinal
        Total RF voltage in [V] used to compute synchrotron tune if not given.
    fr : float or list, optional
        Frequency of the resonator in [Hz].
    RL : float or list, optional
        Loaded shunt impedance of the resonator in [Ohm].
    QL : float or list, optional
        Loaded quality factor of the resonator.
    Z : Impedance, optional
        Longitunial impedance to consider.
    bunch_length : float, optional
        Bunch length in [s].
        Used to computed the form factor for a resonator impedance if given.
        Default is None.

    Returns
    -------
    float
        Coupled bunch instability growth rate for the mode mu in [s-1].
        
    References
    ----------
    [1] : Eq. 51 p139 of Akai, Kazunori. "RF System for Electron Storage 
    Rings." Physics And Engineering Of High Performance Electron Storage Rings 
    And Application Of Superconducting Technology. 2002. 118-149.
    
    [2] : Tavares, P. F., et al. "Beam-based characterization of higher-order-mode 
    driven coupled-bunch instabilities in a fourth-generation storage ring." 
    NIM A (2022): 165945.

    """
    if synchrotron_tune is None and Vrf is None:
        raise ValueError("Either synchrotron_tune or Vrf is needed.")
    if synchrotron_tune is None:
        nu_s = ring.synchrotron_tune(Vrf)
    else:
        nu_s = synchrotron_tune
    factor = ring.eta() * I / (4 * np.pi * ring.E0 * nu_s)

    if isinstance(fr, (float, int)):
        fr = np.array([fr])
    elif isinstance(fr, list):
        fr = np.array(fr)
    if isinstance(RL, (float, int)):
        RL = np.array([RL])
    elif isinstance(RL, list):
        RL = np.array(RL)
    if isinstance(QL, (float, int)):
        QL = np.array([QL])
    elif isinstance(QL, list):
        QL = np.array(QL)

    if Z is None:
        omega_r = 2 * np.pi * fr
        n_max = int(10 * omega_r.max() / (ring.omega0 * M))

        def Zr(omega):
            Z = 0
            for i in range(len(fr)):
                Z += np.real(RL[i] /
                             (1 + 1j * QL[i] *
                              (omega_r[i] / omega - omega / omega_r[i])))
            return Z
    else:
        fmax = Z.data.index.max()
        n_max = int(2 * np.pi * fmax / (ring.omega0 * M))

        def Zr(omega):
            return np.real(Z(omega / (2 * np.pi)))

    n0 = np.arange(n_max)
    n1 = np.arange(1, n_max)
    omega_p = ring.omega0 * (n0*M + mu + nu_s)
    omega_m = ring.omega0 * (n1*M - mu - nu_s)

    if bunch_length is None:
        FFp = 1
        FFm = 1
    else:
        FFp = np.exp(-(omega_p * bunch_length)**2)
        FFm = np.exp(-(omega_m * bunch_length)**2)

    sum_val = np.sum(omega_p * Zr(omega_p) * FFp) - np.sum(
        omega_m * Zr(omega_m) * FFm)

    return factor * sum_val


def lcbi_growth_rate(ring,
                     I,
                     M,
                     synchrotron_tune=None,
                     Vrf=None,
                     fr=None,
                     RL=None,
                     QL=None,
                     Z=None,
                     bunch_length=None):
    """
    Compute the maximum growth rate for longitudinal coupled bunch instability 
    driven an impedance [1-2].
    
    Use either a list of resonators (fr, RL, QL) or an Impedance object (Z).

    Parameters
    ----------
    ring : Synchrotron object
    I : float
        Total beam current in [A].
    M : int
        Nomber of bunches in the beam.
    synchrotron_tune : float, optional
        Synchrotron tune.
    Vrf : float, optinal
        Total RF voltage in [V] used to compute synchrotron tune if not given.
    fr : float or list, optional
        Frequency of the HOM in [Hz].
    RL : float or list, optional
        Loaded shunt impedance of the HOM in [Ohm].
    QL : float or list, optional
        Loaded quality factor of the HOM.
    Z : Impedance, optional
        Longitunial impedance to consider.
    bunch_length : float, optional
        Bunch length in [s].
        Used to computed the form factor for a resonator impedance if given.
        Default is None.

    Returns
    -------
    growth_rate : float
        Maximum coupled bunch instability growth rate in [s-1].
    mu : int
        Coupled bunch mode number corresponding to the maximum coupled bunch 
        instability growth rate.
    growth_rates : array
        Coupled bunch instability growth rates for the different mode numbers 
        in [s-1].
        
    References
    ----------
    [1] : Eq. 51 p139 of Akai, Kazunori. "RF System for Electron Storage 
    Rings." Physics And Engineering Of High Performance Electron Storage Rings 
    And Application Of Superconducting Technology. 2002. 118-149.
    
    [2] : Tavares, P. F., et al. "Beam-based characterization of higher-order-mode 
    driven coupled-bunch instabilities in a fourth-generation storage ring." 
    NIM A (2022): 165945.

    """
    growth_rates = np.zeros(M)
    for i in range(M):
        growth_rates[i] = lcbi_growth_rate_mode(
            ring,
            I,
            M,
            i,
            synchrotron_tune=synchrotron_tune,
            Vrf=Vrf,
            fr=fr,
            RL=RL,
            QL=QL,
            Z=Z,
            bunch_length=bunch_length)

    growth_rate = np.max(growth_rates)
    mu = np.argmax(growth_rates)

    return growth_rate, mu, growth_rates


def lcbi_stability_diagram(ring,
                           I,
                           M,
                           modes,
                           cavity_list,
                           detune_range,
                           synchrotron_tune=None,
                           Vrf=None):
    """
    Plot longitudinal coupled bunch instability stability diagram for a 
    arbitrary list of CavityResonator objects around a detuning range.
    
    Last object in the cavity_list is assumed to be the one with the variable 
    detuning.

    Parameters
    ----------
    ring : Synchrotron object
        Ring parameters.
    I : float
        Total beam current in [A].
    M : int
        Nomber of bunches in the beam.
    modes : list
        Coupled bunch mode numbers to consider.
    cavity_list : list
        List of CavityResonator objects to consider, which can be:
            - active cavities
            - passive cavities
            - HOMs
            - mode dampers
    detune_range : array
        Detuning range (list of points) of the last CavityResonator object.
    synchrotron_tune : float, optional
        Synchrotron tune.
    Vrf : float, optinal
        Total RF voltage in [V] used to compute synchrotron tune if not given.

    Returns
    -------
    fig : Figure
        Show the shunt impedance threshold for different coupled bunch modes.

    """

    Rth = np.zeros_like(detune_range)
    fig, ax = plt.subplots()

    for mu in modes:
        fixed_gr = 0
        for cav in cavity_list[:-1]:
            fixed_gr += lcbi_growth_rate_mode(
                ring,
                I=I,
                mu=mu,
                fr=cav.fr,
                RL=cav.RL,
                QL=cav.QL,
                M=M,
                synchrotron_tune=synchrotron_tune,
                Vrf=Vrf)

        cav = cavity_list[-1]
        for i, det in enumerate(detune_range):
            gr = lcbi_growth_rate_mode(ring,
                                       I=I,
                                       mu=mu,
                                       fr=cav.m * ring.f1 + det,
                                       RL=cav.RL,
                                       QL=cav.QL,
                                       M=M,
                                       synchrotron_tune=synchrotron_tune,
                                       Vrf=Vrf)
            Rth[i] = (1 / ring.tau[2] - fixed_gr) * cav.Rs / gr

        ax.plot(detune_range * 1e-3,
                Rth * 1e-6,
                label="$\mu$ = " + str(int(mu)))

    plt.xlabel(r"$\Delta f$ [kHz]")
    plt.ylabel(r"$R_{s,max}$ $[M\Omega]$")
    plt.yscale("log")
    plt.legend()

    return fig


def rwmbi_growth_rate(ring, current, beff, rho_material, plane='x'):
    """
    Compute the growth rate of the transverse coupled-bunch instability induced
    by resistive wall impedance [1].

    Parameters
    ----------
    ring : Synchrotron object
    current : float
        Total beam current in [A].
    beff : float
        Effective radius of the vacuum chamber in [m].
    rho_material : float
        Resistivity of the chamber's wall material in [Ohm.m].
    plane : str, optional
        The plane in which the instability will be computed. Use 'x' for the 
        horizontal plane, and 'y' for the vertical.

    Reference
    ---------
    [1] Eq. (31) in R. Nagaoka and K. L. F. Bane, "Collective effects in a
    diffraction-limited storage ring", J. Synchrotron Rad. Vol 21, 2014. pp.937-960 

    """
    plane_dict = {'x': 0, 'y': 1}
    index = plane_dict[plane]
    beta0 = ring.optics.local_beta[index]
    omega0 = ring.omega0
    E0 = ring.E0
    R = ring.L / (2 * np.pi)
    frac_tune, int_tune = math.modf(ring.tune[index])
    Z0 = mu_0 * c

    gr = (beta0*omega0*current*R) / (4 * np.pi * E0 * beff**3) * (
        (2*c*Z0*rho_material) / ((1-frac_tune) * omega0))**0.5

    return gr


def rwmbi_threshold(ring, beff, rho_material, plane='x'):
    """
    Compute the threshold current of the transverse coupled-bunch instability 
    induced by resistive wall impedance [1].

    Parameters
    ----------
    ring : Synchrotron object
    beff : float
        Effective radius of the vacuum chamber in [m].
    rho_material : float
        Resistivity of the chamber's wall material in [Ohm.m].
    plane : str, optional
        The plane in which the instability will be computed. Use 'x' for the 
        horizontal plane, and 'y' for the vertical.

    Reference
    ---------
    [1] Eq. (32) in R. Nagaoka and K. L. F. Bane, "Collective effects in a
    diffraction-limited storage ring", J. Synchrotron Rad. Vol 21, 2014. pp.937-960 

    """
    plane_dict = {'x': 0, 'y': 1}
    index = plane_dict[plane]
    beta0 = ring.optics.local_beta[index]
    omega0 = ring.omega0
    E0 = ring.E0
    tau_rad = ring.tau[index]
    frac_tune, int_tune = math.modf(ring.tune[index])
    Z0 = mu_0 * c

    Ith = (4 * np.pi * E0 * beff**3) / (c*beta0*tau_rad) * ((
        (1-frac_tune) * omega0) / (2*c*Z0*rho_material))**0.5

    return Ith


def transverse_gaussian_space_charge_tune_shift(ring, bunch_current, **kwargs):
    """
    Return the (maximum) transverse space charge tune shift for a Gaussian 
    bunch in the linear approximation, see Eq.(1) of [1].

    Parameters
    ----------
    ring : Synchrotron object
        Ring parameters.
    bunch_current : float
        Bunch current in [A].
    sigma_s : float, optional
        RMS bunch length in [s].
        Default is ring.sigma_0.
    emit_x : float, optional
        Horizontal emittance in [m.rad].
        Default is ring.emit[0].
    emit_y : float, optional
        Vertical emittance in [m.rad].
        Default is ring.emit[1].
    use_lattice : bool, optional
        If True, use beta fonctions along the lattice.
        If False, local values of beta fonctions are used.
        Default is ring.optics.use_local_values.
    n_points : int, optional
        Number of points in the lattice to be considered if use_lattice ==
        True. Default is 1000.
    sigma_delta : float, optional
        Relative energy spread.
        Default is ring.sigma_delta.
    gamma : float, optional
        Relativistic Lorentz gamma.
        Default is ring.gamma.

    Returns
    -------
    tune_shift : array of shape (2,)
        Horizontal and vertical space charge tune shift.
        
    Reference
    ---------
    [1] : Antipov, S. A., Gubaidulin, V., Agapov, I., Garcia, E. C., & 
    Gamelin, A. (2024). Space Charge and Future Light Sources. 
    arXiv preprint arXiv:2409.08637.

    """
    sigma_s = kwargs.get('sigma_s', ring.sigma_0)
    emit_x = kwargs.get('emit_x', ring.emit[0])
    emit_y = kwargs.get('emit_y', ring.emit[1])
    use_lattice = kwargs.get('use_lattice', not ring.optics.use_local_values)
    sigma_delta = kwargs.get('sigma_delta', ring.sigma_delta)
    gamma = kwargs.get('gamma', ring.gamma)
    n_points = kwargs.get('n_points', 1000)
    q = ring.particle.charge
    m = ring.particle.mass
    r_0 = 1 / (4*pi*epsilon_0) * q**2 / (m * c**2)
    N = np.abs(bunch_current / ring.f0 / q)
    sigma_z = sigma_s * c

    if use_lattice:
        s = np.linspace(0, ring.L, n_points)
        beta = ring.optics.beta(s)
        sig_x = (emit_x * beta[0] +
                 ring.optics.dispersion(s)[0]**2 * sigma_delta**2)**0.5
        sig_y = (emit_y * beta[1] +
                 ring.optics.dispersion(s)[2]**2 * sigma_delta**2)**0.5
        sig_xy = np.array([sig_x, sig_y])
        return -r_0 * N / ((2 * pi)**1.5 * gamma**3 * sigma_z) * np.trapz(
            beta / (sig_xy * (sig_y+sig_x)), s)
    else:
        beta = ring.optics.local_beta
        sig_x = np.sqrt(beta[0] * emit_x)
        sig_y = np.sqrt(beta[1] * emit_y)
        sig_xy = np.array([sig_x, sig_y])
        return -r_0 * N * ring.L / (
            (2 * pi)**1.5 * gamma**3 * sigma_z) * beta / (sig_xy *
                                                          (sig_y+sig_x))
