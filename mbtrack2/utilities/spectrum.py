# -*- coding: utf-8 -*-
"""
Module where bunch and beam spectrums and profile are defined.
"""

from math import factorial

import numpy as np
from scipy.special import jv, spherical_jn


def spectral_density(frequency, sigma, m=1, k=0, mode="Hermite"):
    """
    Compute the spectral density of different modes for various values of the
    head-tail mode number, based on Table 1 p238 of [1].

    Parameters
    ----------
    frequency : list or numpy array
        sample points of the spectral density in [Hz]
    sigma : float
        RMS bunch length in [s]
    m : int, optional
        head-tail (or azimutal/synchrotron) mode number
    k : int, optional
        radial mode number (such that |q|=m+2k, where |q| is the head-tail mode number)
    mode: str, optional
        type of the mode taken into account for the computation:
        -"Hermite" modes for Gaussian bunches (typical for electrons)
        -"Chebyshev" for airbag bunches
        -"Legendre" for parabolic bunches (typical for protons)
        -"Sacherer" or "Sinusoidal" simplifying approximation of Legendre modes from [3]

    Returns
    -------
    numpy array

    References
    ----------
    [1] : Handbook of accelerator physics and engineering, 3rd printing.
    [2] : Ng, K. Y. (2005). Physics of intensity dependent beam instabilities. WORLD SCIENTIFIC. https://doi.org/10.1142/5835
    [3] : Sacherer, F. J. (1972). Methods for computing bunched beam instabilities. CERN Tech. rep. CERN/SI-BR/72-5 https://cds.cern.ch/record/2291670?ln=en
    """

    if mode == "Hermite":
        return 1 / (factorial(m) * 2**m) * (2 * np.pi * frequency * sigma)**(
            2 * m) * np.exp(-(2 * np.pi * frequency * sigma)**2)
    elif mode == "Chebyshev":
        tau_l = 4 * sigma
        return (jv(m, 2 * np.pi * frequency * tau_l))**2
    elif mode == "Legendre":
        tau_l = 4 * sigma
        return (spherical_jn(m, np.abs(2 * np.pi * frequency * tau_l)))**2
    elif mode == "Sacherer" or mode == "Sinusoidal":
        y = 4 * 2 * np.pi * frequency * sigma / np.pi
        return (2 * (m+1) / np.pi * 1 / np.abs(y**2 - (m + 1)**2) *
                np.sqrt(1 + (-1)**m * np.cos(np.pi * y)))**2
    else:
        raise NotImplementedError("Not implemanted yet.")


def gaussian_bunch_spectrum(frequency, sigma):
    """
    Compute a Gaussian bunch spectrum [1].

    Parameters
    ----------
    frequency : array
        sample points of the beam spectrum in [Hz].
    sigma : float
        RMS bunch length in [s].

    Returns
    -------
    bunch_spectrum : array
        Bunch spectrum sampled at points given in frequency.

    References
    ----------
    [1] : Gamelin, A. (2018). Collective effects in a transient microbunching 
    regime and ion cloud mitigation in ThomX. p86, Eq. 4.19
    """
    return np.exp(-1 / 2 * (2 * np.pi * frequency)**2 * sigma**2)


def gaussian_bunch(time, sigma):
    """
    Compute a Gaussian bunch profile.

    Parameters
    ----------
    time : array
        sample points of the bunch profile in [s].
    sigma : float
        RMS bunch length in [s].

    Returns
    -------
    bunch_profile : array
        Bunch profile in [s**-1] sampled at points given in time.
    """
    return np.exp(-1 / 2 * (time**2 / sigma**2)) / (sigma * np.sqrt(2 * np.pi))


def beam_spectrum(frequency,
                  M,
                  bunch_spacing,
                  sigma=None,
                  bunch_spectrum=None):
    """
    Compute the beam spectrum assuming constant spacing between bunches [1].

    Parameters
    ----------
    frequency : list or numpy array
        sample points of the beam spectrum in [Hz].
    M : int
        Number of bunches.
    bunch_spacing : float
        Time between two bunches in [s].
    sigma : float, optional
        If bunch_spectrum is None then a Gaussian bunch with sigma RMS bunch 
        length in [s] is assumed.
    bunch_spectrum : array, optional
        Bunch spectrum sampled at points given in frequency.

    Returns
    -------
    beam_spectrum : array

    References
    ----------
    [1] Rumolo, G - Beam Instabilities - CAS - CERN Accelerator School: 
        Advanced Accelerator Physics Course - 2014, Eq. 9
    """

    if bunch_spectrum is None:
        bunch_spectrum = gaussian_bunch_spectrum(frequency, sigma)

    beam_spectrum = (bunch_spectrum *
                     np.exp(1j * np.pi * frequency * bunch_spacing * (M-1)) *
                     np.sin(M * np.pi * frequency * bunch_spacing) /
                     np.sin(np.pi * frequency * bunch_spacing))

    return beam_spectrum
