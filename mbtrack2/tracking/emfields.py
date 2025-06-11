"""
A package dealing with particles electromagnetic fields.
For example, it can be applied to space charge, beam-beam force, electron lenses or beam-ion instabilities.
This is largely adapted from a fork of PyHEADTAIL https://github.com/gubaidulinvadim/PyHEADTAIL.
Only the fastest Fadeeva implementation of the error function is used here.
See  Oeftiger, A., de Maria, R., Deniau, L., Li, K., McIntosh, E., Moneta, L., Hegglin, S., Aviral, A. (2016).
Review of CPU and GPU Faddeeva Implementations. https://cds.cern.ch/record/2207430/files/wepoy044.pdf
"""

from functools import wraps

import numpy as np
from scipy.constants import epsilon_0, pi
from scipy.special import wofz as _scipy_wofz


def _wofz(x, y):
    """
    Compute the Faddeeva function w(z) = exp(-z^2) * erfc(-i*z).

    Parameters
    ----------
    x : float
        Real part of the argument.
    y : float
        Imaginary part of the argument.

    Returns
    -------
    tuple
        Real and imaginary parts of the Faddeeva function.
    """
    res = _scipy_wofz(x + 1j*y)
    return res.real, res.imag


def _sqrt_sig(sig_x, sig_y):
    """
    Compute the square root of the difference between the squared transverse rms and vertical rms.

    Parameters
    ----------
    sig_x : float
        Transverse rms of the distribution.
    sig_y : float
        Vertical rms of the distribution.

    Returns
    -------
    float
        Square root of the difference between the squared transverse rms and vertical rms.
    """
    return np.sqrt(2 * (sig_x*sig_x - sig_y*sig_y))


def _efieldn_mit(x, y, sig_x, sig_y):
    """
    Returns electromagnetic fields as E_x/Q, E_y/Q in (V/m/Coulomb).

    Parameters
    ----------
    x : np.ndarray
        x coordinates in meters.
    y : np.ndarray
        y coordinates in meters.
    sig_x : float
        Transverse rms of the distribution in meters.
    sig_y : float
        Vertical rms of the distribution in meters.

    Returns
    -------
    tuple
        Normalized electromagnetic fields Ex/Q, Ey/Q in the units of (V/m/Coulomb).
    """
    sig_sqrt = _sqrt_sig(sig_x, sig_y)
    w1re, w1im = _wofz(x / sig_sqrt, y / sig_sqrt)
    ex = np.exp(-x * x / (2*sig_x*sig_x) - y * y / (2*sig_y*sig_y))
    w2re, w2im = _wofz(x * sig_y / (sig_x*sig_sqrt),
                       y * sig_x / (sig_y*sig_sqrt))
    denom = 2 * epsilon_0 * np.sqrt(pi) * sig_sqrt
    return (w1im - ex*w2im) / denom, (w1re - ex*w2re) / denom


def efieldn_gauss_round(x, y, sig_x, sig_y):
    """
    Computes the electromagnetic field of a round Gaussian distribution.

    Parameters
    ----------
    x : np.ndarray
        x coordinates in meters.
    y : np.ndarray
        y coordinates in meters.
    sig_x : float
        Transverse rms of the distribution in meters.
    sig_y : float
        Vertical rms of the distribution in meters.

    Returns
    -------
    tuple
        Normalized electromagnetic fields Ex/Q, Ey/Q in the units of (V/m/Coulomb).
    """
    r_squared = x*x + y*y
    sig_r = sig_x
    amplitude = -np.expm1(-r_squared /
                          (2*sig_r*sig_r)) / (2*pi*epsilon_0*r_squared)
    return x * amplitude, y * amplitude


def _efieldn_linearized(x, y, sig_x, sig_y):
    """
    Computes linearized electromagnetic field.

    Parameters
    ----------
    x : np.ndarray
        x coordinate in meters.
    y : np.ndarray
        y coordinate in meters.
    sig_x : float
        Vertical rms of the distribution in meters.
    sig_y : float
        Vertical rms of the distribution in meters.

    Returns
    -------
    tuple
        Normalized electromagnetic fields Ex/Q, Ey/Q in the units of (V/m/Coulomb).
    """
    a = np.sqrt(2) * sig_x
    b = np.sqrt(2) * sig_y
    amplitude = 1 / (pi * epsilon_0 * (a+b))
    return x / a * amplitude, y / b * amplitude


def add_sigma_check(efieldn):
    """
    Wrapper for a normalized electromagnetic field function.
    Adds the following actions before calculating the field:
    1) Exchange x and y quantities if sig_x < sig_y.
    2) Apply round beam field formula when sig_x is close to sig_y.

    Parameters
    ----------
    efieldn : function
        Function to calculate normalized electromagnetic field.

    Returns
    -------
    function
        Wrapped function, including round beam and inverted sig_x/sig_y.
    """
    sigmas_ratio_threshold = 1e-3
    absolute_threshold = 1e-10

    @wraps(efieldn)
    def efieldn_checked(x, y, sig_x, sig_y, *args, **kwargs):
        tol_kwargs = dict(rtol=sigmas_ratio_threshold, atol=absolute_threshold)
        if np.allclose(sig_x, sig_y, **tol_kwargs):
            if np.allclose(sig_y, 0, **tol_kwargs):
                en_x = en_y = np.zeros(x.shape, dtype=np.float64)
            else:
                en_x, en_y = efieldn_gauss_round(x, y, sig_x, sig_y, *args,
                                                 **kwargs)
        elif np.all(sig_x < sig_y):
            en_y, en_x = efieldn(y, x, sig_y, sig_x, *args, **kwargs)
        else:
            en_x, en_y = efieldn(x, y, sig_x, sig_y, *args, **kwargs)
        return en_x, en_y

    return efieldn_checked


def get_displaced_efield(efieldn, xr, yr, sig_x, sig_y, mean_x, mean_y):
    """
    Compute the charge-normalized electric field components of a two-dimensional Gaussian charge distribution.

    Parameters
    ----------
    efieldn : function
        Calculates electromagnetic field of a given distribution of charges.
    xr : np.array
        x coordinates in meters.
    yr : np.array
        y coordinates in meters.
    sig_x : float
        Horizontal rms size in meters.
    sig_y : float
        Vertical rms size in meters.
    mean_x : float
        Horizontal mean of the distribution in meters.
    mean_y : float
        Vertical mean of the distribution in meters.

    Returns
    -------
    tuple
        Charge-normalized electromagnetic fields with a displaced center of the distribution.
    """
    x = xr - mean_x
    y = yr - mean_y
    efieldn = add_sigma_check(efieldn)
    en_x, en_y = efieldn(np.abs(x), np.abs(y), sig_x, sig_y)
    en_x = np.abs(en_x) * np.sign(x)
    en_y = np.abs(en_y) * np.sign(y)

    return en_x, en_y
