# -*- coding: utf-8 -*-
"""
This module defines miscellaneous utilities functions.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.special import factorial, gamma, hyp2f1

from mbtrack2.impedance.wakefield import Impedance
from mbtrack2.utilities.spectrum import spectral_density


def effective_impedance(ring,
                        imp,
                        m,
                        mu,
                        sigma,
                        M,
                        tuneS,
                        xi=None,
                        mode="Hermite"):
    """
    Compute the effective (longitudinal or transverse) impedance. 
    Formulas from Eq. (1) and (2) p238 of [1].

    Parameters
    ----------
    ring : Synchrotron object
    imp : Impedance object
    mu : int
        coupled bunch mode number, goes from 0 to (M-1) where M is the
        number of bunches
    m : int
        head-tail (or azimutal/synchrotron) mode number
    sigma : float
        RMS bunch length in [s]
    M : int
        Number of bunches.
    tuneS : float
        Synchrotron tune.
    xi : float, optional
        (non-normalized) chromaticity
    mode: str, optional
        type of the mode taken into account for the computation:
        -"Hermite" modes for Gaussian bunches

    Returns
    -------
    Zeff : float 
        effective impedance in [ohm] or in [ohm/m] depanding on the impedance
        type.

    References
    ----------
    [1] : Handbook of accelerator physics and engineering, 3rd printing.
    """

    if not isinstance(imp, Impedance):
        raise TypeError("{} should be an Impedance object.".format(imp))

    fmin = imp.data.index.min()
    fmax = imp.data.index.max()
    if fmin > 0:
        double_sided_impedance(imp)

    if mode in ["Hermite", "Legendre", "Sinusoidal", "Sacherer", "Chebyshev"]:

        def h(f):
            return spectral_density(frequency=f, sigma=sigma, m=m, mode=mode)
    else:
        raise NotImplementedError("Not implemanted yet.")

    if imp.component_type == "long":
        pmax = fmax / (ring.f0 * M) - 1
        pmin = fmin / (ring.f0 * M) + 1
        p = np.arange(pmin, pmax + 1)

        fp = ring.f0 * (p*M + mu + m*tuneS)
        fp = fp[np.nonzero(fp)]  # Avoid division by 0
        num = np.sum(imp(fp) * h(fp) / (fp * 2 * np.pi))
        den = np.sum(h(fp))
        Zeff = num / den

    elif imp.component_type == "xdip" or imp.component_type == "ydip":
        if imp.component_type == "xdip":
            tuneXY = ring.tune[0] - np.floor(ring.tune[0])
            if xi is None:
                xi = ring.chro[0]
        elif imp.component_type == "ydip":
            tuneXY = ring.tune[1] - np.floor(ring.tune[1])
            if xi is None:
                xi = ring.chro[1]
        pmax = fmax / (ring.f0 * M) - 1
        pmin = fmin / (ring.f0 * M) + 1
        p = np.arange(pmin, pmax + 1)
        fp = ring.f0 * (p*M + mu + tuneXY + m*tuneS)
        f_xi = xi / ring.eta() * ring.f0
        num = np.sum(imp(fp) * h(fp - f_xi))
        den = np.sum(h(fp - f_xi))
        Zeff = num / den
    else:
        raise TypeError("Effective impedance is only defined for long, xdip"
                        " and ydip impedance type.")

    return Zeff


def head_tail_form_factor(ring,
                          imp,
                          m,
                          sigma,
                          tuneS,
                          xi=None,
                          mode="Hermite",
                          mu=0):
    M = 1
    if not isinstance(imp, Impedance):
        raise TypeError("{} should be an Impedance object.".format(imp))

    fmin = imp.data.index.min()
    fmax = imp.data.index.max()
    if fmin > 0:
        double_sided_impedance(imp)

    if mode in ["Hermite", "Legendre", "Sinusoidal", "Sacherer", "Chebyshev"]:

        def h(f):
            return spectral_density(frequency=f, sigma=sigma, m=m, mode=mode)
    else:
        raise NotImplementedError("Not implemanted yet.")

    pmax = np.floor(fmax / (ring.f0 * M))
    pmin = np.ceil(fmin / (ring.f0 * M))

    p = np.arange(pmin, pmax + 1)

    if imp.component_type == "long":
        fp = ring.f0 * (p*M + mu + m*tuneS)
        fp = fp[np.nonzero(fp)]  # Avoid division by 0
        den = np.sum(h(fp))

    elif imp.component_type == "xdip" or imp.component_type == "ydip":
        if imp.component_type == "xdip":
            tuneXY = ring.tune[0] - np.floor(ring.tune[0])
            if xi is None:
                xi = ring.chro[0]
        elif imp.component_type == "ydip":
            tuneXY = ring.tune[1] - np.floor(ring.tune[0])
            if xi is None:
                xi = ring.chro[1]
        fp = ring.f0 * (p*M + mu + tuneXY + m*tuneS)
        f_xi = xi / ring.eta() * ring.f0
        den = np.sum(h(fp - f_xi))
    else:
        raise TypeError("Effective impedance is only defined for long, xdip"
                        " and ydip impedance type.")

    return den


def tune_shift_from_effective_impedance(Zeff):
    pass


def yokoya_elliptic(x_radius, y_radius):
    """
    Compute Yokoya factors for an elliptic beam pipe.

    Parameters
    ----------
    x_radius : float
        Horizontal semi-axis of the ellipse in [m].
    y_radius : float
        Vertical semi-axis of the ellipse in [m].

    Returns
    -------
    yoklong : float
        Yokoya factor for the longitudinal impedance.
    yokxdip : float
        Yokoya factor for the dipolar horizontal impedance.
    yokydip : float
        Yokoya factor for the dipolar vertical impedance.
    yokxquad : float
        Yokoya factor for the quadrupolar horizontal impedance.
    yokyquad : float
        Yokoya factor for the quadrupolar vertical impedance.
    
    References
    ----------
    [1] : M. Migliorati, L. Palumbo, C. Zannini, N. Biancacci, and
    V. G. Vaccaro, "Resistive wall impedance in elliptical multilayer vacuum
    chambers." Phys. Rev. Accel. Beams 22, 121001 (2019).
    [2] : R.L. Gluckstern, J. van Zeijts, and B. Zotter, "Coupling impedance
    of beam pipes of general cross section." Phys. Rev. E 47, 656 (1993).

    """

    if (x_radius <= 0) or (y_radius <= 0):
        raise ValueError("Both radii must be non-zero positive values.")
    elif (x_radius == np.inf) and (y_radius == np.inf):
        raise ValueError("Both radii have infinite values.")
    elif x_radius == np.inf:
        yoklong = 1.0
        yokxdip = np.pi**2 / 24
        yokydip = np.pi**2 / 12
        yokxquad = -np.pi**2 / 24
        yokyquad = np.pi**2 / 24
    elif y_radius == np.inf:
        yoklong = 1.0
        yokxdip = np.pi**2 / 12
        yokydip = np.pi**2 / 24
        yokxquad = np.pi**2 / 24
        yokyquad = -np.pi**2 / 24
    else:
        if y_radius < x_radius:
            small_semiaxis = y_radius
            large_semiaxis = x_radius
        else:
            small_semiaxis = x_radius
            large_semiaxis = y_radius

        qr = (large_semiaxis-small_semiaxis) / (large_semiaxis+small_semiaxis)

        if qr == 0:
            yoklong = 1.0
            yokxdip = 1.0
            yokydip = 1.0
            yokxquad = 0.0
            yokyquad = 0.0

        else:
            # Define form factor functions
            def function_ff(small_semiaxis, F, mu_b, ip, il):
                coeff_fflong = 2 * np.sqrt(2) * small_semiaxis / (np.pi * F)
                coeff_fftrans = np.sqrt(2) * small_semiaxis**3 / (np.pi * F**3)

                fflong = (coeff_fflong * (-1)**ip / np.cosh(2 * ip * mu_b) *
                          (-1)**il / np.cosh(2 * il * mu_b))
                ffdx = (coeff_fftrans * (-1)**ip * (2*ip + 1) / np.cosh(
                    (2*ip + 1) * mu_b) * (-1)**il * (2*il + 1) / np.cosh(
                        (2*il + 1) * mu_b))
                ffdy = (coeff_fftrans * (-1)**ip * (2*ip + 1) / np.sinh(
                    (2*ip + 1) * mu_b) * (-1)**il * (2*il + 1) / np.sinh(
                        (2*il + 1) * mu_b))
                ffquad = (coeff_fftrans * (-1)**ip * (2 * ip)**2 /
                          np.cosh(2 * ip * mu_b) * (-1)**il /
                          np.cosh(2 * il * mu_b))

                return (fflong, ffdx, ffdy, ffquad)

            def function_L(mu_b, ip, il):
                common_L = (np.sqrt(2) * np.pi *
                            np.exp(-(2 * abs(ip - il) + 1) * mu_b) *
                            gamma(0.5 + abs(ip - il)) /
                            (gamma(0.5) * factorial(abs(ip - il))) *
                            hyp2f1(0.5,
                                   abs(ip - il) + 0.5,
                                   abs(ip - il) + 1, np.exp(-4 * mu_b)))
                val_m = (
                    np.sqrt(2) * np.pi * np.exp(-(2*ip + 2*il + 1) * mu_b) *
                    gamma(0.5 + ip + il) / (gamma(0.5) * factorial(ip + il)) *
                    hyp2f1(0.5, ip + il + 0.5, ip + il + 1, np.exp(-4 * mu_b)))
                val_d = (
                    np.sqrt(2) * np.pi * np.exp(-(2*ip + 2*il + 3) * mu_b) *
                    gamma(1.5 + ip + il) /
                    (gamma(0.5) * factorial(ip + il + 1)) *
                    hyp2f1(0.5, ip + il + 1.5, ip + il + 2, np.exp(-4 * mu_b)))

                Lm = common_L + val_m
                Ldx = common_L + val_d
                Ldy = common_L - val_d

                return (Lm, Ldx, Ldy)

            ip_range = np.arange(51)
            il_range = np.arange(51)
            ip, il = np.meshgrid(ip_range, il_range, indexing="ij")

            coeff_long = np.where((ip == 0) & (il == 0), 0.25,
                                  np.where((ip == 0) | (il == 0), 0.5, 1.0))
            coeff_quad = np.where(il == 0, 0.5, 1.0)

            # Our equations are approximately valid only for qr (ratio) values
            # less than or equal to 0.8.
            qr_th = 0.8

            if qr <= qr_th:
                F = np.sqrt(large_semiaxis**2 - small_semiaxis**2)
                mu_b = np.arccosh(large_semiaxis / F)

                ff_values = np.array(
                    function_ff(small_semiaxis, F, mu_b, ip, il))
                L_values = np.array(function_L(mu_b, ip, il))

                yoklong = np.sum(coeff_long * ff_values[0] * L_values[0])
                yokxdip = np.sum(ff_values[1] * L_values[1])
                yokydip = np.sum(ff_values[2] * L_values[2])
                yokxquad = -np.sum(coeff_quad * ff_values[3] * L_values[0])
                yokyquad = -yokxquad

                if y_radius > x_radius:
                    yokxdip, yokydip = yokydip, yokxdip
                    yokxquad, yokyquad = yokyquad, yokxquad

            # Beyond the threshold (qr > qr_th), they may not be valid,
            # but should converge to Yokoya form factors of parallel plates.
            # Fortunately, beyond this threshold, they should show asymptotic behavior,
            # so we perform linear interpolation.
            else:
                yoklong_pp = 1.0
                yokxdip_pp = np.pi**2 / 24
                yokydip_pp = np.pi**2 / 12
                yokxquad_pp = -np.pi**2 / 24
                yokyquad_pp = np.pi**2 / 24

                small_semiaxis_th = large_semiaxis * (1-qr_th) / (1+qr_th)
                F_th = np.sqrt(large_semiaxis**2 - small_semiaxis_th**2)
                mu_b_th = np.arccosh(large_semiaxis / F_th)

                ff_values_th = np.array(
                    function_ff(small_semiaxis_th, F_th, mu_b_th, ip, il))
                L_values_th = np.array(function_L(mu_b_th, ip, il))

                yoklong_th = np.sum(coeff_long * ff_values_th[0] *
                                    L_values_th[0])
                yokxdip_th = np.sum(ff_values_th[1] * L_values_th[1])
                yokydip_th = np.sum(ff_values_th[2] * L_values_th[2])
                yokxquad_th = -np.sum(
                    coeff_quad * ff_values_th[3] * L_values_th[0])
                yokyquad_th = -yokxquad_th

                if y_radius > x_radius:
                    yokxdip_th, yokydip_th = yokydip_th, yokxdip_th
                    yokxquad_th, yokyquad_th = yokyquad_th, yokxquad_th
                    yokxdip_pp, yokydip_pp = yokydip_pp, yokxdip_pp
                    yokxquad_pp, yokyquad_pp = yokyquad_pp, yokxquad_pp

                qr_array = np.array([qr_th, 1.0])
                yoklong_array = np.array([yoklong_th, yoklong_pp])
                yokxdip_array = np.array([yokxdip_th, yokxdip_pp])
                yokydip_array = np.array([yokydip_th, yokydip_pp])
                yokxquad_array = np.array([yokxquad_th, yokxquad_pp])
                yokyquad_array = np.array([yokyquad_th, yokyquad_pp])

                yoklong = np.interp(qr, qr_array, yoklong_array)
                yokxdip = np.interp(qr, qr_array, yokxdip_array)
                yokydip = np.interp(qr, qr_array, yokydip_array)
                yokxquad = np.interp(qr, qr_array, yokxquad_array)
                yokyquad = np.interp(qr, qr_array, yokyquad_array)

    return (yoklong, yokxdip, yokydip, yokxquad, yokyquad)


def beam_loss_factor(impedance, frequency, spectrum, ring):
    """
    Compute "beam" loss factor using the beam spectrum, uses a sum instead of 
    integral compared to loss_factor [1].

    Parameters
    ----------
    impedance : Impedance of type "long"
    frequency : array
        Sample points of spectrum.
    spectrum : array
        Beam spectrum to consider.
    ring : Synchrotron object

    Returns
    -------
    kloss_beam : float
        Beam loss factor in [V/C].

    References
    ----------
    [1] : Handbook of accelerator physics and engineering, 3rd printing. 
        Eq (3) p239.
    """
    pmax = np.floor(impedance.data.index.max() / ring.f0)
    pmin = np.floor(impedance.data.index.min() / ring.f0)

    if pmin >= 0:
        double_sided_impedance(impedance)
        pmin = -1 * pmax

    p = np.arange(pmin + 1, pmax)
    pf0 = p * ring.f0
    ReZ = np.real(impedance(pf0))
    spectral_density = np.abs(spectrum)**2
    # interpolation of the spectrum is needed to avoid problems liked to
    # division by 0
    # computing the spectrum directly to the frequency points gives
    # wrong results
    spect = interp1d(frequency, spectral_density)
    kloss_beam = ring.f0 * np.sum(ReZ * spect(pf0))

    return kloss_beam


def double_sided_impedance(impedance):
    """
    Add negative frequency points to single sided impedance spectrum following
    symetries depending on impedance type.

    Parameters
    ----------
    impedance : Impedance object
        Single sided impedance.
    """
    fmin = impedance.data.index.min()

    if fmin >= 0:
        negative_index = impedance.data.index * -1
        negative_data = impedance.data.set_index(negative_index)

        imp_type = impedance.component_type

        if imp_type == "long":
            negative_data["imag"] = -1 * negative_data["imag"]

        elif (imp_type == "xdip") or (imp_type == "ydip"):
            negative_data["real"] = -1 * negative_data["real"]

        elif (imp_type == "xquad") or (imp_type == "yquad"):
            negative_data["real"] = -1 * negative_data["real"]

        else:
            raise ValueError("Wrong impedance type")

        try:
            negative_data = negative_data.drop(0)
        except KeyError:
            pass

        all_data = pd.concat([impedance.data, negative_data])
        all_data = all_data.sort_index()
        impedance.data = all_data
