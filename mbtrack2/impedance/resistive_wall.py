# -*- coding: utf-8 -*-
"""
Define resistive wall elements based on the WakeField class.
"""

import numpy as np
from scipy.constants import c, epsilon_0, mu_0

from mbtrack2.impedance.wakefield import Impedance, WakeField, WakeFunction
from mbtrack2.tracking.emfields import _wofz


def skin_depth(frequency, rho, mu_r=1, epsilon_r=1):
    """
    General formula for the skin depth.

    Parameters
    ----------
    frequency : array of float
        Frequency points in [Hz].
    rho : float
        Resistivity in [ohm.m].
    mu_r : float, optional
        Relative magnetic permeability.
    epsilon_r : float, optional
        Relative electric permittivity.

    Returns
    -------
    delta : array of float
        Skin depth in [m].

    """

    delta = (np.sqrt(2 * rho / (np.abs(2 * np.pi * frequency) * mu_r * mu_0)) *
             np.sqrt(
                 np.sqrt(1 + (rho * np.abs(2 * np.pi * frequency) * epsilon_r *
                              epsilon_0)**2) +
                 rho * np.abs(2 * np.pi * frequency) * epsilon_r * epsilon_0))
    return delta


class CircularResistiveWall(WakeField):
    """
    Resistive wall WakeField element for a circular beam pipe.

    Impedance from approximated formulas from Eq. (2.77) of Chao book [1].
    Wake function formulas from [2, 3] and the fundamental theorem of
    beam loading from [4].

    Parameters
    ----------
    time : array of float
        Time points where the wake function will be evaluated in [s].
    frequency : array of float
        Frequency points where the impedance will be evaluated in [Hz].
    length : float
        Beam pipe length in [m].
    rho : float
        Resistivity in [ohm.m].
    radius : float
        Beam pipe radius in [m].
    exact : bool, optional
        If False, approxmiated formulas are used for the wake function
        computations.
        The default is True.

    References
    ----------
    [1] : Chao, A. W. (1993). Physics of collective beam instabilities in high
    energy accelerators. Wiley.
    [2] : Ivanyan, Mikayel I., and Vasili M. Tsakanov. "Analytical treatment of
    resistive wake potentials in round pipes." Nuclear Instruments and Methods
    in Physics Research Section A: Accelerators, Spectrometers, Detectors and
    Associated Equipment 522, no. 3 (2004): 223-229.
    [3] : Skripka, Galina, et al. "Simultaneous computation of intrabunch and
    interbunch collective beam motions in storage rings." Nuclear Instruments
    and Methods in Physics Research Section A: Accelerators, Spectrometers,
    Detectors and Associated Equipment 806 (2016): 221-230.
    [4] : Zotter, Bruno W., and Semyon A. Kheifets (1998). Impedances and wakes
    in high-energy particle accelerators. World Scientific.

    """

    def __init__(self, time, frequency, length, rho, radius, exact=True):
        super().__init__()

        self.length = length
        self.rho = rho
        self.radius = radius
        self.Z0 = mu_0 * c
        self.t0 = (2 * self.rho * self.radius**2 / self.Z0)**(1 / 3) / c

        omega = 2 * np.pi * frequency
        Z1 = length * (1 + np.sign(frequency) * 1j) * rho / (
            2 * np.pi * radius * skin_depth(frequency, rho))
        Z2 = c / omega * length * (1 + np.sign(frequency) * 1j) * rho / (
            np.pi * radius**3 * skin_depth(frequency, rho))

        Wl = self.LongitudinalWakeFunction(time, exact)
        Wt = self.TransverseWakeFunction(time, exact)

        Zlong = Impedance(variable=frequency,
                          function=Z1,
                          component_type='long')
        Zxdip = Impedance(variable=frequency,
                          function=Z2,
                          component_type='xdip')
        Zydip = Impedance(variable=frequency,
                          function=Z2,
                          component_type='ydip')
        Wlong = WakeFunction(variable=time, function=Wl, component_type="long")
        Wxdip = WakeFunction(variable=time, function=Wt, component_type="xdip")
        Wydip = WakeFunction(variable=time, function=Wt, component_type="ydip")

        super().append_to_model(Zlong)
        super().append_to_model(Zxdip)
        super().append_to_model(Zydip)
        super().append_to_model(Wlong)
        super().append_to_model(Wxdip)
        super().append_to_model(Wydip)

    def LongitudinalWakeFunction(self, time, exact=True):
        """
        Compute the longitudinal wake function of a circular resistive wall
        using Eq. (11), of [1], or approxmiated expression Eq. (24), of [2].
        The approxmiated expression is valid if the time is large compared
        to the characteristic time t0.

        Eq. (11) in [1] is completely identical to Eq. (22) in [2].

        The real parts of the last two terms of Eq. (11) in [1] are the same,
        and the imaginary parts have the same magnitude but opposite signs.
        Therefore, the former term was doubled, the latter term was eliminated,
        and only the real part was taken to speed up the calculation.

        The fundamental theorem of beam loading [3] is applied for the exact
        expression of the longitudinal wake function: Wl(0) = Wl(0+)/2.

        Parameters
        ----------
        time : array of float
            Time points where the wake function is evaluated in [s].
        exact : bool, optional
            If True, the exact expression is used. The default is True.

        Returns
        -------
        wl : array of float
            Longitudinal wake function in [V/C].

        References
        ----------
        [1] : Ivanyan, Mikayel I., and Vasili M. Tsakanov. "Analytical treatment of
        resistive wake potentials in round pipes." Nuclear Instruments and Methods
        in Physics Research Section A: Accelerators, Spectrometers, Detectors and
        Associated Equipment 522, no. 3 (2004): 223-229.
        [2] : Skripka, Galina, et al. "Simultaneous computation of intrabunch and
        interbunch collective beam motions in storage rings." Nuclear Instruments
        and Methods in Physics Research Section A: Accelerators, Spectrometers,
        Detectors and Associated Equipment 806 (2016): 221-230.
        [3] : Zotter, Bruno W., and Semyon A. Kheifets (1998). Impedances and wakes
        in high-energy particle accelerators. World Scientific.
        """
        wl = np.zeros_like(time)
        if exact == True:
            idx1 = time > 0
            idx2 = time == 0
            factor = (self.Z0 * c / (3 * np.pi * self.radius**2) * self.length)
            wl[idx1] = self.__LongWakeExact(time[idx1], factor)

            # fundamental theorem of beam loading
            wl[idx2] = 3 * factor / 2
        else:
            idx = time >= 0
            wl[idx] = self.__LongWakeApprox(time[idx])
        return wl

    def TransverseWakeFunction(self, time, exact=True):
        """
        Compute the transverse wake function of a circular resistive wall
        using Eq. (11), of [1], or approxmiated expression Eq. (26), of [2].
        The approxmiated expression is valid if the time is large compared
        to the characteristic time t0.

        Eq. (11) in [1] is completely identical to Eq. (25) in [2].

        There are typos in both Eq. (11) in [1] and Eq. (25) in [2].
        Corrected the typos in the last two terms of exact expression for
        transverse wake function in Eq. (11), of [1].
        It is necessary to multiply Eq. (25) in [2] by c*t0.

        The real parts of the last two terms of Eq. (11) in [1] are the same,
        and the imaginary parts have the same magnitude but opposite signs.
        Therefore, the former term was doubled, the latter term was eliminated,
        and only the real part was taken to speed up the calculation.

        Parameters
        ----------
        time : array of float
            Time points where the wake function is evaluated in [s].
        exact : bool, optional
            If True, the exact expression is used. The default is True.

        Returns
        -------
        wt : array of float
            Transverse wake function in [V/C/m].

        References
        ----------
        [1] : Ivanyan, Mikayel I., and Vasili M. Tsakanov. "Analytical treatment of
        resistive wake potentials in round pipes." Nuclear Instruments and Methods
        in Physics Research Section A: Accelerators, Spectrometers, Detectors and
        Associated Equipment 522, no. 3 (2004): 223-229.
        [2] : Skripka, Galina, et al. "Simultaneous computation of intrabunch and
        interbunch collective beam motions in storage rings." Nuclear Instruments
        and Methods in Physics Research Section A: Accelerators, Spectrometers,
        Detectors and Associated Equipment 806 (2016): 221-230.
        """
        wt = np.zeros_like(time)
        if exact == True:
            idx = time > 0
            factor = ((self.Z0 * c**2 * self.t0) /
                      (3 * np.pi * self.radius**4) * self.length)
            wt[idx] = self.__TransWakeExact(time[idx], factor)
        else:
            idx = time >= 0
            wt[idx] = self.__TransWakeApprox(time[idx])
        return wt

    def __LongWakeExact(self, t, factor):
        w1re, _ = _wofz(0, np.sqrt(2 * t / self.t0))
        w2re, _ = _wofz(
            np.cos(np.pi / 6) * np.sqrt(2 * t / self.t0),
            np.sin(np.pi / 6) * np.sqrt(2 * t / self.t0))

        wl = factor * (4 * np.exp(-1 * t / self.t0) *
                       np.cos(np.sqrt(3) * t / self.t0) + w1re - 2*w2re)
        return wl

    def __TransWakeExact(self, t, factor):
        w1re, _ = _wofz(0, np.sqrt(2 * t / self.t0))
        w2re, w2im = _wofz(
            np.cos(np.pi / 6) * np.sqrt(2 * t / self.t0),
            np.sin(np.pi / 6) * np.sqrt(2 * t / self.t0))

        wt = factor * (2 * np.exp(-1 * t / self.t0) *
                       (np.sqrt(3) * np.sin(np.sqrt(3) * t / self.t0) -
                        np.cos(np.sqrt(3) * t / self.t0)) + w1re + 2 *
                       (np.cos(-np.pi / 3) * w2re - np.sin(-np.pi / 3) * w2im))
        return wt

    def __LongWakeApprox(self, t):
        wl = -1 * (1 / (4 * np.pi * self.radius) *
                   np.sqrt(self.Z0 * self.rho /
                           (c * np.pi * t**3)) * self.length)
        return wl

    def __TransWakeApprox(self, t):
        wt = (1 / (np.pi * self.radius**3) *
              np.sqrt(self.Z0 * c * self.rho / (np.pi * t)) * self.length)
        return wt


class Coating(WakeField):

    def __init__(self,
                 frequency,
                 length,
                 rho1,
                 rho2,
                 radius,
                 thickness,
                 approx=False):
        """
        WakeField element for a coated circular beam pipe.

        The longitudinal and tranverse impedances are computed using formulas
        from [1].

        Parameters
        ----------
        f : array of float
            Frequency points where the impedance is evaluated in [Hz].
        length : float
            Length of the beam pipe to consider in [m].
        rho1 : float
            Resistivity of the coating in [ohm.m].
        rho2 : float
            Resistivity of the bulk material in [ohm.m].
        radius : float
            Radius of the beam pipe to consier in [m].
        thickness : float
            Thickness of the coating in [m].
        approx : bool, optional
            If True, used approxmiated formula. The default is False.

        References
        ----------
        [1] : Migliorati, M., E. Belli, and M. Zobov. "Impact of the resistive
        wall impedance on beam dynamics in the Future Circular e+ e− Collider."
        Physical Review Accelerators and Beams 21.4 (2018): 041001.

        """
        super().__init__()

        self.length = length
        self.rho1 = rho1
        self.rho2 = rho2
        self.radius = radius
        self.thickness = thickness

        Zl = self.LongitudinalImpedance(frequency, approx)
        Zt = self.TransverseImpedance(frequency, approx)

        Zlong = Impedance(variable=frequency,
                          function=Zl,
                          component_type='long')
        Zxdip = Impedance(variable=frequency,
                          function=Zt,
                          component_type='xdip')
        Zydip = Impedance(variable=frequency,
                          function=Zt,
                          component_type='ydip')

        super().append_to_model(Zlong)
        super().append_to_model(Zxdip)
        super().append_to_model(Zydip)

    def LongitudinalImpedance(self, f, approx):
        """
        Compute the longitudinal impedance of a coating using Eq. (5), or
        approxmiated expression Eq. (8), of [1]. The approxmiated expression
        is valid if the skin depth of the coating is large compared to the
        coating thickness.

        Parameters
        ----------
        f : array of float
            Frequency points where the impedance is evaluated in [Hz].
        approx : bool
            If True, used approxmiated formula.

        Returns
        -------
        Zl : array
            Longitudinal impedance values in [ohm].

        References
        ----------
        [1] : Migliorati, M., E. Belli, and M. Zobov. "Impact of the resistive
        wall impedance on beam dynamics in the Future Circular e+ e− Collider."
        Physical Review Accelerators and Beams 21.4 (2018): 041001.

        """

        Z0 = mu_0 * c
        factor = Z0 * f / (2 * c * self.radius) * self.length
        skin1 = skin_depth(f, self.rho1)
        skin2 = skin_depth(f, self.rho2)

        if approx == False:
            alpha = skin1 / skin2
            tanh = np.tanh((1 + 1j * np.sign(f)) * self.thickness / skin1)
            bracket = ((np.sign(f) + 1j) * skin1 * (alpha*tanh + 1) /
                       (alpha+tanh))
        else:
            valid_approx = self.thickness / np.min(skin1)
            if valid_approx < 0.01:
                print(
                    "Approximation is not valid. Returning impedance anyway.")
            bracket = ((np.sign(f) + 1j) * skin2 + 2 * 1j * self.thickness *
                       (1 - self.rho2 / self.rho1))

        Zl = factor * bracket

        return Zl

    def TransverseImpedance(self, f, approx):
        """
        Compute the transverse impedance of a coating using Eq. (6), or
        approxmiated expression Eq. (9), of [1]. The approxmiated expression
        is valid if the skin depth of the coating is large compared to the
        coating thickness.

        Parameters
        ----------
        f : array of float
            Frequency points where the impedance is evaluated in [Hz].
        approx : bool
            If True, used approxmiated formula.

        Returns
        -------
        Zt : array
            Transverse impedance values in [ohm].

        References
        ----------
        [1] : Migliorati, M., E. Belli, and M. Zobov. "Impact of the resistive
        wall impedance on beam dynamics in the Future Circular e+ e− Collider."
        Physical Review Accelerators and Beams 21.4 (2018): 041001.

        """

        Z0 = mu_0 * c
        factor = Z0 / (2 * np.pi * self.radius**3) * self.length
        skin1 = skin_depth(f, self.rho1)
        skin2 = skin_depth(f, self.rho2)

        if approx == False:
            alpha = skin1 / skin2
            tanh = np.tanh((1 + 1j * np.sign(f)) * self.thickness / skin1)
            bracket = ((1 + 1j * np.sign(f)) * skin1 * (alpha*tanh + 1) /
                       (alpha+tanh))
        else:
            valid_approx = self.thickness / np.min(skin1)
            if valid_approx < 0.01:
                print(
                    "Approximation is not valid. Returning impedance anyway.")
            bracket = ((1 + 1j * np.sign(f)) * skin2 +
                       2 * 1j * self.thickness * np.sign(f) *
                       (1 - self.rho2 / self.rho1))

        Zt = factor * bracket

        return Zt
