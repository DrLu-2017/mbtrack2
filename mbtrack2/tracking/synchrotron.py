# -*- coding: utf-8 -*-
"""
Module where the Synchrotron class is defined.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c, e


class Synchrotron:
    """
    Synchrotron class to store main properties.

    Optional parameters are optional only if the Optics object passed to the
    class uses a loaded lattice.

    Parameters
    ----------
    h : int
        Harmonic number of the accelerator.
    optics : Optics object
        Object where the optic functions are stored.
    particle : Particle object
        Particle which is accelerated.
    tau : array of shape (3,)
        Horizontal, vertical and longitudinal damping times in [s].
    sigma_delta : float
        Equilibrium energy spread.
    sigma_0 : float
        Natural bunch length in [s].
    emit : array of shape (2,)
        Horizontal and vertical equilibrium emittance in [m.rad].
    L : float, optional
        Ring circumference in [m].
    E0 : float, optional
        Nominal (total) energy of the ring in [eV].
    ac : float, optional
        Momentum compaction factor.
    tune : array of shape (2,), optional
        Horizontal and vertical tunes.
    chro : array of shape (2,), optional
        Horizontal and vertical (non-normalized) chromaticities.
    U0 : float, optional
        Energy loss per turn in [eV].
    adts : list of arrays or None, optional
        List that contains arrays of polynomial's coefficients, in decreasing
        powers, used to determine Amplitude-Dependent Tune Shifts (ADTS).
        The order of the elements strictly needs to be
        [coef_xx, coef_yx, coef_xy, coef_yy], where x and y denote the horizontal
        and the vertical plane, respectively, and coef_PQ means the polynomial's
        coefficients of the ADTS in plane P due to the amplitude in plane Q.

        For example, if the tune shift in y due to the Courant-Snyder invariant
        Jx is characterized by the equation dQy(x) = 3*Jx**2 + 2*Jx + 1, then
        coef_yx takes the form np.array([3, 2, 1]).

        Use None, to exclude the ADTS from calculation.
    mcf_order : array, optional
        Higher-orders momentum compaction factor in decreasing powers.
        Input here the coefficients considering only the derivatives and which
        do not include any factorial factor.
        The 1st order should be included in the array and be at the last
        position.

    Attributes
    ----------
    T0 : float
        Revolution time in [s].
    f0 : float
        Revolution frequency in [Hz].
    omega0 : float
        Angular revolution frequency in [Hz.rad]
    T1 : flaot
        Fundamental RF period in [s].
    f1 : float
        Fundamental RF frequency in [Hz].
    omega1 : float
        Fundamental RF angular frequency in [Hz.rad].
    k1 : float
        Fundamental RF wave number in [m**-1].
    gamma : float
        Relativistic Lorentz gamma.
    beta : float
        Relativistic Lorentz beta.
    long_alpha : float
        Longitudinal alpha Twiss parameter at the tracking location.
        Initialized at zero.
    long_beta : float
        Longitudinal beta Twiss parameter at the tracking location in [s].
        Initialized at zero.
    long_gamma : float
        Longitudinal gamma Twiss parameter at the tracking location in [s-1].
        Initialized at zero.


    Methods
    -------
    synchrotron_tune(V)
        Compute the (unperturbed) synchrotron tune from main RF voltage.
    sigma(position)
        Return the RMS beam size at equilibrium in [m].
    eta(delta)
        Momentum compaction taking into account higher orders if provided in
        mcf_order.
    mcf(delta)
        Momentum compaction factor taking into account higher orders if
        provided in mcf_order.
    get_adts()
        Compute and add Amplitude-Dependent Tune Shifts (ADTS) sextupolar
        componenet from AT lattice.
    get_chroma()
        Compute chromaticity (linear and nonlinear) from AT lattice and
        update the property.
    get_mcf_order()
        Compute momentum compaction factor up to 3rd order from AT lattice.
    get_longitudinal_twiss(V)
        Compute the longitudinal Twiss parameters and the synchrotron tune for
        single or multi-harmonic RF systems.
    to_pyat(Vrf)
        Return a pyAT simple_ring element from the Synchrotron element data.

    """

    def __init__(self, h, optics, particle, **kwargs):
        self._h = h
        self.particle = particle
        self.optics = optics
        self.long_alpha = 0
        self.long_beta = 0
        self.long_gamma = 0

        if not self.optics.use_local_values:
            self.L = kwargs.get("L", self.optics.lattice.circumference)
            self.E0 = kwargs.get("E0", self.optics.lattice.energy)
            self.ac = kwargs.get("ac", self.optics.ac)
            self.tune = kwargs.get("tune", self.optics.tune)
            self.chro = kwargs.get("chro", self.optics.chro)
            self.U0 = kwargs.get("U0", self.optics.lattice.energy_loss)
        else:
            self.L = kwargs.get("L")  # Ring circumference [m]
            # Nominal (total) energy of the ring [eV]
            self.E0 = kwargs.get("E0")
            self.ac = kwargs.get("ac")  # Momentum compaction factor
            self.tune = kwargs.get("tune")  # X/Y/S tunes
            # X/Y (non-normalized) chromaticities
            self.chro = kwargs.get("chro")
            self.U0 = kwargs.get("U0")  # Energy loss per turn [eV]

        self.tau = kwargs.get("tau")  # X/Y/S damping times [s]
        # Equilibrium energy spread
        self.sigma_delta = kwargs.get("sigma_delta")
        self.sigma_0 = kwargs.get("sigma_0")  # Natural bunch length [s]
        self.emit = kwargs.get("emit")  # X/Y emittances in [m.rad]
        self.adts = kwargs.get("adts")  # Amplitude-Dependent Tune Shift (ADTS)
        self.mcf_order = kwargs.get(
            "mcf_order", self.ac)  # Higher-orders momentum compaction factor

    @property
    def h(self):
        """Harmonic number"""
        return self._h

    @h.setter
    def h(self, value):
        self._h = value
        self.L = self.L  # call setter

    @property
    def L(self):
        """Ring circumference [m]"""
        return self._L

    @L.setter
    def L(self, value):
        self._L = value
        self._T0 = self.L / c
        self._T1 = self.T0 / self.h
        self._f0 = 1 / self.T0
        self._omega0 = 2 * np.pi * self.f0
        self._f1 = self.h * self.f0
        self._omega1 = 2 * np.pi * self.f1
        self._k1 = self.omega1 / c

    @property
    def T0(self):
        """Revolution time [s]"""
        return self._T0

    @T0.setter
    def T0(self, value):
        self.L = c * value

    @property
    def T1(self):
        """ "Fundamental RF period [s]"""
        return self._T1

    @T1.setter
    def T1(self, value):
        self.L = c * value * self.h

    @property
    def f0(self):
        """Revolution frequency [Hz]"""
        return self._f0

    @f0.setter
    def f0(self, value):
        self.L = c / value

    @property
    def omega0(self):
        """Angular revolution frequency [Hz rad]"""
        return self._omega0

    @omega0.setter
    def omega0(self, value):
        self.L = 2 * np.pi * c / value

    @property
    def f1(self):
        """Fundamental RF frequency [Hz]"""
        return self._f1

    @f1.setter
    def f1(self, value):
        self.L = self.h * c / value

    @property
    def omega1(self):
        """Fundamental RF angular frequency[Hz rad]"""
        return self._omega1

    @omega1.setter
    def omega1(self, value):
        self.L = 2 * np.pi * self.h * c / value

    @property
    def k1(self):
        """Fundamental RF wave number [m**-1]"""
        return self._k1

    @k1.setter
    def k1(self, value):
        self.L = 2 * np.pi * self.h / value

    @property
    def gamma(self):
        """Relativistic gamma"""
        return self._gamma

    @gamma.setter
    def gamma(self, value):
        self._gamma = value
        self._beta = np.sqrt(1 - self.gamma**-2)
        self._E0 = self.gamma * self.particle.mass * c**2 / e

    @property
    def beta(self):
        """Relativistic beta"""
        return self._beta

    @beta.setter
    def beta(self, value):
        self.gamma = 1 / np.sqrt(1 - value**2)

    @property
    def E0(self):
        """Nominal (total) energy of the ring [eV]"""
        return self._E0

    @E0.setter
    def E0(self, value):
        self.gamma = value / (self.particle.mass * c**2 / e)

    @property
    def mcf_order(self):
        """Higher-orders momentum compaction factor"""
        return self._mcf_order

    @mcf_order.setter
    def mcf_order(self, value):
        self._mcf_order = value
        self.mcf = np.poly1d(self.mcf_order)

    def eta(self, delta=0):
        """
        Momentum compaction taking into account higher orders if provided in
        mcf_order.

        Parameters
        ----------
        delta : float or array like bunch["delta"], optional
            Relative energy deviation. The default is 0.

        Returns
        -------
        float or array
            Momentum compaction.

        """
        return self.mcf(delta) - 1 / (self.gamma**2)

    def sigma(self, position=None):
        """
        Return the RMS beam size at equilibrium in [m].

        Parameters
        ----------
        position : float or array, optional
            Longitudinal position in [m] where the beam size is computed.
            If None, the local values are used.

        Returns
        -------
        sigma : array
            RMS beam size in [m] at position location or at local positon if
            position is None.

        """
        if position is None:
            sigma = np.zeros((4, ))
            sigma[0] = (
                self.emit[0] * self.optics.local_beta[0] +
                self.optics.local_dispersion[0]**2 * self.sigma_delta**2)**0.5
            sigma[1] = (
                self.emit[0] * self.optics.local_gamma[0] +
                self.optics.local_dispersion[1]**2 * self.sigma_delta**2)**0.5
            sigma[2] = (
                self.emit[1] * self.optics.local_beta[1] +
                self.optics.local_dispersion[2]**2 * self.sigma_delta**2)**0.5
            sigma[3] = (
                self.emit[1] * self.optics.local_gamma[1] +
                self.optics.local_dispersion[3]**2 * self.sigma_delta**2)**0.5
        else:
            if isinstance(position, (float, int)):
                n = 1
            else:
                n = len(position)
            sigma = np.zeros((4, n))
            sigma[0, :] = (self.emit[0] * self.optics.beta(position)[0] +
                           self.optics.dispersion(position)[0]**2 *
                           self.sigma_delta**2)**0.5
            sigma[1, :] = (self.emit[0] * self.optics.gamma(position)[0] +
                           self.optics.dispersion(position)[1]**2 *
                           self.sigma_delta**2)**0.5
            sigma[2, :] = (self.emit[1] * self.optics.beta(position)[1] +
                           self.optics.dispersion(position)[2]**2 *
                           self.sigma_delta**2)**0.5
            sigma[3, :] = (self.emit[1] * self.optics.gamma(position)[1] +
                           self.optics.dispersion(position)[3]**2 *
                           self.sigma_delta**2)**0.5
        return sigma

    def synchrotron_tune(self, V):
        """
        Compute the (unperturbed) synchrotron tune from main RF voltage.

        Parameters
        ----------
        V : float
            Main RF voltage in [V].

        Returns
        -------
        tuneS : float
            Synchrotron tune.

        """
        Vsum = V * np.sin(np.arccos(self.U0 / V))
        phi = np.arccos(1 - self.eta(0) * np.pi * self.h / self.E0 * Vsum)
        tuneS = phi / (2 * np.pi)
        return tuneS

    def get_adts(self,
                 xm=1e-4,
                 ym=1e-4,
                 npoints=9,
                 plot=False,
                 ax=None,
                 **kwargs):
        """
        Compute and add Amplitude-Dependent Tune Shifts (ADTS) sextupolar
        componenet from AT lattice.

        Parameters
        ----------
        xm : float, optional
            Maximum horizontal amplitude in [m].
            Default is 1e-4.
        ym : float, optional
            Maximum vertical amplitude in [m].
            Default is 1e-4.
        npoints : int, optional
            Number of points in each plane.
            Default is 9.
        plot : bool, optional
            If True, plot the computed tune shift with amplitude.
            Default is False.
        ax : array of shape (2,) of matplotlib.axes.Axes, optional
            Axes where to plot the figures.
            Default is None.

        Returns
        -------
        ax : array of shape (2,) of matplotlib.axes.Axes
            Only if plot is True.

        See at.physics.nonlinear.detuning for **kwargs
        """
        import at

        if self.optics.use_local_values:
            raise ValueError("ADTS needs to be provided manualy as no AT" +
                             " lattice file is loaded.")
        r0, r1, x, q_dx, y, q_dy = at.physics.nonlinear.detuning(
            self.optics.lattice.radiation_off(copy=True),
            npoints=npoints,
            xm=xm,
            ym=ym,
            **kwargs)
        coef_xx = np.array([r1[0][0] / 2, 0])
        coef_yx = np.array([r1[0][1] / 2, 0])
        coef_xy = np.array([r1[0][1] / 2, 0])
        coef_yy = np.array([r1[1][1] / 2, 0])
        self.adts = [coef_xx, coef_yx, coef_xy, coef_yy]

        if plot:
            if ax is None:
                fig, ax = plt.subplots(2, 1)

            ax[0].scatter(x * 1e6, q_dx[:, 0])
            ax[0].scatter(y * 1e6, q_dy[:, 0])
            ax[0].set_ylabel("Delta Tune X")
            ax[0].set_xlabel("Amplitude [um]")
            ax[0].legend(["dqx/dx", "dqx/dy"])
            ax[0].grid()

            ax[1].scatter(x * 1e6, q_dx[:, 1])
            ax[1].scatter(y * 1e6, q_dy[:, 1])
            ax[1].set_ylabel("Delta Tune Y")
            ax[1].set_xlabel("Amplitude [um]")
            ax[1].legend(["dqy/dx", "dqy/dy"])
            ax[1].grid()

            return ax

    def get_chroma(self, order=4, dpm=0.02, n_points=100):
        """
        Compute chromaticity (linear and nonlinear) from AT lattice and update the property.
        """
        import at

        if self.optics.use_local_values:
            raise ValueError("Value needs to be provided manualy as no AT" +
                             " lattice file is loaded.")
        fit, dpa, tune = at.physics.nonlinear.chromaticity(
            self.optics.lattice,
            method="linopt",
            dpm=dpm,
            n_points=n_points,
            order=order,
        )
        chrox, chroy = fit
        self.chro = [
            elem for pair in zip(chrox[1:], chroy[1:]) for elem in pair
        ]
        return chrox[1:], chroy[1:]

    def get_mcf_order(self, add=True, show_fit=False):
        """
        Compute momentum compaction factor up to 3rd order from AT lattice.

        Parameters
        ----------
        add : bool, optional
            If True, add the computed coefficient to the Synchrotron object.
            If False, the result is returned.
            The default is True.
        show_fit : bool, optional
            If True, show a plot with the data and fit.
            The default is False.

        Return
        ------
        pvalue : array, optional
            Momentum compaction factor up to 3rd order
        """
        import at

        if self.optics.use_local_values:
            raise ValueError("Value needs to be provided manualy as no AT" +
                             " lattice file is loaded.")
        deltamin = -1e-4
        deltamax = 1e-4

        delta = np.linspace(deltamin, deltamax, 7)
        delta4eval = np.linspace(deltamin, deltamax, 21)
        alpha = np.zeros_like(delta)

        for i in range(len(delta)):
            alpha[i] = at.physics.revolution.get_mcf(self.optics.lattice,
                                                     delta[i])
        pvalue = np.polyfit(delta, alpha, 2)

        if show_fit:
            pvalue = np.polyfit(delta, alpha, 2)
            palpha = np.polyval(pvalue, delta4eval)

            plt.plot(delta * 100, alpha, "k.")
            plt.grid()
            plt.plot(delta4eval * 100, palpha, "r")
            plt.xlabel("Energy (%)")
            plt.ylabel("Momemtum compaction factor")
            plt.legend(["Data", "polyfit"])

        if add:
            self.mcf_order = pvalue
        else:
            return pvalue

    def get_longitudinal_twiss(self, V, phase=None, harmonics=None, add=True):
        """
        Compute the longitudinal Twiss parameters and the synchrotron tune for
        single or multi-harmonic RF systems.

        Hypthesis:
        - Use the linear approximation (tau ~ 0).
        - For higher haromics only, cos(phi) ~ 0 must be true.

        Parameters
        ----------
        V : float or array-like of float
            RF voltage in [V].
        phase : float or array-like of float, optional
            RF phase using cosine convention in [rad].
            Default is None.
        harmonics : int or array-like of int, optional
            Harmonics of the cavities to consider.
            Default is None.
        add : bool, optional
            If True, add the computed longitudinal Twiss parameters as class
            arguements.

        Usage
        -----
        - If a single voltage value is given, assume a single RF system set at
        the synchronous phase.
        - If a single pair of voltage/phase value is given, assume a single RF
        system at this setting.
        - Otherwise, compute the synchrotron tune for a multi-harmonic RF
        system.

        Returns
        -------
        tuneS : float
            Synchrotron tune in the linear approximation.
        long_alpha : float
            Longitudinal alpha Twiss parameter at the tracking location.
        long_beta : float
            Longitudinal beta Twiss parameter at the tracking location in [s].
        long_gamma : float
            Longitudinal gamma Twiss parameter at the tracking location in [s-1].
        """

        if isinstance(V, float) or isinstance(V, int):
            V = [V]
            if phase is None:
                phase = [np.arccos(self.U0 / V[0])]
            elif isinstance(phase, float) or isinstance(phase, int):
                phase = [phase]
            if harmonics is None:
                harmonics = [1]

        if not (len(V) == len(phase) == len(harmonics)):
            raise ValueError("You must provide array of the same length for"
                             " V, phase and harmonics")

        Vsum = 0
        for i in range(len(V)):
            Vsum += harmonics[i] * V[i] * np.sin(phase[i])
        phi = np.arccos(1 - self.eta(0) * np.pi * self.h / self.E0 * Vsum)
        long_alpha = -self.eta(0) * np.pi * self.h / (self.E0 *
                                                      np.sin(phi)) * Vsum
        long_beta = self.eta(0) * self.T0 / np.sin(phi)
        long_gamma = self.omega1 * Vsum / (self.E0 * np.sin(phi))
        tuneS = phi / (2 * np.pi)
        if add:
            self.tuneS = tuneS
            self.long_alpha = long_alpha
            self.long_beta = long_beta
            self.long_gamma = long_gamma
        else:
            return tuneS, long_alpha, long_beta, long_gamma

    def to_pyat(self, Vrf, harmonic_number=None, TimeLag=False):
        """
        Return a pyAT simple_ring element from the Synchrotron element data.

        See pyAT documentation for informations about simple_ring.

        Parameters
        ----------
        Vrf : float or array-like of float
            RF Voltage in [V]. If sevral cavities are provided, harmonic_number
            and TimeLag should be provided.
        harmonic_number : float or array-like of float, optional
            Harmonic numbers of the RF cavities. The default is None.
        TimeLag : float or array-like of float, optional
            Set the timelag of the cavities in pyAT definition.
            The default is False.

        Returns
        -------
        at_simple_ring : at.physics.fastring.simple_ring
            A pyAT simple_ring element.

        """
        from at import simple_ring

        optics = self.optics

        if (harmonic_number is None) and isinstance(Vrf, (float, int)):
            harmonic_number = self.h

        if isinstance(Vrf, (list, np.ndarray)):
            if (harmonic_number is None) or (TimeLag is None):
                raise ValueError(
                    "If sevral cavities are provided, "
                    "harmonic_number and TimeLag should be provided.")

        if self.adts is not None:
            try:
                A1 = self.adts[0][-2] * 2
                A2 = self.adts[1][-2] * 2
                A3 = self.adts[3][-2] * 2
            except IndexError:
                A1 = None
                A2 = None
                A3 = None
        else:
            A1 = None
            A2 = None
            A3 = None

        at_simple_ring = simple_ring(
            energy=self.E0,
            circumference=self.L,
            harmonic_number=harmonic_number,
            Qx=self.tune[0],
            Qy=self.tune[1],
            Vrf=Vrf,
            alpha=self.ac,
            betax=optics.local_beta[0],
            betay=optics.local_beta[1],
            alphax=optics.local_alpha[0],
            alphay=optics.local_alpha[1],
            Qpx=self.chro[0],
            Qpy=self.chro[1],
            A1=A1,
            A2=A2,
            A3=A3,
            emitx=self.emit[0],
            emity=self.emit[1],
            espread=self.sigma_delta,
            taux=self.tau[0] / self.T0,
            tauy=self.tau[1] / self.T0,
            tauz=self.tau[2] / self.T0,
            U0=self.U0,
            TimeLag=TimeLag,
        )
        return at_simple_ring
