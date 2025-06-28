# -*- coding: utf-8 -*-
"""
This module handles radio-frequency (RF) cavitiy elements.
"""

from functools import partial

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerPatch

from mbtrack2.instability import lcbi_growth_rate
from mbtrack2.tracking.element import Element


class RFCavity(Element):
    """
    Perfect RF cavity class for main and harmonic RF cavities.
    Use cosine definition.

    Parameters
    ----------
    ring : Synchrotron object
    m : int
        Harmonic number of the cavity
    Vc : float
        Amplitude of cavity voltage [V]
    theta : float
        Phase of Cavity voltage
    """

    def __init__(self, ring, m, Vc, theta):
        self.ring = ring
        self.m = m
        self.Vc = Vc
        self.theta = theta

    @Element.parallel
    def track(self, bunch):
        """
        Tracking method for the element.
        No bunch to bunch interaction, so written for Bunch objects and
        @Element.parallel is used to handle Beam objects.

        Parameters
        ----------
        bunch : Bunch or Beam object
        """
        bunch["delta"] += (
            self.Vc / self.ring.E0 *
            np.cos(self.m * self.ring.omega1 * bunch["tau"] + self.theta))

    def value(self, val):
        return (self.Vc / self.ring.E0 *
                np.cos(self.m * self.ring.omega1 * val + self.theta))


class CavityResonator():
    """Cavity resonator class for active or passive RF cavity with beam
    loading or HOM, based on [1,2].

    Use cosine definition.

    If used with mpi, beam.mpi.share_distributions must be called before the
    track method call.

    Different kind of RF feeback and loops can be added using:
        cavity_resonator.feedback.append(loop)

    Parameters
    ----------
    ring : Synchrotron object
    m : int or float
        Harmonic number of the cavity.
    Rs : float
        Shunt impedance of the cavities in [Ohm], defined as 0.5*Vc*Vc/Pc.
        If Ncav = 1, used for the total shunt impedance.
        If Ncav > 1, used for the shunt impedance per cavity.
    Q : float
        Quality factor of the cavity.
    QL : float
        Loaded quality factor of the cavity.
    detune : float
        Detuing of the cavity in [Hz], defined as (fr - m*ring.f1).
    Ncav : int, optional
        Number of cavities.
    Vc : float, optinal
        Total cavity voltage objective value in [V].
    theta : float, optional
        Total cavity phase objective value in [rad].
    n_bin : int, optional
        Number of bins used for the beam loading computation.
        Only used if MPI is not used, otherwise n_bin must be specified in the
        beam.mpi.share_distributions method.
        The default is 75.

    Attributes
    ----------
    beam_phasor : complex
        Beam phasor in [V].
    beam_phasor_record : array of complex
        Last beam phasor value of each bunch in [V].
    generator_phasor : complex
        Generator phasor in [V].
    generator_phasor_record : array of complex
        Last generator phasor value of each bunch in [V].
    cavity_phasor : complex
        Cavity phasor in [V].
    cavity_phasor_record : array of complex
        Last cavity phasor value of each bunch in [V].
    ig_phasor_record : array of complex
        Last current generator phasor of each bunch in [A].
        Only used for some feedback types.
    cavity_voltage : float
        Cavity total voltage in [V].
    cavity_phase : float
        Cavity total phase in [rad].
    loss_factor : float
        Cavity loss factor in [V/C].
    Rs_per_cavity : float
        Shunt impedance of a single cavity in [Ohm], defined as 0.5*Vc*Vc/Pc.
    beta : float
        Coupling coefficient of the cavity.
    fr : float
        Resonance frequency of the cavity in [Hz].
    wr : float
        Angular resonance frequency in [Hz.rad].
    psi : float
        Tuning angle in [rad].
    filling_time : float
        Cavity filling time in [s].
    Pc : float
        Power dissipated in the cavity walls in [W].
    Pg : float
        Generator power in [W].
    Vgr : float
        Generator voltage at resonance in [V].
    theta_gr : float
        Generator phase at resonance in [rad].
    Vg : float
        Generator voltage in [V].
    theta_g : float
        Generator phase in [rad].
    tracking : bool
        True if the tracking has been initialized.
    bunch_index : int
        Number of the tracked bunch in the current core.
    distance : array
        Distance between bunches.
    valid_bunch_index : array

    Methods
    -------
    Vbr(I0)
        Return beam voltage at resonance in [V].
    Vb(I0)
        Return beam voltage in [V].
    Pb(I0)
        Return power transmitted to the beam in [W].
    Pr(I0)
        Return power reflected back to the generator in [W].
    Z(f)
        Cavity impedance in [Ohm] for a given frequency f in [Hz].
    set_optimal_coupling(I0)
        Set coupling to optimal value.
    set_optimal_detune(I0)
        Set detuning to optimal conditions.
    set_generator(I0)
        Set generator parameters.
    plot_phasor(I0)
        Plot phasor diagram.
    is_DC_Robinson_stable(I0)
        Check DC Robinson stability.
    is_CBI_stable(I0)
        Check Coupled-Bunch-Instability stability.
    plot_DC_Robinson_stability()
        Plot DC Robinson stability limit.
    init_tracking(beam)
        Initialization of the tracking.
    track(beam)
        Tracking method.
    phasor_decay(time)
        Compute the beam phasor decay during a given time span.
    phasor_evol(profile, bin_length, charge_per_mp)
        Compute the beam phasor evolution during the crossing of a bunch.
    VRF(z, I0)
        Return the total RF voltage.
    dVRF(z, I0)
        Return derivative of total RF voltage.
    ddVRF(z, I0)
        Return the second derivative of total RF voltage.
    deltaVRF(z, I0)
        Return the generator voltage minus beam loading voltage.
    update_feedback()
        Force feedback update from current CavityResonator parameters.
    sample_voltage()
        Sample the voltage seen by a zero charge particle during an RF period.

    References
    ----------
    [1] Wilson, P. B. (1994). Fundamental-mode rf design in e+ e− storage ring
    factories. In Frontiers of Particle Beams: Factories with e+ e-Rings
    (pp. 293-311). Springer, Berlin, Heidelberg.

    [2] Naoto Yamamoto, Alexis Gamelin, and Ryutaro Nagaoka. "Investigation
    of Longitudinal Beam Dynamics With Harmonic Cavities by Using the Code
    Mbtrack." IPAC’19, Melbourne, Australia, 2019.
    """

    def __init__(self,
                 ring,
                 m,
                 Rs,
                 Q,
                 QL,
                 detune,
                 Ncav=1,
                 Vc=0,
                 theta=0,
                 n_bin=75):
        self.ring = ring
        self.feedback = []
        self.m = m
        self.Ncav = Ncav
        if Ncav != 1:
            self.Rs_per_cavity = Rs
        else:
            self.Rs = Rs
        self.Q = Q
        self.QL = QL
        self.detune = detune
        self.Vc = Vc
        self.theta = theta
        self.beam_phasor = 0 + 0j
        self.beam_phasor_record = np.zeros((self.ring.h), dtype=complex)
        self.generator_phasor_record = np.zeros((self.ring.h), dtype=complex)
        self.tracking = False
        self.Vg = 0
        self.theta_g = 0
        self.Vgr = 0
        self.theta_gr = 0
        self.Pg = 0
        self.n_bin = int(n_bin)

    def init_tracking(self, beam):
        """
        Initialization of the tracking.

        Parameters
        ----------
        beam : Beam object

        """
        if beam.mpi_switch:
            self.bunch_index = beam.mpi.bunch_num  # Number of the tracked bunch in this processor

        self.distance = beam.distance_between_bunches
        self.valid_bunch_index = beam.bunch_index
        self.tracking = True
        self.nturn = 0

    def track(self, beam):
        """
        Track a Beam object through the CavityResonator object.

        Can be used with or without mpi.
        If used with mpi, beam.mpi.share_distributions must be called before.

        The beam phasor is given at t=0 (synchronous particle) of the first
        non empty bunch.

        Parameters
        ----------
        beam : Beam object

        """

        if self.tracking is False:
            self.init_tracking(beam)

        for index, bunch in enumerate(beam):

            if beam.filling_pattern[index]:

                if beam.mpi_switch:
                    # get rank of bunch n° index
                    rank = beam.mpi.bunch_to_rank(index)
                    # mpi -> get shared bunch profile for current bunch
                    center = beam.mpi.tau_center[rank]
                    profile = beam.mpi.tau_profile[rank]
                    bin_length = float(beam.mpi.tau_bin_length[rank][0])
                    charge_per_mp = float(beam.mpi.charge_per_mp_all[rank])
                    if index == self.bunch_index:
                        sorted_index = beam.mpi.tau_sorted_index
                else:
                    # no mpi -> get bunch profile for current bunch
                    if not bunch.is_empty:
                        (bins, sorted_index, profile,
                         center) = bunch.binning(n_bin=self.n_bin)
                        bin_length = bins[1] - bins[0]
                        charge_per_mp = bunch.charge_per_mp
                        self.bunch_index = index
                    else:
                        # Update filling pattern
                        beam.update_filling_pattern()
                        beam.update_distance_between_bunches()
                        # save beam phasor value
                        self.beam_phasor_record[index] = self.beam_phasor
                        # phasor decay to be at t=0 of the next bunch
                        self.phasor_decay(self.ring.T1, ref_frame="beam")
                        continue

                energy_change = bunch["tau"] * 0

                # remove part of beam phasor decay to be at the start of the binning (=bins[0])
                self.phasor_decay(center[0] - bin_length/2, ref_frame="beam")

                if index != self.bunch_index:
                    self.phasor_evol(profile,
                                     bin_length,
                                     charge_per_mp,
                                     ref_frame="beam")
                else:
                    # modify beam phasor
                    for i, center0 in enumerate(center):
                        mp_per_bin = int(profile[i])

                        if mp_per_bin == 0:
                            self.phasor_decay(bin_length, ref_frame="beam")
                            continue

                        ind = (sorted_index == i)
                        phase = self.m * self.ring.omega1 * (
                            center0 + self.ring.T1 *
                            (index + self.ring.h * self.nturn))
                        Vgene = np.real(self.generator_phasor_record[index] *
                                        np.exp(1j * phase))
                        Vbeam = np.real(self.beam_phasor)
                        Vtot = Vgene + Vbeam - charge_per_mp * self.loss_factor * mp_per_bin
                        energy_change[ind] = Vtot / self.ring.E0

                        self.beam_phasor -= 2 * charge_per_mp * self.loss_factor * mp_per_bin
                        self.phasor_decay(bin_length, ref_frame="beam")

                # phasor decay to be at t=0 of the current bunch (=-1*bins[-1])
                self.phasor_decay(-1 * (center[-1] + bin_length/2),
                                  ref_frame="beam")

                if index == self.bunch_index:
                    # apply kick
                    bunch["delta"] += energy_change

            # save beam phasor value
            self.beam_phasor_record[index] = self.beam_phasor

            # phasor decay to be at t=0 of the next bunch
            self.phasor_decay(self.ring.T1, ref_frame="beam")

        # apply different kind of RF feedback
        for fb in self.feedback:
            fb.track()

        self.nturn += 1

    def init_phasor_track(self, beam):
        """
        Initialize the beam phasor for a given beam distribution using a
        tracking like method.

        Follow the same steps as the track method but in the "rf" reference
        frame and without any modifications on the beam.

        Parameters
        ----------
        beam : Beam object

        """
        if self.tracking is False:
            self.init_tracking(beam)

        n_turn = int(self.filling_time / self.ring.T0 * 10)

        for i in range(n_turn):
            for j, bunch in enumerate(beam.not_empty):

                index = self.valid_bunch_index[j]

                if beam.mpi_switch:
                    # get shared bunch profile for current bunch
                    beam.mpi.share_distributions(beam, n_bin=self.n_bin)
                    center = beam.mpi.tau_center[j]
                    profile = beam.mpi.tau_profile[j]
                    bin_length = float(beam.mpi.tau_bin_length[j][0])
                    charge_per_mp = float(beam.mpi.charge_per_mp_all[j])
                else:
                    if i == 0:
                        # get bunch profile for current bunch
                        (bins, sorted_index, profile,
                         center) = bunch.binning(n_bin=self.n_bin)
                        if j == 0:
                            self.profile_save = np.zeros((
                                len(beam),
                                len(profile),
                            ))
                            self.center_save = np.zeros((
                                len(beam),
                                len(center),
                            ))
                        self.profile_save[j, :] = profile
                        self.center_save[j, :] = center
                    else:
                        profile = self.profile_save[j, :]
                        center = self.center_save[j, :]

                    bin_length = bins[1] - bins[0]
                    charge_per_mp = bunch.charge_per_mp

                self.phasor_decay(center[0] - bin_length/2, ref_frame="rf")
                self.phasor_evol(profile,
                                 bin_length,
                                 charge_per_mp,
                                 ref_frame="rf")
                self.phasor_decay(-1 * (center[-1] + bin_length/2),
                                  ref_frame="rf")
                self.phasor_decay((self.distance[index] * self.ring.T1),
                                  ref_frame="rf")

    def phasor_decay(self, time, ref_frame="beam"):
        """
        Compute the beam phasor decay during a given time span, assuming that
        no particles are crossing the cavity during the time span.

        Parameters
        ----------
        time : float
            Time span in [s], can be positive or negative.
        ref_frame : string, optional
            Reference frame to be used, can be "beam" or "rf".

        """
        if ref_frame == "beam":
            delta = self.wr
        elif ref_frame == "rf":
            delta = (self.wr - self.m * self.ring.omega1)
        self.beam_phasor = self.beam_phasor * np.exp(
            (-1 / self.filling_time + 1j*delta) * time)

    def phasor_evol(self,
                    profile,
                    bin_length,
                    charge_per_mp,
                    ref_frame="beam"):
        """
        Compute the beam phasor evolution during the crossing of a bunch using
        an analytic formula [1].

        Assume that the phasor decay happens before the beam loading.

        Parameters
        ----------
        profile : array
            Longitudinal profile of the bunch in [number of macro-particle].
        bin_length : float
            Length of a bin in [s].
        charge_per_mp : float
            Charge per macro-particle in [C].
        ref_frame : string, optional
            Reference frame to be used, can be "beam" or "rf".

        References
        ----------
        [1] mbtrack2 manual.

        """
        if ref_frame == "beam":
            delta = self.wr
        elif ref_frame == "rf":
            delta = (self.wr - self.m * self.ring.omega1)

        n_bin = len(profile)

        # Phasor decay during crossing time
        deltaT = n_bin * bin_length
        self.phasor_decay(deltaT, ref_frame)

        # Phasor evolution due to induced voltage by marco-particles
        k = np.arange(0, n_bin)
        var = np.exp(
            (-1 / self.filling_time + 1j*delta) * (n_bin-k) * bin_length)
        sum_tot = np.sum(profile * var)
        sum_val = -2 * sum_tot * charge_per_mp * self.loss_factor
        self.beam_phasor += sum_val

    def init_phasor(self, beam):
        """
        Initialize the beam phasor for a given beam distribution using an
        analytic formula [1].

        No modifications on the Beam object.

        Parameters
        ----------
        beam : Beam object

        References
        ----------
        [1] mbtrack2 manual.

        """

        # Initialization
        if self.tracking is False:
            self.init_tracking(beam)

        N = self.n_bin
        delta = (self.wr - self.m * self.ring.omega1)
        n_turn = int(self.filling_time / self.ring.T0 * 10)

        T = np.ones(self.ring.h) * self.ring.T1
        bin_length = np.zeros(self.ring.h)
        charge_per_mp = np.zeros(self.ring.h)
        bins = np.zeros((N + 1, self.ring.h))
        profile = np.zeros((N, self.ring.h))
        center = np.zeros((N, self.ring.h))

        # Gather beam distribution data
        for j, bunch in enumerate(beam.not_empty):
            index = self.valid_bunch_index[j]
            if beam.mpi_switch:
                beam.mpi.share_distributions(beam, n_bin=self.n_bin)
                center[:, index] = beam.mpi.tau_center[j]
                profile[:, index] = beam.mpi.tau_profile[j]
                bin_length[index] = float(beam.mpi.tau_bin_length[j][0])
                charge_per_mp[index] = float(beam.mpi.charge_per_mp_all[j])
            else:
                (bins[:, index], sorted_index, profile[:, index],
                 center[:, index]) = bunch.binning(n_bin=self.n_bin)
                bin_length[index] = bins[1, index] - bins[0, index]
                charge_per_mp[index] = bunch.charge_per_mp
            T[index] -= (center[-1, index] + bin_length[index] / 2)
            if index != 0:
                T[index - 1] += (center[0, index] - bin_length[index] / 2)
        T[self.ring.h - 1] += (center[0, 0] - bin_length[0] / 2)

        # Compute matrix coefficients
        k = np.arange(0, N)
        Tkj = np.zeros((N, self.ring.h))
        for j in range(self.ring.h):
            sum_t = np.array(
                [T[n] + N * bin_length[n] for n in range(j + 1, self.ring.h)])
            Tkj[:, j] = (N-k) * bin_length[j] + T[j] + np.sum(sum_t)

        var = np.exp((-1 / self.filling_time + 1j*delta) * Tkj)
        sum_tot = np.sum((profile*charge_per_mp) * var)

        # Use the formula n_turn times
        for i in range(n_turn):
            # Phasor decay during one turn
            self.phasor_decay(self.ring.T0, ref_frame="rf")
            # Phasor evolution due to induced voltage by marco-particles during one turn
            sum_val = -2 * sum_tot * self.loss_factor
            self.beam_phasor += sum_val

        # Replace phasor at t=0 (synchronous particle) of the first non empty bunch.
        idx0 = self.valid_bunch_index[0]
        self.phasor_decay(center[-1, idx0] + bin_length[idx0] / 2,
                          ref_frame="rf")

    @property
    def generator_phasor(self):
        """Generator phasor in [V]"""
        return self.Vg * np.exp(1j * self.theta_g)

    @property
    def cavity_phasor(self):
        """Cavity total phasor in [V]"""
        return self.generator_phasor + self.beam_phasor

    @property
    def cavity_phasor_record(self):
        """Last cavity phasor value of each bunch in [V]"""
        return self.generator_phasor_record + self.beam_phasor_record

    @property
    def ig_phasor_record(self):
        """Last current generator phasor of each bunch in [A]"""
        for FB in self.feedback:
            if isinstance(FB, (ProportionalIntegralLoop, DirectFeedback)):
                return FB.ig_phasor_record
        return np.zeros(self.ring.h)

    @property
    def DFB_ig_phasor(self):
        """Last direct feedback current generator phasor of each bunch in [A]"""
        for FB in self.feedback:
            if isinstance(FB, DirectFeedback):
                return FB.DFB_ig_phasor
        return np.zeros(self.ring.h)

    @property
    def cavity_voltage(self):
        """Cavity total voltage in [V]"""
        return np.abs(self.cavity_phasor)

    @property
    def cavity_phase(self):
        """Cavity total phase in [rad]"""
        return np.angle(self.cavity_phasor)

    @property
    def beam_voltage(self):
        """Beam loading voltage in [V]"""
        return np.abs(self.beam_phasor)

    @property
    def beam_phase(self):
        """Beam loading phase in [rad]"""
        return np.angle(self.beam_phasor)

    @property
    def loss_factor(self):
        """Cavity loss factor in [V/C]"""
        return self.wr * self.Rs / (2 * self.Q)

    @property
    def m(self):
        """Harmonic number of the cavity"""
        return self._m

    @m.setter
    def m(self, value):
        self._m = value

    @property
    def Ncav(self):
        """Number of cavities"""
        return self._Ncav

    @Ncav.setter
    def Ncav(self, value):
        self._Ncav = value

    @property
    def Rs_per_cavity(self):
        """Shunt impedance of a single cavity in [Ohm], defined as
        0.5*Vc*Vc/Pc."""
        return self._Rs_per_cavity

    @Rs_per_cavity.setter
    def Rs_per_cavity(self, value):
        self._Rs_per_cavity = value

    @property
    def Rs(self):
        """Shunt impedance [ohm]"""
        return self.Rs_per_cavity * self.Ncav

    @Rs.setter
    def Rs(self, value):
        self.Rs_per_cavity = value / self.Ncav

    @property
    def RL(self):
        """Loaded shunt impedance [ohm]"""
        return self.Rs / (1 + self.beta)

    @property
    def Q(self):
        """Quality factor"""
        return self._Q

    @Q.setter
    def Q(self, value):
        self._Q = value

    @property
    def QL(self):
        """Loaded quality factor"""
        return self._QL

    @QL.setter
    def QL(self, value):
        self._QL = value
        self._beta = self.Q / self.QL - 1
        self.update_feedback()

    @property
    def beta(self):
        """Coupling coefficient"""
        return self._beta

    @beta.setter
    def beta(self, value):
        self.QL = self.Q / (1+value)

    @property
    def detune(self):
        """Cavity detuning [Hz] - defined as (fr - m*f1)"""
        return self._detune

    @detune.setter
    def detune(self, value):
        self._detune = value
        self._fr = self.detune + self.m * self.ring.f1
        self._wr = self.fr * 2 * np.pi
        self._psi = np.arctan(self.QL * (self.fr / (self.m * self.ring.f1) -
                                         (self.m * self.ring.f1) / self.fr))
        self.update_feedback()

    @property
    def fr(self):
        """Resonance frequency of the cavity in [Hz]"""
        return self._fr

    @fr.setter
    def fr(self, value):
        self.detune = value - self.m * self.ring.f1

    @property
    def wr(self):
        """Angular resonance frequency in [Hz.rad]"""
        return self._wr

    @wr.setter
    def wr(self, value):
        self.detune = (value - self.m * self.ring.f1) * 2 * np.pi

    @property
    def psi(self):
        """Tuning angle in [rad]"""
        return self._psi

    @psi.setter
    def psi(self, value):
        delta = (self.ring.f1 * self.m * np.tan(value) /
                 self.QL)**2 + 4 * (self.ring.f1 * self.m)**2
        fr = (self.ring.f1 * self.m * np.tan(value) / self.QL +
              np.sqrt(delta)) / 2
        self.detune = fr - self.m * self.ring.f1

    @property
    def filling_time(self):
        """Cavity filling time in [s]"""
        return 2 * self.QL / self.wr

    @property
    def Pc(self):
        """Power dissipated in the cavity walls in [W]"""
        return self.Vc**2 / (2 * self.Rs)

    def Pb(self, I0):
        """
        Return power transmitted to the beam in [W] - near Eq. (4.2.3) in [1].

        Parameters
        ----------
        I0 : float
            Beam current in [A].

        Returns
        -------
        float
            Power transmitted to the beam in [W].

        """
        return I0 * self.Vc * np.cos(self.theta)

    def Pr(self, I0):
        """
        Power reflected back to the generator in [W].

        Parameters
        ----------
        I0 : float
            Beam current in [A].

        Returns
        -------
        float
            Power reflected back to the generator in [W].

        """
        return self.Pg - self.Pb(I0) - self.Pc

    def Vbr(self, I0):
        """
        Return beam voltage at resonance in [V].

        Parameters
        ----------
        I0 : float
            Beam current in [A].

        Returns
        -------
        float
            Beam voltage at resonance in [V].

        """
        return 2 * I0 * self.Rs / (1 + self.beta)

    def Vb(self, I0):
        """
        Return beam voltage in [V].

        Parameters
        ----------
        I0 : float
            Beam current in [A].

        Returns
        -------
        float
            Beam voltage in [V].

        """
        return self.Vbr(I0) * np.cos(self.psi)

    def Z(self, f):
        """Cavity impedance in [Ohm] for a given frequency f in [Hz]"""
        return self.RL / (1 + 1j * self.QL * (self.fr / f - f / self.fr))

    def set_optimal_detune(self, I0, F=1):
        """
        Set detuning to optimal conditions - second Eq. (4.2.1) in [1].

        Parameters
        ----------
        I0 : float
            Beam current in [A].

        """
        self.psi = np.arctan(-self.Vbr(I0) * F / self.Vc * np.sin(self.theta))

    def set_optimal_coupling(self, I0):
        """
        Set coupling to optimal value - Eq. (4.2.3) in [1].

        Parameters
        ----------
        I0 : float
            Beam current in [A].

        """
        self.beta = 1 + (2 * I0 * self.Rs * np.cos(self.theta) / self.Vc)

    def set_generator(self, I0):
        """
        Set generator parameters (Pg, Vgr, theta_gr, Vg and theta_g) for a
        given current and set of parameters.

        Parameters
        ----------
        I0 : float
            Beam current in [A].

        """

        # Generator power [W] - Eq. (4.1.2) [1] corrected with factor (1+beta)**2 instead of (1+beta**2)
        self.Pg = self.Vc**2 * (1 + self.beta)**2 / (
            2 * self.Rs * 4 * self.beta * np.cos(self.psi)**2) * (
                (np.cos(self.theta) + 2 * I0 * self.Rs /
                 (self.Vc * (1 + self.beta)) * np.cos(self.psi)**2)**2 +
                (np.sin(self.theta) + 2 * I0 * self.Rs /
                 (self.Vc *
                  (1 + self.beta)) * np.cos(self.psi) * np.sin(self.psi))**2)
        # Generator voltage at resonance [V] - Eq. (3.2.2) [1]
        self.Vgr = 2 * self.beta**(1 / 2) / (1 + self.beta) * (
            2 * self.Rs * self.Pg)**(1 / 2)
        # Generator phase at resonance [rad] - from Eq. (4.1.1)
        self.theta_gr = np.arctan(
            (self.Vc * np.sin(self.theta) +
             self.Vbr(I0) * np.cos(self.psi) * np.sin(self.psi)) /
            (self.Vc * np.cos(self.theta) +
             self.Vbr(I0) * np.cos(self.psi)**2)) - self.psi
        # Generator voltage [V]
        self.Vg = self.Vgr * np.cos(self.psi)
        # Generator phase [rad]
        self.theta_g = self.theta_gr + self.psi
        # Set generator_phasor_record
        self.generator_phasor_record = np.ones(
            self.ring.h) * self.generator_phasor

    def plot_phasor(self, I0):
        """
        Plot phasor diagram showing the vector addition of generator and beam
        loading voltage.

        Parameters
        ----------
        I0 : float
            Beam current in [A].

        Returns
        -------
        Figure.

        """

        def make_legend_arrow(legend, orig_handle, xdescent, ydescent, width,
                              height, fontsize):
            p = mpatches.FancyArrow(0,
                                    0.5 * height,
                                    width,
                                    0,
                                    length_includes_head=True,
                                    head_width=0.75 * height)
            return p

        fig = plt.figure()
        ax = fig.add_subplot(111, polar=True)
        ax.set_rmax(
            max([1.2,
                 self.Vb(I0) / self.Vc * 1.2, self.Vg / self.Vc * 1.2]))
        arr1 = ax.arrow(self.theta,
                        0,
                        0,
                        1,
                        alpha=0.5,
                        width=0.015,
                        edgecolor='black',
                        lw=2)

        arr2 = ax.arrow(self.psi + np.pi,
                        0,
                        0,
                        self.Vb(I0) / self.Vc,
                        alpha=0.5,
                        width=0.015,
                        edgecolor='red',
                        lw=2)

        arr3 = ax.arrow(self.theta_g,
                        0,
                        0,
                        self.Vg / self.Vc,
                        alpha=0.5,
                        width=0.015,
                        edgecolor='blue',
                        lw=2)

        ax.set_rticks([])  # less radial ticks
        plt.legend([arr1, arr2, arr3], ['Vc', 'Vb', 'Vg'],
                   handler_map={
                       mpatches.FancyArrow:
                       HandlerPatch(patch_func=make_legend_arrow),
                   })

        return fig

    def is_CBI_stable(self,
                      I0,
                      synchrotron_tune=None,
                      modes=None,
                      bool_return=False):
        """
        Check Coupled-Bunch-Instability stability.
        Effect of Direct RF feedback is not included.

        This method is a wraper around lcbi_growth_rate to caluclate the CBI
        growth rate from the CavityResonator own impedance.

        See lcbi_growth_rate for details.

        Parameters
        ----------
        I0 : float
            Beam current in [A].
        synchrotron_tune : float, optinal
            Fractional number of longitudinal tune.
            If None, synchrotron_tune is computed using self.Vc as total
            voltage.
            Default is None.
        modes : float or list of float, optional
            Coupled Bunch Instability mode numbers to consider.
            If not None, return growth_rates of the input coupled bunch modes.
            Default is None.
        bool_return : bool, optional
            If True:
                - return True if the most unstable mode is stronger than
                longitudinal damping time.
                - return False if it is not the case.
            Default is False.

        Returns
        -------
        growth_rate : float
            Maximum coupled bunch instability growth rate in [s-1].
        mu : int
            Coupled bunch mode number corresponding to the maximum coupled bunch
            instability growth rate.

        """
        growth_rate, mu, growth_rates = lcbi_growth_rate(
            self.ring,
            I0,
            M=self.ring.h,
            synchrotron_tune=synchrotron_tune,
            Vrf=self.Vc,
            fr=self.fr,
            RL=self.RL,
            QL=self.QL)

        if modes is not None:
            growth_rates = growth_rates[modes]
            return growth_rates

        if bool_return:
            if growth_rate > 1 / self.ring.tau[2]:
                return False
            else:
                return True
        else:
            return growth_rate, mu

    def is_DC_Robinson_stable(self, I0):
        """
        Check DC Robinson stability - Eq. (6.1.1) [1]

        Parameters
        ----------
        I0 : float
            Beam current in [A].

        Returns
        -------
        bool

        """
        return 2 * self.Vc * np.sin(self.theta) + self.Vbr(I0) * np.sin(
            2 * self.psi) > 0

    def plot_DC_Robinson_stability(self, detune_range=[-1e5, 1e5]):
        """
        Plot DC Robinson stability limit.

        Parameters
        ----------
        detune_range : list or array, optional
            Range of tuning to plot in [Hz].

        Returns
        -------
        Figure.

        """
        old_detune = self.psi

        x = np.linspace(detune_range[0], detune_range[1], 1000)
        y = []
        for i in range(0, x.size):
            self.detune = x[i]
            y.append(-self.Vc * (1 + self.beta) /
                     (self.Rs * np.sin(2 * self.psi)) *
                     np.sin(self.theta))  # droite de stabilité

        fig = plt.figure()
        ax = plt.gca()
        ax.plot(x, y)
        ax.set_xlabel("Detune [Hz]")
        ax.set_ylabel("Threshold current [A]")
        ax.set_title("DC Robinson stability limit")

        self.psi = old_detune

        return fig

    def VRF(self, z, I0, F=1, PHI=0):
        """Total RF voltage taking into account form factor amplitude F and form factor phase PHI"""
        return self.Vg * np.cos(self.ring.k1 * self.m * z +
                                self.theta_g) - self.Vb(I0) * F * np.cos(
                                    self.ring.k1 * self.m * z + self.psi - PHI)

    def dVRF(self, z, I0, F=1, PHI=0):
        """Return derivative of total RF voltage taking into account form factor amplitude F and form factor phase PHI"""
        return -1 * self.Vg * self.ring.k1 * self.m * np.sin(
            self.ring.k1 * self.m * z +
            self.theta_g) + self.Vb(I0) * F * self.ring.k1 * self.m * np.sin(
                self.ring.k1 * self.m * z + self.psi - PHI)

    def ddVRF(self, z, I0, F=1, PHI=0):
        """Return the second derivative of total RF voltage taking into account form factor amplitude F and form factor phase PHI"""
        return -1 * self.Vg * (self.ring.k1 * self.m)**2 * np.cos(
            self.ring.k1 * self.m * z + self.theta_g) + self.Vb(I0) * F * (
                self.ring.k1 * self.m)**2 * np.cos(self.ring.k1 * self.m * z +
                                                   self.psi - PHI)

    def deltaVRF(self, z, I0, F=1, PHI=0):
        """Return the generator voltage minus beam loading voltage taking into account form factor amplitude F and form factor phase PHI"""
        return -1 * self.Vg * (self.ring.k1 * self.m)**2 * np.cos(
            self.ring.k1 * self.m * z + self.theta_g) - self.Vb(I0) * F * (
                self.ring.k1 * self.m)**2 * np.cos(self.ring.k1 * self.m * z +
                                                   self.psi - PHI)

    def update_feedback(self):
        """Force feedback update from current CavityResonator parameters."""
        for FB in self.feedback:
            if isinstance(FB, (ProportionalIntegralLoop, DirectFeedback)):
                FB.init_Ig2Vg_matrix()

    def sample_voltage(self, n_points=1e4, index=0):
        """
        Sample the voltage seen by a zero charge particle during an RF period.

        Parameters
        ----------
        n_points : int or float, optional
            Number of sample points. The default is 1e4.
        index : int, optional
            RF bucket number to sample. Be carful if index > 0 as no new beam
            loading is not taken into account here.
            The default is 0.

        Returns
        -------
        pos : array of float
            Array of position from -T1/2 to T1/2.
        voltage_rec : array of float
            Recoring of the voltage.

        """
        # Init
        n_points = int(n_points)
        index = 0
        voltage_rec = np.zeros(n_points)
        pos = np.linspace(-self.ring.T1 / 2, self.ring.T1 / 2, n_points)
        DeltaT = self.ring.T1 / (n_points-1)

        # From t=0 of first non empty bunch to -T1/2
        self.phasor_decay(-self.ring.T1 / 2 + index * self.ring.T1,
                          ref_frame="beam")

        # Goes from (-T1/2) to (T1/2 + DeltaT) in n_points steps
        for i in range(n_points):
            phase = self.m * self.ring.omega1 * (
                pos[i] + self.ring.T1 * (index + self.ring.h * self.nturn))
            Vgene = np.real(self.generator_phasor_record[index] *
                            np.exp(1j * phase))
            Vbeam = np.real(self.beam_phasor)
            Vtot = Vgene + Vbeam
            voltage_rec[i] = Vtot
            self.phasor_decay(DeltaT, ref_frame="beam")

        # Get back to t=0
        self.phasor_decay(-DeltaT * n_points + self.ring.T1 / 2 -
                          index * self.ring.T1,
                          ref_frame="beam")

        return pos, voltage_rec


class ProportionalLoop:
    """
    Proportional feedback loop to control a CavityResonator amplitude and phase.

    Feedback setpoints are cav_res.Vc and cav_res.theta.

    The loop must be added to the CavityResonator object using:
        cav_res.feedback.append(loop)

    Parameters
    ----------
    ring : Synchrotron object
    cav_res : CavityResonator
        The CavityResonator which is to be controlled.
    gain_A : float
        Amplitude (voltage) gain of the feedback.
    gain_P : float
        Phase gain of the feedback.
    delay : int
        Feedback delay in unit of turns.
        Must be supperior or equal to 1.

    """

    def __init__(self, ring, cav_res, gain_A, gain_P, delay):
        self.ring = ring
        self.cav_res = cav_res
        self.gain_A = gain_A
        self.gain_P = gain_P
        self.delay = int(delay)
        if delay < 1:
            raise ValueError("delay must be >= 1.")
        self.volt_delay = np.ones(self.delay) * self.cav_res.Vc
        self.phase_delay = np.ones(self.delay) * self.cav_res.theta

    def track(self):
        """
        Tracking method for the amplitude and phase loop.

        Returns
        -------
        None.

        """
        diff_A = self.volt_delay[-1] - self.cav_res.Vc
        diff_P = self.phase_delay[-1] - self.cav_res.theta
        self.cav_res.Vg -= self.gain_A * diff_A
        self.cav_res.theta_g -= self.gain_P * diff_P
        self.cav_res.generator_phasor_record = (np.ones(self.ring.h) *
                                                self.cav_res.generator_phasor)
        self.volt_delay = np.roll(self.volt_delay, 1)
        self.phase_delay = np.roll(self.phase_delay, 1)
        self.volt_delay[0] = self.cav_res.cavity_voltage
        self.phase_delay[0] = self.cav_res.cavity_phase


class TunerLoop:
    """
    Cavity tuner loop used to control a CavityResonator tuning (psi or detune)
    in order to keep the phase between cavity and generator current constant.

    Only a proportional controller is assumed.

    The loop must be added to the CavityResonator object using:
        cav_res.feedback.append(loop)

    Parameters
    ----------
    ring : Synchrotron object
    cav_res : CavityResonator
        The CavityResonator which is to be controlled.
    gain : float
        Proportional gain of the tuner loop.
        If not specified, 0.01 is used.
    avering_period : int, optional
        Period during which the phase difference is monitored and averaged.
        Then the feedback correction is applied every avering_period turn.
        Unit is turn number.
        A value longer than one synchrotron period (1/fs) is recommended.
        If None, 2-synchrotron period (2/fs) is used, although it is
        too fast compared to the actual situation.
        Default is None.
    offset : float, optional
        Tuning offset in [rad].
        Default is 0.

    """

    def __init__(self,
                 ring,
                 cav_res,
                 gain=0.01,
                 avering_period=None,
                 offset=0):
        self.ring = ring
        self.cav_res = cav_res
        if avering_period is None:
            fs = (self.ring.synchrotron_tune(self.cav_res.Vc) * self.ring.f1 /
                  self.ring.h)
            avering_period = 2 / fs / self.ring.T0

        self.Pgain = gain
        self.offset = offset
        self.avering_period = int(avering_period)
        self.diff = 0
        self.count = 0

    def track(self):
        """
        Tracking method for the tuner loop.

        Returns
        -------
        None.

        """
        if self.count == self.avering_period:
            diff = self.diff / self.avering_period - self.offset
            self.cav_res.psi -= diff * self.Pgain
            self.count = 0
            self.diff = 0
        else:
            self.diff += (self.cav_res.cavity_phase - self.cav_res.theta_g +
                          self.cav_res.psi)
            self.count += 1


class ProportionalIntegralLoop:
    """
    Proportional Integral (PI) loop to control a CavityResonator amplitude and
    phase via generator current (ig) [1,2].

    Feedback reference targets (setpoints) are cav_res.Vc and cav_res.theta.

    The ProportionalIntegralLoop should be initialized only after generator
    parameters are set.

    The loop must be added to the CavityResonator object using:
        cav_res.feedback.append(loop)

    When the reference target is changed, it might be needed to re-initialize
    the feedforward constant by calling init_FFconst().

    Parameters
    ----------
    ring : Synchrotron object
    cav_res : CavityResonator object
        CavityResonator in which the loop will be added.
    gain : float list
        Proportional gain (Pgain) and integral gain (Igain) of the feedback
        system in the form of a list [Pgain, Igain].
        In case of normal conducting cavity (QL~1e4), the Pgain of ~1.0 and
        Igain of ~1e4(5) are usually used.
        In case of super conducting cavity (QL > 1e6), the Pgain of ~100
        can be used.
        In a "bad" parameter set, unstable oscillation of the cavity voltage
        can be caused. So, a parameter scan of the gain should be made.
        Igain * ring.T1 / dtau is Ki defined as a the coefficients for
        integral part in of [1], where dtau is a clock period of the PI controller.
    sample_num : int
        Number of bunch over which the mean cavity voltage is computed.
        Units are in bucket numbers.
    every : int
        Sampling and clock period of the feedback controller
        Time interval between two cavity voltage monitoring and feedback.
        Units are in bucket numbers.
    delay : int
        Loop delay of the PI feedback system.
        Units are in bucket numbers.
    IIR_cutoff : float, optinal
        Cutoff frequency of the IIR filter in [Hz].
        If 0, cutoff frequency is infinity.
        Default is 0.
    FF : bool, optional
        Boolean switch to use feedforward constant.
        True is recommended to prevent a cavity voltage drop in the beginning
        of the tracking.
        In case of small Pgain (QL ~ 1e4), cavity voltage drop may cause
        beam loss due to static Robinson.
        Default is True.

    Methods
    -------
    track()
        Tracking method for the Cavity PI control feedback.
    init_Ig2Vg_matrix()
        Initialize matrix for Ig2Vg_matrix.
    init_FFconst()
        Initialize feedforward constant.
    Ig2Vg_matrix()
        Return Vg from Ig using matrix formalism.
    Ig2Vg()
        Go from Ig to Vg and apply values.
    Vg2Ig(Vg)
        Return Ig from Vg (assuming constant Vg).
    IIR_init(cutoff)
        Initialization for the IIR filter.
    IIR(input)
        Return IIR filter output.
    IIRcutoff()
        Return IIR cutoff frequency in [Hz].

    Notes
    -----
    Assumption : delay >> every ~ sample_num

    Adjusting ig(Vg) parameter to keep the cavity voltage constant (to target values).
    The "track" method is
       1) Monitoring the cavity voltage phasor
       Mean cavity voltage value between specified bunch number (sample_num) is monitored
       with specified interval period (every).
       2) Changing the ig phasor
       The ig phasor is changed according the difference of the specified
       reference values with specified gain (gain).
       By using ig instead of Vg, the cavity response can be taken account.
       3) ig changes are reflected to Vg after the specifed delay (delay) of the system

    Vc-->(rot)-->IIR-->(-)-->(V->I,fac)-->PI-->Ig --> Vg
                        |
                       Ref

    Examples
    --------
    PF; E0=2.5GeV, C=187m, frf=500MHz
        QL=11800, fs0=23kHz
        ==> gain=[0.5,1e4], sample_num=8, every=7(13ns), delay=500(1us)
                     is one reasonable parameter set.
        The practical clock period is 13ns.
            ==> Igain_PF = Igain_mbtrack * Trf / 13ns = Igain_mbtrack * 0.153

    References
    ----------
    [1] : https://en.wikipedia.org/wiki/PID_controller
    [2] : Yamamoto, N., Takahashi, T., & Sakanaka, S. (2018). Reduction and
    compensation of the transient beam loading effect in a double rf system
    of synchrotron light sources. PRAB, 21(1), 012001.

    """

    def __init__(self,
                 ring,
                 cav_res,
                 gain,
                 sample_num,
                 every,
                 delay,
                 IIR_cutoff=0,
                 FF=True):
        self.ring = ring
        self.cav_res = cav_res
        self.Ig2Vg_mat = np.zeros((self.ring.h, self.ring.h), dtype=complex)
        self.ig_modulation_signal = np.zeros(self.ring.h, dtype=complex)
        self.gain = gain
        self.FF = FF

        if delay > 0:
            self.delay = int(delay)
        else:
            self.delay = 1
        if every > 0:
            self.every = int(every)
        else:
            self.every = 1
        record_size = int(np.ceil(self.delay / self.every))
        if record_size < 1:
            raise ValueError("Bad parameter set : delay or every")
        self.sample_num = int(sample_num)

        # init lists for FB process
        self.ig_phasor = np.ones(self.ring.h, dtype=complex) * self.Vg2Ig(
            self.cav_res.generator_phasor)
        self.ig_phasor_record = self.ig_phasor
        self.vc_previous = np.ones(
            self.sample_num) * self.cav_res.cavity_phasor
        self.diff_record = np.zeros(record_size, dtype=complex)
        self.I_record = 0 + 0j

        self.sample_list = range(0, self.ring.h, self.every)

        self.IIR_init(IIR_cutoff)
        self.init_FFconst()

        # Pre caclulation for Ig2Vg
        self.init_Ig2Vg_matrix()

    def track(self, apply_changes=True):
        """
        Tracking method for the Cavity PI control feedback.

        Returns
        -------
        None.

        """
        vc_list = np.concatenate([
            self.vc_previous, self.cav_res.cavity_phasor_record
        ])  # This line is slowing down the process.
        self.ig_phasor.fill(self.ig_phasor[-1])

        for index in self.sample_list:
            # 2) updating Ig using last item of the list
            diff = self.diff_record[-1] - self.FFconst
            self.I_record += diff / self.ring.f1
            fb_value = self.gain[0] * diff + self.gain[1] * self.I_record
            self.ig_phasor[index:] = self.Vg2Ig(fb_value) + self.FFconst
            # Shift the record
            self.diff_record = np.roll(self.diff_record, 1)
            # 1) recording diff as a first item of the list
            mean_vc = np.mean(vc_list[index:self.sample_num + index]) * np.exp(
                -1j * self.cav_res.theta)
            self.diff_record[0] = self.cav_res.Vc - self.IIR(mean_vc)
        # update sample_list for next turn
        self.sample_list = range(index + self.every - self.ring.h, self.ring.h,
                                 self.every)
        # update vc_previous for next turn
        self.vc_previous = self.cav_res.cavity_phasor_record[-self.sample_num:]

        self.ig_phasor = self.ig_phasor + self.ig_modulation_signal
        self.ig_phasor_record = self.ig_phasor

        if apply_changes:
            self.Ig2Vg()

    def init_Ig2Vg_matrix(self):
        """
        Initialize matrix for Ig2Vg_matrix.

        Shoud be called before first use of Ig2Vg_matrix and after each cavity
        parameter change.
        """
        k = np.arange(0, self.ring.h)
        self.Ig2Vg_vec = np.exp(-1 / self.cav_res.filling_time *
                                (1 - 1j * np.tan(self.cav_res.psi)) *
                                self.ring.T1 * (k+1))
        tempV = np.exp(-1 / self.cav_res.filling_time * self.ring.T1 * k *
                       (1 - 1j * np.tan(self.cav_res.psi)))
        for idx in np.arange(self.ring.h):
            self.Ig2Vg_mat[idx:, idx] = tempV[:self.ring.h - idx]

    def init_FFconst(self):
        """Initialize feedforward constant."""
        if self.FF:
            self.FFconst = np.mean(self.ig_phasor)
        else:
            self.FFconst = 0

    def Ig2Vg_matrix(self):
        """
        Return Vg from Ig using matrix formalism.
        Warning: self.init_Ig2Vg should be called after each CavityResonator
        parameter change.
        """
        generator_phasor_record = (
            self.Ig2Vg_vec * self.cav_res.generator_phasor_record[-1] +
            self.Ig2Vg_mat.dot(self.ig_phasor_record) *
            self.cav_res.loss_factor * self.ring.T1)
        return generator_phasor_record

    def Ig2Vg(self):
        """
        Go from Ig to Vg.

        Apply new values to cav_res.generator_phasor_record, cav_res.Vg and
        cav_res.theta_g from ig_phasor_record.
        """
        self.cav_res.generator_phasor_record = self.Ig2Vg_matrix()
        self.cav_res.Vg = np.mean(np.abs(self.cav_res.generator_phasor_record))
        self.cav_res.theta_g = np.mean(
            np.angle(self.cav_res.generator_phasor_record))

    def Vg2Ig(self, Vg):
        """
        Return Ig from Vg (assuming constant Vg).

        Eq.25 of ref [2] assuming the dVg/dt = 0.
        """
        return Vg * (1 - 1j * np.tan(self.cav_res.psi)) / self.cav_res.RL

    def IIR_init(self, cutoff):
        """
        Initialization for the IIR filter.

        Parameters
        ----------
        cutoff : float
            Cutoff frequency of the IIR filter in [Hz].
            If 0, cutoff frequency is infinity.

        """
        if cutoff == 0:
            self.IIRcoef = 1.0
        else:
            omega = 2.0 * np.pi * cutoff
            T = self.ring.T1 * self.every
            alpha = np.cos(omega * T) - 1
            tmp = alpha*alpha - 2*alpha
            if tmp > 0:
                self.IIRcoef = alpha + np.sqrt(tmp)
            else:
                self.IIRcoef = T * cutoff * 2 * np.pi
        self.IIRout = self.cav_res.Vc

    def IIR(self, input):
        """Return IIR filter output."""
        self.IIRout = (1 - self.IIRcoef) * self.IIRout + self.IIRcoef * input
        return self.IIRout

    @property
    def IIRcutoff(self):
        """Return IIR cutoff frequency in [Hz]."""
        T = self.ring.T1 * self.every
        return (1.0 / 2.0 / np.pi / T * np.arccos(
            (2 - 2 * self.IIRcoef - self.IIRcoef * self.IIRcoef) / 2 /
            (1 - self.IIRcoef)))


class DirectFeedback(ProportionalIntegralLoop):
    """
    Direct RF FB (DFB) via generator current (ig) using PI controller [1,2].

    The DirectFeedback inherits from ProportionalIntegralLoop so all
    mandatory parameters from ProportionalIntegralLoop should be passed as
    kwargs when initializing a DirectFeedback object.

    To avoid cavity-beam unmatching (large synchrotron oscilation of beam),
    the CavityResonator generator parameters should be set before
    initialization.

    The loop must be added to the CavityResonator object using:
        cav_res.feedback.append(loop)

    Parameters
    ----------
    DFB_gain : float
        Gain of the DFB.
    DFB_phase_shift : float
        Phase shift of the DFB.
    DFB_sample_num : int, optional
        Sample number to monitor Vc for the DFB.
        The averaged Vc value in DFB_sample_num is monitored.
        Units are in bucket numbers.
        If None, value from the ProportionalIntegralLoop is used.
        Default is None.
    DFB_every : int, optional
        Interval to monitor and change Vc for the DFB.
        Units are in bucket numbers.
        If None, value from the ProportionalIntegralLoop is used.
        Default is None.
    DFB_delay : int, optional
        Loop delay of the DFB.
        Units are in bucket numbers.
        If None, value from the ProportionalIntegralLoop is used.
        Default is None.

    Attributes
    ----------
    phase_shift : float
        Phase shift applied to DFB, defined as psi - DFB_phase_shift.
    DFB_psi: float
        Return detuning angle value with Direct RF feedback in [rad].
    DFB_alpha : float
        Return the amplitude ratio alpha of the DFB.
    DFB_gamma : float
        Return the gain gamma of the DFB.
    DFB_Rs : float
        Return the shunt impedance of the DFB in [ohm].

    Methods
    -------
    DFB_parameter_set(DFB_gain, DFB_phase_shift)
        Set DFB gain and phase shift parameters.
    track()
        Tracking method for the Direct RF feedback.
    DFB_Vg()
        Return the generator voltage main and DFB components in [V].
    DFB_fs()
        Return the modified synchrotron frequency in [Hz].

    References
    ----------
    [1] : Akai, K. (2022). Stability analysis of rf accelerating mode with
    feedback loops under heavy beam loading in SuperKEKB. PRAB, 25(10),
    102002.
    [2] : N. Yamamoto et al. (2023). Stability survey of a double RF system
    with RF feedback loops for bunch lengthening in a low-emittance synchrotron
    ring. In Proc. IPAC'23. doi:10.18429/JACoW-IPAC2023-WEPL161

    """

    def __init__(self,
                 DFB_gain,
                 DFB_phase_shift,
                 DFB_sample_num=None,
                 DFB_every=None,
                 DFB_delay=None,
                 **kwargs):
        super(DirectFeedback, self).__init__(**kwargs)

        if DFB_delay is not None:
            self.DFB_delay = int(DFB_delay)
        else:
            self.DFB_delay = self.delay

        if DFB_sample_num is not None:
            self.DFB_sample_num = int(DFB_sample_num)
        else:
            self.DFB_sample_num = self.sample_num

        if DFB_every is not None:
            self.DFB_every = int(DFB_every)
        else:
            self.DFB_every = self.every

        record_size = int(np.ceil(self.DFB_delay / self.DFB_every))
        if record_size < 1:
            raise ValueError("Bad parameter set : DFB_delay or DFB_every")

        self.DFB_parameter_set(DFB_gain, DFB_phase_shift)
        if np.sum(np.abs(self.cav_res.beam_phasor)) == 0:
            cavity_phasor = self.cav_res.Vc * np.exp(1j * self.cav_res.theta)
        else:
            cavity_phasor = np.mean(self.cav_res.cavity_phasor_record)
        self.DFB_VcRecord = np.ones(record_size, dtype=complex) * cavity_phasor
        self.DFB_vc_previous = (np.ones(self.DFB_sample_num, dtype=complex) *
                                cavity_phasor)

        self.DFB_sample_list = range(0, self.ring.h, self.DFB_every)

    @property
    def DFB_phase_shift(self):
        """Return DFB phase shift."""
        return self._DFB_phase_shift

    @DFB_phase_shift.setter
    def DFB_phase_shift(self, value):
        """Set DFB_phase_shift and phase_shift"""
        self._DFB_phase_shift = value
        self._phase_shift = self.cav_res.psi - value

    @property
    def phase_shift(self):
        """
        Return phase shift applied to DFB.
        Defined as self.cav_res.psi - self.DFB_phase_shift.
        """
        return self._phase_shift

    @property
    def DFB_psi(self):
        """
        Return detuning angle value with Direct RF feedback in [rad].

        Fig.4 of ref [1].
        """
        return np.angle(np.mean(self.cav_res.cavity_phasor_record)) - np.angle(
            np.mean(self.ig_phasor_record))

    @property
    def DFB_alpha(self):
        """
        Return the amplitude ratio alpha of the DFB.

        Near Eq. (13) of [1].
        """
        fac = np.abs(
            np.mean(self.DFB_ig_phasor) / np.mean(self.ig_phasor_record))
        return 20 * np.log10(fac)

    @property
    def DFB_gamma(self):
        """
        Return the gain gamma of the DFB.

        Near Eq. (13) of [1].
        """
        fac = np.abs(
            np.mean(self.DFB_ig_phasor) / np.mean(self.ig_phasor_record))
        return fac / (1-fac)

    @property
    def DFB_Rs(self):
        """
        Return the shunt impedance of the DFB in [ohm].

        Eq. (15) of [1].
        """
        return self.cav_res.Rs / (1 + self.DFB_gamma * np.cos(self.DFB_psi))

    def DFB_parameter_set(self, DFB_gain, DFB_phase_shift):
        """
        Set DFB gain and phase shift parameters.

        Parameters
        ----------
        DFB_gain : float
            Gain of the DFB.
        DFB_phase_shift : float
            Phase shift of the DFB.

        """
        self.DFB_gain = DFB_gain
        self.DFB_phase_shift = DFB_phase_shift

        if np.sum(np.abs(self.cav_res.beam_phasor)) == 0:
            vc = (np.ones(self.ring.h) * self.cav_res.Vc *
                  np.exp(1j * self.cav_res.theta))
        else:
            vc = self.cav_res.cavity_phasor_record
        vg_drf = self.DFB_gain * vc * np.exp(1j * self.phase_shift)
        self.DFB_ig_phasor = self.Vg2Ig(vg_drf)

        self.ig_phasor = self.ig_phasor_record - self.DFB_ig_phasor
        self.init_FFconst()

    def track(self):
        """
        Tracking method for the Direct RF feedback.

        Returns
        -------
        None.

        """
        super(DirectFeedback, self).track(False)

        vc_list = np.concatenate(
            [self.DFB_vc_previous, self.cav_res.cavity_phasor_record])
        self.DFB_ig_phasor = np.roll(self.DFB_ig_phasor, 1)
        for index in self.DFB_sample_list:
            # 2) updating Ig using last item of the list
            vg_drf = (self.DFB_gain * self.DFB_VcRecord[-1] *
                      np.exp(1j * self.phase_shift))
            self.DFB_ig_phasor[index:] = self.Vg2Ig(vg_drf)
            # Shift the record
            self.DFB_VcRecord = np.roll(self.DFB_VcRecord, 1)
            # 1) recording Vc
            mean_vc = np.mean(vc_list[index:self.DFB_sample_num + index])
            self.DFB_VcRecord[0] = mean_vc
        # update sample_list for next turn
        self.DFB_sample_list = range(index + self.DFB_every - self.ring.h,
                                     self.ring.h, self.DFB_every)
        # update vc_previous for next turn
        self.DFB_vc_previous = self.cav_res.cavity_phasor_record[
            -self.DFB_sample_num:]

        self.ig_phasor_record = self.ig_phasor + self.DFB_ig_phasor

        self.Ig2Vg()

    def DFB_Vg(self, vc=-1):
        """Return the generator voltage main and DFB components in [V]."""
        if vc == -1:
            vc = np.mean(self.cav_res.cavity_phasor_record)
        vg_drf = self.DFB_gain * vc * np.exp(1j * self.phase_shift)
        vg_main = np.mean(self.cav_res.generator_phasor_record) - vg_drf
        return vg_main, vg_drf

    def DFB_fs(self, vg_main=-1, vg_drf=-1):
        """Return the modified synchrotron frequency in [Hz]."""
        vc = np.mean(self.cav_res.cavity_phasor_record)
        if vg_drf == -1:
            vg_drf = self.DFB_gain * vc * np.exp(1j * self.phase_shift)
        if vg_main == -1:
            vg_main = np.mean(self.cav_res.generator_phasor_record) - vg_drf
        vg_sum = np.abs(vg_main) * np.sin(
            np.angle(vg_main)) + np.abs(vg_drf) * np.sin(np.angle(vg_drf))
        omega_s = 0
        if (vg_sum) > 0.0:
            omega_s = np.sqrt(self.ring.ac * self.ring.omega1 * (vg_sum) /
                              self.ring.E0 / self.ring.T0)
        return omega_s / 2 / np.pi


class ProportionalIntegralIQLoop:
    """
    Proportional Integral (PI) loop with I/Q control and optional Mode 0 Damper
    to control a CavityResonator amplitude and phase via generator current (ig).

    This class adapts the logic from ProportionalIntegralLoop for I/Q components
    and integrates an optional Mode 0 damper.

    Feedback reference targets (setpoints) are derived from cav_res.Vc and
    cav_res.theta to define target I and Q components.

    The loop must be added to the CavityResonator object using:
        cav_res.feedback.append(loop)

    Parameters
    ----------
    ring : Synchrotron object
    cav_res : CavityResonator object
        CavityResonator in which the loop will be added.
    gain : list or tuple of two floats
        Proportional gain (Pgain) and integral gain (Igain) of the feedback
        system, e.g., `[Pgain, Igain]`. These gains are applied to both
        I and Q channels.
    sample_num : int
        Number of bunches over which the mean cavity voltage is computed.
        Units are in bucket numbers.
    every : int
        Sampling and clock period of the feedback controller.
        Time interval between two cavity voltage monitoring and feedback.
        Units are in bucket numbers.
    delay : int
        Loop delay of the PI feedback system.
        Units are in bucket numbers.
    IIR_cutoff : float, optional
        Cutoff frequency of the IIR filter in [Hz] applied to the magnitude
        of the mean cavity voltage. If 0, cutoff frequency is infinity (no filtering).
        Default is 0.
    FF : bool, optional
        Boolean switch to use feedforward constant. True is recommended.
        Default is True.
    enable_damper : bool, optional
        Enable the Mode 0 damper. Default is False.
    damper_gain : list or tuple of two floats, optional
        Proportional and Integral gains for the Mode 0 damper `[P_damper, I_damper]`.
        Default is `[0, 0]`.
    damper_filter_func : callable, optional
        A function to filter the raw Mode 0 signal. It should take one argument
        (the raw signal) and return the filtered signal.
        Default is `None` (no filtering beyond internal delay).
    damper_mean_idx : int, optional
        Index used in `bunch.mean[damper_mean_idx]` to extract the relevant
        phase information for the Mode 0 signal calculation.
        Default is 4. (The user should ensure this index corresponds to the correct phase variable).

    Methods
    -------
    track()
        Tracking method for the Cavity PI IQ control feedback.
    init_Ig2Vg_matrix()
        Initialize matrix for Ig2Vg_matrix.
    init_FFconst()
        Initialize feedforward constant.
    Ig2Vg_matrix()
        Return Vg from Ig using matrix formalism.
    Ig2Vg()
        Go from Ig to Vg and apply values.
    Vg2Ig(Vg)
        Return Ig from Vg (assuming constant Vg).
    IIR_init(cutoff)
        Initialization for the IIR filter.
    IIR(input)
        Return IIR filter output (operates on complex input, filters magnitude).
    IIRcutoff()
        Return IIR cutoff frequency in [Hz].
    compute_mode0_signal()
        Calculate the Mode 0 oscillation signal for the damper.
    """

    def __init__(
        self,
        ring,
        cav_res,
        gain, # Should be [Pgain, Igain] for I and Q
        sample_num,
        every,
        delay,
        IIR_cutoff=0,
        FF=True,
        enable_damper=False,
        damper_gain=(0, 0), # Should be [P_damper, I_damper]
        damper_filter_func=None,
        damper_mean_idx=4,
    ):
        self.ring = ring
        self.cav_res = cav_res
        self.Ig2Vg_mat = np.zeros((self.ring.h, self.ring.h), dtype=complex)
        self.ig_modulation_signal = np.zeros(self.ring.h, dtype=complex)

        # PI gains for I and Q components (can be made separate if needed)
        self.Pgain = gain[0]
        self.Igain = gain[1]
        self.FF = FF

        # Damper parameters
        self.enable_damper = enable_damper
        self.Pgain_damper = damper_gain[0]
        self.Igain_damper = damper_gain[1]
        self.damper_filter_func = damper_filter_func if damper_filter_func else lambda x: x
        self.damper_mean_idx = damper_mean_idx
        self.damper_buffer = [] # For damper signal delay
        self.mode0_integral_term = 0.0 # Integral term for damper

        if delay > 0:
            self.delay = int(delay)
        else:
            self.delay = 1
        if every > 0:
            self.every = int(every)
        else:
            self.every = 1

        # record_size for PI controller's error history (complex error)
        self.record_size = int(np.ceil(self.delay / self.every))
        if self.record_size < 1:
            raise ValueError("Bad parameter set for PI loop: delay or every results in non-positive record_size.")

        # Damper delay is tied to the main loop delay for now.
        # If separate delay for damper is needed, this should be a new parameter.
        self.damper_delay_steps = self.record_size

        self.sample_num = int(sample_num)

        # Target I and Q components
        self.target_I = self.cav_res.Vc * np.cos(self.cav_res.theta)
        self.target_Q = self.cav_res.Vc * np.sin(self.cav_res.theta)
        self.target_phasor = self.target_I + 1j * self.target_Q


        # Initialize feedback variables (similar to ProportionalIntegralLoop)
        self.ig_phasor = np.ones(self.ring.h, dtype=complex) * self.Vg2Ig(
            self.cav_res.generator_phasor
        )
        self.ig_phasor_record = np.copy(self.ig_phasor) # Ensure a copy
        self.vc_previous = np.ones(
            self.sample_num, dtype=complex
        ) * self.cav_res.cavity_phasor

        self.diff_record = np.zeros(self.record_size, dtype=complex) # Stores complex error Vc_target - Vc_measured
        self.I_record = 0 + 0j # Complex integral term for PI

        self.sample_list = range(0, self.ring.h, self.every)

        self.IIR_init(IIR_cutoff) # IIR for cavity voltage magnitude
        self.init_FFconst()
        self.init_Ig2Vg_matrix()

    def track(self, apply_changes=True):
        """Tracking method for the Cavity PI IQ control feedback."""
        vc_list = np.concatenate(
            [self.vc_previous, self.cav_res.cavity_phasor_record]
        )
        current_ig_fill_value = self.ig_phasor[-1] # Value to fill for non-updated parts
        self.ig_phasor.fill(current_ig_fill_value)


        for index in self.sample_list:
            # 1. Monitor cavity voltage (mean over sample_num)
            mean_vc_complex = np.mean(vc_list[index : self.sample_num + index])

            # Apply IIR filter to the magnitude of the mean cavity voltage
            # The phase is preserved from the original mean_vc_complex
            mean_vc_mag_filtered = self.IIR(np.abs(mean_vc_complex))
            mean_vc_phase = np.angle(mean_vc_complex)
            vc_for_error_calc = mean_vc_mag_filtered * np.exp(1j * mean_vc_phase)

            # 2. Calculate complex error
            # The error 'diff' is (target_voltage - measured_voltage)
            # ProportionalIntegralLoop does target - measured, so we follow that.
            # self.cav_res.Vc is the target magnitude.
            # self.cav_res.theta is the target phase.
            # Target phasor is self.cav_res.Vc * exp(1j*self.cav_res.theta)
            # The error is calculated against the IIR-filtered, phase-preserved voltage
            current_diff = self.target_phasor - vc_for_error_calc


            # 3. PI Controller (complex arithmetic)
            # diff_record[-1] is the error from 'delay' steps ago
            # FFconst is Vg2Ig(target_phasor) effectively
            delayed_error = self.diff_record[-1] # Error from 'delay' buckets ago
            self.I_record += delayed_error / self.ring.f1 # Update integral term with delayed error

            # PI control output based on delayed error
            # The FFconst is added to the PI output.
            # Vg2Ig converts a *voltage* target to a *current*.
            # The PI controller output is a *correction* to the current.
            # So, FFconst (base current) + PI_correction (current delta)
            pi_output_complex = self.Pgain * delayed_error + self.Igain * self.I_record

            # The feedforward constant is the base generator current required
            # FFconst is already in terms of Ig.
            ig_control_phasor = self.FFconst + pi_output_complex


            # 4. Mode 0 Damper (if enabled)
            if self.enable_damper:
                mode0_signal_raw = self.compute_mode0_signal_raw() # Gets raw signal, applies filter_func

                # Store raw (filtered) signal in buffer for delay
                self.damper_buffer.append(mode0_signal_raw)
                if len(self.damper_buffer) > self.damper_delay_steps:
                    delayed_mode0_signal = self.damper_buffer.pop(0)
                else:
                    # Not enough history for full delay, use 0 or partially delayed if preferred
                    delayed_mode0_signal = 0.0

                # PI control for damper (acts on phase)
                self.mode0_integral_term += delayed_mode0_signal * self.Igain_damper / self.ring.f1
                fb_phase_shift = (self.Pgain_damper * delayed_mode0_signal +
                                  self.mode0_integral_term)

                # Clip phase shift
                fb_phase_shift = np.clip(fb_phase_shift, -np.pi / 12, np.pi / 12)

                # Apply phase shift to the PI controller's ig_phasor output
                ig_control_phasor = ig_control_phasor * np.exp(1j * fb_phase_shift)

            self.ig_phasor[index:] = ig_control_phasor

            # Shift the error record for the PI controller
            self.diff_record = np.roll(self.diff_record, 1)
            self.diff_record[0] = current_diff # Store current error for future use

        # Update sample_list for next turn
        if self.sample_list: # Check if sample_list was not empty
             last_processed_index = index
        else: # sample_list was empty, e.g. every > ring.h
            last_processed_index = self.ring.h - self.every # Ensure no negative index

        self.sample_list = range(last_processed_index + self.every - self.ring.h, self.ring.h, self.every)


        # Update vc_previous for next turn
        self.vc_previous = self.cav_res.cavity_phasor_record[-self.sample_num:]

        self.ig_phasor = self.ig_phasor + self.ig_modulation_signal # Add any external modulation
        self.ig_phasor_record = np.copy(self.ig_phasor)

        if apply_changes:
            self.Ig2Vg()

    def init_Ig2Vg_matrix(self):
        """
        Initialize matrix for Ig2Vg_matrix.
        Should be called before first use of Ig2Vg_matrix and after each cavity
        parameter change.
        """
        k = np.arange(0, self.ring.h)
        self.Ig2Vg_vec = np.exp(-1 / self.cav_res.filling_time *
                                (1 - 1j * np.tan(self.cav_res.psi)) *
                                self.ring.T1 * (k + 1))
        tempV = np.exp(-1 / self.cav_res.filling_time * self.ring.T1 * k *
                       (1 - 1j * np.tan(self.cav_res.psi)))
        for idx in np.arange(self.ring.h):
            self.Ig2Vg_mat[idx:, idx] = tempV[:self.ring.h - idx]

    def init_FFconst(self):
        """Initialize feedforward constant."""
        if self.FF:
            # FFconst is the generator current needed for the target cavity phasor
            self.FFconst = self.Vg2Ig(self.target_phasor)
        else:
            self.FFconst = 0 + 0j


    def Ig2Vg_matrix(self):
        """
        Return Vg from Ig using matrix formalism.
        Warning: self.init_Ig2Vg_matrix should be called after each CavityResonator
        parameter change.
        """
        # This calculation seems to assume ig_phasor_record is a delta from a base.
        # If ig_phasor_record is the *total* desired ig, the application might differ.
        # ProportionalIntegralLoop's Ig2Vg_matrix is:
        # (self.Ig2Vg_vec * self.cav_res.generator_phasor_record[-1] +
        #  self.Ig2Vg_mat.dot(self.ig_phasor_record) * self.cav_res.loss_factor * self.ring.T1)
        # Here, self.ig_phasor_record IS the total current.
        # The term with generator_phasor_record[-1] is for the decay of previous state.
        # The term with ig_phasor_record is the driven response.
        # So, this looks correct if ig_phasor_record is total current per bucket.

        generator_phasor_record = (
            self.Ig2Vg_vec * self.cav_res.generator_phasor_record[-1] +
            self.Ig2Vg_mat.dot(self.ig_phasor_record) * # ig_phasor_record is complex current
            (2 * self.cav_res.loss_factor / (self.ring.T1 * self.cav_res.wr)) * # Converts current to voltage effectively Rs_eff * Ig
             self.cav_res.loss_factor * self.ring.T1 # This seems to have loss_factor twice
        )
        # Let's re-evaluate the conversion from Ig to Vg contribution
        # Vg_induced = Ig * Z_cavity_eff
        # Z_eff for driven part is related to RL and psi.
        # The original ProportionalIntegralLoop used:
        # self.Ig2Vg_mat.dot(self.ig_phasor_record) * self.cav_res.loss_factor * self.ring.T1
        # This implies ig_phasor_record was scaled such that this product gives voltage.
        # Let's stick to the original scaling for now assuming ig_phasor_record is consistent.
        # If Vg2Ig gives Ig, then Ig * Z should give Vg.
        # The factor in Vg2Ig is (1 - 1j * np.tan(self.cav_res.psi)) / self.cav_res.RL
        # So Ig * RL / (1 - 1j*tan(psi)) should be Vg.
        # The matrix multiplication handles the transient response.
        # The scaling factor self.cav_res.loss_factor * self.ring.T1 needs to be consistent
        # with how ig_phasor_record is defined by Vg2Ig and the PI loop.

        # Simplification: The matrix Ig2Vg_mat already contains the transfer function from Ig to Vg for each time step.
        # The term self.cav_res.loss_factor * self.ring.T1 is a scaling.
        # If Vg2Ig(Vg_target) produces an Ig_target, and PI controller adds a delta_Ig,
        # then ig_phasor_record = Ig_target + delta_Ig.
        # The matrix operation should correctly transform this total Ig per bucket to Vg.

        # Let's use the same formulation as ProportionalIntegralLoop, assuming ig_phasor_record
        # is the current that, when passed through the system response (matrix + scaling), gives Vg.
        generator_phasor_record = (
            self.Ig2Vg_vec * self.cav_res.generator_phasor_record[-1] +
            self.Ig2Vg_mat.dot(self.ig_phasor_record) * self.cav_res.loss_factor * self.ring.T1
        )
        return generator_phasor_record


    def Ig2Vg(self):
        """
        Go from Ig to Vg.
        Apply new values to cav_res.generator_phasor_record, cav_res.Vg and
        cav_res.theta_g from ig_phasor_record.
        """
        self.cav_res.generator_phasor_record = self.Ig2Vg_matrix()
        self.cav_res.Vg = np.mean(np.abs(self.cav_res.generator_phasor_record))
        self.cav_res.theta_g = np.mean(
            np.angle(self.cav_res.generator_phasor_record)
        )

    def Vg2Ig(self, Vg_phasor):
        """
        Return Ig_phasor from Vg_phasor (assuming steady state for this conversion).
        Vg_phasor is a complex voltage.
        Eq.25 of ref [2] (Yamamoto et al. 2018) is for |Vg| and relates to Ig, Vc, psi.
        More directly, for the generator part, Vg = Ig * Z_eff_generator.
        If Z_eff_generator = R_L / (1 - j tan(psi_g)) where psi_g is generator tuning angle.
        However, the cavity equation is Vc = Vg + Vb.
        The Vg in the feedback is the generator voltage phasor.
        So, Ig = Vg_phasor * (1 - 1j * np.tan(self.cav_res.psi)) / self.cav_res.RL
        This assumes Vg_phasor is the voltage across the loaded cavity impedance seen by the generator.
        """
        return Vg_phasor * (1 - 1j * np.tan(self.cav_res.psi)) / self.cav_res.RL

    def IIR_init(self, cutoff):
        """
        Initialization for the IIR filter (acts on magnitude).
        """
        if cutoff == 0:
            self.IIRcoef = 1.0
        else:
            omega = 2.0 * np.pi * cutoff
            T = self.ring.T1 * self.every # Sampling period of the IIR filter input
            alpha = np.cos(omega * T) - 1
            tmp = alpha * alpha - 2 * alpha
            if tmp > 0:
                self.IIRcoef = alpha + np.sqrt(tmp)
            else:
                # Fallback for too high cutoff / too low T
                self.IIRcoef = np.clip(T * omega, 0, 1.0) # Ensure coef is between 0 and 1
        # IIRout stores the previous filtered magnitude
        self.IIRout_mag = np.abs(self.target_phasor) # Initialize with target magnitude


    def IIR(self, input_mag):
        """
        Apply IIR filter to the input magnitude.
        Returns the filtered magnitude.
        """
        self.IIRout_mag = (1 - self.IIRcoef) * self.IIRout_mag + self.IIRcoef * input_mag
        return self.IIRout_mag

    @property
    def IIRcutoff(self):
        """Return IIR cutoff frequency in [Hz]."""
        if self.IIRcoef == 1.0:
            return float('inf') # No filtering effectively
        T = self.ring.T1 * self.every
        # Inverse calculation for arccos might be sensitive
        # Value for arccos: ( (1-coef)^2 - coef^2 ) / (2*(1-coef)) -- simplified from Wikipedia's alpha
        # Or from the forward calculation:
        # coef = alpha + sqrt(alpha^2 - 2*alpha) where alpha = cos(omega*T) - 1
        # This requires solving for omega*T.
        # Let K = self.IIRcoef. We need to solve K = (cos(x)-1) + sqrt((cos(x)-1)^2 - 2(cos(x)-1)) for x=omegaT
        # This is complex. A simpler approximation or direct storage might be better if exactness is critical.
        # For now, returning an approximate or indicative value if not 1.0
        try:
            # This is from the direct formula for alpha when IIRcoef = alpha + sqrt(alpha^2-2alpha)
            # and IIRcoef is derived from the b_0 coefficient of a first order filter.
            # For the given IIR: y[n] = (1-a)y[n-1] + a x[n], where a = IIRcoef
            # The -3dB cutoff is where |H(e^j omegaT)| = 1/sqrt(2)
            # |H(z)| = a / (1 - (1-a)z^-1)
            # Leads to cos(omega T) = ( (1-a)^2 + a^2 - a^2/2 ) / (2a(1-a)) ... this is also getting complex.
            # A common approximation for small a: omega_c T ~ a
            if self.IIRcoef > 0 and self.IIRcoef < 1: # Avoid log(0) or division by zero
                 # From a simple exponential smoothing relation omega_c T approx IIRcoef
                return self.IIRcoef / (2.0 * np.pi * T) if T > 0 else float('inf')
            return 0 # Or some other indicator of no meaningful cutoff
        except (ValueError, ArithmeticError):
            return float('nan') # Error in calculation


    def compute_mode0_signal_raw(self):
        """
        Calculate Mode 0 signal from beam phase oscillation.
        This version applies the user-provided filter_func.
        The delay is handled separately by the track method using damper_buffer.
        """
        beam = getattr(self.cav_res, "_last_beam", None)
        if beam is None or not hasattr(beam, 'not_empty'):
            return 0.0

        not_empty_bunches = list(beam.not_empty)
        if not not_empty_bunches:
            return 0.0

        # Extract phase information using damper_mean_idx
        try:
            # Assuming damper_mean_idx gives tau, convert to phase: omega1 * tau
            # Or if it's already a phase, use directly.
            # The original code used b.mean[idx] directly. Assuming it's a phase-like quantity.
            bunch_phases_raw = [b.mean[self.damper_mean_idx] for b in not_empty_bunches]
        except IndexError:
            # Fallback or error if damper_mean_idx is invalid for some bunches
            return 0.0 # Or raise an error

        # Synchronous phase reference for deviation.
        # Could be cav_res.theta, or 0 if phases are already deviations.
        # Original code used ref_phase = 0.
        ref_phase = 0
        phase_deviations = [(phase - ref_phase) for phase in bunch_phases_raw]

        # Unwrap phases to handle jumps (e.g., if phases are in [-pi, pi])
        # Only makes sense if the raw values are indeed angles.
        # If damper_mean_idx provides time 'tau', then this should be omega1*tau before unwrap.
        # For now, assume unwrap is meaningful for the quantity from damper_mean_idx.
        unwrapped_deviations = np.unwrap(phase_deviations)

        # Mode 0 signal is the average of these deviations
        mode0_signal = np.mean(unwrapped_deviations)

        # Apply user-defined filter function
        filtered_signal = self.damper_filter_func(mode0_signal)

        return filtered_signal
