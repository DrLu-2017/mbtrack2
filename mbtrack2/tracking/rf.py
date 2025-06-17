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
        Also sets the generator phasor such that the total cavity voltage
        matches self.Vc and self.theta immediately after this method.

        No modifications on the Beam object.

        Parameters
        ----------
        beam : Beam object

        References
        ----------
        [1] mbtrack2 manual.

        """

        if self.tracking is False:
            self.init_tracking(beam)

        self.beam_phasor = 0 + 0j
        self.beam_phasor_record = np.zeros((self.ring.h), dtype=complex)

        if len(self.valid_bunch_index) == 0:
            target_cavity_phasor = self.Vc * np.exp(1j * self.theta)
            self.Vg = np.abs(target_cavity_phasor)
            self.theta_g = np.angle(target_cavity_phasor)
            current_generator_phasor = self.Vg * np.exp(1j * self.theta_g)
            self.generator_phasor_record = np.ones(self.ring.h, dtype=complex) * current_generator_phasor
            return

        N = self.n_bin
        delta_freq_term = (self.wr - self.m * self.ring.omega1)

        n_turn = 1
        if self.ring.T0 > 1e-15 and self.filling_time > 1e-15:
            n_turn = max(1, int(self.filling_time / self.ring.T0 * 10))

        profile_all_buckets = np.zeros((N, self.ring.h))
        bin_length_all_buckets = np.zeros(self.ring.h)
        charge_per_mp_all_buckets = np.zeros(self.ring.h)

        for ring_idx in self.valid_bunch_index:
            bunch = beam[ring_idx] # CORRECTED LINE

            if bunch is None:
                bin_length_all_buckets[ring_idx] = 0.0; charge_per_mp_all_buckets[ring_idx] = 0.0; continue

            if beam.mpi_switch:
                try:
                    mpi_local_idx = list(self.valid_bunch_index).index(ring_idx)
                except ValueError:
                    bin_length_all_buckets[ring_idx] = 0.0; charge_per_mp_all_buckets[ring_idx] = 0.0; continue

                mpi_attrs_present = hasattr(beam.mpi, 'tau_center') and beam.mpi.tau_center is not None and \
                                  hasattr(beam.mpi, 'tau_profile') and beam.mpi.tau_profile is not None and \
                                  hasattr(beam.mpi, 'tau_bin_length') and beam.mpi.tau_bin_length is not None and \
                                  hasattr(beam.mpi, 'charge_per_mp_all') and beam.mpi.charge_per_mp_all is not None

                needs_share = not mpi_attrs_present or \
                              mpi_local_idx >= len(beam.mpi.tau_center) or \
                              mpi_local_idx >= len(beam.mpi.tau_profile) or \
                              mpi_local_idx >= len(beam.mpi.tau_bin_length) or \
                              mpi_local_idx >= len(beam.mpi.charge_per_mp_all)

                if needs_share:
                     beam.mpi.share_distributions(beam, n_bin=self.n_bin)

                if hasattr(beam.mpi, 'tau_profile') and beam.mpi.tau_profile is not None and mpi_local_idx < len(beam.mpi.tau_profile) and \
                   hasattr(beam.mpi, 'tau_bin_length') and beam.mpi.tau_bin_length is not None and mpi_local_idx < len(beam.mpi.tau_bin_length) and \
                   hasattr(beam.mpi, 'charge_per_mp_all') and beam.mpi.charge_per_mp_all is not None and mpi_local_idx < len(beam.mpi.charge_per_mp_all):
                    profile_all_buckets[:, ring_idx] = beam.mpi.tau_profile[mpi_local_idx]
                    bin_length_all_buckets[ring_idx] = float(beam.mpi.tau_bin_length[mpi_local_idx][0])
                    charge_per_mp_all_buckets[ring_idx] = float(beam.mpi.charge_per_mp_all[mpi_local_idx])
                else:
                    bin_length_all_buckets[ring_idx] = 0.0; charge_per_mp_all_buckets[ring_idx] = 0.0; continue
            else:
                current_bins_1d, _, current_profile_1d, _ = bunch.binning(n_bin=self.n_bin)
                profile_all_buckets[:, ring_idx] = current_profile_1d
                bin_length_all_buckets[ring_idx] = (current_bins_1d[1] - current_bins_1d[0]) if len(current_bins_1d) > 1 else 0.0
                charge_per_mp_all_buckets[ring_idx] = bunch.charge_per_mp

        k_indices = np.arange(0, N)
        Tkj_eff = np.zeros((N, len(self.valid_bunch_index)))

        for i, current_ring_idx in enumerate(self.valid_bunch_index):
            if bin_length_all_buckets[current_ring_idx] < 1e-15:
                Tkj_eff[:, i] = 0.0
                continue

            time_from_slice_k_to_bunch_end = (N - 1.0 - k_indices) * bin_length_all_buckets[current_ring_idx]
            time_after_current_bunch_to_eot_ref = 0.0

            if len(self.valid_bunch_index) > 1:
                temp_sum_ring_idx = current_ring_idx
                current_enumerated_idx_in_valid = i
                for _iteration_count in range(len(self.valid_bunch_index) - 1):
                    time_to_next_active_bunch_start = self.distance[temp_sum_ring_idx] * self.ring.T1
                    time_after_current_bunch_to_eot_ref += time_to_next_active_bunch_start

                    current_enumerated_idx_in_valid = (current_enumerated_idx_in_valid + 1) % len(self.valid_bunch_index)
                    next_active_ring_idx_in_sum = self.valid_bunch_index[current_enumerated_idx_in_valid]

                    if next_active_ring_idx_in_sum == current_ring_idx:
                        break

                    if bin_length_all_buckets[next_active_ring_idx_in_sum] > 1e-15:
                        time_after_current_bunch_to_eot_ref += N * bin_length_all_buckets[next_active_ring_idx_in_sum]
                    temp_sum_ring_idx = next_active_ring_idx_in_sum

            Tkj_eff[:, i] = time_from_slice_k_to_bunch_end + time_after_current_bunch_to_eot_ref

        active_profile = profile_all_buckets[:, self.valid_bunch_index]
        active_charge_per_mp = charge_per_mp_all_buckets[self.valid_bunch_index]

        exp_decay_part = -1.0 / self.filling_time if abs(self.filling_time) > 1e-15 else 0.0

        exp_factor_complex = exp_decay_part + 1j * delta_freq_term
        var = np.exp(exp_factor_complex * Tkj_eff)

        term_to_sum = (active_profile * active_charge_per_mp[np.newaxis, :]) * var
        sum_tot = np.sum(term_to_sum)

        self.beam_phasor = 0j
        for _ in range(n_turn):
            self.phasor_decay(self.ring.T0, ref_frame="rf")
            sum_val = -2.0 * sum_tot * self.loss_factor
            self.beam_phasor += sum_val

        self.beam_phasor_record.fill(self.beam_phasor)

        target_cavity_phasor = self.Vc * np.exp(1j * self.theta)
        required_generator_phasor = target_cavity_phasor - self.beam_phasor

        self.Vg = np.abs(required_generator_phasor)
        self.theta_g = np.angle(required_generator_phasor)

        current_generator_phasor = self.Vg * np.exp(1j * self.theta_g)
        self.generator_phasor_record = np.ones(self.ring.h, dtype=complex) * current_generator_phasor

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
    Proportional Integral (PI) loop to control a CavityResonator amplitude and
    phase via generator current (ig) using separate I/Q signal components.

    Feedback reference targets (setpoints) are cav_res.Vc and cav_res.theta.

    The ProportionalIntegralIQLoop should be initialized only after generator
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
        system in the form of a list [Pgain_I, Pgain_Q, Igain_I, Igain_Q].
        If only two values are provided, Pgain_I = Pgain_Q and Igain_I = Igain_Q.
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
        Default is True.
    target_I : float, optional
        Target I component of the cavity voltage. If None, calculated from cav_res.Vc and cav_res.theta.
    target_Q : float, optional
        Target Q component of the cavity voltage. If None, calculated from cav_res.Vc and cav_res.theta.

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
    """

    def __init__(
        self,
        ring,
        cav_res,
        gain,
        sample_num,
        every,
        delay,
        IIR_cutoff=0,
        FF=True,
        target_I=None,
        target_Q=None,
    ):
        self.ring = ring
        self.cav_res = cav_res
        self.Ig2Vg_mat = np.zeros((self.ring.h, self.ring.h), dtype=complex)
        self.ig_modulation_signal = np.zeros(self.ring.h, dtype=complex)

        if len(gain) == 2:
            self.gain_I = gain[0]  # Proportional gain for I component
            self.gain_Q = gain[0]  # Proportional gain for Q component
            self.Igain_I = gain[1]  # Integral gain for I component
            self.Igain_Q = gain[1]  # Integral gain for Q component
        elif len(gain) == 4:
            self.gain_I = gain[0]
            self.gain_Q = gain[1]
            self.Igain_I = gain[2]
            self.Igain_Q = gain[3]
        else:
            raise ValueError("gain must be a list of 2 or 4 floats")

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

        if target_I is None or target_Q is None:
            self.target_I = self.cav_res.Vc * np.cos(self.cav_res.theta)
            self.target_Q = self.cav_res.Vc * np.sin(self.cav_res.theta)
        else:
            self.target_I = target_I
            self.target_Q = target_Q

        # Initialize feedback variables
        self.ig_phasor = np.ones(self.ring.h, dtype=complex) * self.Vg2Ig(
            self.cav_res.generator_phasor)
        self.ig_phasor_record = self.ig_phasor
        self.vc_previous = np.ones(
            self.sample_num) * self.cav_res.cavity_phasor

        # Separate I/Q error records
        self.diff_record_I = np.zeros(record_size)
        self.diff_record_Q = np.zeros(record_size)
        self.I_record_I = 0.0  # Integral term for I component
        self.I_record_Q = 0.0  # Integral term for Q component

        self.sample_list = range(0, self.ring.h, self.every)

        self.IIR_init(IIR_cutoff)
        self.init_FFconst()
        self.init_Ig2Vg_matrix()

    def track(self, apply_changes=True):
        """Tracking method for the Cavity PI control feedback using IQ components."""

        vc_list = np.concatenate(
            [self.vc_previous, self.cav_res.cavity_phasor_record])
        self.ig_phasor.fill(self.ig_phasor[-1])

        for index in self.sample_list:
            # get cavity voltage
            mean_vc = np.mean(vc_list[index:self.sample_num + index])
            # Apply IIR filter
            mean_vc_filtered = self.IIR(mean_vc) # Apply IIR to the complex voltage

            # calcul I/Q errors from filtered complex voltage
            error_I = self.target_I - mean_vc_filtered.real
            error_Q = self.target_Q - mean_vc_filtered.imag

            # update integral terms
            self.I_record_I += error_I / self.ring.f1
            self.I_record_Q += error_Q / self.ring.f1

            # calculate feedback values (PI control for both I and Q)
            fb_I = self.gain_I * error_I + self.Igain_I * self.I_record_I
            fb_Q = self.gain_Q * error_Q + self.Igain_Q * self.I_record_Q

            fb_value = fb_I + 1j*fb_Q

            self.ig_phasor[index:] = self.Vg2Ig(fb_value) + self.FFconst

        # update next turn
        self.sample_list = range(index + self.every - self.ring.h, self.ring.h,
                                 self.every)
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
            # Initialize IIRout with a complex value based on target I and Q
            self.IIRout = self.target_I + 1j * self.target_Q
        else:
            omega = 2.0 * np.pi * cutoff
            T = self.ring.T1 * self.every
            alpha = np.cos(omega * T) - 1
            tmp = alpha*alpha - 2*alpha
            if tmp > 0:
                self.IIRcoef = alpha + np.sqrt(tmp)
            else:
                # Fallback if parameters lead to non-real sqrt
                # This is a simple heuristic, might need adjustment
                self.IIRcoef = T * omega # Simplified from T * cutoff * 2 * np.pi
            # Initialize IIRout with a complex value
            self.IIRout = self.target_I + 1j * self.target_Q


    def IIR(self, input_complex): # input is complex
        """Return IIR filter output."""
        # Ensure IIRout is complex. If it's real (e.g. from old init), make it complex.
        if not isinstance(self.IIRout, complex):
            self.IIRout = complex(self.IIRout, 0)
        self.IIRout = (1 - self.IIRcoef) * self.IIRout + self.IIRcoef * input_complex
        return self.IIRout

    @property
    def IIRcutoff(self):
        """Return IIR cutoff frequency in [Hz]."""
        if self.IIRcoef == 1.0: # Means cutoff was 0
            return 0.0
        T = self.ring.T1 * self.every
        # Ensure the argument for arccos is within [-1, 1]
        arccos_arg = (2 - 2 * self.IIRcoef - self.IIRcoef * self.IIRcoef) / (2 * (1 - self.IIRcoef))
        if arccos_arg > 1.0:
            arccos_arg = 1.0
        elif arccos_arg < -1.0:
            arccos_arg = -1.0
        return (1.0 / 2.0 / np.pi / T * np.arccos(arccos_arg))


class Mode0Damper:
    """
    A Mode 0 damper to suppress coherent bunch oscillations.

    This class calculates a phase correction based on the average phase
    deviation of the bunches (Mode 0 oscillation) and applies proportional
    and integral control.
    """

    def __init__(
        self,
        ring,
        cav_res,
        damper_gain,
        damper_delay,
        filter_func=None,
        mean_idx=4, # Default from ProportionalIntegralIQLoopMode0Damper
    ):
        self.ring = ring
        self.cav_res = cav_res  # CavityResonator instance
        self.gain_mode0_proportional = damper_gain[0]
        self.gain_mode0_integral = damper_gain[1]
        self.damper_delay = int(damper_delay)
        self.filter_func = filter_func if filter_func else lambda x: x
        self.mean_idx = mean_idx

        self.buffer = []  # Buffer for implementing delay
        self.mode0_integral = 0.0  # Integral term for Mode 0 damper

    def compute_mode0_signal(self):
        """
        Calculate Mode 0 signal from beam phase oscillation.
        This method is copied from ProportionalIntegralIQLoopMode0Damper.
        """
        # Get beam reference from the CavityResonator
        # It's assumed CavityResonator's track method stores the last processed beam
        # in an attribute like _last_beam. This needs to be ensured by CavityResonator.
        beam = getattr(self.cav_res, "_last_beam", None)
        if beam is None:
            # Try to get beam from ring if not in cav_res (e.g. if cav_res is a PI loop)
            # This is a fallback and might not be the most robust way
            beam = getattr(self.ring, "beam", None)
            if beam is None:
                 return 0.0


        not_empty_bunches = list(beam.not_empty)
        if len(not_empty_bunches) == 0:
            return 0.0

        # Calculate phase deviation from synchronous phase
        # For Mode 0, we are interested in the average phase of the beam centroid.
        # The exact definition of "phase" can vary. Here, using `bunch.mean[self.mean_idx]`
        # which was the practice in ProportionalIntegralIQLoopMode0Damper.
        # A more direct phase might be `self.ring.m * self.ring.omega1 * bunch.mean["tau"]`
        # but sticking to the original implementation's choice for `mean_idx`.
        ref_phase = 0 # Assuming synchronous phase is 0 for deviation calculation
        bunch_phases = [b.mean[self.mean_idx] for b in not_empty_bunches]
        phase_deviations = [(phase - ref_phase) for phase in bunch_phases]

        # Unwrap phases to handle phase jumps correctly if phases are angular
        # If mean_idx refers to tau directly, unwrap might not be needed or interpreted differently
        # Given it's from phase space, unwrap is likely relevant.
        phase_deviations = np.unwrap(phase_deviations)

        # Calculate mode 0 signal (average phase deviation)
        mode0_signal_raw = np.mean(phase_deviations)

        # Apply filter if specified
        filtered_signal = self.filter_func(mode0_signal_raw)

        # Implement delay buffer
        self.buffer.append(filtered_signal)
        if len(self.buffer) > self.damper_delay:
            delayed_signal = self.buffer.pop(0)
        else:
            # If buffer is not full yet, effectively the signal is zero or not ready
            delayed_signal = 0.0

        return delayed_signal

    def get_damper_correction(self):
        """
        Calculates and returns the phase shift correction from the Mode 0 damper.
        """
        mode0_signal = self.compute_mode0_signal()

        # Proportional part
        fb_phase_shift_proportional = self.gain_mode0_proportional * mode0_signal

        # Integral part
        self.mode0_integral += (mode0_signal *
                                self.gain_mode0_integral /
                                self.ring.f1)

        # Total correction
        fb_phase_shift = fb_phase_shift_proportional + self.mode0_integral

        # Apply clipping (saturation)
        fb_phase_shift = np.clip(fb_phase_shift, -np.pi / 12, np.pi / 12)  # Limit to ±15 degrees

        return fb_phase_shift


class ProportionalIntegralIQLoopMode0Damper:
    """
    A PI controller using IQ components with an optional Mode 0 damper for RF cavity control.
    This class now acts as a wrapper, composing ProportionalIntegralIQLoop and Mode0Damper.
    """

    def __init__(
        self,
        ring,
        cav_res,
        gain,
        sample_num,
        every,
        delay,
        damper_delay=None,
        IIR_cutoff=0,
        FF=True,
        enable_damper=False,
        damper_gain=[0, 0],
        filter_func=None,
        mean_idx=4,
    ):
        self.ring = ring
        self.cav_res = cav_res
        self.enable_damper = enable_damper
        self.ig_modulation_signal = np.zeros(self.ring.h, dtype=complex)

        self.pi_loop = ProportionalIntegralIQLoop(
            ring=self.ring,
            cav_res=self.cav_res,
            gain=gain, # Pass the original gain, PIQL will handle 2 or 4 floats
            sample_num=sample_num,
            every=every,
            delay=delay,
            IIR_cutoff=IIR_cutoff,
            FF=FF
        )

        if self.enable_damper:
            actual_damper_delay = damper_delay if damper_delay is not None else delay
            self.damper = Mode0Damper(
                ring=self.ring,
                cav_res=self.cav_res, # Mode0Damper needs cav_res for _last_beam
                damper_gain=damper_gain,
                damper_delay=actual_damper_delay,
                filter_func=filter_func,
                mean_idx=mean_idx
            )
        else:
            self.damper = None

    def track(self, apply_changes=True):
        """Tracking method for the Cavity PI control feedback with optional Mode 0 Damper."""

        # Run the core PI loop first to determine the base ig_phasor
        # apply_changes is set to False because we might modify ig_phasor further
        self.pi_loop.track(apply_changes=False)

        if self.damper is not None: # self.enable_damper can also be used
            correction_angle = self.damper.get_damper_correction()

            # The correction should be applied to the feedback component of ig_phasor,
            # not the feed-forward part (FFconst).
            component_to_rotate = self.pi_loop.ig_phasor - self.pi_loop.FFconst
            rotated_component = component_to_rotate * np.exp(1j * correction_angle)
            self.pi_loop.ig_phasor = rotated_component + self.pi_loop.FFconst
            self.pi_loop.ig_phasor_record = self.pi_loop.ig_phasor # Update record

        # Apply any external modulation signal
        self.pi_loop.ig_phasor = self.pi_loop.ig_phasor + self.ig_modulation_signal
        self.pi_loop.ig_phasor_record = self.pi_loop.ig_phasor # Update record again

        # Finally, apply the (potentially modified) ig_phasor to the cavity
        if apply_changes:
            self.pi_loop.Ig2Vg()
