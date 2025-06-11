# -*- coding: utf-8 -*-
"""
This module defines the most basic elements for tracking, including Element,
an abstract base class which is to be used as mother class to every elements
included in the tracking.
"""

from abc import ABCMeta, abstractmethod
from copy import deepcopy
from functools import wraps

import numpy as np
from scipy.special import factorial

from mbtrack2.tracking.particles import Beam


class Element(metaclass=ABCMeta):
    """
    Abstract Element class used for subclass inheritance to define all kinds
    of objects which intervene in the tracking.
    """

    @abstractmethod
    def track(self, beam):
        """
        Track a beam object through this Element.
        This method needs to be overloaded in each Element subclass.

        Parameters
        ----------
        beam : Beam object
        """
        raise NotImplementedError

    @staticmethod
    def parallel(track):
        """
        Defines the decorator @parallel which handles the embarrassingly
        parallel case which happens when there is no bunch to bunch
        interaction in the tracking routine.

        Adding @Element.parallel allows to write the track method of the
        Element subclass for a Bunch object instead of a Beam object.

        Parameters
        ----------
        track : function, method of an Element subclass
            track method of an Element subclass which takes a Bunch object as
            input

        Returns
        -------
        track_wrapper: function, method of an Element subclass
            track method of an Element subclass which takes a Beam object or a
            Bunch object as input
        """

        @wraps(track)
        def track_wrapper(*args, **kwargs):
            if isinstance(args[1], Beam):
                self = args[0]
                beam = args[1]
                if beam.mpi_switch:
                    track(self, beam[beam.mpi.bunch_num], *args[2:], **kwargs)
                else:
                    for bunch in beam.not_empty:
                        track(self, bunch, *args[2:], **kwargs)
            else:
                self = args[0]
                bunch = args[1]
                track(self, bunch, *args[2:], **kwargs)

        return track_wrapper

    @staticmethod
    def track_bunch_if_non_empty(track):
        """
        Defines the decorator @track_bunch_if_non_empty which handles the case
        where a track method should not be called if the bunch is empty.

        Should be added only the track method defined for Bunch elements.

        Parameters
        ----------
        track : function, method of an Element subclass
            track method of an Element subclass which takes a Bunch object as
            input

        Returns
        -------
        track_wrapper: function, method of an Element subclass
            track method of an Element subclass which takes a Bunch object as
            input

        """

        @wraps(track)
        def track_wrapper(*args):
            # self = args[0]
            bunch = args[1]
            if bunch.is_empty:
                pass
            else:
                track(*args)

        return track_wrapper


class LongitudinalMap(Element):
    """
    Longitudinal map for a single turn in the synchrotron.

    Parameters
    ----------
    ring : Synchrotron object
    """

    def __init__(self, ring):
        self.ring = ring

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
        bunch["delta"] -= self.ring.U0 / self.ring.E0
        bunch["tau"] += self.ring.eta(
            bunch["delta"]) * self.ring.T0 * bunch["delta"]


class SynchrotronRadiation(Element):
    """
    Element to handle synchrotron radiation, radiation damping and quantum
    excitation, for a single turn in the synchrotron.

    Parameters
    ----------
    ring : Synchrotron object
    switch : bool array of shape (3,), optional
        If False in one plane (long, x, y), the synchrotron radiation is turned
        off.
        The default is True, in all three planes.
    qexcitation : bool, optional
        If False, the quantum excitation is turned off.
        The default is True.

    """

    def __init__(self,
                 ring,
                 switch=np.ones((3, ), dtype=bool),
                 qexcitation=True):
        self.ring = ring
        self.switch = switch
        self.qexcitation = qexcitation

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
        N = len(bunch)

        excitation = 0
        if self.switch[0]:

            if self.qexcitation:
                rand = np.random.standard_normal(size=N)
                excitation = (2 * self.ring.sigma_delta *
                              (self.ring.T0 / self.ring.tau[2])**0.5 * rand)

            bunch["delta"] = (1 - 2 * self.ring.T0 /
                              self.ring.tau[2]) * bunch["delta"] + excitation

        if self.switch[1]:

            if self.qexcitation:
                rand = np.random.standard_normal(size=N)
                excitation = (2 * self.ring.sigma()[1] *
                              (self.ring.T0 / self.ring.tau[0])**0.5 * rand)

            bunch["xp"] = (1 - 2 * self.ring.T0 /
                           self.ring.tau[0]) * bunch["xp"] + excitation

        if self.switch[2]:

            if self.qexcitation:
                rand = np.random.standard_normal(size=N)
                excitation = (2 * self.ring.sigma()[3] *
                              (self.ring.T0 / self.ring.tau[1])**0.5 * rand)

            bunch["yp"] = (1 - 2 * self.ring.T0 /
                           self.ring.tau[1]) * bunch["yp"] + excitation


class SkewQuadrupole:
    """
    Thin skew quadrupole element used to introduce betatron coupling (the
    length of the quadrupole is neglected).

    Parameters
    ----------
    strength : float
        Integrated strength of the skew quadrupole [m].

    """

    def __init__(self, strength):
        self.strength = strength

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

        bunch["xp"] = bunch["xp"] - self.strength * bunch["y"]
        bunch["yp"] = bunch["yp"] - self.strength * bunch["x"]


class TransverseMapSector(Element):
    """
    Transverse map for a sector of the synchrotron, from an initial
    position s0 to a final position s1.

    Parameters
    ----------
    ring : Synchrotron object
        Ring parameters.
    alpha0 : array of shape (2,)
        Alpha Twiss function at the initial location of the sector.
    beta0 : array of shape (2,)
        Beta Twiss function at the initial location of the sector.
    dispersion0 : array of shape (4,)
        Dispersion function at the initial location of the sector.
    alpha1: array of shape (2,)
        Alpha Twiss function at the final location of the sector.
    beta1 : array of shape (2,)
        Beta Twiss function at the final location of the sector.
    dispersion1 : array of shape (4,)
        Dispersion function at the final location of the sector.
    phase_diff : array of shape (2,)
        Phase difference between the initial and final location of the
        sector.
    chro_diff : array of shape (2,)
        Chromaticity difference between the initial and final location of
        the sector.
    adts : array of shape (4,), optional
        Amplitude-dependent tune shift of the sector, see Synchrotron class
        for details. The default is None.

    """

    def __init__(
        self,
        ring,
        alpha0,
        beta0,
        dispersion0,
        alpha1,
        beta1,
        dispersion1,
        phase_diff,
        chro_diff,
        adts=None,
    ):
        self.ring = ring
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.gamma0 = (1 + self.alpha0**2) / self.beta0
        self.dispersion0 = dispersion0
        self.alpha1 = alpha1
        self.beta1 = beta1
        self.gamma1 = (1 + self.alpha1**2) / self.beta1
        self.dispersion1 = dispersion1
        self.tune_diff = phase_diff / (2 * np.pi)
        self.chro_diff = chro_diff
        if adts is not None:
            self.adts_poly = [
                np.poly1d(adts[0]),
                np.poly1d(adts[1]),
                np.poly1d(adts[2]),
                np.poly1d(adts[3]),
            ]
        else:
            self.adts_poly = None

    def _compute_chromatic_tune_advances(self, bunch):
        order = len(self.chro_diff) // 2
        if order == 1:
            tune_advance_x = self.chro_diff[0] * bunch["delta"]
            tune_advance_y = self.chro_diff[1] * bunch["delta"]
        elif order == 2:
            tune_advance_x = (self.chro_diff[0] * bunch["delta"] +
                              self.chro_diff[2] / 2 * bunch["delta"]**2)
            tune_advance_y = (self.chro_diff[1] * bunch["delta"] +
                              self.chro_diff[3] / 2 * bunch["delta"]**2)
        elif order == 3:
            tune_advance_x = (self.chro_diff[0] * bunch["delta"] +
                              self.chro_diff[2] / 2 * bunch["delta"]**2 +
                              self.chro_diff[4] / 6 * bunch["delta"]**3)
            tune_advance_y = (self.chro_diff[1] * bunch["delta"] +
                              self.chro_diff[3] / 2 * bunch["delta"]**2 +
                              self.chro_diff[5] / 6 * bunch["delta"]**3)
        elif order == 4:
            tune_advance_x = (self.chro_diff[0] * bunch["delta"] +
                              self.chro_diff[2] / 2 * bunch["delta"]**2 +
                              self.chro_diff[4] / 6 * bunch["delta"]**3 +
                              self.chro_diff[6] / 24 * bunch["delta"]**4)
            tune_advance_y = (self.chro_diff[1] * bunch["delta"] +
                              self.chro_diff[3] / 2 * bunch["delta"]**2 +
                              self.chro_diff[5] / 6 * bunch["delta"]**3 +
                              self.chro_diff[7] / 24 * bunch["delta"]**4)
        else:
            coefs = np.array([1 / factorial(i) for i in range(order + 1)])
            coefs[0] = 0
            self.chro_diff = np.concatenate(([0, 0], self.chro_diff))
            tune_advance_x = np.polynomial.polynomial.Polynomial(
                self.chro_diff[::2] * coefs)(bunch["delta"])
            tune_advance_y = np.polynomial.polynomial.Polynomial(
                self.chro_diff[1::2] * coefs)(bunch["delta"])
        return tune_advance_x, tune_advance_y

    def _compute_new_coords(self, bunch, tune_advance, plane):
        if plane == "x":
            i, j, coord, mom = 0, 0, "x", "xp"
        elif plane == "y":
            i, j, coord, mom = 1, 2, "y", "yp"
        else:
            raise ValueError("plane should be either x or y")
        c_u = np.cos(2 * np.pi * tune_advance)
        s_u = np.sin(2 * np.pi * tune_advance)
        M00 = np.sqrt(
            self.beta1[i] / self.beta0[i]) * (c_u + self.alpha0[i] * s_u)
        M01 = np.sqrt(self.beta0[i] * self.beta1[i]) * s_u
        M02 = (self.dispersion1[j] - M00 * self.dispersion0[j] -
               M01 * self.dispersion0[j + 1])
        M10 = ((self.alpha0[i] - self.alpha1[i]) * c_u -
               (1 + self.alpha0[i] * self.alpha1[i]) * s_u) / np.sqrt(
                   self.beta0[i] * self.beta1[i])
        M11 = np.sqrt(
            self.beta0[i] / self.beta1[i]) * (c_u - self.alpha1[i] * s_u)
        M12 = (self.dispersion1[j + 1] - M10 * self.dispersion0[j] -
               M11 * self.dispersion0[j + 1])
        u = M00 * bunch[coord] + M01 * bunch[mom] + M02 * bunch["delta"]
        up = M10 * bunch[coord] + M11 * bunch[mom] + M12 * bunch["delta"]
        return u, up

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
        tune_advance_x = self.tune_diff[0]
        tune_advance_y = self.tune_diff[1]
        # Compute tune advance which depends on energy via chromaticity and
        # ADTS
        if (np.array(self.chro_diff) != 0).any():
            tune_advance_x_chro, tune_advance_y_chro = (
                self._compute_chromatic_tune_advances(bunch))
            tune_advance_x += tune_advance_x_chro
            tune_advance_y += tune_advance_y_chro

        if self.adts_poly is not None:
            Jx = ((self.gamma0[0] * bunch["x"]**2) +
                  (2 * self.alpha0[0] * bunch["x"] * bunch["xp"]) +
                  (self.beta0[0] * bunch["xp"]**2))
            Jy = ((self.gamma0[1] * bunch["y"]**2) +
                  (2 * self.alpha0[1] * bunch["y"] * bunch["yp"]) +
                  (self.beta0[1] * bunch["yp"]**2))
            tune_advance_x += self.adts_poly[0](Jx) + self.adts_poly[2](Jy)
            tune_advance_y += self.adts_poly[1](Jx) + self.adts_poly[3](Jy)

        bunch["x"], bunch["xp"] = self._compute_new_coords(
            bunch, tune_advance_x, "x")
        bunch["y"], bunch["yp"] = self._compute_new_coords(
            bunch, tune_advance_y, "y")


class TransverseMap(TransverseMapSector):
    """
    Transverse map for a single turn in the synchrotron.

    Parameters
    ----------
    ring : Synchrotron object
    """

    def __init__(self, ring):
        super().__init__(
            ring,
            ring.optics.local_alpha,
            ring.optics.local_beta,
            ring.optics.local_dispersion,
            ring.optics.local_alpha,
            ring.optics.local_beta,
            ring.optics.local_dispersion,
            2 * np.pi * ring.tune,
            ring.chro,
            ring.adts,
        )


def transverse_map_sector_generator(ring, positions):
    """
    Convenience function which generate a list of TransverseMapSector elements
    from a ring:
        - if an AT lattice is loaded, the optics functions and chromaticity is
        computed at the given positions.
        - if no AT lattice is loaded, the local optics are used everywhere.

    Tracking through all the sectors is equivalent to a full turn (and thus to
    the TransverseMap object).

    Parameters
    ----------
    ring : Synchrotron object
        Ring parameters.
    positions : array
        List of longitudinal positions in [m] to use as starting and end points
        of the TransverseMapSector elements.
        The array should contain the initial position (s=0) but not the end
        position (s=ring.L), so like position = np.array([0, pos1, pos2, ...]).

    Returns
    -------
    sectors : list
        List of TransverseMapSector elements.

    """
    N_sec = len(positions)
    sectors = []
    if hasattr(ring, "adts") and ring.adts is not None:
        adts = np.array([val / N_sec for val in ring.adts])
    else:
        adts = None
    if ring.optics.use_local_values:
        for i in range(N_sec):
            sectors.append(
                TransverseMapSector(
                    ring,
                    ring.optics.local_alpha,
                    ring.optics.local_beta,
                    ring.optics.local_dispersion,
                    ring.optics.local_alpha,
                    ring.optics.local_beta,
                    ring.optics.local_dispersion,
                    2 * np.pi * ring.tune / N_sec,
                    np.asarray(ring.chro) / N_sec,
                    adts=adts,
                ))
    else:
        import at

        def _compute_chro(ring, N_sec, dp=1e-2, order=4):
            lat = deepcopy(ring.optics.lattice)
            lat.append(at.Marker("END"))
            fit, _, _ = at.physics.nonlinear.chromaticity(lat,
                                                          method="linopt",
                                                          dpm=dp,
                                                          n_points=100,
                                                          order=order)
            Chrox, Chroy = fit[0, 1:], fit[1, 1:]
            chrox = np.array([np.linspace(0, val, N_sec) for val in Chrox])
            chroy = np.array([np.linspace(0, val, N_sec) for val in Chroy])
            return np.array([chrox, chroy])

        _chro = _compute_chro(ring, N_sec)
        for i in range(N_sec):
            alpha0 = ring.optics.alpha(positions[i])
            beta0 = ring.optics.beta(positions[i])
            dispersion0 = ring.optics.dispersion(positions[i])
            mu0 = ring.optics.mu(positions[i])
            chro0 = _chro[:, i]
            if i != (N_sec - 1):
                alpha1 = ring.optics.alpha(positions[i + 1])
                beta1 = ring.optics.beta(positions[i + 1])
                dispersion1 = ring.optics.dispersion(positions[i + 1])
                mu1 = ring.optics.mu(positions[i + 1])
                chro1 = _chro[:, i + 1]
            else:
                alpha1 = ring.optics.alpha(positions[0])
                beta1 = ring.optics.beta(positions[0])
                dispersion1 = ring.optics.dispersion(positions[0])
                mu1 = ring.optics.mu(ring.L)
                chro1 = _chro[:, -1]
            phase_diff = mu1 - mu0
            chro_diff = chro1 - chro0
            sectors.append(
                TransverseMapSector(
                    ring,
                    alpha0,
                    beta0,
                    dispersion0,
                    alpha1,
                    beta1,
                    dispersion1,
                    phase_diff,
                    chro_diff,
                    adts=adts,
                ))
    return sectors
