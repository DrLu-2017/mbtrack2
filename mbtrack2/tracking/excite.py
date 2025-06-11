# -*- coding: utf-8 -*-
"""
Module to deal with different kinds of beam excitation.
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import chirp

from mbtrack2.tracking.element import Element
from mbtrack2.tracking.particles import Beam, Bunch


class Sweep(Element):
    """
    Element which excite the beam in between two frequencies, i.e. apply
    frequency sweep (chirp) on all or a given bunch in the chosen plane.

    If applied to a full beam, the excitation is the same (and at the same time)
    for all bunches, so it drives a growth of coupled bunch mode 0.

    Parameters
    ----------
    ring : Synchrotron
        Synchrotron object.
    f0 : float
        Initial frequency of the sweep in [Hz].
    f1 : float
        Final frequency of the sweep in [Hz].
    t1 : float
        Time duration of the sweep in [s].
    level : float
        Kick level to apply in [V].
    plane : "x", "y" or "tau"
        Plane on which to apply the kick.
    bunch_to_sweep : int, optional
        Bunch number to which the sweeping is applied.
        If None, the sweeping is applied for all bunches.
        Default is None.

    Methods
    -------
    track(bunch_or_beam)
        Tracking method for the element.
    plot()
        Plot the sweep voltage applied.

    """

    def __init__(self, ring, f0, f1, t1, level, plane, bunch_to_sweep=None):
        self.ring = ring
        self.t = np.arange(0, t1, ring.T0)
        self.N = len(self.t)
        self.count = 0
        self.level = level
        self.sweep = chirp(self.t, f0, t1, f1)
        if plane == "x":
            self.apply = "xp"
        elif plane == "y":
            self.apply = "yp"
        elif plane == "tau":
            self.apply = "delta"
        else:
            raise ValueError("plane should be 'x', 'y' or 'tau'.")
        self.bunch_to_sweep = bunch_to_sweep

    def track(self, bunch_or_beam):
        """
        Tracking method for this element.

        Parameters
        ----------
        bunch_or_beam : Bunch or Beam
        """
        if isinstance(bunch_or_beam, Bunch):
            bunch = bunch_or_beam
            self._track_bunch(bunch)
        elif isinstance(bunch_or_beam, Beam):
            beam = bunch_or_beam
            if beam.mpi_switch:
                if self.bunch_to_sweep is not None:
                    if beam.mpi.bunch_num == self.bunch_to_sweep:
                        self._track_bunch(beam[beam.mpi.bunch_num])
                else:
                    self._track_bunch(beam[beam.mpi.bunch_num])
            else:
                if self.bunch_to_sweep is not None:
                    self._track_bunch(beam[self.bunch_to_sweep])
                else:
                    for bunch in beam.not_empty:
                        self._track_bunch(bunch, False)
                    self.count += 1
                    if self.count >= self.N:
                        self.count = 0
        else:
            raise TypeError("bunch_or_beam should be a Beam or Bunch object.")

    def _track_bunch(self, bunch, count_step=True):
        """
        Tracking method for a bunch for this element.

        Parameters
        ----------
        bunch : Bunch
        """
        sweep_val = self.sweep[self.count]
        bunch[self.apply] += self.level / self.ring.E0 * sweep_val
        if count_step:
            self.count += 1
            if self.count >= self.N:
                self.count = 0

    def plot(self):
        """Plot the sweep voltage applied."""
        fig, ax = plt.subplots()
        ax.plot(self.t, self.sweep)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Sweep voltage [V]")
        return fig
