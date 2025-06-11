"""
Module for transverse space charge calculations.
"""

from scipy.constants import c

from mbtrack2.tracking.element import Element
from mbtrack2.tracking.emfields import _efieldn_mit, get_displaced_efield


class TransverseSpaceCharge(Element):
    """
    Class representing a transverse space charge element.

    Parameters
    ----------
    ring : Synchrotron
        The synchrotron object representing the particle accelerator ring.
    interaction_length : float
        The interaction length of the space charge effect in meters.
    n_bins : int, optional
        The number of bins (longitudinal) used for space charge calculations. Default is 100.

    Attributes
    ----------
    ratio_threshold : float
        The ratio numerical threshold for the space charge element to decide if the beam is transversely round.
    absolute_threshold : float
        The absolute numerical threshold for the space charge element to decide if the beam is transversely round.
    ring: Synchrotron
        Ring object with information about the ring. This class uses ring.E0 and ring.gamma for calculations.
    efieldn : function
        The electric field function.

    Methods
    -------
    track(bunch)
        Perform the tracking of the bunch through the space charge element.

    """

    ratio_threshold = 1e-3
    absolute_threshold = 1e-10

    def __init__(self, ring, interaction_length, n_bins=100):
        """
        Initialize the SpaceCharge object.

        Parameters
        ----------
        ring : Synchrotron
            The synchrotron object representing the particle accelerator ring.
        interaction_length : float
            The interaction length of the space charge effect in meters.
        n_bins : int, optional
            The number of bins (longitudinal) used for space charge calculations. Default is 100.
        """
        self.ring = ring
        self.n_bins = n_bins
        self.interaction_length = interaction_length
        self.efieldn = _efieldn_mit

    @Element.parallel
    @Element.track_bunch_if_non_empty
    def track(self, bunch):
        """
        Perform the tracking of the bunch through the space charge element.

        Parameters
        ----------
        bunch : Bunch
            The bunch of particles to be tracked.

        """
        prefactor = self.interaction_length / (self.ring.E0 *
                                               self.ring.gamma**2)
        (bins, sorted_index, profile,
         center) = bunch.binning(n_bin=self.n_bins, return_full_length=True)
        dz = (bins[1] - bins[0]) * c
        charge_density = bunch.charge_per_mp * profile / dz
        for bin_index in range(self.n_bins - 1):
            particle_ids = bin_index == sorted_index
            if bunch.track_alive:
                particle_ids = particle_ids & bunch.alive
            if len(particle_ids) == 0:
                continue
            x = bunch.particles["x"][particle_ids]
            y = bunch.particles["y"][particle_ids]

            if len(x) != 0 and len(y) != 0:
                mean_x, std_x = x.mean(), x.std()
                mean_y, std_y = y.mean(), y.std()

                en_x, en_y = get_displaced_efield(
                    self.efieldn,
                    bunch.particles["x"][particle_ids],
                    bunch.particles["y"][particle_ids],
                    std_x,
                    std_y,
                    mean_x,
                    mean_y,
                )

                kicks_x = prefactor * en_x * charge_density[bin_index]
                kicks_y = prefactor * en_y * charge_density[bin_index]

                bunch.particles["xp"][particle_ids] += kicks_x
                bunch.particles["yp"][particle_ids] += kicks_y
