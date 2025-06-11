"""
Module implementing necessary functionalities for beam-ion interactions.
Classes:
BeamIonElement
IonMonitor
IonAperture
IonParticles
"""

from abc import ABCMeta
from functools import wraps
from itertools import count

import h5py as hp
import numpy as np
from numpy.random import choice, normal, uniform
from scipy.constants import c, e

from mbtrack2.tracking.aperture import ElipticalAperture
from mbtrack2.tracking.element import Element
from mbtrack2.tracking.emfields import _efieldn_mit, get_displaced_efield
from mbtrack2.tracking.monitors import Monitor
from mbtrack2.tracking.particles import Beam, Bunch


class IonMonitor(Monitor, metaclass=ABCMeta):
    """
    A class representing an ion monitor.

    Parameters
    ----------
    save_every : int
        The number of steps between each save operation.
    buffer_size : int
        The size of the buffer to store intermediate data.
    total_size : int
        The total number of steps to be simulated.
    file_name : str, optional
        The name of the HDF5 file to store the data. If not provided, a new
        file will be created.
        Defaults to None.

    Methods
    -------
    monitor_init(group_name, save_every, buffer_size, total_size, dict_buffer,
                 dict_file, file_name=None, dict_dtype=None)
        Initialize the monitor object.
    track(bunch)
        Tracking method for the element.

    Raises
    ------
    ValueError
        If total_size is not divisible by buffer_size.
    """

    _n_monitors = count(0)
    file = None

    def __init__(self, save_every, buffer_size, total_size, file_name=None):
        group_name = "IonData_" + str(next(self._n_monitors))
        dict_buffer = {
            "mean": (4, buffer_size),
            "std": (4, buffer_size),
            "charge": (buffer_size, ),
            "mp_number": (buffer_size, ),
        }
        dict_file = {
            "mean": (4, total_size),
            "std": (4, total_size),
            "charge": (total_size, ),
            "mp_number": (buffer_size, ),
        }
        self.monitor_init(
            group_name,
            save_every,
            buffer_size,
            total_size,
            dict_buffer,
            dict_file,
            file_name,
        )

        self.dict_buffer = dict_buffer
        self.dict_file = dict_file

    def monitor_init(
        self,
        group_name,
        save_every,
        buffer_size,
        total_size,
        dict_buffer,
        dict_file,
        file_name=None,
        dict_dtype=None,
    ):
        """
        Initialize the monitor object.

        Parameters
        ----------
        group_name : str
            The name of the HDF5 group to store the data.
        save_every : int
            The number of steps between each save operation.
        buffer_size : int
            The size of the buffer to store intermediate data.
        total_size : int
            The total number of steps to be simulated.
        dict_buffer : dict
            A dictionary containing the names and sizes of the attribute
            buffers.
        dict_file : dict
            A dictionary containing the names and shapes of the datasets to be
            created.
        file_name : str, optional
            The name of the HDF5 file to store the data. If not provided, a new
            file will be created.
            Defaults to None.
        dict_dtype : dict, optional
            A dictionary containing the names and data types of the datasets.
            Defaults to None.

        Raises
        ------
        ValueError
            If total_size is not divisible by buffer_size.
        """
        if self.file is None:
            self.file = hp.File(file_name, "a", libver="earliest")

        self.group_name = group_name
        self.save_every = int(save_every)
        self.total_size = int(total_size)
        self.buffer_size = int(buffer_size)
        if total_size % buffer_size != 0:
            raise ValueError("total_size must be divisible by buffer_size.")
        self.buffer_count = 0
        self.write_count = 0
        self.track_count = 0

        # setup attribute buffers from values given in dict_buffer
        for key, value in dict_buffer.items():
            if dict_dtype is None:
                self.__setattr__(key, np.zeros(value))
            else:
                self.__setattr__(key, np.zeros(value, dtype=dict_dtype[key]))
        self.time = np.zeros((self.buffer_size, ), dtype=int)
        # create HDF5 groups and datasets to save data from group_name and
        # dict_file
        self.g = self.file.require_group(self.group_name)
        self.g.require_dataset("time", (self.total_size, ), dtype=int)
        for key, value in dict_file.items():
            if dict_dtype is None:
                self.g.require_dataset(key, value, dtype=float)
            else:
                self.g.require_dataset(key, value, dtype=dict_dtype[key])

        # create a dictionary which handle slices
        slice_dict = {}
        for key, value in dict_file.items():
            slice_dict[key] = []
            for i in range(len(value) - 1):
                slice_dict[key].append(slice(None))
        self.slice_dict = slice_dict

    def track(self, object_to_save):
        if self.track_count % self.save_every == 0:
            self.to_buffer(object_to_save)
        self.track_count += 1


class IonAperture(ElipticalAperture):
    """
    Class representing an ion aperture.

    Inherits from ElipticalAperture. Unlike in ElipticalAperture, ions are
    removed from IonParticles instead of just being flagged as not "alive".
    For beam-ion simulations there are too many lost particles and it is better
    to remove them.

    Attributes
    ----------
    X_radius_squared : float
        The squared radius of the aperture in the x-direction.
    Y_radius_squared : float
        The squared radius of the aperture in the y-direction.

    Methods
    -------
    track(bunch)
        Tracking method for the element.

    """

    @Element.parallel
    def track(self, bunch):
        """
        Tracking method for the element.
        No bunch to bunch interaction, so written for Bunch objects and
        @Element.parallel is used to handle Beam objects.

        Parameters
        ----------
        bunch : Bunch or Beam object
            The bunch object to be tracked.

        """
        alive = (bunch.particles["x"]**2 / self.X_radius_squared +
                 bunch.particles["y"]**2 / self.Y_radius_squared <= 1)
        for stat in bunch:
            bunch.particles[stat] = bunch.particles[stat][alive]
        bunch.mp_number = len(bunch.particles["x"])
        bunch.alive = np.ones((bunch.mp_number, ))


class IonParticles(Bunch):
    """
    Class representing a collection of ion particles.

    Parameters:
    ----------
    mp_number : int
        The number of particles.
    ion_element_length : float
        The length of the ion segment.
    ring : Synchrotron class object
        The ring object representing the accelerator ring.
    track_alive : bool, optional
        Flag indicating whether to track the alive particles.
        Default is False.
    alive : bool, optional
        Flag indicating whether the particles are alive.
        Default is True.

    Attributes
    ----------
    charge : float
        Bunch charge in [C].
    mean : array of shape (4,)
        Charge-weighted mean values for x, xp, y and yp coordinates.
    mean_xy : tuple of 2 float
        Charge-weighted mean values for x and y coordinates.
    std : array of shape (4,)
        Charge-weighted std values for x, xp, y and yp coordinates.
    mean_std_xy : tuple of 4 float
        Charge-weighted mean and std values for x and y coordinates.

    Methods:
    --------
    generate_as_a_distribution(electron_bunch, charge)
        Generates the particle positions based on a normal distribution, taking
        distribution parameters from an electron bunch.
    generate_from_random_samples(electron_bunch, charge)
        Generates the particle positions and times based on random samples from
        electron positions.
    """

    def __init__(self,
                 mp_number,
                 ion_element_length,
                 ring,
                 track_alive=False,
                 alive=True):
        self.ring = ring
        mp_number = int(mp_number)
        self._mp_number = mp_number
        self.alive = np.ones((self.mp_number, ), dtype=bool)
        if not alive:
            self.alive = np.zeros((self.mp_number, ), dtype=bool)
            mp_number = 1
        self.ion_element_length = ion_element_length
        self.track_alive = track_alive
        self.current = 0
        self.particles = {
            "x": np.zeros((mp_number, ), dtype=np.float64),
            "xp": np.zeros((mp_number, ), dtype=np.float64),
            "y": np.zeros((mp_number, ), dtype=np.float64),
            "yp": np.zeros((mp_number, ), dtype=np.float64),
            "tau": np.zeros((mp_number, ), dtype=np.float64),
            "delta": np.zeros((mp_number, ), dtype=np.float64),
            "charge": np.zeros((mp_number, ), dtype=np.float64),
        }
        self.charge_per_mp = 0

    @property
    def mp_number(self):
        """Macro-particle number"""
        return self._mp_number

    @mp_number.setter
    def mp_number(self, value):
        self._mp_number = int(value)

    @property
    def charge(self):
        """Bunch charge in [C]"""
        return self["charge"].sum()

    def _mean_weighted(self, coord):
        if self.charge == 0:
            return np.zeros_like(coord, dtype=float)
        else:
            mean = [[np.average(self[name], weights=self["charge"])]
                    for name in coord]
            return np.squeeze(np.array(mean))

    def _mean_std_weighted(self, coord):
        if self.charge == 0:
            return np.zeros_like(coord,
                                 dtype=float), np.zeros_like(coord,
                                                             dtype=float)
        else:
            mean = [[np.average(self[name], weights=self["charge"])]
                    for name in coord]
            var = []
            for i, name in enumerate(coord):
                var.append(
                    np.average((self[name] - mean[i])**2,
                               weights=self["charge"]))
            var = np.array(var)
            return np.squeeze(np.array(mean)), np.squeeze(np.sqrt(var))

    @property
    def mean(self):
        """Return charge-weighted mean values for x, xp, y and yp coordinates."""
        coord = ["x", "xp", "y", "yp"]
        return self._mean_weighted(coord)

    @property
    def mean_xy(self):
        """Return charge-weighted mean values for x and y coordinates."""
        coord = ["x", "y"]
        mean = self._mean_weighted(coord)
        return mean[0], mean[1]

    @property
    def std(self):
        """Return charge-weighted std values for x, xp, y and yp coordinates."""
        coord = ["x", "xp", "y", "yp"]
        _, std = self._mean_std_weighted(coord)
        return std

    @property
    def mean_std_xy(self):
        """Return charge-weighted mean and std values for x and y coordinates."""
        coord = ["x", "y"]
        mean, std = self._mean_std_weighted(coord)
        return mean[0], mean[1], std[0], std[1]

    def generate_as_a_distribution(self, electron_bunch, charge):
        """
        Generates the particle positions based on a normal distribution, taking
        distribution parameters from an electron bunch.

        Parameters:
        ----------
        electron_bunch : Bunch
            An instance of the Bunch class representing the electron bunch.
        charge : float
            Total ion charge generated in [C].
        """
        if electron_bunch.is_empty:
            raise ValueError("Electron bunch is empty.")

        self["x"], self["y"] = (
            normal(
                loc=electron_bunch["x"].mean(),
                scale=(electron_bunch["x"]).std(),
                size=self.mp_number,
            ),
            normal(
                loc=electron_bunch["y"].mean(),
                scale=(electron_bunch["y"]).std(),
                size=self.mp_number,
            ),
        )
        self["xp"], self["yp"], self["delta"] = (
            np.zeros((self.mp_number, )),
            np.zeros((self.mp_number, )),
            np.zeros((self.mp_number, )),
        )

        self["tau"] = uniform(
            low=-self.ion_element_length / c,
            high=self.ion_element_length / c,
            size=self.mp_number,
        )
        self["charge"] = np.ones((self.mp_number, )) * charge / self.mp_number

    def generate_from_random_samples(self, electron_bunch, charge):
        """
        Generates the particle positions and times based on random samples from
        electron positions in the bunch.

        Parameters:
        ----------
        electron_bunch : Bunch
            An instance of the Bunch class representing the electron bunch.
        charge : float
            Total ion charge generated in [C].
        """
        if electron_bunch.is_empty:
            raise ValueError("Electron bunch is empty.")

        self["x"], self["y"] = (
            choice(electron_bunch["x"], size=self.mp_number),
            choice(electron_bunch["y"], size=self.mp_number),
        )
        self["xp"], self["yp"], self["delta"] = (
            np.zeros((self.mp_number, )),
            np.zeros((self.mp_number, )),
            np.zeros((self.mp_number, )),
        )
        self["tau"] = uniform(
            low=-self.ion_element_length / c,
            high=self.ion_element_length / c,
            size=self.mp_number,
        )
        self["charge"] = np.ones((self.mp_number, )) * charge / self.mp_number

    def __add__(self, new_particles):
        self.mp_number += new_particles.mp_number
        for t in new_particles:
            self.particles[t] = np.append(self.particles[t],
                                          new_particles.particles[t])
        self.alive = np.append(
            self.alive, np.ones((new_particles.mp_number, ), dtype=bool))
        return self


class BeamIonElement(Element):
    """
    Represents an element for simulating beam-ion interactions.

    Apertures and monitors for the ion beam (instances of IonAperture and
    IonMonitor) can be added to tracking after the BeamIonElement object has
    been initialized by using:
        beam_ion.apertures.append(aperture)
        beam_ion.monitors.append(monitor)

    If the a IonMonitor object is used to record the ion beam, at the end of
    tracking the user should close the monitor, for example by calling:
        beam_ion.monitors[0].close()

    Parameters
    ----------
    ion_mass : float
        The mass of the ions in [kg].
    ion_charge : float
        The charge of the ions in [C].
    ionization_cross_section : float
        The cross section of ionization in [m^2].
    residual_gas_density : float
        The residual gas density in [m^-3].
    ring : instance of Synchrotron()
        The ring.
    ion_field_model : {'weak', 'strong', 'PIC'}
        The ion field model used to update electron beam coordinates:
            - 'weak': ion field acts on each macroparticle.
            - 'strong': ion field acts on electron bunch c.m.
            - 'PIC': a PIC solver is used to get the ion electric field and the
            result is interpolated on the electron bunch coordinates.
        For both 'weak' and 'strong' models, the electric field is computed
        using the Bassetti-Erskine formula [1], so assuming a Gaussian beam
        distribution.
        For 'PIC' the PyPIC package is required.
    electron_field_model : {'weak', 'strong', 'PIC'}
        The electron field model, defined in the same way as ion_field_model.
    ion_element_length : float
        The length of the beam-ion interaction region. For example, if only a
        single interaction point is used this should be equal to ring.L.
    n_ion_macroparticles_per_bunch : int, optional
        The number of ion macroparticles generated per electron bunch passage.
        Defaults to 30.
    generate_method : {'distribution', 'samples'}, optional
        The method to generate the ion macroparticles:
            - 'distribution' generates a distribution statistically equivalent
            to the distribution of electrons.
            - 'samples' generates ions from random samples of electron
            positions.
        Defaults to 'distribution'.
    generate_ions : bool, optional
        If True, generate ions during BeamIonElement.track calls.
        Default is True.
    beam_ion_interaction : bool, optional
        If True, update both beam and ion beam coordinate due to beam-ion
        interaction during BeamIonElement.track calls.
        Default is True.
    ion_drift : bool, optional
        If True, update ion beam coordinate due to drift time between bunches.
        Default is True.

    Methods
    -------
    __init__(ion_mass, ion_charge, ionization_cross_section,
             residual_gas_density, ring, ion_field_model, electron_field_model,
             ion_element_length, n_ion_macroparticles_per_bunch=30,
             generate_method='distribution')
        Initializes the BeamIonElement object.
    parallel(track)
        Defines the decorator @parallel to handle tracking of Beam() objects.
    clear_ions()
        Clear the ion particles in the ion beam.
    track_ions_in_a_drift(drift_length)
        Tracks the ions in a drift.
    generate_new_ions(electron_bunch)
        Generate new ions based on the given electron bunch.
    track(electron_bunch)
        Beam-ion interaction kicks.

    Raises
    ------
    UserWarning
        If the BeamIonMonitor object is used, the user should call the close()
        method at the end of tracking.

    References
    ----------
    [1] : Bassetti, M., & Erskine, G. A. (1980). Closed expression for the
    electrical field of a two-dimensional Gaussian charge (No. ISR-TH-80-06).
    """

    def __init__(
        self,
        ion_mass,
        ion_charge,
        ionization_cross_section,
        residual_gas_density,
        ring,
        ion_field_model,
        electron_field_model,
        ion_element_length,
        n_ion_macroparticles_per_bunch=30,
        generate_method="distribution",
        generate_ions=True,
        beam_ion_interaction=True,
        ion_drift=True,
    ):
        self.ring = ring
        self.bunch_spacing = ring.L / ring.h
        self.ion_mass = ion_mass
        self.ionization_cross_section = ionization_cross_section
        self.residual_gas_density = residual_gas_density
        self.ion_charge = ion_charge
        self.electron_field_model = electron_field_model
        self.ion_field_model = ion_field_model
        self.ion_element_length = ion_element_length
        self.generate_method = generate_method
        if self.generate_method not in ["distribution", "samples"]:
            raise ValueError("Wrong generate_method.")
        self.n_ion_macroparticles_per_bunch = n_ion_macroparticles_per_bunch
        self.ion_beam = IonParticles(
            mp_number=1,
            ion_element_length=self.ion_element_length,
            ring=self.ring)

        self.generate_ions = generate_ions
        self.beam_ion_interaction = beam_ion_interaction
        self.ion_drift = ion_drift

        # interfaces for apertures and montiors
        self.apertures = []
        self.monitors = []

    def parallel(track):
        """
        Defines the decorator @parallel which handle the embarrassingly
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
                    raise ValueError(
                        "Tracking through beam-ion element is performed sequentially."
                    )
                for bunch in beam:
                    track(self, bunch, *args[2:], **kwargs)
            else:
                self = args[0]
                bunch = args[1]
                track(self, bunch, *args[2:], **kwargs)

        return track_wrapper

    def clear_ions(self):
        """
        Clear the ion particles in the ion beam.
        """
        self.ion_beam.particles = IonParticles(
            mp_number=1,
            ion_element_length=self.ion_element_length,
            ring=self.ring)

    def track_ions_in_a_drift(self, drift_length):
        """
        Tracks the ions in a drift.

        Parameters
        ----------
        drift_length : float
            The drift length in meters.
        """
        drifted_ions_x = self.ion_beam["x"] + drift_length * self.ion_beam["xp"]
        drifted_ions_y = self.ion_beam["y"] + drift_length * self.ion_beam["yp"]

        self.ion_beam["x"] = drifted_ions_x
        self.ion_beam["y"] = drifted_ions_y

    def _get_efields(self, first_beam, second_beam, field_model):
        """
        Calculates the electromagnetic field of the second beam acting on the
        first beam for a given field model.

        Parameters
        ----------
        first_beam : IonParticles or Bunch
            The first beam, which is being acted on.
        second_beam : IonParticles or Bunch
            The second beam, which is generating an electric field.
        field_model : {'weak', 'strong', 'PIC'}
            The field model used for the interaction.

        Returns
        -------
        en_x : numpy.ndarray
            The x component of the electric field.
        en_y : numpy.ndarray
            The y component of the electric field.
        """
        if field_model not in ["weak", "strong", "PIC"]:
            raise ValueError(
                f"The implementation for required beam-ion interaction model \
                {field_model} is not implemented")
        if isinstance(second_beam, IonParticles):
            sb_mx, sb_my, sb_stdx, sb_stdy = second_beam.mean_std_xy
        else:
            sb_mx, sb_stdx = (
                second_beam["x"].mean(),
                second_beam["x"].std(),
            )
            sb_my, sb_stdy = (
                second_beam["y"].mean(),
                second_beam["y"].std(),
            )
        if field_model == "weak":
            en_x, en_y = get_displaced_efield(
                _efieldn_mit,
                first_beam["x"],
                first_beam["y"],
                sb_stdx,
                sb_stdy,
                sb_mx,
                sb_my,
            )

        elif field_model == "strong":
            if isinstance(first_beam, IonParticles):
                fb_mx, fb_my = first_beam.mean_xy
            else:
                fb_mx, fb_my = (
                    first_beam["x"].mean(),
                    first_beam["y"].mean(),
                )
            en_x, en_y = get_displaced_efield(_efieldn_mit, fb_mx, fb_my,
                                              sb_stdx, sb_stdy, sb_mx, sb_my)

        elif field_model == "PIC":
            from PyPIC import FFT_OpenBoundary
            from PyPIC import geom_impact_ellip as ellipse

            qe = e
            Dx = 0.1 * sb_stdx
            Dy = 0.1 * sb_stdy
            x_aper = 10 * sb_stdx
            y_aper = 10 * sb_stdy
            chamber = ellipse.ellip_cham_geom_object(x_aper=x_aper,
                                                     y_aper=y_aper)
            picFFT = FFT_OpenBoundary.FFT_OpenBoundary(
                x_aper=chamber.x_aper,
                y_aper=chamber.y_aper,
                dx=Dx,
                dy=Dy,
                fftlib="pyfftw",
            )
            nel_part = 0 * second_beam["x"] + 1.0
            picFFT.scatter(second_beam["x"], second_beam["y"], nel_part)
            picFFT.solve()
            en_x, en_y = picFFT.gather(first_beam["x"], first_beam["y"])
            en_x /= qe * second_beam["x"].shape[0]
            en_y /= qe * second_beam["x"].shape[0]
        return en_x, en_y

    def _get_new_beam_momentum(self,
                               first_beam,
                               second_beam,
                               prefactor,
                               field_model="strong"):
        """
        Calculates the new momentum of the first beam due to the interaction
        with the second beam.

        Parameters
        ----------
        first_beam : IonParticles or Bunch
            The first beam, which is being acted on.
        second_beam : IonParticles or Bunch
            The second beam, which is generating an electric field.
        prefactor : float
            A scaling factor applied to the calculation of the new momentum.
        field_model : {'weak', 'strong', 'PIC'}
            The field model used for the interaction.
            Default is "strong".

        Returns
        -------
        new_xp : numpy.ndarray
            The new x momentum of the first beam.
        new_yp : numpy.ndarray
            The new y momentum of the first beam.
        """

        en_x, en_y = self._get_efields(first_beam,
                                       second_beam,
                                       field_model=field_model)
        kicks_x = prefactor * en_x
        kicks_y = prefactor * en_y
        new_xp = first_beam["xp"] + kicks_x
        new_yp = first_beam["yp"] + kicks_y
        return new_xp, new_yp

    def _update_beam_momentum(self, beam, new_xp, new_yp):
        beam["xp"] = new_xp
        beam["yp"] = new_yp

    def generate_new_ions(self, electron_bunch):
        """
        Generate new ions based on the given electron bunch.

        Parameters
        ----------
        electron_bunch : Bunch
            The electron bunch used to generate new ions.

        Returns
        -------
        None
        """
        new_ion_particles = IonParticles(
            mp_number=self.n_ion_macroparticles_per_bunch,
            ion_element_length=self.ion_element_length,
            ring=self.ring,
        )
        new_ion_charge = (electron_bunch.charge *
                          self.ionization_cross_section *
                          self.residual_gas_density * self.ion_element_length)
        if self.generate_method == "distribution":
            new_ion_particles.generate_as_a_distribution(
                electron_bunch=electron_bunch, charge=new_ion_charge)
        elif self.generate_method == "samples":
            new_ion_particles.generate_from_random_samples(
                electron_bunch=electron_bunch, charge=new_ion_charge)
        self.ion_beam += new_ion_particles

    @parallel
    def track(self, electron_bunch):
        """
        Beam-ion interaction kicks.

        Parameters
        ----------
        electron_bunch : Bunch() or Beam() class object
            An electron bunch to be interacted with.
        """

        if electron_bunch.is_empty:
            empty_bucket = True
        else:
            empty_bucket = False

        if not empty_bucket and self.generate_ions:
            self.generate_new_ions(electron_bunch=electron_bunch)

        for aperture in self.apertures:
            aperture.track(self.ion_beam)

        for monitor in self.monitors:
            monitor.track(self.ion_beam)

        if not empty_bucket and self.beam_ion_interaction:
            prefactor_to_ion_field = -self.ion_beam.charge / (self.ring.E0)
            prefactor_to_electron_field = -electron_bunch.charge * (
                e / (self.ion_mass * c**2))
            new_xp_ions, new_yp_ions = self._get_new_beam_momentum(
                self.ion_beam,
                electron_bunch,
                prefactor_to_electron_field,
                field_model=self.electron_field_model,
            )
            new_xp_electrons, new_yp_electrons = self._get_new_beam_momentum(
                electron_bunch,
                self.ion_beam,
                prefactor_to_ion_field,
                field_model=self.ion_field_model,
            )
            self._update_beam_momentum(self.ion_beam, new_xp_ions, new_yp_ions)
            self._update_beam_momentum(electron_bunch, new_xp_electrons,
                                       new_yp_electrons)

        if self.ion_drift:
            self.track_ions_in_a_drift(drift_length=self.bunch_spacing)
