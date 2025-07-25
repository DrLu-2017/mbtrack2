class ProportionalIntegralIQLoop:
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

    def __init__(
        self, ring, cav_res, gain, sample_num, every, delay, IIR_cutoff=0, FF=True
    ):
        self.ring = ring
        self.cav_res = cav_res
        self.Ig2Vg_mat = np.zeros((self.ring.h, self.ring.h), dtype=complex)
        self.ig_modulation_signal = np.zeros(self.ring.h, dtype=complex)
        self.gain_I = gain[0]  # Proportional gain for I component
        self.gain_Q = gain[0]  # Proportional gain for Q component
        self.Igain_I = gain[1]  # Integral gain for I component
        self.Igain_Q = gain[1]  # Integral gain for Q component
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

        # Convert target voltage to IQ components
        # target_complex = self.cav_res.Vc * np.exp(1j * self.cav_res.theta)
        self.target_I = self.cav_res.Vc * np.cos(self.cav_res.theta)
        self.target_Q = self.cav_res.Vc * np.sin(self.cav_res.theta)
        print("Target Vc", self.cav_res.Vc, "Target theta", self.cav_res.theta)
        # Initialize feedback variables
        self.ig_phasor = np.ones(self.ring.h, dtype=complex) * self.Vg2Ig(
            self.cav_res.generator_phasor
        )
        self.ig_phasor_record = self.ig_phasor
        self.vc_previous = np.ones(self.sample_num) * self.cav_res.cavity_phasor

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
        vc_list = np.concatenate([self.vc_previous, self.cav_res.cavity_phasor_record])
        self.ig_phasor.fill(self.ig_phasor[-1])

        for index in self.sample_list:
            # Get current cavity voltage in IQ form
            mean_vc = np.mean(vc_list[index : self.sample_num + index])
            mean_vc_mag = np.abs(mean_vc)
            mean_vc_phase = np.arctan2(mean_vc.imag, mean_vc.real)

            mean_vc_filtered = self.IIR(mean_vc)

            # Calculate I/Q errors
            error_I = self.target_I - mean_vc_mag * np.cos(mean_vc_phase)
            error_Q = self.target_Q - mean_vc_mag * np.sin(mean_vc_phase)

            # Update integral terms
            self.I_record_I += error_I / self.ring.f1
            self.I_record_Q += error_Q / self.ring.f1

            # Calculate feedback values using PI control for both I and Q
            fb_I = self.gain_I * error_I + self.Igain_I * self.I_record_I
            fb_Q = self.gain_Q * error_Q + self.Igain_Q * self.I_record_Q

            # Combine I/Q components and apply feedback
            fb_value_mag = np.sqrt(fb_I**2 + fb_Q**2)
            fb_value_phase = np.arctan2(fb_Q, fb_I)
            fb_value = fb_value_mag * np.exp(1j * fb_value_phase)
            self.ig_phasor[index:] = self.Vg2Ig(fb_value) + self.FFconst

        # Update for next turn
        self.sample_list = range(
            index + self.every - self.ring.h, self.ring.h, self.every
        )
        self.vc_previous = self.cav_res.cavity_phasor_record[-self.sample_num :]

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
        self.Ig2Vg_vec = np.exp(
            -1
            / self.cav_res.filling_time
            * (1 - 1j * np.tan(self.cav_res.psi))
            * self.ring.T1
            * (k + 1)
        )
        tempV = np.exp(
            -1
            / self.cav_res.filling_time
            * self.ring.T1
            * k
            * (1 - 1j * np.tan(self.cav_res.psi))
        )
        for idx in np.arange(self.ring.h):
            self.Ig2Vg_mat[idx:, idx] = tempV[: self.ring.h - idx]

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
            self.Ig2Vg_vec * self.cav_res.generator_phasor_record[-1]
            + self.Ig2Vg_mat.dot(self.ig_phasor_record)
            * self.cav_res.loss_factor
            * self.ring.T1
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
        self.cav_res.theta_g = np.mean(np.angle(self.cav_res.generator_phasor_record))

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
            tmp = alpha * alpha - 2 * alpha
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
        return (
            1.0
            / 2.0
            / np.pi
            / T
            * np.arccos(
                (2 - 2 * self.IIRcoef - self.IIRcoef * self.IIRcoef)
                / 2
                / (1 - self.IIRcoef)
            )
        )