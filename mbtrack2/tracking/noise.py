import numpy as np


class PhaseNoiseGenerator:
    """
    Example class for generating phase noise sequences.
    Different types of noise models can be defined as needed:
      - White noise
      - 1/f noise
      - Time-domain noise generated from a given PSD (Phase Noise Spectrum)
      - Etc.

    Parameters
    ----------
    fs : float
        Sampling frequency (if generating the entire noise sequence offline at once)
    n_samples : int
        Number of noise points to generate at once (can also be non-fixed)
    noise_amplitude : float
        Noise amplitude factor (can be understood as the standard deviation or used to calibrate noise magnitude)
    model : str
        Type of noise model, customizable as "white", "1/f", "custom_psd", etc.
    custom_psd : array-like, optional
        User-defined phase noise power spectral density (if needed)
    freq_vector : array-like, optional
        Frequency coordinates corresponding to custom_psd

    Usage
    -----
    1. Initialize PhaseNoiseGenerator(fs, n_samples, noise_amplitude, ...)
    2. Call generate_noise() during each tracking step/turn to get noise values (or sequences)
    """

    def __init__(
        self,
        fs=1.0e5,
        n_samples=1024,
        noise_amplitude=1e-3,
        model="white",
        custom_psd=None,
        freq_vector=None,
    ):
        self.fs = fs
        self.n_samples = n_samples
        self.noise_amplitude = noise_amplitude
        self.model = model
        self.custom_psd = custom_psd
        self.freq_vector = freq_vector

        #  if generated_noise offline, generate it here
        self.generated_noise = None
        self.index = 0  # record the current sample point

        if model == "white":
            self._generate_white_noise()
        elif model == "1/f":
            self._generate_1overf_noise()
        elif model == "custom_psd" and (custom_psd is not None):
            self._generate_custom_psd_noise()
        else:
            # default to white noise if model is not recognized
            self._generate_white_noise()

    def _generate_white_noise(self):
        """wite noise generation"""
        self.generated_noise = np.random.normal(loc=0.0,
                                                scale=self.noise_amplitude,
                                                size=self.n_samples)

    def _generate_1overf_noise(self):
        """1/f noise generation by filter in frequement domain"""
        # white noise generation in frequency domain
        white = np.fft.rfft(np.random.normal(0.0, 1.0, self.n_samples))
        freqs = np.fft.rfftfreq(self.n_samples, 1.0 / self.fs)
        #  ~ 1/f filtering
        H = 1.0 / (freqs+1e-6)
        colored = white * H
        # transform back to time domain
        out = np.fft.irfft(colored, n=self.n_samples)
        # adjust amplitude to user specified level
        out = out / np.std(out) * self.noise_amplitude
        self.generated_noise = out

    def _generate_custom_psd_noise(self):
        """
        according to the user-defined phase noise power spectral density custom_psd to generate time-domain noise (example).
        Here, it is assumed that custom_psd and freq_vector are of the same length,
        and freq_vector[0] corresponds to DC, freq_vector[-1] corresponds to the highest frequency.
        """

        # generate random phase in frequency domain
        phase_random = np.random.uniform(0, 2 * np.pi, len(self.custom_psd))
        # amplitude = np.sqrt(self.custom_psd * df)  df = bandwidth =
        # (freq_vector[-1] - freq_vector[0]) / (len(freq_vector)-1)
        df = (self.freq_vector[-1] -
              self.freq_vector[0]) / (len(self.freq_vector) - 1)
        amplitude = np.sqrt(self.custom_psd * df)  # 简化近似
        freq_domain_signal = amplitude * np.exp(1j * phase_random)

        # if need to be symmetric, add negative frequency part
        # here we assume the custom_psd is symmetric, so we can just mirror the positive frequency part to negative frequency part
        # ...
        # FFT to time domain
        time_domain_signal = np.fft.irfft(freq_domain_signal, n=self.n_samples)
        # ajuste amplitude to user specified level
        time_domain_signal = (time_domain_signal / np.std(time_domain_signal) *
                              self.noise_amplitude)
        self.generated_noise = time_domain_signal

    def generate_noise(self, size=1):
        """

        return size of noise points (default 1).
        if the sequence generated at once is used up, generate again (or loop).
        Returns
        -------
        float or np.ndarray
            returns phase noise value (in radians), can be directly added to cavity or generator phase.
        Examples
        --------
        >>> noise = PhaseNoiseGenerator(fs=1.0e5, n_samples=1024, noise_amplitude=1e-3)
        >>> noise.generate_noise()
        """
        if self.generated_noise is None:
            # if not defined, generate white noise
            val = np.random.normal(0.0, self.noise_amplitude, size=size)
            return val if size > 1 else val[0]

        # if pre-generated noise is used up, generate again or loop
        if self.index + size <= self.n_samples:
            val = self.generated_noise[self.index:self.index + size]
            self.index += size
        else:
            # loop back to the beginning of the generated noise sequence
            self.index = 0
            val = self.generated_noise[self.index:self.index + size]
            self.index += size

        if size == 1:
            return val[0]
        else:
            return val
