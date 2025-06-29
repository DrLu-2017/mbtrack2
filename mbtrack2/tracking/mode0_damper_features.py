# -*- coding: utf-8 -*-
"""
This module defines the Mode 0 Damper features for RF cavity control.
"""
import numpy as np

class Mode0DamperFeatures:
    """
    Encapsulates Mode 0 damper functionality for an RF system.

    This class calculates a phase correction based on the Mode 0 oscillation
    of the beam, which can then be applied to the RF generator's current phasor.
    It includes its own PI controller for the damper loop.
    """

    def __init__(
        self,
        ring, # Synchrotron object, needed for ring.f1
        beam_interface, # Interface to get beam bunch data
        gain_damper, # [P_damper, I_damper]
        delay_steps, # Damper loop delay in terms of processing steps (e.g., number of turns or 'every' cycles)
        filter_func=None, # User-provided function to filter raw Mode 0 signal
        mean_idx_for_mode0=4, # Index in bunch.mean[] for phase information
        phase_shift_limit=np.pi / 12, # Max phase shift to apply
    ):
        self.ring = ring
        self.beam_interface = beam_interface # Provides access to beam bunches
        self.Pgain_damper = gain_damper[0]
        self.Igain_damper = gain_damper[1]
        self.delay_steps = int(delay_steps)
        self.filter_func = filter_func if filter_func else lambda x: x
        self.mean_idx_for_mode0 = mean_idx_for_mode0
        self.phase_shift_limit = phase_shift_limit

        # Buffer for delaying the Mode 0 signal fed into the damper's PI controller
        # Length of buffer determines the delay.
        self.mode0_signal_buffer = [0.0] * self.delay_steps
        self.integral_term_damper = 0.0  # Integral term for the damper's PI controller

    def _calculate_raw_mode0_signal(self):
        """
        Calculates the raw Mode 0 signal from beam bunch data.
        The 'raw' signal here is before the user-provided filter_func is applied.
        """
        bunches = self.beam_interface.get_non_empty_bunches()
        if not bunches:
            return 0.0

        try:
            # Extract phase-like information using mean_idx_for_mode0
            # This assumes bunch.mean[idx] provides a quantity related to phase deviation.
            # If it's absolute tau, conversion to phase (omega1 * tau) might be needed,
            # or if it's already a phase, it can be used directly.
            # The original ProportionalIntegralIQLoop used bunch.mean[idx] directly.
            bunch_phase_values_raw = [b.mean[self.mean_idx_for_mode0] for b in bunches]
        except IndexError:
            # Fallback or error if mean_idx_for_mode0 is invalid
            # Consider logging a warning here if a logger is available
            return 0.0

        # Reference phase for deviation.
        # The original code implies deviations are calculated from a mean, or the values are already deviations.
        # If bunch_phase_values_raw are absolute phases, they should be relative to synchronous phase.
        # For simplicity, assuming they are already comparable or deviations.
        # np.unwrap handles phase jumps if values are angles in a fixed range (e.g., -pi to pi).
        # This step is crucial if mean_idx_for_mode0 gives, e.g., tau values that get large.
        # However, if it's already a small deviation, unwrap might not be necessary or could be harmful.
        # Let's assume the values from bunch.mean[mean_idx_for_mode0] are suitable for averaging
        # after potential unwrapping if they represent wrapped phases.
        # If they are, for instance, time offsets (tau), then their average is directly the mode 0 signal.

        # Decision: Follow original logic which directly averages, assuming values are appropriate.
        # If unwrap is needed, it should be part of a more sophisticated signal processing step,
        # potentially within the filter_func or explicitly handled if values are known to be angles.
        # For now, let's average directly, as in the snippet `np.mean(unwrapped_deviations)`
        # where `unwrapped_deviations` came from `bunch.mean[damper_mean_idx]`.

        # If the values are actual phases that could wrap, unwrap is important:
        # unwrapped_values = np.unwrap(bunch_phase_values_raw)
        # mode0_signal = np.mean(unwrapped_values)
        # If they are already small deviations or time offsets, direct mean is fine:
        mode0_signal = np.mean(bunch_phase_values_raw)

        return mode0_signal

    def get_phase_correction(self):
        """
        Computes the phase correction amount based on the Mode 0 signal
        and the damper's PI controller.

        This method should be called once per damper processing cycle (e.g., per turn
        or per 'every' interval of the main loop).
        """
        # 1. Calculate current raw Mode 0 signal
        current_raw_mode0 = self._calculate_raw_mode0_signal()

        # 2. Apply user-defined filter function
        current_filtered_mode0 = self.filter_func(current_raw_mode0)

        # 3. Get delayed Mode 0 signal from the buffer
        # The oldest signal in the buffer is at index 0.
        delayed_mode0_signal = self.mode0_signal_buffer[0]

        # 4. Update buffer: shift old values, add new filtered signal
        self.mode0_signal_buffer = self.mode0_signal_buffer[1:] + [current_filtered_mode0]

        # 5. PI control for damper based on the delayed signal
        # Update integral term (scaled by ring.f1 for unit consistency if Igain_damper assumes it)
        self.integral_term_damper += delayed_mode0_signal * self.Igain_damper / self.ring.f1

        # Proportional term + Integral term
        phase_shift_correction = (self.Pgain_damper * delayed_mode0_signal +
                                  self.integral_term_damper)

        # 6. Clip the phase shift to limits
        phase_shift_correction = np.clip(phase_shift_correction,
                                         -self.phase_shift_limit,
                                         self.phase_shift_limit)

        return phase_shift_correction

    def reset_integral_term(self):
        """Resets the integral term of the damper's PI controller."""
        self.integral_term_damper = 0.0

    def reset_buffer(self):
        """Resets the mode0 signal buffer."""
        self.mode0_signal_buffer = [0.0] * self.delay_steps


    # Interface for beam data needed by this class
    class BeamInterface:
        def get_non_empty_bunches(self):
            """Should return a list or iterator of non-empty Bunch objects."""
            raise NotImplementedError

# Helper class to adapt a Beam object for Mode0DamperFeatures
class BeamAdapter(Mode0DamperFeatures.BeamInterface):
    def __init__(self, beam_obj):
        self.beam = beam_obj # Assumes beam_obj has a `not_empty` attribute or similar

    def get_non_empty_bunches(self):
        if hasattr(self.beam, 'not_empty'):
            return list(self.beam.not_empty) # Convert iterator to list if necessary
        elif hasattr(self.beam, '__iter__'): # Basic check if beam_obj is iterable
            # This might be too general, depends on how Beam objects are structured
            # and if filtering for "non-empty" is implicit or needs to be done here.
            # Assuming direct iteration gives non-empty, or pre-filtered bunches.
            return [bunch for bunch in self.beam if not bunch.is_empty] # Example
        return []

"""
Design Notes for Mode0DamperFeatures:
- BeamInterface: Decouples from the concrete Beam object structure. The main loop or `CavityResonator` would provide an object implementing this.
- `_calculate_raw_mode0_signal`: Gets data from bunches via `beam_interface`.
- `get_phase_correction`:
    - Calls `_calculate_raw_mode0_signal`.
    - Applies `filter_func`.
    - Manages a simple shift-register buffer (`mode0_signal_buffer`) for delay.
    - Applies PI control (`Pgain_damper`, `Igain_damper`) to the delayed signal.
    - Clips the resulting phase shift.
- State Management:
    - `mode0_signal_buffer`: Stores history of filtered Mode 0 signals for delay.
    - `integral_term_damper`: Accumulates the integral part for the damper's PI controller.
- The main `ProportionalIntegralIQLoop` will:
    - Instantiate `Mode0DamperFeatures`.
    - Provide a `BeamAdapter` (or similar).
    - In its `track` method, if the damper is enabled:
        - Call `mode0_damper.get_phase_correction()`.
        - Apply the returned phase shift to the `ig_control_phasor` (output of the PI IQ features).
"""
