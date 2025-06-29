# -*- coding: utf-8 -*-
"""
This module defines the Proportional-Integral IQ loop features for RF cavity control.
"""
import numpy as np

class ProportionalIntegralIQFeatures:
    """
    Encapsulates Proportional-Integral (PI) control logic for I/Q components
    of an RF cavity.

    This class is intended to be used by a higher-level controller managing
    an RF cavity (e.g., CavityResonator via a loop object).
    """
    def __init__(
        self,
        ring,
        cav_res_interface, # Interface to access cavity_resonator properties (Vc, theta, etc.)
                           # and methods like Vg2Ig.
        gain_pi, # [Pgain, Igain]
        sample_num,
        every, # Sampling period for PI controller updates (in buckets)
        delay_steps, # PI loop delay in terms of 'every' units (number of records in diff_record)
        IIR_filter, # An initialized IIR filter object with an `apply(value)` method
        use_FF=True,
    ):
        self.ring = ring
        self.cav_res = cav_res_interface # Provides Vc, theta, Vg2Ig, cavity_phasor_record
        self.Pgain = gain_pi[0]
        self.Igain = gain_pi[1]
        self.sample_num = int(sample_num)
        self.every = int(every) # Number of buckets between updates
        self.delay_steps = int(delay_steps) # Number of error records to keep for delay
        self.IIR_filter = IIR_filter # Expects an object with an apply(value) method
        self.use_FF = use_FF

        # Target I and Q components based on cavity resonator's Vc and theta
        self._update_target_phasor()

        # Initialize feedback variables
        # vc_previous stores the last self.sample_num cavity phasors from the previous turn
        self.vc_previous = np.ones(
            self.sample_num, dtype=complex
        ) * self.cav_res.get_cavity_phasor() # Initial estimate

        # diff_record stores the history of complex errors (target - measured)
        # The length is self.delay_steps, corresponding to the loop delay.
        self.diff_record = np.zeros(self.delay_steps, dtype=complex)
        self.integral_term = 0 + 0j # Complex integral term for PI

        self.FFconst = 0 + 0j
        if self.use_FF:
            self._calculate_ff_constant()

    def _update_target_phasor(self):
        """Updates the target I and Q phasor based on cav_res attributes."""
        vc, theta = self.cav_res.get_vc_theta_target()
        self.target_I = vc * np.cos(theta)
        self.target_Q = vc * np.sin(theta)
        self.target_phasor = self.target_I + 1j * self.target_Q
        # If FF is used, it should be recalculated if target changes
        if self.use_FF:
            self._calculate_ff_constant()


    def _calculate_ff_constant(self):
        """Calculates the feedforward constant (base generator current)."""
        if self.use_FF:
            self.FFconst = self.cav_res.Vg2Ig(self.target_phasor)
        else:
            self.FFconst = 0 + 0j

    def reset_integral_term(self):
        """Resets the integral term of the PI controller."""
        self.integral_term = 0 + 0j

    def update_target(self):
        """Call this if the cavity's Vc or theta targets change."""
        self._update_target_phasor()

    def process_turn(self, current_cavity_phasor_record, bucket_indices_to_update):
        """
        Processes one turn of PI IQ control for specified bucket indices.

        Parameters
        ----------
        current_cavity_phasor_record : np.ndarray (complex)
            The full record of cavity phasors for all h buckets in the current turn.
        bucket_indices_to_update : list or range
            A list or range of bucket indices for which the PI control output
            (ig_control_phasor) should be calculated in this step. This is
            determined by the `every` parameter of the main loop.

        Returns
        -------
        dict
            A dictionary where keys are bucket indices and values are the
            calculated complex `ig_control_phasor` for that bucket.
            The main loop will then fill its `ig_phasor` array with these values.
        """
        output_ig_phasors = {}

        # Concatenate previous turn's tail end of cavity phasors with current turn's record
        # This allows averaging across turn boundaries if sample_num is large enough.
        vc_list_for_averaging = np.concatenate(
            [self.vc_previous, current_cavity_phasor_record]
        )

        for bucket_idx in bucket_indices_to_update:
            # 1. Monitor cavity voltage (mean over sample_num)
            # The `bucket_idx` here is relative to the start of `current_cavity_phasor_record`.
            # So, `vc_list_for_averaging` index needs to be offset by `self.sample_num`.
            start_avg_idx = self.sample_num + bucket_idx
            mean_vc_complex = np.mean(
                vc_list_for_averaging[start_avg_idx : start_avg_idx + self.sample_num]
            )

            # Apply IIR filter to the magnitude of the mean cavity voltage
            mean_vc_mag_filtered = self.IIR_filter.apply(np.abs(mean_vc_complex))
            mean_vc_phase = np.angle(mean_vc_complex)
            vc_for_error_calc = mean_vc_mag_filtered * np.exp(1j * mean_vc_phase)

            # 2. Calculate current complex error
            current_error = self.target_phasor - vc_for_error_calc

            # 3. PI Controller (complex arithmetic)
            # The error from `self.delay_steps` ago is `self.diff_record[-1]`
            # if diff_record is shifted *after* this calculation.
            # If diff_record was shifted at the end of the *previous* call for this bucket,
            # then diff_record[0] would be the oldest.
            # Let's assume diff_record[-1] is the appropriately delayed error.
            delayed_error = self.diff_record[-1]

            # Update integral term with the delayed error
            # Normalization by f1 (revolution frequency) makes Igain units more standard.
            self.integral_term += delayed_error / self.ring.f1

            # PI control output based on delayed error
            pi_correction_complex = self.Pgain * delayed_error + self.Igain * self.integral_term

            # The feedforward constant is the base generator current required
            ig_control_phasor = self.FFconst + pi_correction_complex
            output_ig_phasors[bucket_idx] = ig_control_phasor

            # Store current error for future use (becomes delayed_error later)
            # This shift assumes that for each `bucket_idx` in `bucket_indices_to_update`,
            # we are making one step in the PI controller for that "slice" of the feedback.
            # This might need careful handling if `bucket_indices_to_update` doesn't cover all
            # phases of the `every` cycle.
            # For simplicity, the main loop will manage rolling self.diff_record once per its "every" cycle.
            # This method just provides the latest error.
            # The main loop will call: self.diff_record = np.roll(self.diff_record, 1); self.diff_record[0] = new_error

        # Update vc_previous for the next turn's calculation
        # This should be done once per turn by the main loop, not here.
        # self.vc_previous = current_cavity_phasor_record[-self.sample_num:]

        # Return the calculated ig_phasors and the latest error calculated
        # The main loop will handle updating self.diff_record
        # For the last bucket_idx processed, that's the 'current_error' we want to record.
        # This is slightly tricky because this function is called for a batch of indices.
        # The design is that the main loop calls this, gets the ig_phasors, and then updates
        # its own diff_record (which it passes to this class or this class owns).
        # Let's make this class own self.diff_record and self.vc_previous.

        # The error to be stored is `current_error` from the last `bucket_idx` processed.
        # This assumes that `bucket_indices_to_update` are processed sequentially and represent
        # one "tick" of the `every` clock.
        if bucket_indices_to_update: # If list is not empty
            self.last_calculated_error = current_error
        else:
            self.last_calculated_error = 0j # Or previous error if no updates

        return output_ig_phasors

    def get_last_calculated_error(self):
        """Returns the most recent error calculated by process_turn."""
        return getattr(self, "last_calculated_error", 0j)

    def update_history(self, new_error, current_cavity_phasor_record_tail):
        """
        Updates the error history (diff_record) and vc_previous.
        To be called by the main loop once per its "every" cycle.

        Parameters:
        -----------
        new_error: complex
            The error (target - measured) calculated for the current PI step.
        current_cavity_phasor_record_tail: np.ndarray
            The last `self.sample_num` elements of the `cavity_phasor_record` from the
            current turn, to be used as `vc_previous` for the next turn.
        """
        self.diff_record = np.roll(self.diff_record, 1)
        self.diff_record[0] = new_error
        self.vc_previous = current_cavity_phasor_record_tail

    def get_ff_const(self):
        return self.FFconst

    def get_integral_term(self):
        return self.integral_term

    def set_integral_term(self, integral_term):
        self.integral_term = integral_term

    # Interface for CavityResonator properties needed by this class
    # This avoids passing the full cav_res object if not needed, promoting loose coupling.
    # The main loop object (e.g. ProportionalIntegralIQLoop) will implement this interface.
    class CavityInterface:
        def get_vc_theta_target(self):
            """Should return (target_Vc, target_theta)"""
            raise NotImplementedError

        def Vg2Ig(self, Vg_phasor):
            """Should convert Vg_phasor to Ig_phasor for FF calculation"""
            raise NotImplementedError

        def get_cavity_phasor(self):
            """Should return the current cavity phasor (e.g., for initialization)"""
            raise NotImplementedError

# Example of a simple IIR filter class that ProportionalIntegralIQFeatures can use
class SimpleIIRFilter:
    def __init__(self, IIR_coefficient, initial_output_value=0.0):
        self.coeff = IIR_coefficient
        self.output = initial_output_value

    def apply(self, input_value):
        self.output = (1 - self.coeff) * self.output + self.coeff * input_value
        return self.output

    def get_output(self):
        return self.output

    def set_output(self, value): # For re-initialization if needed
        self.output = value

    def get_coefficient(self):
        return self.coeff

    def set_coefficient(self, coeff):
        self.coeff = coeff

# Helper class to adapt CavityResonator for ProportionalIntegralIQFeatures
class CavityResonatorAdapter(ProportionalIntegralIQFeatures.CavityInterface):
    def __init__(self, cav_res_obj, ring_obj):
        self.cav_res = cav_res_obj
        self.ring = ring_obj # Needed for Vg2Ig if it uses ring.f1 indirectly

    def get_vc_theta_target(self):
        return self.cav_res.Vc, self.cav_res.theta

    def Vg2Ig(self, Vg_phasor):
        # This directly calls the method on CavityResonator.
        # Ensure this Vg2Ig is compatible (takes complex phasor, returns complex phasor)
        return Vg_phasor * (1 - 1j * np.tan(self.cav_res.psi)) / self.cav_res.RL

    def get_cavity_phasor(self):
        return self.cav_res.cavity_phasor # Current overall cavity phasor

    # Provide access to ring.f1 for PI controller if needed for integral term scaling
    def get_ring_f1(self):
        return self.ring.f1

# Example usage (conceptual, would be in the main loop)
# Assuming cav_res is a CavityResonator instance and ring is a Synchrotron instance
# adapter = CavityResonatorAdapter(cav_res, ring)
# iir_filter = SimpleIIRFilter(IIR_coefficient=0.1, initial_output_value=np.abs(cav_res.Vc))
#
# pi_iq_features = ProportionalIntegralIQFeatures(
#     ring=ring,
#     cav_res_interface=adapter,
#     gain_pi=[Pgain, Igain],
#     sample_num=sample_num,
#     every=every, # This 'every' is for internal logic if it were to run independently
#     delay_steps=record_size, # record_size from main loop
#     IIR_filter=iir_filter,
#     use_FF=True
# )
#
# In the main loop's track method, before iterating `sample_list`:
#   cav_phasor_record = self.cav_res.cavity_phasor_record
#   indices_to_update_this_step = self.sample_list # or however it's determined
#
#   calculated_igs = pi_iq_features.process_turn(cav_phasor_record, indices_to_update_this_step)
#
#   for bucket_idx, ig_val in calculated_igs.items():
#       self.ig_phasor[bucket_idx:] = ig_val # Or fill appropriately
#
#   # After processing all relevant buckets for this 'every' cycle:
#   latest_error = pi_iq_features.get_last_calculated_error()
#   tail_phasors = cav_phasor_record[-self.sample_num:]
#   pi_iq_features.update_history(latest_error, tail_phasors)

"""
Design Notes for ProportionalIntegralIQFeatures:
- CavityResonatorInterface: To decouple from the concrete CavityResonator, allowing easier testing and flexibility. The main `ProportionalIntegralIQLoop` will provide an object that implements this interface.
- IIR Filter: Passed as an object. This allows different IIR filter implementations. The `SimpleIIRFilter` is provided as an example.
- `process_turn`: This is the core method. It takes the current turn's full cavity phasor record and the specific bucket indices that need their `ig_phasor` calculated in this step.
- State Management:
    - `vc_previous`: Stores the tail of the previous turn's cavity phasors for averaging. Updated by the main loop.
    - `diff_record`: History of complex errors for implementing the delay. Updated by the main loop.
    - `integral_term`: The accumulated integral part of the PI controller. Managed internally but can be reset.
- FF Constant: Calculated based on the target phasor and `cav_res_interface.Vg2Ig`.
- Target Update: `update_target()` method to be called if `cav_res.Vc` or `cav_res.theta` change, as this affects `target_phasor` and `FFconst`.

The main `ProportionalIntegralIQLoop` will be responsible for:
- Instantiating this `ProportionalIntegralIQFeatures` class.
- Instantiating a suitable IIR filter object.
- Providing the `CavityResonatorAdapter`.
- In its `track` method:
    - Determining which bucket indices to update based on `self.every`.
    - Calling `pi_iq_features.process_turn()`.
    - Updating its own `ig_phasor` array with the results.
    - Managing the `diff_record` shift and update using `pi_iq_features.get_last_calculated_error()` and `pi_iq_features.update_history()`.
    - Managing the update of `pi_iq_features.vc_previous` via `update_history()`.
"""
