# -*- coding: utf-8 -*-
"""
Example script to demonstrate the usage of the refactored
ProportionalIntegralIQLoopMode0Damper with its components now consolidated
within rf.py.

This script outlines the setup and instantiation of the RF feedback loop.
It is a conceptual guide and requires specific physics parameters for a
full simulation.
"""
import numpy as np

# Assuming mbtrack2 is installed or PYTHONPATH is set up correctly
from mbtrack2.tracking.synchrotron import Synchrotron
from mbtrack2.tracking.particles import Beam, Bunch, Proton
from mbtrack2.tracking.rf import (
    CavityResonator,
    ProportionalIntegralIQLoopMode0Damper,
    # Note: ProportionalIntegralIQFeatures, Mode0DamperFeatures, SimpleIIRFilter,
    # CavityResonatorAdapterImpl, BeamAdapterImpl are now defined within rf.py
    # but are not typically imported directly by the user, as they are
    # components of ProportionalIntegralIQLoopMode0Damper.
)

# --- 1. Setup Synchrotron Parameters (Illustrative) ---
ring_params = {
    "ACO": 6.0 * np.pi / 180.0,
    "E0": 2.5e9,
    "N": 1,
    "h": 312,
    "atomic_A": 1,
    "charge_q": 1,
    "U0": 0.0,
    "tau": np.array([1e-3, 1e-3, 1e-3]),
    "tune": np.array([0.18, 0.28, 0.005]),
    "chromaticity": np.array([0.0, 0.0]),
}
ring = Synchrotron(**ring_params)
ring.particle = Proton()

# --- 2. Setup CavityResonator (Illustrative) ---
cavity = CavityResonator(
    ring=ring,
    m=ring.h,
    Rs=10e6,
    Q=20000,
    QL=10000,
    detune=0,
    Vc=1.5e6,
    theta=0,
    n_bin = 100
)
I0_initial_estimate = 10e-3
cavity.set_generator(I0_initial_estimate)

# --- 3. Setup Beam (Illustrative) ---
n_macroparticles = 10000
n_bunches = 1
bunch_intensity = 1e10

beam = Beam(ring, n_macroparticles, n_bunches)
for i in range(n_bunches):
    bucket_index = i * int(ring.h / n_bunches)
    beam[bucket_index] = Bunch(ring, n_macroparticles)
    beam[bucket_index]["tau"] = np.random.normal(0, 10e-12, n_macroparticles)
    beam[bucket_index]["delta"] = np.random.normal(0, 1e-4, n_macroparticles)
    beam[bucket_index].intensity = bunch_intensity / n_macroparticles

beam.update_filling_pattern()
beam.update_distance_between_bunches()

# --- 4. Instantiate ProportionalIntegralIQLoopMode0Damper ---
pi_iq_gain = [0.5, 1e4]
sample_num = 8
every_buckets = 7
delay_buckets = 50

enable_damper_flag = True
damper_pi_gain = [0.1, 100]
damper_filter_func = None
damper_mean_idx_val = 4 # Check Bunch.mean_vars for correct index, usually 4 for tau

rf_loop = ProportionalIntegralIQLoopMode0Damper(
    ring=ring,
    cav_res=cavity,
    gain=pi_iq_gain,
    sample_num=sample_num,
    every=every_buckets,
    delay=delay_buckets,
    IIR_cutoff=10e3,
    FF=True,
    enable_damper=enable_damper_flag,
    damper_gain=damper_pi_gain,
    damper_filter_func=damper_filter_func,
    damper_mean_idx=damper_mean_idx_val,
    damper_phase_shift_limit=np.pi / 18
)
cavity.feedback.append(rf_loop)

# --- 5. Conceptual Tracking Loop ---
n_turns = 100

print(f"Starting conceptual tracking for {n_turns} turns...")
print(f"Initial Cavity Vc: {cavity.cavity_voltage:.2f} V, Phase: {cavity.cavity_phase:.3f} rad")
print(f"PI IQ Target Vc: {cavity.Vc:.2f} V, Phase: {cavity.theta:.3f} rad")
if enable_damper_flag:
    print("Mode 0 Damper is ENABLED.")

for turn in range(n_turns):
    # In a real simulation, other elements would be tracked here.
    # cavity.track(beam) will now correctly make the beam available via
    # cavity._current_beam_in_track for the damper, due to changes in rf.py.
    cavity.track(beam)

    if (turn + 1) % 10 == 0:
        mean_tau_turn = 0.0
        mean_delta_turn = 0.0
        num_active_bunches = 0

        active_bunches = list(beam.not_empty)
        if active_bunches:
            for b in active_bunches:
                 mean_tau_turn += b.mean[damper_mean_idx_val]
                 mean_delta_turn += b.mean[5] # Assuming index 5 is delta
                 num_active_bunches+=1
            if num_active_bunches > 0:
                mean_tau_turn /= num_active_bunches
                mean_delta_turn /= num_active_bunches

        print(
            f"Turn {turn + 1:3d}: "
            f"Cav Vc={cavity.cavity_voltage:.2f} V, Ph={cavity.cavity_phase:.3f} rad; "
            f"Beam <tau>={mean_tau_turn*1e12:.2f} ps (using mean_idx={damper_mean_idx_val}), "
            f"<delta>={mean_delta_turn*1e3:.3f}e-3"
        )

print("Conceptual tracking finished.")
print("\nScript structure demonstrates instantiation and conceptual use of the RF loop with consolidated classes.")
print("The CavityResonator.track method in rf.py has been updated to support damper beam access.")

# To run this example:
# 1. Ensure mbtrack2 is installed and importable.
# 2. Fill in realistic physics parameters for `ring_params`, `cavity`, and `beam`.
# 3. Potentially add more lattice elements and tracking logic.
```
