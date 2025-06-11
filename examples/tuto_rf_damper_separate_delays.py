# -*- coding: utf-8 -*-
"""
Tutorial: Demonstrating Separate Delays in ProportionalIntegralIQLoopMode0Damper

This script shows how to configure the ProportionalIntegralIQLoopMode0Damper
with a separate delay for its Mode 0 damper functionality (`damper_delay`)
compared to the main PI loop's delay (`delay`).
"""

import numpy as np

# Assuming mbtrack2 components are accessible in the Python path
# Adjust imports based on your project structure if necessary
from mbtrack2.tracking.synchrotron import Synchrotron
from mbtrack2.utilities.optics import Optics # Corrected import path
from mbtrack2.tracking.particles import Particle, Beam, Bunch # Particle needed for Synchrotron

from mbtrack2.tracking.rf import (
    CavityResonator,
    ProportionalIntegralIQLoopMode0Damper
)

# Simple placeholder for Particle and Optics if detailed setup is not required by the example
class MinimalParticle:
    def __init__(self):
        from scipy.constants import e, m_e
        self.mass = m_e # Electron mass
        self.charge = -e # Electron charge

class MinimalOptics:
    def __init__(self, circumference=None):
        self.use_local_values = True # Simplify Synchrotron's expectations
        self.lattice = None
        if circumference:
            # Mocking a lattice attribute if circumference is directly needed by Synchrotron via optics.lattice.circumference
            class MockLattice:
                def __init__(self, circ):
                    self.circumference = circ
            self.lattice = MockLattice(circumference)


def setup_basic_ring_and_cavity():
    """Helper function to set up a basic ring and cavity resonator."""

    harmonic_number = 396
    particle = MinimalParticle()

    # Calculate circumference from f0 if needed, or set L directly
    # f0 = 350e3 means T0 = 1/f0. L = c * T0
    from scipy.constants import c
    f0_val = 350e3
    L_val = c / f0_val

    optics = MinimalOptics(circumference=L_val)

    # Basic ring parameters
    ring = Synchrotron(
        h=harmonic_number,
        optics=optics,
        particle=particle,
        E0=3e9,
        U0=1e6,
        ac=1.7e-4,
        tau=[10e-3, 10e-3, 5e-3],
        # f0=f0_val, # f0 is derived from L, L is now from optics mock or passed directly
        L=L_val, # Pass L directly
        sigma_delta=1e-3, # Example value
        sigma_0=10e-12,   # Example value
        emit=[1e-9, 1e-10] # Example values
    )
    # ring.omega1 is automatically calculated by property setters if h and L (or f0) are set.

    # Basic cavity resonator
    cav_res = CavityResonator(
        ring=ring,
        m=ring.h,      # Harmonic number (main RF)
        Rs=2.5e6 * ring.h, # Shunt impedance
        Q=5e8,         # Quality factor
        QL=2e5,        # Loaded quality factor
        detune=0,      # Detuning in Hz
        Vc=1.5e6,      # Target cavity voltage
        theta=np.pi/2  # Target cavity phase
    )
    cav_res.set_generator(I0=0.1) # Set some initial generator parameters for 100mA
    return ring, cav_res

def main():
    print("--- Demonstrating ProportionalIntegralIQLoopMode0Damper Delays ---")

    ring, cav_res = setup_basic_ring_and_cavity()

    # Common parameters for the PI IQ Loop
    gain = [0.8, 1e5]  # [Pgain, Igain]
    sample_num = 16     # Number of bunches for voltage averaging
    every = 10          # Sampling period for feedback (bucket units)
    enable_damper = True

    # --- Case 1: Explicitly setting different delay and damper_delay ---
    main_loop_delay_case1 = 20  # Delay for the main PI controller
    damper_specific_delay = 7   # Specific delay for the Mode 0 damper

    print(f"\nCase 1: Initializing with main delay = {main_loop_delay_case1} and damper_delay = {damper_specific_delay}")

    pi_iq_loop_explicit_damper_delay = ProportionalIntegralIQLoopMode0Damper(
        ring=ring,
        cav_res=cav_res,
        gain=gain,
        sample_num=sample_num,
        every=every,
        delay=main_loop_delay_case1,
        damper_delay=damper_specific_delay, # Explicitly set
        IIR_cutoff=0,
        FF=True,
        enable_damper=enable_damper,
        damper_gain=[0.1, 100] # Example damper gains
    )

    print(f"  Loop initialized: main self.delay = {pi_iq_loop_explicit_damper_delay.delay}")
    print(f"  Loop initialized: damper self.damper_delay = {pi_iq_loop_explicit_damper_delay.damper_delay}")
    assert pi_iq_loop_explicit_damper_delay.delay == main_loop_delay_case1
    assert pi_iq_loop_explicit_damper_delay.damper_delay == damper_specific_delay
    print("  Delays configured as expected.")

    # --- Case 2: Damper delay defaults to main loop delay ---
    main_loop_delay_case2 = 15 # Delay for the main PI controller
    # damper_delay is not provided, or explicitly None

    print(f"\nCase 2: Initializing with main delay = {main_loop_delay_case2} and damper_delay not set (should default)")

    pi_iq_loop_default_damper_delay = ProportionalIntegralIQLoopMode0Damper(
        ring=ring,
        cav_res=cav_res,
        gain=gain,
        sample_num=sample_num,
        every=every,
        delay=main_loop_delay_case2,
        # damper_delay is omitted, so it will use main_loop_delay_case2
        IIR_cutoff=0,
        FF=True,
        enable_damper=enable_damper,
        damper_gain=[0.1, 100]
    )

    print(f"  Loop initialized: main self.delay = {pi_iq_loop_default_damper_delay.delay}")
    print(f"  Loop initialized: damper self.damper_delay = {pi_iq_loop_default_damper_delay.damper_delay}")
    assert pi_iq_loop_default_damper_delay.delay == main_loop_delay_case2
    assert pi_iq_loop_default_damper_delay.damper_delay == main_loop_delay_case2
    print("  Damper delay correctly defaulted to main delay.")

    print("\n--- Example Setup Complete ---")
    print("This script demonstrates the configuration of delays.")
    print("To run a full simulation, you would typically add a Beam object,")
    print("append the loop to cav_res.feedback, and run a tracking loop.")

    # Example of how you might add it to feedback (not run in this simple demo)
    # cav_res.feedback.append(pi_iq_loop_explicit_damper_delay)
    #
    # beam = Beam(ring, n_macroparticles=1000, n_bunches=ring.h)
    # for bunch_idx in range(ring.h):
    #     if bunch_idx % 2 == 0: # Example: Fill every other bunch
    #         bunch = Bunch(ring, n_macroparticles=1000)
    #         bunch.tau = np.random.normal(0, 100e-12, bunch.n_macroparticles)
    #         bunch.delta = np.random.normal(0, 1e-4, bunch.n_macroparticles)
    #         beam[bunch_idx] = bunch
    # beam.update_filling_pattern()
    #
    # print("\nSimulating a few turns (conceptual)...")
    # for turn in range(5):
    #     cav_res.track(beam) # This would involve the feedback loop's track method
    #     # Add other ring elements tracking here
    #     print(f"Turn {turn+1} - Cavity Vc: {cav_res.cavity_voltage:.2f}, Vg: {cav_res.Vg:.2f}")


if __name__ == "__main__":
    main()
