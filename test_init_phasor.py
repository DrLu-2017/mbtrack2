import numpy as np
from mbtrack2.tracking import Synchrotron, Electron, Beam, CavityResonator
from mbtrack2.utilities import Optics # Optics needs to be imported from utilities

# Define ring parameters (simplified from notebook)
h = 20               # Harmonic number
L = 100              # Circumference [m]
E0 = 1.5e9           # Nominal energy [eV]
particle = Electron()
ac = 1e-3            # Momentum compaction factor
U0 = 200e3           # Energy loss per turn [eV] # Made this up, notebook calculates it based on Vc usually
# For the purpose of testing init_phasor, precise U0 might not be critical for Vc=1e6 target if theta is calculated
# but it's good to have a value. Notebook uses Vc to find U0 for theta, here Vc is target.
# Let's set a plausible theta directly or calculate U0 from a plausible synchronous phase.
# The notebook calculates theta = np.arccos(ring.U0/Vc). If Vc is 1e6, and U0 is 200e3, theta is arccos(0.2)

# Local optics at tracking point (simplified)
local_beta = np.array([3, 2])
local_alpha = np.array([0, 0])
local_dispersion = np.array([0, 0, 0, 0])
optics = Optics(local_beta=local_beta, local_alpha=local_alpha, local_dispersion=local_dispersion)

# Create Synchrotron object
tau_ring = np.array([1e-3, 1e-3, 2e-3])
tune_ring = np.array([12.2, 15.3])
emit_ring = np.array([10e-9, 10e-12])
sigma_0_ring = 15e-12
sigma_delta_ring = 1e-3

ring = Synchrotron(h=h, optics=optics, particle=particle, L=L, E0=E0, ac=ac, U0=U0,
                   tau=tau_ring, emit=emit_ring, tune=tune_ring,
                   sigma_delta=sigma_delta_ring, sigma_0=sigma_0_ring)
# Note: Other ring params like chro are not strictly needed for this specific test.

# Create a Beam
filling_pattern = np.zeros(ring.h, dtype=bool)
filling_pattern[0] = True  # Single bunch for simplicity
beam = Beam(ring)
beam.init_beam(filling_pattern=filling_pattern, current_per_bunch=1e-3, mp_per_bunch=100)

# CavityResonator parameters
m_cav = 1 # Assuming fundamental cavity for simplicity, notebook also uses m=1 for CavityResonator
Rs_cav = 1e6            # Shunt impedance [Ohm]
Q_cav = 10000
QL_cav = 5000
detune_cav = 100        # Detuning in Hz
Ncav_cav = 1
Vc_target = 1.0e6       # Target cavity voltage [V]
# Calculate theta based on U0 and Vc_target, as in the notebook for consistency
# Ensure U0 is less than Vc_target for arccos to be valid
if U0 >= Vc_target:
    print(f"Warning: U0 ({U0}) is greater than or equal to Vc_target ({Vc_target}). Theta calculation might be invalid.")
    # Fallback theta if U0 is too high, though this indicates inconsistent parameters
    theta_target = 0.0
else:
    theta_target = np.arccos(U0 / Vc_target)

n_bin_cav = 75

cavity = CavityResonator(ring, m=m_cav, Rs=Rs_cav, Q=Q_cav, QL=QL_cav, detune=detune_cav,
                         Ncav=Ncav_cav, Vc=Vc_target, theta=theta_target, n_bin=n_bin_cav)

print(f"CavityResonator initialized. Target Vc = {cavity.Vc}, Target Theta = {cavity.theta}")
print(f"Initial beam_phasor: {cavity.beam_phasor}")
print(f"Initial generator_phasor: {cavity.generator_phasor}")
print(f"Initial cavity_phasor: {cavity.cavity_phasor}")
print(f"Initial cavity_voltage: {cavity.cavity_voltage}")
print(f"Initial cavity_phase: {cavity.cavity_phase}")

# Call the modified init_phasor
cavity.init_phasor(beam)

print("\nAfter init_phasor(beam):")
print(f"Calculated beam_phasor: {cavity.beam_phasor}")
print(f"Set generator_phasor (Vg*exp(j*theta_g)): {cavity.Vg * np.exp(1j * cavity.theta_g)}")
print(f"Resulting cavity_phasor (gen + beam): {cavity.cavity_phasor}")
print(f"Resulting cavity_voltage: {cavity.cavity_voltage}")
print(f"Resulting cavity_phase: {cavity.cavity_phase}")
print(f"Target Vc: {Vc_target}")
print(f"Target Theta: {theta_target}")

# Assertions
voltage_check = np.isclose(cavity.cavity_voltage, Vc_target, rtol=1e-5) # Relative tolerance for voltage
# For phase, ensure it's wrapped correctly if there are differences near +/- pi
# A simple way is to check the complex difference
cavity_phasor_complex = cavity.cavity_voltage * np.exp(1j * cavity.cavity_phase)
target_phasor_complex = Vc_target * np.exp(1j * theta_target)
# Check difference of unit phasors to compare phase, or check angle difference carefully
phase_diff = np.angle(cavity_phasor_complex / target_phasor_complex) # Phase difference
phase_check = np.isclose(phase_diff, 0.0, atol=1e-5) # Absolute tolerance for phase difference in radians

print(f"\nVoltage check (isclose to target {Vc_target}?): {voltage_check}")
print(f"Phase check (isclose to target {theta_target}?): {phase_check}")

assert voltage_check, f"Cavity voltage {cavity.cavity_voltage} is not close to target {Vc_target}"
assert phase_check, f"Cavity phase {cavity.cavity_phase} (diff {phase_diff} rad) is not close to target {theta_target}"

print("\nUnit test passed: init_phasor correctly sets cavity voltage and phase.")
