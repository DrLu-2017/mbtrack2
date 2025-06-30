import numpy as np
from tqdm import tqdm
import os
from mbtrack2.tracking.rf import CavityResonator, ProportionalIntegralIQLoop
from mbtrack2.tracking.particles import Beam
from mbtrack2 import Synchrotron, Electron, Optics, LongitudinalMap, SynchrotronRadiation
# from mbtrack2.tracking.monitor import CavityMonitor  # Uncomment if needed
import matplotlib.pyplot as plt

def restart(ring, I0=0.001, tot_turns=500):
    h = ring.h
    fill_ptrn = np.zeros(h)
    fill_ptrn[0:h] = I0 / h
    mybeam = Beam(ring)
    mybeam.init_beam(fill_ptrn, mp_per_bunch=1)

    m = 1  # Harmonic number of the cavity
    Rs = 5e6  # Shunt impedance [Ohm]
    Q = 35e3
    QL = 5e3
    detune = -100e3  # Hz
    Ncav = 4
    MC = CavityResonator(ring, m, Rs, Q, QL, detune, Ncav=Ncav)

    MC.Vc = 1e6
    MC.theta = np.arccos(ring.U0 / MC.Vc)
    MC.set_optimal_detune(I0)
    MC.set_generator(I0)

    # If MCmon exists, close and delete old HDF5 file
    if 'MCmon' in globals():
        try:
            globals()["MCmon"].close()
        except Exception as e:
            print(f"Warning: MCmon close failed: {e}")

    if os.path.exists("save.hdf5"):
        try:
            os.remove("save.hdf5")
        except Exception as e:
            print(f"Warning: Failed to delete save.hdf5: {e}")

    # Uncomment if you want to use CavityMonitor
    # total_size = int(tot_turns / 5)
    # MCmon = CavityMonitor("MC", ring, file_name="save",
    #                       total_size=total_size, save_every=5,
    #                       buffer_size=10, mpi_mode=False)
    # return mybeam, MC, MCmon
    return mybeam, MC

def run_pi_iq_test(
    ring, I0, gain, n_turns=500, with_beam_physics=False, long=None, rad=None, MCmon=None
):
    mybeam, MC = restart(ring, I0, tot_turns=n_turns)
    MC.feedback = []
    iq_loop = ProportionalIntegralIQLoop(ring, MC, gain=gain, sample_num=8, every=2, delay=1)
    MC.feedback.append(iq_loop)
    setpoint = (MC.Vc, MC.theta)
    iq_loop.update_references(Vc=MC.Vc, theta=MC.theta)
    errors = []
    for i in range(n_turns):
        if with_beam_physics:
            if long is not None:
                long.track(mybeam)
            if rad is not None:
                rad.track(mybeam)
        MC.track(mybeam)
        if MCmon is not None:
            MCmon.track(mybeam, MC)
        error = np.abs(MC.cavity_voltage - MC.Vc)
        errors.append(error)
    return np.mean(errors), np.max(errors)

def optimize_pi_gain(ring, I0, P_range, I_range, n_turns=500):
    best_rms = float('inf')
    best_gain = None
    best_max = float('inf')
    results = []
    for P in tqdm(P_range, desc=f"PI tuning for I0={I0}"):
        for I in I_range:
            rms_err, max_err = run_pi_iq_test(ring, I0, gain=[P, I], n_turns=n_turns)
            results.append((P, I, rms_err, max_err))
            if rms_err < best_rms:
                best_rms = rms_err
                best_max = max_err
                best_gain = (P, I)
    print(f"Best gain for I0={I0}: P={best_gain[0]}, I={best_gain[1]}, RMS error={best_rms:.2f}, Max error={best_max:.2f}")
    return best_gain, results

# --- Setup ring as in your example ---
from mbtrack2.tracking.synchrotron import Synchrotron
from mbtrack2.tracking.particles import Proton

# --- Custom ring and optics parameters ---
h = 20  # Harmonic number
L = 100  # Ring circumference [m]
E0 = 1.5e9  # Energy [eV]
particle = Electron()
ac = 1e-3
U0 = 200e3
tau = np.array([1e-3, 1e-3, 2e-3])
tune = np.array([12.2, 15.3])
emit = np.array([10e-9, 10e-12])
sigma_0 = 15e-12
sigma_delta = 1e-3
chro = [2.0, 3.0]
local_beta = np.array([3, 2])
local_alpha = np.array([0, 0])
local_dispersion = np.array([0, 0, 0, 0])
optics = Optics(local_beta=local_beta, local_alpha=local_alpha, local_dispersion=local_dispersion)

ring = Synchrotron(
    h=h, optics=optics, particle=particle, L=L, E0=E0, ac=ac,
    U0=U0, tau=tau, emit=emit, tune=tune,
    sigma_delta=sigma_delta, sigma_0=sigma_0, chro=chro
)

long = LongitudinalMap(ring) # define the LongitudinalMap element with the ring parameters
rad = SynchrotronRadiation(ring)


# --- PI gain sweep ranges ---
P_range = np.linspace(0.5, 5, 6)   # e.g., [0.5, 1, 2, 3, 4, 5]
I_range = np.logspace(3, 5, 5)     # e.g., [1e3, 3e3, 1e4, 3e4, 1e5]

# --- Optimize for I0 = 0.001 A ---
best_gain_1, results_1 = optimize_pi_gain(ring, I0=0.001, P_range=P_range, I_range=I_range, n_turns=500)

# --- Optimize for I0 = 0.2 A ---
best_gain_2, results_2 = optimize_pi_gain(ring, I0=0.2, P_range=P_range, I_range=I_range, n_turns=500)

# --- Plotting and Summary Report ---
def plot_results(results, P_range, I_range, title, filename):
    # Convert results to arrays for plotting
    P_vals = np.array([r[0] for r in results])
    I_vals = np.array([r[1] for r in results])
    rms_vals = np.array([r[2] for r in results])
    max_vals = np.array([r[3] for r in results])
    # Reshape for contour plotting
    rms_grid = rms_vals.reshape(len(P_range), len(I_range))
    max_grid = max_vals.reshape(len(P_range), len(I_range))
    I_mesh, P_mesh = np.meshgrid(I_range, P_range)
    plt.figure(figsize=(8, 6))
    cp = plt.contourf(I_mesh, P_mesh, rms_grid, levels=20, cmap='viridis')
    plt.colorbar(cp, label='RMS Error [V]')
    plt.xlabel('Integral Gain (I)')
    plt.ylabel('Proportional Gain (P)')
    plt.title(f'PI-IQ Loop RMS Error: {title}')
    plt.xscale('log')
    plt.savefig(filename + '_rms.png', dpi=150)
    plt.show()
    plt.figure(figsize=(8, 6))
    cp2 = plt.contourf(I_mesh, P_mesh, max_grid, levels=20, cmap='magma')
    plt.colorbar(cp2, label='Max Error [V]')
    plt.xlabel('Integral Gain (I)')
    plt.ylabel('Proportional Gain (P)')
    plt.title(f'PI-IQ Loop Max Error: {title}')
    plt.xscale('log')
    plt.savefig(filename + '_max.png', dpi=150)
    plt.show()

print("\n--- PI-IQ Loop Gain Optimization Summary ---")
print(f"Best gain for I0=0.001 A: P={best_gain_1[0]}, I={best_gain_1[1]}")
print(f"Best gain for I0=0.2   A: P={best_gain_2[0]}, I={best_gain_2[1]}")

plot_results(results_1, P_range, I_range, 'I0 = 0.001 A', 'piiq_I0_0p001')
plot_results(results_2, P_range, I_range, 'I0 = 0.2 A', 'piiq_I0_0p2')

print("\nFull results for I0=0.001 A:")
for r in results_1:
    print(f"P={r[0]:.2f}, I={r[1]:.1f}, RMS={r[2]:.2f}, Max={r[3]:.2f}")
print("\nFull results for I0=0.2 A:")
for r in results_2:
    print(f"P={r[0]:.2f}, I={r[1]:.1f}, RMS={r[2]:.2f}, Max={r[3]:.2f}")