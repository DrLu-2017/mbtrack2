import pytest

pytestmark = pytest.mark.unit
import numpy as np
from utility_test_functions import assert_attr_changed

from mbtrack2 import (
    CavityResonator,
    DirectFeedback,
    ProportionalIntegralLoop,
    ProportionalIntegralIQLoopMode0Damper, # Added here
    ProportionalLoop,
    RFCavity,
    TunerLoop,
)


class TestRFCavity:

    # Initialize RFCavity with valid parameters and verify attributes are changed when tracked
    def test_rf_cavity(self, demo_ring, small_bunch):
        cavity = RFCavity(ring=demo_ring, m=1, Vc=1e6, theta=np.pi / 4)
        assert_attr_changed(cavity, small_bunch, attrs_changed=["delta"])


@pytest.fixture
def cav_res(demo_ring):
    cav_res = CavityResonator(demo_ring,
                              m=1,
                              Rs=1e6,
                              Q=1e4,
                              QL=1e3,
                              detune=-20e3,
                              Ncav=4)
    return cav_res


@pytest.fixture
def cav_res_tracking(demo_ring, cav_res):
    cav_res.Vc = 1e6
    cav_res.theta = np.arccos(demo_ring.U0 / cav_res.Vc)
    cav_res.set_generator(0.5)
    return cav_res


class TestCavityResonator:

    @pytest.mark.parametrize("n_bin", [(75), (1)])
    def test_track_mpi(self, cav_res_tracking, beam_1bunch_mpi, n_bin):
        cav_res_tracking.n_bin = n_bin
        initial_phasor = cav_res_tracking.beam_phasor
        initial_beam_phasor_record = cav_res_tracking.beam_phasor_record.copy()

        beam_1bunch_mpi.mpi.share_distributions(beam_1bunch_mpi, n_bin=n_bin)
        assert_attr_changed(cav_res_tracking,
                            beam_1bunch_mpi,
                            attrs_changed=["delta"])

        assert not np.array_equal(cav_res_tracking.beam_phasor, initial_phasor)
        assert not np.array_equal(cav_res_tracking.beam_phasor_record,
                                  initial_beam_phasor_record)

    @pytest.mark.parametrize("n_bin", [(75), (1)])
    def test_track_no_mpi(self, cav_res_tracking, beam_uniform, n_bin):
        cav_res_tracking.n_bin = n_bin
        initial_phasor = cav_res_tracking.beam_phasor
        initial_beam_phasor_record = cav_res_tracking.beam_phasor_record.copy()

        assert_attr_changed(cav_res_tracking,
                            beam_uniform,
                            attrs_changed=["delta"])

        assert not np.array_equal(cav_res_tracking.beam_phasor, initial_phasor)
        assert not np.array_equal(cav_res_tracking.beam_phasor_record,
                                  initial_beam_phasor_record)

    @pytest.mark.parametrize("n_bin", [(75), (1)])
    def test_track_no_mpi_non_uniform(self, cav_res_tracking, beam_non_uniform,
                                      n_bin):
        cav_res_tracking.n_bin = n_bin
        initial_phasor = cav_res_tracking.beam_phasor
        initial_beam_phasor_record = cav_res_tracking.beam_phasor_record.copy()

        assert_attr_changed(cav_res_tracking,
                            beam_non_uniform,
                            attrs_changed=["delta"])

        assert not np.array_equal(cav_res_tracking.beam_phasor, initial_phasor)
        assert not np.array_equal(cav_res_tracking.beam_phasor_record,
                                  initial_beam_phasor_record)

    # Track a beam with zero charge per macro-particle
    def test_track_zero_charge_per_mp(self, cav_res_tracking, generate_beam):
        beam = generate_beam(current_per_bunch=0)
        assert_attr_changed(cav_res_tracking, beam, change=False)

    # Apply RF feedback loops correctly during tracking
    def test_apply_rf_feedback(self, cav_res_tracking, beam_uniform, mocker):
        cav_res_tracking.feedback.append(mocker.Mock())
        cav_res_tracking.track(beam_uniform)
        cav_res_tracking.feedback[0].track.assert_called_once(
        ), "RF feedback should be applied"

    @pytest.mark.parametrize("ref_frame", [("beam"), ("rf")])
    def test_phasor_decay(self, cav_res, ref_frame):
        init_phasor = 1 + 1j
        cav_res.beam_phasor = init_phasor
        cav_res.phasor_decay(10e-6, ref_frame)
        assert cav_res.beam_phasor != init_phasor

    @pytest.mark.parametrize("ref_frame", [("beam"), ("rf")])
    def test_phasor_evol(self, cav_res, ref_frame):
        init_phasor = 1 + 1j
        cav_res.beam_phasor = init_phasor
        profile = np.random.randint(1000, size=(10, ))
        bin_length = 1e-12
        charge_per_mp = 1e-15
        cav_res.phasor_evol(profile, bin_length, charge_per_mp, ref_frame)
        assert cav_res.beam_phasor != init_phasor

    def test_phasor_init(self, cav_res, beam_non_uniform):
        init_phasor = cav_res.beam_phasor
        cav_res.init_phasor_track(beam_non_uniform)
        phasor_init_phasor_track = cav_res.beam_phasor

        cav_res.beam_phasor = init_phasor
        cav_res.init_phasor(beam_non_uniform)
        phasor_init_phasor = cav_res.beam_phasor

        assert phasor_init_phasor != init_phasor
        assert phasor_init_phasor_track != init_phasor
        assert np.allclose(phasor_init_phasor,
                           phasor_init_phasor_track,
                           rtol=1e-2)

    def test_phasor_init_mpi(self, cav_res, beam_1bunch_mpi):
        init_phasor = cav_res.beam_phasor
        cav_res.init_phasor_track(beam_1bunch_mpi)
        phasor_init_phasor_track = cav_res.beam_phasor

        cav_res.beam_phasor = init_phasor
        cav_res.init_phasor(beam_1bunch_mpi)
        phasor_init_phasor = cav_res.beam_phasor

        assert phasor_init_phasor != init_phasor
        assert phasor_init_phasor_track != init_phasor
        assert np.allclose(phasor_init_phasor,
                           phasor_init_phasor_track,
                           rtol=1e-2)

    # Setting detune updates _detune, _fr, _wr, and _psi correctly
    def test_detune(self, cav_res):
        detune_value = 1000
        cav_res.detune = detune_value
        assert cav_res._detune == detune_value
        assert cav_res._fr == detune_value + cav_res.m * cav_res.ring.f1
        assert cav_res._wr == cav_res._fr * 2 * np.pi
        assert cav_res._psi == np.arctan(
            cav_res.QL * (cav_res._fr / (cav_res.m * cav_res.ring.f1) -
                          (cav_res.m * cav_res.ring.f1) / cav_res._fr))

    # update_feedback is called after setting detune
    def test_update_feedback_called(self, mocker, cav_res):
        mocker.spy(cav_res, 'update_feedback')
        cav_res.detune = 1000
        cav_res.update_feedback.assert_called_once()

    def test_plot_phasor_diagram(self, cav_res_tracking):
        fig = cav_res_tracking.plot_phasor(0.5)
        assert fig is not None

    # Calculating generator phasor using Vg and theta_g
    def test_generator_phasor_calculation(self, cav_res):
        cav_res.Vg = 1000
        cav_res.theta_g = np.pi / 4
        expected_phasor = 1000 * np.exp(1j * np.pi / 4)
        assert cav_res.generator_phasor == expected_phasor

    # Computing cavity phasor by summing generator and beam phasors
    def test_cavity_phasor_computation(self, cav_res):
        cav_res.Vg = 1000
        cav_res.theta_g = np.pi / 4
        cav_res.beam_phasor = 500 * np.exp(1j * np.pi / 6)
        expected_phasor = cav_res.generator_phasor + cav_res.beam_phasor
        assert cav_res.cavity_phasor == expected_phasor

    # Retrieving cavity phasor record for each bunch
    def test_cavity_phasor_record_retrieval(self, cav_res):
        cav_res.generator_phasor_record = np.array([1 + 1j, 2 + 2j])
        cav_res.beam_phasor_record = np.array([0.5 + 0.5j, 1 + 1j])
        expected_record = np.array([1.5 + 1.5j, 3 + 3j])
        assert np.allclose(cav_res.cavity_phasor_record, expected_record)

    # Accessing ig phasor record when feedback is present
    def test_ig_phasor_record_with_feedback(self, cav_res):
        pi_loop = ProportionalIntegralLoop(cav_res.ring, cav_res, [1.0, 1.0],
                                           10, 10, 10)
        pi_loop.ig_phasor_record = np.array([1 + 1j, 2 + 2j])
        cav_res.feedback.append(pi_loop)
        assert np.allclose(cav_res.ig_phasor_record, pi_loop.ig_phasor_record)

    # Determining cavity voltage and phase from cavity phasor
    def test_cavity_voltage_and_phase_determination(self, cav_res):
        cav_res.Vg = 1000
        cav_res.theta_g = np.pi / 4
        cav_res.beam_phasor = 500 * np.exp(1j * np.pi / 6)
        expected_voltage = np.abs(cav_res.cavity_phasor)
        expected_phase = np.angle(cav_res.cavity_phasor)
        assert cav_res.cavity_voltage == expected_voltage
        assert cav_res.cavity_phase == expected_phase

    # Calculating beam voltage and phase from beam phasor
    def test_beam_voltage_and_phase_calculation(self, cav_res):
        cav_res.beam_phasor = 500 * np.exp(1j * np.pi / 6)
        expected_voltage = np.abs(cav_res.beam_phasor)
        expected_phase = np.angle(cav_res.beam_phasor)
        assert cav_res.beam_voltage == expected_voltage
        assert cav_res.beam_phase == expected_phase

    # Handling feedback list with no ProportionalIntegralLoop or DirectFeedback
    def test_ig_phasor_record_no_feedback(self, cav_res):
        cav_res.feedback = []
        expected_record = np.zeros(cav_res.ring.h)
        assert np.allclose(cav_res.ig_phasor_record, expected_record)

    # Calculating loaded shunt impedance RL
    def test_loaded_shunt_impedance_rl(self, cav_res):
        # Set up the initial conditions
        cav_res.Rs = 1e6  # Shunt impedance
        cav_res.QL = 1e3  # Loaded quality factor
        cav_res.Q = 1e4  # Quality factor

        # Calculate expected RL
        expected_RL = cav_res.Rs / (1 + (cav_res.Q / cav_res.QL - 1))

        # Assert that the calculated RL matches the expected value
        assert cav_res.RL == expected_RL

    # Set optimal coupling and detuning and verify changes in attributes
    def test_set_optimal_coupling_and_detuning(self, demo_ring, cav_res):
        initial_beta = cav_res.beta
        initial_psi = cav_res.psi
        cav_res.set_optimal_coupling(0.5)
        cav_res.set_optimal_detune(0.5)
        assert cav_res.beta != initial_beta
        assert cav_res.psi != initial_psi

    # Verify that is_CBI_stable correctly calculates growth rate and mode number
    def test_calculate_growth_rate_and_mode(self, cav_res_tracking):
        growth_rate, mu = cav_res_tracking.is_CBI_stable(I0=0.5)
        assert isinstance(growth_rate, float)
        assert np.issubdtype(type(mu), np.integer)

    # Ensure is_CBI_stable returns growth rates for specified modes when modes is provided
    def test_return_growth_rates_for_specified_modes(self, cav_res_tracking):
        modes = [0, 1, 2]
        growth_rates = cav_res_tracking.is_CBI_stable(I0=0.5, modes=modes)
        assert len(growth_rates) == len(modes)

    def test_is_DC_Robinson_stable(self, cav_res_tracking):
        assert isinstance(cav_res_tracking.is_DC_Robinson_stable(0.5),
                          (bool, np.bool_))

    def test_plot_DC_Robinson_stability(self, cav_res_tracking):
        fig = cav_res_tracking.plot_DC_Robinson_stability()
        assert fig is not None

    @pytest.mark.parametrize("z,expected_type",
                             [(1e-3, float),
                              (np.linspace(-1e-3, 1e-3, 100), np.ndarray)])
    def test_VRF(self, cav_res_tracking, z, expected_type):
        assert isinstance(cav_res_tracking.VRF(z, 0.5), expected_type)
        assert isinstance(cav_res_tracking.dVRF(z, 0.5), expected_type)
        assert isinstance(cav_res_tracking.ddVRF(z, 0.5), expected_type)
        assert isinstance(cav_res_tracking.deltaVRF(z, 0.5), expected_type)

    def test_sample_voltage_default_parameters(self, cav_res_tracking,
                                               beam_uniform):
        cav_res_tracking.init_tracking(beam_uniform)
        n_points = 1e4
        beam_phasor_init = 1 + 1j
        cav_res_tracking.beam_phasor = beam_phasor_init
        pos, voltage_rec = cav_res_tracking.sample_voltage(n_points=n_points)
        assert len(pos) == n_points
        assert len(voltage_rec) == n_points
        assert np.allclose(cav_res_tracking.beam_phasor, beam_phasor_init)

    def test_set_generator(self, cav_res, demo_ring):
        cav_res.Vc = 1e6
        cav_res.theta = np.arccos(demo_ring.U0 / cav_res.Vc)
        cav_res.set_generator(0.5)
        assert cav_res.Pg is not None
        assert cav_res.Vgr is not None
        assert cav_res.theta_gr is not None
        assert cav_res.Vg is not None
        assert cav_res.theta_g is not None
        assert cav_res.generator_phasor_record is not None

    def test_functions(self, cav_res_tracking):
        assert isinstance(cav_res_tracking.Pb(0.5), float)
        assert isinstance(cav_res_tracking.Pr(0.5), float)
        assert isinstance(cav_res_tracking.Vbr(0.5), float)
        assert isinstance(cav_res_tracking.Vb(0.5), float)
        assert isinstance(cav_res_tracking.Z(1e9), complex)

    def test_properties(self, cav_res_tracking):
        assert cav_res_tracking.Pc > 0
        assert cav_res_tracking.filling_time > 0
        assert cav_res_tracking.loss_factor > 0


class TestProportionalLoop:

    @pytest.fixture
    def prop_loop(self, demo_ring, cav_res_tracking):
        return ProportionalLoop(demo_ring,
                                cav_res_tracking,
                                gain_A=0.5,
                                gain_P=0.5,
                                delay=2)

    # track method updates cav_res.Vg and cav_res.theta_g correctly
    def test_track_delay(self, prop_loop):
        initial_Vg = prop_loop.cav_res.Vg
        initial_theta_g = prop_loop.cav_res.theta_g
        prop_loop.track()
        assert prop_loop.cav_res.Vg == initial_Vg
        assert prop_loop.cav_res.theta_g == initial_theta_g

    # track method updates cav_res.Vg and cav_res.theta_g correctly
    def test_track_update(self, prop_loop):
        initial_Vg = prop_loop.cav_res.Vg
        initial_theta_g = prop_loop.cav_res.theta_g
        prop_loop.track()
        prop_loop.track()  # delay=2 is 3 turns to take action
        prop_loop.track()
        assert prop_loop.cav_res.Vg != initial_Vg
        assert prop_loop.cav_res.theta_g != initial_theta_g

    # Initialization with delay less than 1 raises ValueError
    def test_initialization_delay_less_than_one(self, demo_ring, cav_res):
        with pytest.raises(ValueError):
            ProportionalLoop(demo_ring,
                             cav_res,
                             gain_A=0.5,
                             gain_P=0.5,
                             delay=0)

    # Initialization with non-integer delay converts it to integer
    def test_initialization_non_integer_delay(self, demo_ring, cav_res):
        loop = ProportionalLoop(demo_ring,
                                cav_res,
                                gain_A=0.5,
                                gain_P=0.5,
                                delay=2.7)
        assert loop.delay == 2


class TestTunerLoop:

    @pytest.fixture
    def tuner_loop(self, demo_ring, cav_res_tracking):
        return TunerLoop(demo_ring,
                         cav_res_tracking,
                         gain=1,
                         avering_period=2,
                         offset=0)

    def test_track_once(self, tuner_loop):
        initial_psi = tuner_loop.cav_res.psi
        tuner_loop.track()
        assert tuner_loop.diff != 0
        assert tuner_loop.count == 1
        assert tuner_loop.cav_res.psi == initial_psi

    # Track method correctly adjusts cav_res.psi when count equals avering_period
    def test_track_adjusts_psi(self, tuner_loop):
        initial_psi = tuner_loop.cav_res.psi
        for _ in range(tuner_loop.avering_period + 1):
            tuner_loop.track()
        assert tuner_loop.diff == 0
        assert tuner_loop.cav_res.psi != initial_psi

    # Initialize with avering_period set to None and verify default calculation
    def test_default_averaging_period_calculation(self, demo_ring,
                                                  cav_res_tracking):
        tuner_loop = TunerLoop(demo_ring,
                               cav_res_tracking,
                               avering_period=None)
        fs = demo_ring.synchrotron_tune(cav_res_tracking.Vc) * demo_ring.f0
        expected_period = int(2 / fs / demo_ring.T0)
        assert tuner_loop.avering_period == expected_period

    # Test track method with zero gain to ensure no changes to cav_res.psi
    def test_track_with_zero_gain(self, tuner_loop):
        tuner_loop.Pgain = 0
        initial_psi = tuner_loop.cav_res.psi
        for _ in range(tuner_loop.avering_period):
            tuner_loop.track()
        assert tuner_loop.cav_res.psi == initial_psi


class TestProportionalIntegralLoop:

    @pytest.fixture
    def pi_loop(self, demo_ring, cav_res_tracking):
        loop = ProportionalIntegralLoop(demo_ring, cav_res_tracking,
                                        [0.5, 1e4], 8, 7, 5)
        return loop

    def test_track(self, pi_loop):
        initial_ig_phasor = pi_loop.ig_phasor.copy()
        generator_phasor_record_init = pi_loop.cav_res.generator_phasor_record.copy(
        )
        pi_loop.track()
        assert not np.array_equal(initial_ig_phasor, pi_loop.ig_phasor)
        assert not np.array_equal(generator_phasor_record_init,
                                  pi_loop.cav_res.generator_phasor_record)

    # Ig2Vg method correctly updates generator phasor record and cavity parameters
    def test_Ig2Vg_updates_generator_phasor(self, pi_loop):
        generator_phasor_record_init = pi_loop.cav_res.generator_phasor_record.copy(
        )
        Vg_init = pi_loop.cav_res.Vg
        theta_g_init = pi_loop.cav_res.theta_g
        pi_loop.Ig2Vg()
        assert not np.allclose(generator_phasor_record_init,
                               pi_loop.cav_res.generator_phasor_record)
        assert pi_loop.cav_res.Vg != Vg_init
        assert pi_loop.cav_res.theta_g != theta_g_init

    # IIR filter processes input correctly and returns expected output
    def test_IIR_filter_output(self, pi_loop):
        input_signal = np.array([1.0 + 1.0j])
        output = pi_loop.IIR(input_signal)
        assert output is not None

    # Init_FFconst initializes feedforward constant based on FF flag
    def test_init_FFconst_with_FF_true(self, pi_loop):
        pi_loop.init_FFconst()
        assert pi_loop.FFconst != 0

    def test_parameters(self, pi_loop):
        assert isinstance(pi_loop.Vg2Ig(1e6), complex)
        pi_loop.IIR_init(1e9)
        assert isinstance(pi_loop.IIRcutoff, float)
        assert isinstance(pi_loop.IIR(1.0 + 1.0j), complex)


class TestDirectFeedback:

    @pytest.fixture
    def drf_fb(self, demo_ring, cav_res_tracking):
        drf_fb = DirectFeedback(DFB_gain=1.0,
                                DFB_phase_shift=0.1,
                                ring=demo_ring,
                                cav_res=cav_res_tracking,
                                gain=[0.5, 1e4],
                                sample_num=8,
                                every=7,
                                delay=5)
        return drf_fb

    def test_properties(self, drf_fb):
        assert isinstance(drf_fb.DFB_phase_shift, float)
        assert isinstance(drf_fb.phase_shift, float)
        assert isinstance(drf_fb.DFB_psi, float)
        assert isinstance(drf_fb.DFB_alpha, float)
        assert isinstance(drf_fb.DFB_gamma, float)
        assert isinstance(drf_fb.DFB_Rs, float)

    def test_methods(self, drf_fb):
        vg_main, vg_drf = drf_fb.DFB_Vg()
        assert isinstance(vg_main, complex)
        assert isinstance(vg_drf, complex)
        assert isinstance(drf_fb.DFB_fs(), float)

    def test_track(self, drf_fb):
        initial_ig_phasor_record = drf_fb.ig_phasor_record.copy()
        initial_dfb_ig_phasor = drf_fb.DFB_ig_phasor.copy()
        generator_phasor_record_init = drf_fb.cav_res.generator_phasor_record.copy(
        )
        drf_fb.track()
        assert not np.array_equal(drf_fb.ig_phasor_record,
                                  initial_ig_phasor_record)
        assert not np.array_equal(drf_fb.DFB_ig_phasor, initial_dfb_ig_phasor)
        assert not np.array_equal(generator_phasor_record_init,
                                  drf_fb.cav_res.generator_phasor_record)

    # Set DFB parameters using DFB_parameter_set and verify changes in DFB_ig_phasor
    def test_set_dfb_parameters(self, drf_fb):
        initial_DFB_ig_phasor = drf_fb.DFB_ig_phasor.copy()
        initial_ig_phasor = drf_fb.ig_phasor.copy()

        drf_fb.DFB_parameter_set(DFB_gain=2.0, DFB_phase_shift=0.2)

        assert not np.array_equal(drf_fb.DFB_ig_phasor, initial_DFB_ig_phasor)
        assert not np.array_equal(drf_fb.ig_phasor, initial_ig_phasor)
        assert drf_fb.DFB_gain == 2.0
        assert drf_fb.DFB_phase_shift == 0.2


@pytest.fixture
def pi_iq_loop_damper(demo_ring, cav_res_tracking):
    # Basic parameters for the loop
    gain = [0.5, 1e4]
    sample_num = 8
    every = 7
    delay = 10 # Main delay
    damper_delay = 5 # Specific damper delay
    loop = ProportionalIntegralIQLoopMode0Damper(
        ring=demo_ring,
        cav_res=cav_res_tracking,
        gain=gain,
        sample_num=sample_num,
        every=every,
        delay=delay,
        damper_delay=damper_delay,
        enable_damper=True
    )
    return loop


class TestProportionalIntegralIQLoopMode0Damper:
    def test_damper_delay_explicit(self, pi_iq_loop_damper):
        assert pi_iq_loop_damper.delay == 10
        assert pi_iq_loop_damper.damper_delay == 5, "damper_delay should be set to the explicit value"

    def test_damper_delay_defaults_to_main_delay(self, demo_ring, cav_res_tracking):
        gain = [0.5, 1e4]
        sample_num = 8
        every = 7
        delay = 12 # Main delay
        # damper_delay is not provided
        loop_no_specific_damper_delay = ProportionalIntegralIQLoopMode0Damper(
            ring=demo_ring,
            cav_res=cav_res_tracking,
            gain=gain,
            sample_num=sample_num,
            every=every,
            delay=delay,
            # damper_delay=None, # Explicitly None or omitted
            enable_damper=True
        )
        assert loop_no_specific_damper_delay.delay == 12
        assert loop_no_specific_damper_delay.damper_delay == delay, \
            "damper_delay should default to main delay when not provided"

        # Test with damper_delay explicitly set to None
        loop_damper_delay_none = ProportionalIntegralIQLoopMode0Damper(
            ring=demo_ring,
            cav_res=cav_res_tracking,
            gain=gain,
            sample_num=sample_num,
            every=every,
            delay=delay,
            damper_delay=None,
            enable_damper=True
        )
        assert loop_damper_delay_none.damper_delay == delay, \
            "damper_delay should default to main delay when set to None"

    def test_compute_mode0_signal_uses_damper_delay(self, pi_iq_loop_damper, demo_ring, cav_res_tracking):
        # Set up a beam mock or a simple beam if needed for compute_mode0_signal
        # For simplicity, we'll directly manipulate the buffer and check its size against damper_delay

        loop = pi_iq_loop_damper
        loop.damper_delay = 3 # Set a small, distinct damper_delay for testing
        loop.delay = 10 # Ensure main delay is different

        # Mock the _last_beam attribute if it's accessed and causes errors
        # For this test, we focus on the buffer logic based on damper_delay
        class MockBunch:
            def __init__(self, mean_val):
                self.mean = [0,0,0,0,mean_val] # mean_idx is 4 by default

        class MockBeam:
            def __init__(self, num_bunches):
                self.not_empty = [MockBunch(i * 0.1) for i in range(num_bunches)]

        loop.cav_res._last_beam = MockBeam(5) # Attach a mock beam

        # Fill the buffer up to damper_delay
        for i in range(loop.damper_delay):
            loop.buffer = [] # Clear buffer for precise testing of this call
            for j in range(i + 1):
                 loop.buffer.append((j + 1) * 0.01) # Add NON-ZERO dummy values
            # Call compute_mode0_signal, its internal value doesn't matter as much as buffer interaction
            val = loop.compute_mode0_signal()
            if i < loop.damper_delay -1 : # Before buffer is full (according to damper_delay)
                 assert val == 0.0, f"Should return 0.0 when buffer not full (i={i}, buffer_len={len(loop.buffer)})"
            else: # Buffer is full or overflowing according to damper_delay
                 assert val != 0.0, f"Should return a value when buffer full (i={i}, buffer_len={len(loop.buffer)})"

        # Check that buffer length does not exceed damper_delay after multiple calls
        # (it should pop if len > damper_delay)
        loop.buffer = []
        for _ in range(loop.damper_delay + 2): # Call more times than damper_delay
            loop.compute_mode0_signal()

        assert len(loop.buffer) == loop.damper_delay, \
            f"Buffer length ({len(loop.buffer)}) should be equal to damper_delay ({loop.damper_delay}) after multiple calls"
