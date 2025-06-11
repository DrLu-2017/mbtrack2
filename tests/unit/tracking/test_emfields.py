import numpy as np
import pytest

from mbtrack2.tracking.emfields import (
    _efieldn_linearized,
    _efieldn_mit,
    _sqrt_sig,
    _wofz,
    add_sigma_check,
    efieldn_gauss_round,
    get_displaced_efield,
)

pytestmark = pytest.mark.unit


class Test_particles_electromagnetic_fields:

    def test_wofz_return_float(self):
        real, imag = _wofz(1.0, 1.0)
        assert isinstance(real, float)
        assert isinstance(imag, float)

    def test_sqrt_sig_return_positive(self):
        val = _sqrt_sig(1.0, 1.0)
        assert isinstance(val, float)
        assert val >= 0

    @pytest.mark.parametrize("func", [(_efieldn_mit), (efieldn_gauss_round),
                                      (_efieldn_linearized)])
    def test_efieldn_return_float(self, func):
        ex, ey = func(1.0, 1.0, 1.0, 1.0)
        assert isinstance(ex, float)
        assert isinstance(ey, float)

    # Maintains original function behavior when sig_x is greater than sig_y
    def test_add_sigma_check_maintain_original_behavior_when_sig_x_greater_than_sig_y(
            self):

        def mock_efieldn(x, y, sig_x, sig_y):
            return x + sig_x, y + sig_y

        wrapped_function = add_sigma_check(mock_efieldn)
        result = wrapped_function(np.array([1.0]), np.array([1.0]), 2.0, 1.0)
        assert result == (3.0, 2.0)

    # Exchanges x and y when sig_x is less than sig_y
    def test_add_sigma_check_exchange_x_y_when_sig_x_less_than_sig_y(self):

        def mock_efieldn(x, y, sig_x, sig_y):
            return x + sig_x, y + sig_y

        wrapped_function = add_sigma_check(mock_efieldn)
        result = wrapped_function(np.array([1.0]), np.array([1.0]), 1.0, 2.0)
        assert result == (2.0, 3.0)

    # Applies round beam field formula when sig_x is close to sig_y
    def test_add_sigma_check_apply_round_beam_formula_when_sigmas_close(self):
        wrapped_function = add_sigma_check(efieldn_gauss_round)
        result = wrapped_function(np.array([1.0]), np.array([1.0]), 1.0,
                                  1.00001)
        assert np.allclose(
            result,
            efieldn_gauss_round(np.array([1.0]), np.array([1.0]), 1.0,
                                1.00001))

    # Returns zero fields when sig_x and sig_y are both close to zero
    def test_add_sigma_check_zero_fields_when_sigmas_close_to_zero(self):
        wrapped_function = add_sigma_check(efieldn_gauss_round)
        result = wrapped_function(np.array([1.0]), np.array([1.0]), 1e-11,
                                  1e-11)
        assert np.allclose(result, (np.zeros(1), np.zeros(1)))

    def test_add_sigma_check_empty_arrays(self):
        # Define a mock efieldn function
        def mock_efieldn(x, y, sig_x, sig_y):
            return np.zeros_like(x), np.zeros_like(y)

        # Wrap the mock function with add_sigma_check
        wrapped_function = add_sigma_check(mock_efieldn)

        # Create empty arrays for x and y
        x = np.array([])
        y = np.array([])
        sig_x = 1.0
        sig_y = 1.0

        # Call the wrapped function
        en_x, en_y = wrapped_function(x, y, sig_x, sig_y)

        # Assert that the output is also empty arrays
        assert en_x.size == 0
        assert en_y.size == 0

    # Computes electric field for typical Gaussian charge distribution
    def test_get_displaced_efield_return_shape(self):

        def mock_efieldn(x, y, sig_x, sig_y):
            return np.ones_like(x), np.ones_like(y)

        xr = np.array([1.0, 2.0, 3.0])
        yr = np.array([1.0, 2.0, 3.0])
        sig_x = 2.0
        sig_y = 1.0
        mean_x = 0.0
        mean_y = 0.0

        en_x, en_y = get_displaced_efield(mock_efieldn, xr, yr, sig_x, sig_y,
                                          mean_x, mean_y)

        assert np.allclose(en_x, [1.0, 1.0, 1.0])
        assert np.allclose(en_y, [1.0, 1.0, 1.0])
