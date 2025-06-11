import numpy as np
import pytest

pytestmark = pytest.mark.unit
from mbtrack2 import ComplexData, Impedance, WakeField, WakeFunction

component_list = ComplexData.name_and_coefficients_table().columns.to_list()


class TestComplexData:

    @pytest.fixture
    def complex_data(self):
        complex_data = ComplexData(variable=np.array([0, 1]),
                                   function=np.array([1 + 1j, 2 + 2j]))
        return complex_data

    # Add two ComplexData objects using 'zero' method and check resulting DataFrame
    def test_add_zero_method(self, complex_data):
        complex_data2 = ComplexData(variable=np.array([0, 1]),
                                    function=np.array([3 + 3j, 4 + 4j]))
        result = complex_data + complex_data2
        expected_real = np.array([4, 6])
        expected_imag = np.array([4, 6])
        assert np.allclose(result.data['real'], expected_real)
        assert np.allclose(result.data['imag'], expected_imag)

    # Multiply ComplexData by a scalar and verify the resulting DataFrame
    def test_multiply_by_scalar(self, complex_data):
        result = complex_data * 2
        expected_real = np.array([2, 4])
        expected_imag = np.array([2, 4])
        assert np.allclose(result.data['real'], expected_real)
        assert np.allclose(result.data['imag'], expected_imag)

    # Interpolate ComplexData using __call__ and verify the interpolated values
    def test_interpolation_call(self):
        complex_data = ComplexData(
            variable=np.array([0, 0.25, 0.75, 1]),
            function=np.array([1 + 1j, 1.25 + 1.25j, 1.75 + 1.75j, 2 + 2j]))
        interpolated_values = complex_data(np.array([0.5]))
        expected_values = np.array([1.5 + 1.5j])
        assert np.allclose(interpolated_values, expected_values)

    # Set component_type and verify coefficients and plane attributes
    def test_set_component_type(self):
        complex_data = ComplexData()
        complex_data.component_type = 'xdip'
        assert complex_data.a == 1
        assert complex_data.b == 0
        assert complex_data.c == 0
        assert complex_data.d == 0
        assert complex_data.plane == 'x'

    # Multiply ComplexData by a non-numeric type and expect a warning
    def test_multiply_by_non_numeric_warning(self):
        complex_data = ComplexData()
        with pytest.warns(UserWarning):
            result = complex_data.multiply('non-numeric')
            assert result is complex_data

    # Interpolate ComplexData with values outside the index range
    def test_interpolation_outside_range(self, complex_data):
        with pytest.raises(ValueError):
            complex_data(np.array([-0.5, 1.5]))

    # Interpolate ComplexData with values outside the index range
    def test_wrong_component_type(self, complex_data):
        with pytest.raises(KeyError):
            complex_data.component_type = "wrong"


@pytest.fixture(params=component_list)
def wf_param(request):
    wf = WakeFunction(variable=np.array([0, 1]),
                      function=np.array([1, 2]),
                      component_type=request.param)
    return wf


@pytest.fixture
def generate_wf():

    def generate(
            variable=np.array([0, 1]),
            function=np.array([1, 2]),
            component_type='long',
    ):
        wf = WakeFunction(variable=variable,
                          function=function,
                          component_type=component_type)
        return wf

    return generate


class TestWakeFunction:

    # Add two WakeFunction objects with the same component type and verify result
    def test_add_same_component_type(self, generate_wf):
        wf1 = generate_wf()
        wf2 = generate_wf(function=np.array([3, 4]))
        result = wf1 + wf2
        assert np.allclose(result.data['real'], [4, 6])
        assert result.component_type == 'long'

    # Multiply WakeFunction by a scalar and verify the result
    def test_multiply_by_scalar(self, generate_wf):
        wf = generate_wf()
        result = wf * 2
        assert np.allclose(result.data['real'], [2, 4])
        assert result.component_type == 'long'

    # Add WakeFunction objects with different component types and check for warnings
    def test_add_different_component_types_warning(self, generate_wf):
        wf1 = generate_wf(component_type='long')
        wf2 = generate_wf(component_type='xdip')
        with pytest.warns(
                UserWarning,
                match="do not have the same coordinates or plane or type"):
            result = wf1 + wf2
        assert result.data.equals(wf1.data)

    # Convert WakeFunction to Impedance and verify the transformation
    def test_convert_to_impedance(self, generate_wf):
        for component_type in ["long", "xdip", "ydip"]:
            wf = generate_wf(component_type=component_type)
            imp = wf.to_impedance(freq_lim=10)
            assert imp.component_type == component_type
            assert isinstance(imp, Impedance)

    # Test deconvolution
    def test_deconvolution(self, generate_wf):
        for component_type in ["long", "xdip", "ydip"]:
            wf = generate_wf(component_type=component_type)
            deconv_wf = wf.deconvolution(freq_lim=1e9, sigma=0.01, mu=0)
            assert isinstance(deconv_wf, WakeFunction)
            assert deconv_wf.component_type == component_type

    # Plot the WakeFunction data and verify the plot output
    def test_plot_output(self, wf_param):
        wf_param.plot()
        assert True

    # Plot the WakeFunction.loss_factor
    def test_loss_factor(self, wf_param):
        assert wf_param.loss_factor(1) > 0


@pytest.fixture(params=component_list)
def imp_param(request):
    imp = Impedance(variable=np.array([1, 2]),
                    function=np.array([3 + 4j, 5 + 6j]),
                    component_type=request.param)
    return imp


@pytest.fixture
def generate_imp():

    def generate(
            variable=np.array([1, 2]),
            function=np.array([3 + 4j, 5 + 6j]),
            component_type='long',
    ):
        imp = Impedance(variable=variable,
                        function=function,
                        component_type=component_type)
        return imp

    return generate


class TestImpedance:

    # Impedance objects can be added using the add method with matching component types
    def test_add_matching_component_types(self, generate_imp):
        imp1 = generate_imp()
        imp2 = generate_imp(function=np.array([1 + 1j, 1 + 1j]))
        result = imp1 + imp2
        expected_real = np.array([4, 6])
        expected_imag = np.array([5, 7])
        assert np.allclose(result.data['real'], expected_real)
        assert np.allclose(result.data['imag'], expected_imag)
        assert result.component_type == 'long'

    # Impedance objects can be multiplied by a scalar using the multiply method
    def test_multiply_by_scalar(self, generate_imp):
        imp = generate_imp()
        result = imp * 2
        expected_real = np.array([6, 10])
        expected_imag = np.array([8, 12])
        assert np.allclose(result.data['real'], expected_real)
        assert np.allclose(result.data['imag'], expected_imag)
        assert result.component_type == 'long'

    # The loss_factor method computes the correct loss factor for a given sigma
    def test_loss_factor_computation(self, generate_imp):
        for component_type in ["long", "xdip", "ydip", "xquad", "yquad"]:
            imp = generate_imp(component_type=component_type)
            assert imp.loss_factor(1) > 0

    # The to_wakefunction method correctly transforms impedance data to a WakeFunction object
    def test_to_wakefunction_transformation(self, generate_imp):
        for component_type in ["long", "xdip", "ydip"]:
            imp = generate_imp(component_type=component_type)
            wf = imp.to_wakefunction()
            assert isinstance(wf, WakeFunction)
            assert wf.component_type == component_type

    # The plot method generates a plot of the impedance data with appropriate labels
    def test_plot_generation(self, imp_param):
        imp_param.plot()
        assert True


@pytest.fixture
def generate_wakefield(generate_imp, generate_wf):

    def generate(impedance_params=None, wake_function_params=None, name=None):
        structure_list = []
        if impedance_params:
            for params in impedance_params:
                structure_list.append(generate_imp(**params))
        if wake_function_params:
            for params in wake_function_params:
                structure_list.append(generate_wf(**params))
        return WakeField(structure_list=structure_list, name=name)

    return generate


@pytest.fixture(params=component_list)
def wakefield_param(generate_wakefield, request):
    wake = generate_wakefield(impedance_params=[{
        'component_type': request.param
    }],
                              wake_function_params=[{
                                  'component_type':
                                  request.param
                              }])
    return wake


class TestWakeField:

    def test_initialize_with_impedance_wakefunction_list(
            self, generate_wakefield):
        for component_type in component_list:
            wakefield = generate_wakefield(impedance_params=[{
                'component_type':
                component_type
            }],
                                           wake_function_params=[{
                                               'component_type':
                                               component_type
                                           }])
        assert f'Z{component_type}' in wakefield.components
        assert f'W{component_type}' in wakefield.components

    def test_append_to_model(self, generate_imp, generate_wakefield,
                             generate_wf):
        wakefield = generate_wakefield()
        for component_type in component_list:
            imp = generate_imp(component_type=component_type)
            wf = generate_wf(component_type=component_type)
            wakefield.append_to_model(imp)
            assert f'Z{component_type}' in wakefield.components
            wakefield.append_to_model(wf)
            assert f'W{component_type}' in wakefield.components

    def test_save_and_load_wakefield(self, generate_wakefield, tmp_path):
        wakefield = generate_wakefield(
            impedance_params=[{
                'component_type': 'long'
            }])
        file_path = tmp_path / "wakefield.pkl"
        wakefield.save(file_path)
        loaded_wakefield = WakeField.load(file_path)
        assert 'Zlong' in loaded_wakefield.components

    @pytest.mark.parametrize(
        'comp,expected',
        [('long', np.array([4, 6])),
         ('xcst', np.array([3 + np.sqrt(2), 4 + 2 * np.sqrt(2)])),
         ('ycst', np.array([3 + np.sqrt(2), 4 + 2 * np.sqrt(2)])),
         ('xdip', np.array([5, 8])), ('ydip', np.array([5, 8])),
         ('xquad', np.array([5, 8])), ('yquad', np.array([5, 8]))])
    def test_add_wakefields_with_beta_functions(self, generate_wakefield, comp,
                                                expected):
        wake1 = generate_wakefield(
            wake_function_params=[{
                'component_type': comp,
                'function': np.array([1, 2])
            }])
        wake2 = generate_wakefield(
            wake_function_params=[{
                'component_type': comp,
                'function': np.array([3, 4])
            }])
        beta1 = [2, 2]
        beta2 = [1, 1]

        result_wake = WakeField.add_wakefields(wake1, beta1, wake2, beta2)
        wf = getattr(result_wake, f'W{comp}')
        assert np.allclose(wf.data["real"], expected)

        result_wake = WakeField.add_several_wakefields([wake1, wake2],
                                                       np.array([beta1,
                                                                 beta2]))
        wf = getattr(result_wake, f'W{comp}')
        assert np.allclose(wf.data["real"], expected)

    def test_add_duplicate_components_raises_error(self, generate_wakefield):
        wakefield = generate_wakefield(
            impedance_params=[{
                'component_type': 'long'
            }])
        with pytest.raises(ValueError):
            wakefield.append_to_model(wakefield.Zlong)

    def test_drop_non_existent_component_raises_error(self,
                                                      generate_wakefield):
        wakefield = generate_wakefield()
        with pytest.raises(AttributeError):
            wakefield.drop('Znonexistent')

    def test_append_to_model_with_invalid_component(self, generate_wakefield):
        wakefield = generate_wakefield()
        invalid_component = "InvalidComponent"
        with pytest.raises(ValueError,
                           match="is not an Impedance nor a WakeFunction."):
            wakefield.append_to_model(invalid_component)

    def test_drop_method_input_types(self, generate_wakefield):
        wakefield = generate_wakefield(impedance_params=[{
            'component_type':
            'long'
        }],
                                       wake_function_params=[{
                                           'component_type':
                                           'xdip'
                                       }])

        wakefield.drop('Zlong')
        assert 'Zlong' not in wakefield.components

        wakefield.drop(['Wxdip'])
        assert 'Wxdip' not in wakefield.components

        wakefield = generate_wakefield(impedance_params=[{
            'component_type':
            'long'
        }],
                                       wake_function_params=[{
                                           'component_type':
                                           'xdip'
                                       }])
        wakefield.drop('Z')
        assert 'Zlong' not in wakefield.components

        wakefield = generate_wakefield(impedance_params=[{
            'component_type':
            'long'
        }],
                                       wake_function_params=[{
                                           'component_type':
                                           'xdip'
                                       }])
        wakefield.drop('W')
        assert 'Wxdip' not in wakefield.components

        with pytest.raises(TypeError):
            wakefield.drop(123)
