import numpy as np
import pandas as pd

from mbtrack2 import ComplexData, ImpedanceModel, WakeField

component_list = ComplexData.name_and_coefficients_table().columns.to_list()

import pytest

pytestmark = pytest.mark.unit


class TestImpedanceModel:

    # Adding WakeField objects with unique names should succeed without errors
    @pytest.mark.parametrize('pos1,pos2', [([0], [1]), (0, 1), (0.5, 1.5)])
    def test_add_wakefield(self, ring_with_at_lattice, generate_wakefield,
                           pos1, pos2):
        model = ImpedanceModel(ring_with_at_lattice)
        wakefield1 = generate_wakefield(name="wake1")
        wakefield2 = generate_wakefield(name="wake2")
        model.add(wakefield1, pos1)
        model.add(wakefield2, pos2)
        assert "wake1" in model.names
        assert "wake2" in model.names
        assert wakefield1 in model.wakefields
        assert wakefield2 in model.wakefields
        assert pos1 in model.positions
        assert pos2 in model.positions

    # Adding global WakeField objects should correctly append them to the globals list
    def test_add_global_wakefield(self, ring_with_at_lattice,
                                  generate_wakefield):
        model = ImpedanceModel(ring_with_at_lattice)
        global_wakefield = generate_wakefield(name="global_wake")
        model.add_global(global_wakefield)
        assert "global_wake" in model.globals_names
        assert global_wakefield in model.globals

    # Test weighted global sum with average_beta
    def test_global_wakefield_weighted(self, ring_with_at_lattice,
                                       generate_wakefield):
        ring_with_at_lattice.optics.local_beta = np.array([0.5, 1])
        model = ImpedanceModel(ring_with_at_lattice,
                               average_beta=np.array([4, 2]))

        global_wakefield = generate_wakefield(name="global_wake",
                                              wake_function_params=[{
                                                  'component_type':
                                                  "xdip",
                                                  'function':
                                                  np.array([1, 2])
                                              }, {
                                                  'component_type':
                                                  "ydip",
                                                  'function':
                                                  np.array([1, 2])
                                              }],
                                              impedance_params=[{
                                                  'component_type':
                                                  "xdip",
                                                  'function':
                                                  np.array([1, 2])
                                              }, {
                                                  'component_type':
                                                  "ydip",
                                                  'function':
                                                  np.array([1, 2])
                                              }])

        model.add_global(global_wakefield)
        model.compute_sum()

        excepted_x = np.array([4 / 0.5 * 1, 4 / 0.5 * 2])
        excepted_y = np.array([2 / 1 * 1, 2 / 1 * 2])

        np.testing.assert_allclose(model.sum.Wxdip.data["real"], excepted_x)
        np.testing.assert_allclose(model.sum.Zxdip.data["real"], excepted_x)
        np.testing.assert_allclose(model.sum.Wydip.data["real"], excepted_y)
        np.testing.assert_allclose(model.sum.Zydip.data["real"], excepted_y)

    # Test beta weighting summation
    @pytest.mark.parametrize('comp,expected',
                             [('long', np.array([3, 6])),
                              ('xcst',
                               np.array([(2**0.5 + 4**0.5 + 6**0.5) / 2,
                                         (2**0.5 + 4**0.5 + 6**0.5) / 2 * 2])),
                              ('ycst',
                               np.array([(1**0.5 + 2**0.5 + 3**0.5) / 1,
                                         (1**0.5 + 2**0.5 + 3**0.5) / 1 * 2])),
                              ('xdip', np.array([6, 12])),
                              ('ydip', np.array([6, 12])),
                              ('xquad', np.array([6, 12])),
                              ('yquad', np.array([6, 12]))])
    def test_sum_beta(self, demo_ring, generate_wakefield, comp, expected):

        wake_in = generate_wakefield(wake_function_params=[{
            'component_type':
            comp,
            'function':
            np.array([1, 2])
        }],
                                     impedance_params=[{
                                         'component_type':
                                         comp,
                                         'function':
                                         np.array([1, 2])
                                     }])

        demo_ring.optics.local_beta = np.array([2, 1])
        model = ImpedanceModel(demo_ring)
        model.add(wake_in, 0, name="test")
        beta = np.array([[2, 4, 6], [1, 2, 3]])
        result_wake = model.sum_beta(model.wakefields[0], beta)
        wf = getattr(result_wake, f'W{comp}')
        z = getattr(result_wake, f'Z{comp}')
        assert np.allclose(wf.data["real"], expected)
        assert np.allclose(z.data["real"], expected)

    def test_compute_sum_names(self, ring_with_at_lattice, generate_wakefield):
        wake1 = generate_wakefield(name="wake1")
        wake2 = generate_wakefield(name="wake2")
        model = ImpedanceModel(ring_with_at_lattice)
        model.add(wake1, [1, 2, 3])
        model.add(wake2, [5, 6, 7])
        model.compute_sum_names()
        assert hasattr(model, 'sum_wake1')
        assert hasattr(model, 'sum_wake2')
        assert 'sum_wake1' in model.sum_names
        assert 'sum_wake2' in model.sum_names

    # Computing the sum of WakeFields should correctly aggregate all components
    def test_compute_sum(self, ring_with_at_lattice, wakefield_param):
        model = ImpedanceModel(ring_with_at_lattice)
        model.add(wakefield_param, [0], "wake1")
        model.add(wakefield_param, [1], "wake2")
        model.compute_sum()
        assert hasattr(model, 'sum')
        for comp in wakefield_param.components:
            assert comp in model.sum.components

    # Saving the ImpedanceModel to a file should serialize all relevant attributes
    def test_save_impedance_model(self, tmp_path, ring_with_at_lattice,
                                  generate_wakefield):
        model = ImpedanceModel(ring_with_at_lattice)
        wakefield = generate_wakefield(name="wake")
        model.add(wakefield, [0])
        model.compute_sum()
        file_path = tmp_path / "impedance_model.pkl"
        model.save(file_path)
        assert file_path.exists()

    # Loading the ImpedanceModel from a file should restore all attributes accurately
    def test_load_impedance_model(self, tmp_path, ring_with_at_lattice,
                                  generate_wakefield):
        model = ImpedanceModel(ring_with_at_lattice)
        wakefield = generate_wakefield(name="wake")
        global_wake = generate_wakefield(name="global_wake")
        model.add(wakefield, [0])
        model.add_global(global_wake)
        model.compute_sum()
        file_path = tmp_path / "impedance_model.pkl"
        model.save(file_path)

        new_model = ImpedanceModel(ring_with_at_lattice)
        new_model.load(file_path)

        assert "wake" in new_model.names
        assert isinstance(new_model.wakefields[0], WakeField)
        assert [0] in new_model.positions
        assert "global_wake" in new_model.globals_names
        assert isinstance(new_model.globals[0], WakeField)
        assert new_model.average_beta.shape == (2, 1)

    # Plotting the contributions of WakeFields should generate a valid plot
    @pytest.mark.parametrize('comp', [('long'), ('xdip'), ('ydip'), ('xquad'),
                                      ('yquad')])
    def test_plot_area(self, ring_with_at_lattice, generate_wakefield, comp):
        model = ImpedanceModel(ring_with_at_lattice)
        wake = generate_wakefield(impedance_params=[{
            'component_type': comp
        }],
                                  name="wake")
        model.add(wake, [0])
        model.compute_sum()
        fig = model.plot_area(Z_type=f"Z{comp}", sigma=30e-12, zoom=True)
        assert fig is not None

    # Adding a WakeField with a duplicate name should raise a ValueError
    def test_add_duplicate_name_raises_valueerror(self, ring_with_at_lattice,
                                                  generate_wakefield):
        model = ImpedanceModel(ring_with_at_lattice)
        wakefield = generate_wakefield()
        wakefield.name = "duplicate"

        model.add(wakefield, [0], "duplicate")

        with pytest.raises(ValueError):
            model.add(wakefield, [1], "duplicate")

    # Adding a global WakeField with a duplicate name should raise a ValueError
    def test_add_global_duplicate_name_raises_valueerror(
            self, ring_with_at_lattice, generate_wakefield):
        model = ImpedanceModel(ring_with_at_lattice)
        global_wakefield = generate_wakefield()
        global_wakefield.name = "global_duplicate"

        model.add_global(global_wakefield, "global_duplicate")

        with pytest.raises(ValueError):
            model.add_global(global_wakefield, "global_duplicate")

    # Renaming an attribute should update the internal dictionary correctly
    def test_rename_attribute(self, ring_with_at_lattice):
        model = ImpedanceModel(ring_with_at_lattice)

        model.some_attribute = 123

        model.rename_attribute("some_attribute", "new_attribute_name")

        assert hasattr(model, "new_attribute_name")
        assert not hasattr(model, "some_attribute")

    # Grouping attributes should correctly combine specified properties
    def test_group_attributes(self, demo_ring, generate_wakefield):
        # Setup
        model = ImpedanceModel(demo_ring)

        # Create WakeFields with different properties
        wake1 = generate_wakefield(impedance_params=[{
            'component_type': 'long'
        }, {
            'component_type': 'xdip'
        }])
        wake2 = generate_wakefield(impedance_params=[{
            'component_type': 'long'
        }, {
            'component_type': 'xdip'
        }])
        # Add WakeFields to the model
        model.add(wake1, positions=[0.0], name='wake1')
        model.add(wake2, positions=[1.0], name='wake2')

        # Compute sum to initialize sum_names
        model.compute_sum()

        # Group attributes with a common property
        model.group_attributes('wake', property_list=['Zlong', 'Zxdip'])

        # Assertions
        assert 'wake' in model.sum_names
        assert not hasattr(model, 'sum_wake1')
        assert not hasattr(model, 'sum_wake2')

        # Check if the grouped attribute has combined properties
        grouped_wake = getattr(model, 'wake')
        assert hasattr(grouped_wake, 'Zlong')
        assert hasattr(grouped_wake, 'Zxdip')

    # Grouping attributes should correctly combine specified properties
    # def test_group_attributes_combines_properties(self, demo_ring, generate_wakefield):
    #     # Setup
    #     model = ImpedanceModel(demo_ring)

    #     # Create WakeFields with different properties
    #     wake1 = generate_wakefield(impedance_params=[{'component_type': 'long'},
    #                                                  {'component_type': 'xdip'}])
    #     wake2 = generate_wakefield(impedance_params=[{'component_type': 'long'}])
    #     # Add WakeFields to the model
    #     model.add(wake1, positions=[0.0], name='wake1')
    #     model.add(wake2, positions=[1.0], name='wake2')

    #     # Compute sum to initialize sum_names
    #     model.compute_sum()

    #     # Group attributes with a common property
    #     model.group_attributes('wake', property_list=['Zlong', 'Zxdip'])

    #     # Assertions
    #     assert 'wake' in model.sum_names
    #     assert not hasattr(model, 'sum_wake1')
    #     assert not hasattr(model, 'sum_wake2')

    #     # Check if the grouped attribute has combined properties
    #     grouped_wake = getattr(model, 'wake')
    #     assert hasattr(grouped_wake, 'Zlong')
    #     assert hasattr(grouped_wake, 'Zxdip')

    def test_power_loss_spectrum(self, demo_ring, generate_wakefield):
        # Setup
        wakefield = generate_wakefield(
            impedance_params=[{
                'component_type': 'long'
            }])
        model = ImpedanceModel(demo_ring)
        model.add(wakefield, positions=[0], name='test_wakefield')
        model.compute_sum()

        # Parameters
        M = 100
        bunch_spacing = 1e-9
        I = 0.5
        sigma = 1e-9

        # Test without max_overlap
        pf0, power_loss = model.power_loss_spectrum(M,
                                                    bunch_spacing,
                                                    I,
                                                    sigma=sigma)

        # Assertions
        assert len(pf0) == len(power_loss)
        assert np.all(power_loss > 0)

    # The energy_loss method should compute correctly with minimal input data
    def test_energy_loss(self, demo_ring, generate_wakefield):
        # Setup
        wakefield = generate_wakefield(
            impedance_params=[{
                'component_type': 'long'
            }])
        model = ImpedanceModel(demo_ring)
        model.add(wakefield, positions=[0], name='test_wakefield')
        model.compute_sum()

        # Parameters for energy_loss
        M = 1  # Minimal number of bunches
        bunch_spacing = 1.0  # Arbitrary non-zero spacing
        I = 0.1  # Minimal non-zero current
        sigma = 1e-12  # Minimal non-zero sigma

        # Execute
        summary = model.energy_loss(M, bunch_spacing, I, sigma=sigma)

        # Verify
        assert not summary.empty, "The summary should not be empty"
        assert "loss factor (beam) [V/pC]" in summary.columns, "Expected column missing"
        assert "loss factor (bunch) [V/pC]" in summary.columns, "Expected column missing"
