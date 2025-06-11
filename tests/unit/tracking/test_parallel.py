import pytest

pytestmark = pytest.mark.unit
import numpy as np
from mpi4py import MPI

from mbtrack2.tracking.parallel import Mpi


@pytest.fixture
def mpi_size1():
    filling_pattern = np.array([True])
    return Mpi(filling_pattern)


@pytest.fixture
def generate_mpi_mock(mocker):

    def generate(filling_pattern, rank=0):

        size = filling_pattern.sum()
        mocker.patch.object(MPI, 'COMM_WORLD')
        MPI.COMM_WORLD.Get_size.return_value = size
        MPI.COMM_WORLD.Get_rank.return_value = rank

        return Mpi(filling_pattern)

    return generate


@pytest.fixture
def mpi_mock_size4_fill6(generate_mpi_mock):
    filling_pattern = np.ones((6, ), dtype=bool)
    filling_pattern[1] = False
    filling_pattern[4] = False
    return generate_mpi_mock(filling_pattern)


class TestMpi:

    # Verify table creation
    def test_initialize_mpi_size1(self, mpi_size1):
        assert np.array_equal(mpi_size1.table, np.array([[0, 0]]))
        assert mpi_size1.bunch_num == 0
        assert mpi_size1.next_bunch == 0
        assert mpi_size1.previous_bunch == 0

    # Verify correct rank-to-bunch and bunch-to-rank mappings
    def test_rank_to_bunch_and_bunch_to_rank_mappings(self,
                                                      mpi_mock_size4_fill6):
        assert mpi_mock_size4_fill6.rank_to_bunch(0) == 0
        assert mpi_mock_size4_fill6.bunch_to_rank(0) == 0
        assert mpi_mock_size4_fill6.rank_to_bunch(1) == 2
        assert mpi_mock_size4_fill6.bunch_to_rank(1) == None
        assert mpi_mock_size4_fill6.rank_to_bunch(2) == 3
        assert mpi_mock_size4_fill6.bunch_to_rank(2) == 1
        assert mpi_mock_size4_fill6.rank_to_bunch(3) == 5
        assert mpi_mock_size4_fill6.bunch_to_rank(3) == 2
        assert mpi_mock_size4_fill6.bunch_to_rank(4) == None
        assert mpi_mock_size4_fill6.bunch_to_rank(5) == 3

    def test_next_previous_bunch(self, mpi_mock_size4_fill6):
        assert mpi_mock_size4_fill6.rank == 0
        assert mpi_mock_size4_fill6.next_bunch == 1
        assert mpi_mock_size4_fill6.previous_bunch == 3

    @pytest.mark.parametrize("dim", [("tau"), ("delta"), ("x"), ("xp"), ("y"),
                                     ("yp")])
    def test_share_distributions(self, beam_1bunch_mpi, dim):
        n_bin = 75
        beam_1bunch_mpi.mpi.share_distributions(beam_1bunch_mpi,
                                                dimensions=dim,
                                                n_bin=n_bin)
        assert beam_1bunch_mpi.mpi.__getattribute__(dim + "_center").shape == (
            beam_1bunch_mpi.mpi.size, n_bin)
        assert beam_1bunch_mpi.mpi.__getattribute__(
            dim + "_profile").shape == (beam_1bunch_mpi.mpi.size, n_bin)
        assert beam_1bunch_mpi.mpi.__getattribute__(
            dim + "_sorted_index").shape == (len(
                beam_1bunch_mpi[beam_1bunch_mpi.mpi.bunch_num]), )
        assert beam_1bunch_mpi.mpi.charge_per_mp_all is not None

    def test_share_distributions_multidim(self, beam_1bunch_mpi):
        n_bin = [75, 60, 22]
        dims = ["x", "yp", "tau"]
        beam_1bunch_mpi.mpi.share_distributions(beam_1bunch_mpi,
                                                dimensions=dims,
                                                n_bin=n_bin)
        assert beam_1bunch_mpi.mpi.charge_per_mp_all is not None
        for i, dim in enumerate(dims):
            assert beam_1bunch_mpi.mpi.__getattribute__(
                dim + "_center").shape == (beam_1bunch_mpi.mpi.size, n_bin[i])
            assert beam_1bunch_mpi.mpi.__getattribute__(
                dim + "_profile").shape == (beam_1bunch_mpi.mpi.size, n_bin[i])
            assert beam_1bunch_mpi.mpi.__getattribute__(
                dim + "_sorted_index").shape == (len(
                    beam_1bunch_mpi[beam_1bunch_mpi.mpi.bunch_num]), )

    def test_share_distributions_nobunch(self, beam_1bunch_mpi):
        n_bin = 75
        dim = "x"
        beam_1bunch_mpi[beam_1bunch_mpi.mpi.bunch_num].alive[:] = False
        beam_1bunch_mpi.mpi.share_distributions(beam_1bunch_mpi,
                                                dimensions=dim,
                                                n_bin=n_bin)
        assert beam_1bunch_mpi.mpi.charge_per_mp_all is not None
        assert np.allclose(
            beam_1bunch_mpi.mpi.__getattribute__(dim + "_center"),
            np.zeros((n_bin, ), dtype=np.int64))
        assert np.allclose(
            beam_1bunch_mpi.mpi.__getattribute__(dim + "_profile"),
            np.zeros((n_bin, ), dtype=np.float64))
        assert beam_1bunch_mpi.mpi.__getattribute__(dim +
                                                    "_sorted_index") is None

    def test_share_means(self, beam_1bunch_mpi):
        beam_1bunch_mpi.mpi.share_means(beam_1bunch_mpi)
        assert beam_1bunch_mpi.mpi.charge_all is not None
        assert beam_1bunch_mpi.mpi.mean_all.shape == (beam_1bunch_mpi.mpi.size,
                                                      6)
        assert beam_1bunch_mpi.mpi.mean_all[0, :] == pytest.approx(
            beam_1bunch_mpi[beam_1bunch_mpi.mpi.bunch_num].mean)

    def test_share_stds(self, beam_1bunch_mpi):
        beam_1bunch_mpi.mpi.share_stds(beam_1bunch_mpi)
        assert beam_1bunch_mpi.mpi.charge_all is not None
        assert beam_1bunch_mpi.mpi.std_all.shape == (beam_1bunch_mpi.mpi.size,
                                                     6)
        assert beam_1bunch_mpi.mpi.std_all[0, :] == pytest.approx(
            beam_1bunch_mpi[beam_1bunch_mpi.mpi.bunch_num].std)
