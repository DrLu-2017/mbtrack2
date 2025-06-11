import numpy as np
import pytest
from utility_test_growth_rate import get_growth_beam

from mbtrack2 import (
    Beam,
    BeamMonitor,
    LongitudinalMap,
    LongRangeResistiveWall,
    RFCavity,
    TransverseMap,
    rwmbi_growth_rate,
)

pytestmark = pytest.mark.phys


class TestLongRangeResistiveWallPhys:

    def test_llrw_growth_rate_chroma0(self, demo_ring, tmp_path):

        ring = demo_ring
        mp_num = int(1)
        n_turns = int(5e3)
        nt = 50
        rho = 1.7e-8
        radius = 4e-3
        Itot = 1.0
        Vc = 1.0e6
        name = str(tmp_path / 'save')
        ring.chro = [0, 0]

        growth_rate = rwmbi_growth_rate(ring, Itot, radius, rho, "y")

        RF = RFCavity(ring, 1, Vc, np.arccos(ring.U0 / Vc))
        long = LongitudinalMap(ring)
        trans = TransverseMap(ring)

        bb = Beam(ring)
        filling_pattern = np.ones((ring.h, )) * Itot / ring.h

        bb.init_beam(filling_pattern,
                     mp_per_bunch=mp_num,
                     track_alive=False,
                     mpi=False)

        rw = LongRangeResistiveWall(ring,
                                    beam=bb,
                                    length=ring.L,
                                    rho=rho,
                                    radius=radius,
                                    types=["Wydip"],
                                    nt=nt)

        save_every = 10
        bbmon = BeamMonitor(h=ring.h,
                            save_every=save_every,
                            buffer_size=10,
                            total_size=int(n_turns / save_every),
                            file_name=name,
                            mpi_mode=False)

        for i in range(n_turns):
            if (i % 1000 == 0):
                print(i)

            RF.track(bb)
            long.track(bb)
            trans.track(bb)

            rw.track(bb)
            bbmon.track(bb)

        bbmon.close()

        path = name + '.hdf5'
        gr, gr_std = get_growth_beam(ring, path, 1e3, 5e3)
        rel_error = np.abs(growth_rate - gr) * 100 / growth_rate
        rel_error_std = np.abs(gr_std) * 100 / growth_rate
        print(f"Relative error: {rel_error} % (std: {rel_error_std} %)")
        assert rel_error < 2

    # def test_llrw_growth_rate_chroma0_average_beta(self,
    #                                                generate_ring_with_at_lattice,
    #                                                tmp_path):
    #     tracking_loc = np.random.rand()*350
    #     ring = generate_ring_with_at_lattice(tracking_loc=tracking_loc)
    #     mp_num = int(1)
    #     n_turns = int(1e3)
    #     nt = 50
    #     rho=1.7e-8
    #     radius=4e-3
    #     Itot = 2
    #     Vc = 2.8e6
    #     name = str(tmp_path / 'save')
    #     ring.chro = [0,0]

    #     growth_rate = rwmbi_growth_rate(ring, Itot, radius, rho, "y")

    #     RF = RFCavity(ring, 1, Vc, np.arccos(ring.U0/Vc))
    #     long = LongitudinalMap(ring)
    #     trans = TransverseMap(ring)

    #     bb = Beam(ring)
    #     filling_pattern = np.ones((ring.h,))*Itot/ring.h

    #     bb.init_beam(filling_pattern, mp_per_bunch=mp_num, track_alive=False, mpi=False)

    #     rw = LongRangeResistiveWall(ring, beam=bb, length=ring.L, rho=rho,
    #                                 radius=radius, types=["Wydip"], nt=nt)

    #     save_every = 10
    #     bbmon = BeamMonitor(h=ring.h, save_every=save_every, buffer_size=10,
    #                         total_size=int(n_turns/save_every),
    #                         file_name=name, mpi_mode=False)

    #     for i in range(n_turns):
    #         if (i % 100 == 0):
    #             print(i)

    #         RF.track(bb)
    #         long.track(bb)
    #         trans.track(bb)

    #         rw.track(bb)
    #         bbmon.track(bb)

    #     bbmon.close()

    #     path = name + '.hdf5'
    #     gr, gr_std = get_growth_beam(ring, path, 500, 1e3)
    #     rel_error = np.abs(growth_rate-gr)*100/growth_rate
    #     rel_error_std = np.abs(gr_std)*100/growth_rate
    #     print(f"Relative error: {rel_error} % (std: {rel_error_std} %)")
    #     assert rel_error < 2
