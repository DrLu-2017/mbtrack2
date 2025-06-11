# -*- coding: utf-8 -*-
__version__ = "0.8.0"
from mbtrack2.impedance import *
from mbtrack2.instability import *
from mbtrack2.tracking import *
from mbtrack2.utilities import *

try:
    DYNAMIC_VERSIONING = True
    import os
    import subprocess

    worktree = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    gitdir = worktree + "/.git/"
    with open(os.devnull, "w") as devnull:
        __version__ = subprocess.check_output(
            "git --git-dir=" + gitdir + " --work-tree=" + worktree +
            " describe --long --dirty --abbrev=10 --tags",
            shell=True,
            stderr=devnull,
        )
    __version__ = __version__.decode("utf-8").rstrip()  # remove trailing \n
    # remove commit hash to conform to PEP440:
    split_ = __version__.split("-")
    __version__ = split_[0]
    if split_[1] != "0":
        __version__ += "." + split_[1]
    dirty = "dirty" in split_[-1]
except:
    DYNAMIC_VERSIONING = False
    from . import __version__

    dirty = False


def print_version():
    print(("mbtrack2 version " + __version__))
    if dirty:
        print("(dirty git work tree)")
    print(50 * '-')
    print(
        "If used in a publication, please cite mbtrack2 paper and the zenodo archive for the corresponding code version (and other papers for more specific features)."
    )
    print(
        "[1] A. Gamelin, W. Foosang, N. Yamamoto, V. Gubaidulin and R. Nagaoka, “mbtrack2”. Zenodo, Dec. 16, 2024. doi: 10.5281/zenodo.14418989."
    )
    print(
        "[2] A. Gamelin, W. Foosang, and R. Nagaoka, “mbtrack2, a Collective Effect Library in Python”, presented at the 12th Int. Particle Accelerator Conf. (IPAC'21), Campinas, Brazil, May 2021, paper MOPAB070."
    )
    print("\n")


try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank == 0:
        print_version()
except:
    print_version()
