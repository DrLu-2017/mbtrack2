mbtrack2
========

[![GitLab Release](https://img.shields.io/gitlab/v/release/184?gitlab_url=https%3A%2F%2Fgitlab.synchrotron-soleil.fr%2F)](https://gitlab.synchrotron-soleil.fr/PA/collective-effects/mbtrack2/-/releases) [![PyPI - Version](https://img.shields.io/pypi/v/mbtrack2)](https://pypi.org/project/mbtrack2/)
[![PyPI - License](https://img.shields.io/pypi/l/mbtrack2)](https://gitlab.synchrotron-soleil.fr/PA/collective-effects/mbtrack2/-/blob/stable/LICENSE)
[![Read the Docs](https://img.shields.io/readthedocs/mbtrack2)](https://mbtrack2.readthedocs.io/)

mbtrack2 is a coherent object-oriented framework written in python to work on collective effects in synchrotrons.

mbtrack2 is composed of different modules allowing to easily write scripts for single bunch or multi-bunch tracking using MPI parallelization in a transparent way. The base of the tracking model of mbtrack2 is inspired by mbtrack, a C multi-bunch tracking code initially developed at SOLEIL.

Examples
--------
Jupyter notebooks demonstrating mbtrack2 features are available in the example folder and can be opened online using google colab:
+ mbtrack2 base features [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GamelinAl/mbtrack2_examples/blob/main/mbtrack2_demo.ipynb)
+ dealing with RF cavities and longitudinal beam dynamics [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GamelinAl/mbtrack2_examples/blob/main/mbtrack2_cavity_resonator.ipynb)
+ collective effects [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GamelinAl/mbtrack2_examples/blob/main/mbtrack2_collective_effects.ipynb)
+ bunch by bunch feedback [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GamelinAl/mbtrack2_examples/blob/main/mbtrack2_BxB_FB.ipynb)
+ RF loops and feedbacks [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GamelinAl/mbtrack2_examples/blob/main/mbtrack2_RF_feedback.ipynb)


Installation
------------

### Using pip

Run:

```
pip install mbtrack2
```
To test your installation run:
```
from mbtrack2 import *
```

### Using conda

Clone the mbtrack2 repo and enter the repo:
```
git clone https://gitlab.synchrotron-soleil.fr/PA/collective-effects/mbtrack2.git
cd mbtrack2
```

To create a new conda environment for mbtrack2 run:

```
conda env create -f mbtrack2.yml
conda activate mbtrack2
```

Or to update your current conda environment to be able to run mbtrack2:

```
conda env update --file mbtrack2.yml
```

To test your installation run:
```
from mbtrack2 import *
```

### Using docker

A docker image is available:

```
docker pull gitlab-registry.synchrotron-soleil.fr/pa/collective-effects/mbtrack2
```

References
----------
If used in a publication, please cite mbtrack2 paper and the zenodo archive for the corresponding code version (and any other paper in this list for more specific features).

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14418989.svg)](https://doi.org/10.5281/zenodo.14418989)

### mbtrack2 general features
A. Gamelin, W. Foosang, and R. Nagaoka, “mbtrack2, a Collective Effect Library in Python”, presented at the 12th Int. Particle Accelerator Conf. (IPAC'21), Campinas, Brazil, May 2021, paper MOPAB070.

### RF cavities with beam loading and RF feedbacks
Yamamoto, Naoto, Alexis Gamelin, and Ryutaro Nagaoka. "Investigation of Longitudinal Beam Dynamics With Harmonic Cavities by Using the Code Mbtrack." Proc. 10th International Partile Accelerator Conference (IPAC’19), Melbourne, Australia, 19-24 May 2019. 2019.
