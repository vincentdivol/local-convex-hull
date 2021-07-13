# Short tutorial for the article *Minimax adaptive estimation in manifold inference*

I propose a code to play around the concept of *local convex hull*, that is define in [this article](https://arxiv.org/abs/2001.04896). In particular, the method described in the paper to select a ''good'' scale for the local convex hull is implemented.

- [utils.py](https://github.com/vincentdivol/local-convex-hull/blob/main/utils.py) contains auxiliary functions used in the notebooks below
- [create_samples.py](https://github.com/vincentdivol/local-convex-hull/blob/main/create_samples.py) contains different functions to sample according to different distributions on manifolds.
- [local_conv.ipynb](https://github.com/vincentdivol/local-convex-hull/blob/main/local_convex_hull.ipynb) is the tutorial 

The Jupyter Notebook can be lauched on Binder.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/vincentdivol/local-convex-hull/HEAD?filepath=local_convex_hull.ipynb)
