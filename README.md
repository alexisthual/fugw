# Fused unbalanced Gromov-Wasserstein for Python

This package features multiple GPU-compatible PyTorch implementations
to solve Fused Unbalanced Gromov-Wasserstein problems.

## Introduction

In short, this code computes a matrix $P$ that matches points of two distributions $s$ and $t$.

We denote as $n$ and $m$ the number of points (we will also call them vertices or voxels)
of $s$ and $t$ respectively.

Two points $i$ and $j$, from $s$ and $t$ respectively, are matched based on similarity
between their respective features $F_i^s$ and $F_j^t$ (Wasserstein loss).
We fuse this loss with one that tries to preserve the respective underlying geometries of these two
distributions. We represent these geometries as kernel matrices $D_s$ and $D_t$ in the figure below.
In essence, two points which were distant from one another in the first distribution
should be matched to points which are distant from one another
in the second distribution (Gromov-Wasserstein loss).
Finally, the unbalancing part of this problem allows to leave points of the first and/or
the second distribution for which no good match can be found.

Details about the implementation and motivations are available in the original
[NeurIPS 2022 paper presenting this work](https://arxiv.org/abs/2206.09398),
in which we align cortical structures of human individuals using this method.
In this case, we match areas of the cortex based on similarity of their functional activity
(ie how they behave throughout a series of experiments) while trying to preserve the anatomy of the cortex.

![Introduction to FUGW](assets/fugw_intro.png)

## Install

In a dedicated Python env, run:

```bash
pip install -r requirements.txt
pip install -e .
```

## Dev install

On top of running the usual install commands, install dev dependencies with:

```bash
pip install -r requirements-dev.txt
pip install -e '.[dev]'
```

This will allow to run tests locally:

```bash
pytest
```

## Usage and examples

This repo contains multiple solvers to FUGW optimization problems,
as well as scikit-learn transformers wrapping these solvers.
Functions implemented in `./tests` can be useful to understand how to use the solvers and the associated transformers.

### 1 - Transporting distributions that have less than 10k points

The following example is taken from `./tests/test_dense.py`:

```python
import numpy as np
from sklearn.metrics import pairwise_distances
import torch

from fugw import FUGW


np.random.seed(100)
n_vertices_source = 105
n_vertices_target = 95
n_features_train = 10
n_features_test = 5


def init_distribution(n_features, n_vertices):
    weights = np.ones(n_vertices) / n_vertices
    features = np.random.rand(n_features, n_vertices)
    embeddings = np.random.rand(n_vertices, 3)
    geometry = pairwise_distances(embeddings)

    return weights, features, geometry, embeddings


if __name__ == "__main__":
    # Generate random training data for source and target
    _, source_features_train, source_geometry, _ = init_distribution(
        n_features_train, n_vertices_source
    )
    _, target_features_train, target_geometry, _ = init_distribution(
        n_features_train, n_vertices_target
    )

    fugw = FUGW(alpha=0.5)
    fugw.fit(
        source_features_train,
        target_features_train,
        source_geometry=source_geometry,
        target_geometry=target_geometry,
    )

    # Use trained model to transport new features
    source_features_test = np.random.rand(n_features_test, n_vertices_source)
    target_features_test = np.random.rand(n_features_test, n_vertices_target)

    transformed_data = fugw.transform(source_features_test)
    assert transformed_data.shape == target_features_test.shape

    # Compute score
    s = fugw.score(source_features_test, target_features_test)
    assert isinstance(s, int) or isinstance(s, float)
```

### 2 - Transporting samples of the source and target distributions

TODO

### 3 - Transporting distributions that have more than 10k points

Because FUGW computes a matrix $P$ of shape $n \times m$,
the size of $P$ grows quadratically with the number of vertices.

In order to be able to store such a matrix on GPU for high values of $n$, $m$, a sparse
solver is available. It leverages example 2 to compute a dense transport plan between
sub-samples of $s$ and $t$, and uses it to define a sparsity mask of the solution
that will be computed by the sparse solver.

TODO

### 4 - Computing a FUGW barycenter from multiple distributions

See `./tests/test_barycenter.py`.

TODO

## Citing this work

If this package was useful to you, please cite it in your work:

```bibtex
@article{Thual-2022-fugw,
  title={Aligning individual brains with Fused Unbalanced Gromov-Wasserstein},
  author={Thual, Alexis and Tran, Huy and Zemskova, Tatiana and Courty, Nicolas and Flamary, Rémi and Dehaene, Stanislas and Thirion, Bertrand},
  publisher={arXiv},
  doi={10.48550/ARXIV.2206.09398},
  url={https://arxiv.org/abs/2206.09398},
  year={2022},
  copyright={Creative Commons Attribution 4.0 International}
}
```
