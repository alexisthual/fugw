import pickle

import numpy as np
import torch

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

# `rich` console used throughout the codebase
console = Console()


# `rich` progress bar used throughout the codebase
def _get_progress(**kwargs):
    """Return a custom `rich` progress bar."""
    return Progress(
        SpinnerColumn(),
        TaskProgressColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        "<",
        TimeRemainingColumn(),
        **kwargs,
    )


def _make_tensor(x, device=None, dtype=None):
    """Turn x into a torch.Tensor with suited type and device."""
    if isinstance(x, np.ndarray):
        tensor = torch.tensor(x)
    elif isinstance(x, torch.Tensor):
        tensor = x
    else:
        raise Exception(f"Expected np.ndarray or torch.Tensor, got {type(x)}")

    # By default, cast x to float32 or int32
    # depending on its original type
    if dtype is None:
        if tensor.is_floating_point():
            dtype = torch.float32
        else:
            dtype = torch.int32

    return tensor.to(device=device, dtype=dtype)


def _make_sparse_csr_tensor(x, device=None, dtype=torch.float32):
    """Turn x into a sparse CSR torch.Tensor with suited type and device."""
    if x is None:
        return None
    elif isinstance(x, torch.Tensor):
        if hasattr(x, "layout"):
            if x.layout == torch.sparse_coo:
                return x.to(device, dtype).to_sparse_csr()
            elif x.layout == torch.sparse_csr:
                # TODO: See https://github.com/pytorch/pytorch/issues/94679
                # Convert to COO to move to device, then convert back to CSR
                return x.to_sparse_coo().to(device, dtype).to_sparse_csr()
            else:
                raise Exception(
                    f"Matrix's sparse layout is {x.layout}, "
                    "but expected sparse_coo or sparse_csr"
                )
        else:
            raise Exception(
                "Expected a torch sparse matrix, "
                "but attribute 'layout' is missing."
            )
    else:
        raise Exception(f"Expected sparse torch.Tensor, got {type(x)}")


def _make_csr_matrix(crow_indices, col_indices, values, size, device):
    """Make a sparse CSR torch.Tensor from its components."""
    return torch.sparse_csr_tensor(
        crow_indices.to(device),
        col_indices.to(device),
        values.to(device),
        size=size,
    )


def _low_rank_squared_l2(X, Y):
    """
    Write square euclidean distance matrix M
    (ie $M_{i, j} = ||X_i - Y_j||^2_{2}||$)
    as exact product of two low rank matrices A1 and A2
    such that M = A1 A2^t.

    Parameters
    ----------
    X: torch.Tensor(n, d)
    Y: torch.Tensor(m, d)

    Returns
    -------
    A1: torch.Tensor(n, d+2)
    A2: torch.Tensor(m, d+2)
    """
    if not (isinstance(X, torch.Tensor) and isinstance(Y, torch.Tensor)):
        X = torch.Tensor(X)
        Y = torch.Tensor(Y)

    device, dtype = X.device, X.dtype

    n, m = X.shape[0], Y.shape[0]

    norms_X = (X**2).sum(1, keepdim=True)
    norms_y = (Y**2).sum(1, keepdim=True)
    ones_x = torch.ones(n, 1).to(device).to(dtype)
    ones_y = torch.ones(m, 1).to(device).to(dtype)

    A1 = torch.cat([norms_X, ones_x, -(2**0.5) * X], dim=1)
    A2 = torch.cat([ones_y, norms_y, 2**0.5 * Y], dim=1)

    return (A1, A2)


def _sample_multivariate_normal(n_dims=2, n_points=200):
    """
    Samples n_points from a multivariate normal distribution
    with random mean and covariance matrix
    in dimension n_dims.

    Returns
    -------
    x: torch.Tensor of size (n_dims, n_points)
    """
    # Generate random mean
    mean = torch.normal(0, 3, size=(n_dims,))

    # Generate random covariance matrix from Wishart distribution
    m = torch.distributions.wishart.Wishart(
        df=torch.tensor(n_dims),
        covariance_matrix=torch.eye(n_dims),
    )
    cov = m.sample()

    # Generate random multivariate normal
    m = torch.distributions.multivariate_normal.MultivariateNormal(mean, cov)
    x = torch.stack([m.sample() for _ in range(n_points)]).T

    return x


def _init_mock_distribution(
    n_features, n_voxels, should_normalize=False, return_numpy=False
):
    """Initialize mock distribution for testing."""
    weights = torch.ones(n_voxels) / n_voxels
    features = _sample_multivariate_normal(n_features, n_voxels)
    embeddings = torch.rand(n_voxels, 3)
    geometry = torch.cdist(embeddings, embeddings)

    # Normalize outputs if need be
    if should_normalize:
        features = features / torch.linalg.norm(features, dim=1).reshape(-1, 1)
        geometry = geometry / geometry.max()
        embeddings = embeddings / geometry.max()

    distribution = (
        weights,
        features,
        geometry,
        embeddings,
    )

    if return_numpy:
        return tuple(map(lambda x: x.numpy(), distribution))
    else:
        return distribution


def _add_dict(d, new_d):
    """Add values of dictionary new_d to dictionary d."""
    for key, value in new_d.items():
        d.setdefault(key, []).append(value)
    return d


def save_mapping(mapping, fname):
    """Save mapping in pickle file, separating hyperparams and weights.

    Parameters
    ----------
    mapping: fugw.mappings
        FUGW mapping to save
    fname: str or pathlib.Path
        Path to pickle file to save
    """
    with open(fname, "wb") as f:
        # Dump hyperparams first
        pickle.dump(mapping, f)
        # Dump mapping weights
        pickle.dump(mapping.pi, f)


def load_mapping(fname, load_weights=True):
    """Load mapping from pickle file, optionally loading weights.

    Parameters
    ----------
    fname: str or pathlib.Path
        Path to pickle file to load
    load_weights: bool, optional, defaults to True
        If True, load mapping weights from pickle file.

    Returns
    -------
    mapping: fugw.mappings
    """
    with open(fname, "rb") as f:
        mapping = pickle.load(f)
        if load_weights:
            mapping.pi = pickle.load(f)

    return mapping
