from itertools import product

import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering
from scipy.sparse import coo_matrix

from fugw.utils import get_progress, make_tensor


def random_normalizing(X, sample_size=100, repeats=10):
    """
    Normalize X by dividing it by the maximum distance
    between pairs of rows of X.
    This maximum distance is determined by sampling pairs
    in X randomly.

    Parameters
    ----------
    X: torch.Tensor of size (n, k)
        Tensor to normalize.
    sample_size: int, optional, defaults to 100
        Number of vectors to sample from X at each iteration.
    repeats: int, optinal, defaults to 10
        Number of iterations to run.

    Returns
    -------
    X_normalized: torch.Tensor of size (n, k)
        Normalized X.
    d_max: float
        Maximum distance encountered while sampling pairs
        of indices from X.
    """
    d_max = 0
    X_tensor = make_tensor(X)
    for _ in range(repeats):
        idx = torch.randperm(X_tensor.shape[0])[:sample_size]
        distances = torch.cdist(X_tensor[idx, :], X_tensor[idx, :], p=2)
        d = distances.max()
        d_max = max(d, d_max)

    X_normalized = X_tensor / d_max

    return X_normalized, d_max.item()

def mesh_to_graph(coordinates, triangles):
    """
    Compute sparse matrix representing edges of a given mesh.

    Inputs
    ------
    mesh: str or os.PathLike or mesh object

    Outputs
    -------
    connectivity: sparse connectivity matrix
    """

    n_points = coordinates.shape[0]
    edges = np.hstack(
        (
            np.vstack((triangles[:, 0], triangles[:, 1])),
            np.vstack((triangles[:, 0], triangles[:, 2])),
            np.vstack((triangles[:, 1], triangles[:, 0])),
            np.vstack((triangles[:, 1], triangles[:, 2])),
            np.vstack((triangles[:, 2], triangles[:, 0])),
            np.vstack((triangles[:, 2], triangles[:, 1])),
        )
    )
    weights = np.ones(edges.shape[1])

    # Divide data by 2 since all edges i -> j are counted twice
    # because they all belong to exactly two triangles on the mesh
    connectivity = (
        coo_matrix((weights, edges), (n_points, n_points)).tocsr() / 2
    )

    # Making it symmetrical
    connectivity = (connectivity + connectivity.T) / 2

    return connectivity


def uniform_sampling_from_mesh(
        coordinates,
        triangles,
        embeddings,
        n_samples=100
):
    connectivity = mesh_to_graph(coordinates, triangles)
    ward = AgglomerativeClustering(
        n_clusters=n_samples,
        connectivity=connectivity,
        linkage="ward",
    )

    ward.fit_predict(embeddings)
    naive_samples = np.hstack([np.random.choice(np.argwhere(ward.labels_ == label).flatten(), 1) for label in range(n_samples)])
    return naive_samples

def fit(
    coarse_mapping=None,
    coarse_mapping_solver="mm",
    coarse_mapping_solver_params={},
    coarse_pairs_selection_method="topk",
    source_selection_radius=1,
    target_selection_radius=1,
    fine_mapping=None,
    fine_mapping_solver="mm",
    fine_mapping_solver_params={},
    source_sample=None,
    target_sample=None,
    source_features=None,
    target_features=None,
    source_geometry_embeddings=None,
    target_geometry_embeddings=None,
    source_weights=None,
    target_weights=None,
    device="auto",
    verbose=False,
):
    """
    Compute transport plan between source and target distributions
    using a coarse-to-fine approach.

    Parameters
    ----------
    coarse_mapping: fugw.FUGW
        Coarse model to fit
    coarse_mapping_solver: str, defaults to "mm"
        Solver to use to fit the coarse mapping
    coarse_mapping_solver_params: dict
        Parameters to give to the `.fit()` method
        of the coarse model
    coarse_pairs_selection_method: "topk" or "quantile"
        Method used to select pairs of source and target features
        whose neighbourhoods will be used to define
        the sparsity mask of the solution
    source_selection_radius: float
        Radius used to determine the neighbourhood
        of source vertices when defining sparsity mask
        for fine-scale solution
    target_selection_radius: float
        Radius used to determine the neighbourhood
        of target vertices when defining sparsity mask
        for fine-scale solution
    fine_mapping: fugw.FUGWSparse
        Fine-scale model to fit
    fine_mapping_solver: str, defaults to "mm"
        Solver to use to fit the fine-grained mapping
    fine_mapping_solver_params: dict
        Parameters to give to the `.fit()` method
        of the fine-scale model
    source_sample_size: int
        Number of vertices to sample from source
        for coarse step
    target_sample_size: int
        Number of vertices to sample from target
        for coarse step
    source_features: ndarray(n_features, n)
        Feature maps for source subject.
        n_features is the number of contrast maps, it should
        be the same for source and target data.
        n is the number of nodes on the source graph, it
        can be different from m, the number of nodes on the
        target graph.
    target_features: ndarray(n_features, m)
        Feature maps for target subject.
    source_geometry_embeddings: array(n, k)
        Embedding approximating the distance matrix between
        source vertices
    target_geometry_embeddings: array(m, k)
        Embedding approximating the distance matrix between
        target vertices
    source_weights: ndarray(n) or None
        Distribution weights of source nodes.
        Should sum to 1. If None, eahc node's weight
        will be set to 1 / n.
    target_weights: ndarray(n) or None
        Distribution weights of target nodes.
        Should sum to 1. If None, eahc node's weight
        will be set to 1 / m.
    device: "auto" or torch.device
        if "auto": use first available gpu if it's available,
        cpu otherwise.
    verbose: bool, optional, defaults to False
        Log solving process.
    """

    source_geometry_embeddings = make_tensor(source_geometry_embeddings)
    target_geometry_embeddings = make_tensor(target_geometry_embeddings)

    # Compute anatomical kernels
    source_geometry_kernel = torch.cdist(
        source_geometry_embeddings[source_sample],
        source_geometry_embeddings[source_sample],
        p=2,
    )
    source_geometry_kernel /= source_geometry_kernel.max()
    target_geometry_kernel = torch.cdist(
        target_geometry_embeddings[target_sample],
        target_geometry_embeddings[target_sample],
        p=2,
    )
    target_geometry_kernel /= target_geometry_kernel.max()

    # Sampled weights
    if source_weights is None:
        n = source_features.shape[1]
        source_weights = torch.ones(n) / n
    if target_weights is None:
        m = target_features.shape[1]
        target_weights = torch.ones(m) / m

    source_weights_sampled = make_tensor(source_weights)[source_sample]
    source_weights_sampled = (
        source_weights_sampled / source_weights_sampled.sum()
    )
    target_weights_sampled = make_tensor(target_weights)[target_sample]
    target_weights_sampled = (
        target_weights_sampled / target_weights_sampled.sum()
    )

    # Fit coarse model
    coarse_mapping.fit(
        source_features[:, source_sample],
        target_features[:, target_sample],
        source_geometry=source_geometry_kernel,
        target_geometry=target_geometry_kernel,
        source_weights=source_weights_sampled,
        target_weights=target_weights_sampled,
        solver=coarse_mapping_solver,
        solver_params=coarse_mapping_solver_params,
        device=device,
        verbose=verbose,
    )

    # Select best pairs of source and target vertices from coarse alignment
    if coarse_pairs_selection_method == "quantile":
        # Method 1: keep first percentile
        quantile = 99.95

        threshold = np.percentile(coarse_mapping.pi, quantile)
        rows, cols = np.nonzero(coarse_mapping.pi > threshold)
    elif coarse_pairs_selection_method == "topk":
        # Method 2: keep topk indices per line and per column
        # (this should be prefered as it will keep vertices
        # which are particularly unbalanced)
        rows = np.concatenate(
            [
                np.arange(source_sample.shape[0]),
                np.argmax(coarse_mapping.pi, axis=0),
            ]
        )
        cols = np.concatenate(
            [
                np.argmax(coarse_mapping.pi, axis=1),
                np.arange(target_sample.shape[0]),
            ]
        )

    # Build sparsity mask from pairs of coefficients
    def block(i, j, source_radius=5, target_radius=5):
        # find neighbours of voxel i in source subject
        # which are within searchlight radius
        distance_to_i = np.linalg.norm(
            source_geometry_embeddings
            - source_geometry_embeddings[source_sample[i]],
            ord=2,
            axis=1,
        )
        i_neighbours = np.nonzero(distance_to_i <= source_radius)[0]

        distance_to_j = np.linalg.norm(
            target_geometry_embeddings
            - target_geometry_embeddings[target_sample[j]],
            ord=2,
            axis=1,
        )
        j_neighbours = np.nonzero(distance_to_j <= target_radius)[0]

        return list(product(i_neighbours, j_neighbours))

    # Store pairs of source and target indices which are allowed to be matched
    sparsity_mask = []

    with get_progress() as progress:
        pairs = list(zip(rows, cols))
        if verbose:
            task = progress.add_task("Sparsity mask", total=len(pairs))
        for i, j in pairs:
            sparsity_mask.extend(
                block(
                    i,
                    j,
                    source_radius=source_selection_radius,
                    target_radius=target_selection_radius,
                )
            )
            if verbose:
                progress.update(task, advance=1)

    sparsity_mask = np.array(sparsity_mask, dtype=np.int32)

    # Sort 2d array along second axis
    # Takes about 1 minute
    sorted_mask_indices = np.lexsort(
        (sparsity_mask[:, 1], sparsity_mask[:, 0])
    )

    sorted_mask = sparsity_mask[sorted_mask_indices]

    # Keep all indices whose value is different from the value
    # of their predecesor
    kept_indices = (
        np.nonzero(
            np.any(
                (sorted_mask[1:, :] - sorted_mask[:-1, :]) != np.array([0, 0]),
                axis=1,
            )
        )[0]
        # Don't forget to shift indices
        + 1
    )
    # Don't forget first element of array
    kept_indices = np.append(kept_indices, 0)

    deduplicated_mask = sparsity_mask[kept_indices]

    init_plan = torch.sparse_coo_tensor(
        torch.from_numpy(deduplicated_mask.T),
        torch.from_numpy(
            np.ones(deduplicated_mask.shape[0]) / deduplicated_mask.shape[0]
        ),
        (
            source_geometry_embeddings.shape[0],
            target_geometry_embeddings.shape[0],
        ),
    )

    fine_mapping.fit(
        source_features,
        target_features,
        source_geometry_embedding=source_geometry_embeddings,
        target_geometry_embedding=target_geometry_embeddings,
        source_weights=source_weights,
        target_weights=target_weights,
        init_plan=init_plan,
        device=device,
        verbose=verbose,
        solver=fine_mapping_solver,
        solver_params=fine_mapping_solver_params,
    )
