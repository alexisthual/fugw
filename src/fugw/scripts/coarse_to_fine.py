import numpy as np
import torch

from fugw.utils import _make_tensor
from scipy.sparse import coo_matrix
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.image import grid_to_graph


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
    repeats: int, optional, defaults to 10
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
    X_tensor = _make_tensor(X)
    for _ in range(repeats):
        idx = torch.randperm(X_tensor.shape[0])[:sample_size]
        distances = torch.cdist(X_tensor[idx, :], X_tensor[idx, :], p=2)
        d = distances.max()
        d_max = max(d, d_max)

    X_normalized = X_tensor / d_max

    return X_normalized, d_max.item()


def mesh_connectivity_matrix(coordinates, triangles):
    """
    Compute sparse matrix representing edges of a given mesh.

    Parameters
    ----------
    coordinates: ndarray of size (n, k)
    triangles: ndarray of size (e, 3)

    Returns
    -------
    connectivity: sparse coo matrix of size (n, n)
    """
    n_vertices = coordinates.shape[0]
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
        coo_matrix((weights, edges), (n_vertices, n_vertices)).tocsr() / 2
    )

    # Force symmetry
    connectivity = (connectivity + connectivity.T) / 2

    return connectivity


def sample_mesh_uniformly(
    coordinates, triangles, embeddings=None, n_samples=100
):
    """
    Returns random indices on a mesh such that their are
    approximately uniformly spead over the surface.
    It leverages Ward's algorithm to build same-size clusters
    from ``embeddings`` and then samples in these clusters.

    Parameters
    ----------
    coordinates: np.ndarray of size (n, 3)
        3D coordinates of vertices of the mesh
    triangles: np.ndarray of size (e, 3)
        Faces of the mesh
    embeddings: np.ndarray of size (n, d)
        Embeddings approximating the geodesic distance on the mesh
    n_samples: int
        Number of points to sample

    Returns
    -------
    samples: np.ndarray of size (s)
        Indices of sampled points
    """
    if embeddings is None:
        embeddings = np.ones(coordinates.shape[0]).reshape(-1, 1)

    connectivity = mesh_connectivity_matrix(coordinates, triangles)

    ward = AgglomerativeClustering(
        n_clusters=n_samples,
        connectivity=connectivity,
        linkage="ward",
    )
    ward.fit_predict(embeddings)

    naive_samples = np.hstack(
        [
            np.random.choice(np.argwhere(ward.labels_ == label).flatten(), 1)
            for label in range(n_samples)
        ]
    ).astype(np.int32)

    return naive_samples


def sample_volume_uniformly(segmentation, embeddings=None, n_samples=100):
    """
    Returns random indices on a volume/segmentation such that they are
    approximately uniformly spread over the volume.
    It leverages Ward's algorithm to build same-size clusters
    from ``embeddings`` and then samples in these clusters.

    Parameters
    ----------
    segmentation: np.ndarray of size (x, y, z)
        Segmentation mask of the region of interest
    embeddings: np.ndarray of size (n, d)
        Embeddings approximating the geodesic distance in the volume
    n_samples: int
        Number of points to sample

    Returns
    -------
    samples: np.ndarray of size (s)
        Indices of sampled points
    """

    coordinates = np.array(np.nonzero(segmentation)).T

    if embeddings is None:
        embeddings = np.ones(coordinates.shape[0]).reshape(-1, 1)

    x, y, z = segmentation.shape
    connectivity = grid_to_graph(x, y, z, mask=segmentation)

    ward = AgglomerativeClustering(
        n_clusters=n_samples,
        connectivity=connectivity,
        linkage="ward",
    )
    ward.fit_predict(embeddings)

    naive_samples = np.hstack(
        [
            np.random.choice(np.argwhere(ward.labels_ == label).flatten(), 1)
            for label in range(n_samples)
        ]
    ).astype(np.int32)

    return naive_samples


def get_cluster_matrix(clusters, n_samples):
    """
    Computes a sparse matrix C such that C_{i, j} = 1
    if and only if the ``i``-th sampled point
    belongs to cluster ``j``.

    Parameters
    ----------
    clusters: torch.Tensor of size (c,)
        An array such that the index at ``clusters[i]``
        belongs to cluster ``i``
    n_samples: int
        Number of sampled points ``s``

    Returns
    -------
    C: torch.Tensor sparse COO matrix of size (s, c)
    """
    n_clusters = clusters.shape[0]
    C = torch.sparse_coo_tensor(
        torch.stack(
            [
                torch.tensor(clusters).type(torch.int32),
                torch.arange(n_clusters).type(torch.int32),
            ]
        ),
        torch.ones(n_clusters).type(torch.float32),
        size=(n_samples, n_clusters),
    ).coalesce()

    return C


def get_neighbourhood_matrix(embeddings, sample, radius):
    """
    Computes a sparse matrix ``N`` such that ``N_{i, j} = 1``
    if and only if the geodesic distance between vertex ``i``
    and the ``j``-th sampled vertex is less than ``radius``.

    Parameters
    ----------
    embeddings: torch.Tensor of size (n, d)
        Embeddings X such that ``torch.linalg.norm(X[i], X[j], ord=2)``
        approximates the geodesic distance between points ``i`` and ``j``
    sample: torch.Tensor of size (s)
        Indices of points of ``embeddings`` which were sampled
    radius: float
        Radius of computed neighbouhoods

    Returns
    -------
    N: torch.Tensor sparse COO matrix of size (n, s)
    """
    n_vertices = embeddings.shape[0]
    n_samples = sample.shape[0]

    vertices_within_radius = [
        torch.tensor(
            np.argwhere(
                np.linalg.norm(
                    embeddings - embeddings[sample_index],
                    ord=2,
                    axis=1,
                )
                <= radius
            )
        )
        for sample_index in sample
    ]

    rows = torch.concat(vertices_within_radius).flatten().type(torch.int32)
    cols = torch.concat(
        [
            i * torch.ones(len(vertices_within_radius[i]))
            for i in range(n_samples)
        ]
    ).type(torch.int32)
    values = torch.ones_like(rows).type(torch.float32)

    neighbourhood_matrix = torch.sparse_coo_tensor(
        torch.stack([rows, cols]),
        values,
        size=(n_vertices, n_samples),
    ).coalesce()

    return neighbourhood_matrix


def fit(
    coarse_mapping=None,
    coarse_mapping_solver="mm",
    coarse_mapping_solver_params={},
    coarse_callback_bcd=None,
    coarse_pairs_selection_method="topk",
    source_selection_radius=1,
    target_selection_radius=1,
    fine_mapping=None,
    fine_mapping_solver="mm",
    fine_mapping_solver_params={},
    fine_callback_bcd=None,
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
    source_sample: array of size (n1, )
        Indices of vertices sampled on source distribution
        which will be used when fitting the coarse mapping
    target_sample: array of size (m1, )
        Indices of vertices sampled on target distribution
        which will be used when fitting the coarse mapping
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
        Should sum to 1. If None, each node's weight
        will be set to 1 / n.
    target_weights: ndarray(n) or None
        Distribution weights of target nodes.
        Should sum to 1. If None, each node's weight
        will be set to 1 / m.
    device: "auto" or torch.device
        if "auto": use first available gpu if it's available,
        cpu otherwise.
    verbose: bool, optional, defaults to False
        Log solving process.

    Returns
    -------
    source_sample: torch.Tensor of size(source_sample_size)
        Tensor containing the indices which were sampled on the
        source so as to compute the coarse mapping.
    target_sample: torch.Tensor of size(target_sample_size)
        Tensor containing the indices which were sampled on the
        target so as to compute the coarse mapping.
    """
    # 0. Parse input tensors
    source_sample = _make_tensor(source_sample, dtype=torch.int64)
    target_sample = _make_tensor(target_sample, dtype=torch.int64)

    source_features = _make_tensor(source_features)
    target_features = _make_tensor(target_features)

    source_geometry_embeddings = _make_tensor(source_geometry_embeddings)
    target_geometry_embeddings = _make_tensor(target_geometry_embeddings)

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

    source_weights_sampled = _make_tensor(source_weights)[source_sample]
    source_weights_sampled = (
        source_weights_sampled / source_weights_sampled.sum()
    )
    target_weights_sampled = _make_tensor(target_weights)[target_sample]
    target_weights_sampled = (
        target_weights_sampled / target_weights_sampled.sum()
    )

    # 1. Fit coarse mapping
    coarse_mapping.fit(
        source_features[:, source_sample],
        target_features[:, target_sample],
        source_geometry=source_geometry_kernel,
        target_geometry=target_geometry_kernel,
        source_weights=source_weights_sampled,
        target_weights=target_weights_sampled,
        solver=coarse_mapping_solver,
        solver_params=coarse_mapping_solver_params,
        callback_bcd=coarse_callback_bcd,
        device=device,
        verbose=verbose,
    )

    # 2. Build sparsity mask

    # Select best pairs of source and target vertices from coarse alignment
    if coarse_pairs_selection_method == "quantile":
        # Method 1: keep first percentile
        quantile = 99.95

        threshold = np.percentile(coarse_mapping.pi, quantile)
        rows, cols = np.nonzero(coarse_mapping.pi > threshold)

    elif coarse_pairs_selection_method == "topk":
        # Method 2: keep topk indices per line and per column
        # (this should be preferred as it will keep vertices
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

    # Compute mask as a matrix product between:
    # a. neighbourhood matrices that encode
    # which vertex is close to which sampled point
    N_source = get_neighbourhood_matrix(
        source_geometry_embeddings, source_sample, source_selection_radius
    )
    N_target = get_neighbourhood_matrix(
        target_geometry_embeddings, target_sample, target_selection_radius
    )
    # b. cluster matrices that encode
    # which sampled point belongs to which cluster
    C_source = get_cluster_matrix(rows, source_sample.shape[0])
    C_target = get_cluster_matrix(cols, target_sample.shape[0])

    mask = (N_source @ C_source) @ (N_target @ C_target).T

    # Define init plan from spasity mask
    init_plan = torch.sparse_coo_tensor(
        mask.indices(),
        torch.ones_like(mask.values()) / mask.values().shape[0],
        (
            source_geometry_embeddings.shape[0],
            target_geometry_embeddings.shape[0],
        ),
    ).coalesce()

    # 3. Fit fine-grained mapping
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
        callback_bcd=fine_callback_bcd,
    )

    return source_sample, target_sample
