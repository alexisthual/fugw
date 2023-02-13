from itertools import product

import numpy as np
import pytest
import torch

from fugw import FUGWSparse

from .utils import init_distribution

np.random.seed(100)
n_voxels_source = 105
n_voxels_target = 95
n_features_train = 10
n_features_test = 5

sparse_layouts = ["coo", "csr"]

devices = [torch.device("cpu")]
if torch.cuda.is_available():
    devices.append(torch.device("cuda:0"))


def test_fugw_sparse():
    # Generate random training data for source and target
    _, source_features_train, _, source_embeddings = init_distribution(
        n_features_train, n_voxels_source
    )
    _, target_features_train, _, target_embeddings = init_distribution(
        n_features_train, n_voxels_target
    )

    fugw = FUGWSparse()
    fugw.fit(
        source_features_train,
        target_features_train,
        source_geometry_embedding=source_embeddings,
        target_geometry_embedding=target_embeddings,
    )

    # Use trained model to transport new features
    source_features_test = np.random.rand(n_features_test, n_voxels_source)
    target_features_test = np.random.rand(n_features_test, n_voxels_target)

    transformed_data = fugw.transform(source_features_test)
    assert transformed_data.shape == target_features_test.shape

    # Compute score
    s = fugw.score(source_features_test, target_features_test)
    assert isinstance(s, int) or isinstance(s, float)


@pytest.mark.parametrize(
    "device,sparse_layout",
    product(devices, sparse_layouts),
)
def test_fugw_sparse_with_init(device, sparse_layout):
    # Generate random training data for source and target
    _, source_features_train, _, source_embeddings = init_distribution(
        n_features_train, n_voxels_source
    )
    _, target_features_train, _, target_embeddings = init_distribution(
        n_features_train, n_voxels_target
    )

    rows = []
    cols = []
    items_per_row = 3
    for i in range(n_voxels_source):
        for _ in range(items_per_row):
            rows.append(i)
        cols.extend(np.random.permutation(n_voxels_target)[:items_per_row])

    if sparse_layout == "coo":
        init_plan = torch.sparse_coo_tensor(
            np.array([rows, cols]),
            np.ones(len(rows)) / len(rows),
            size=(n_voxels_source, n_voxels_target),
        ).to(device)
    elif sparse_layout == "csr":
        init_plan = torch.sparse_coo_tensor(
            np.array([rows, cols]),
            np.ones(len(rows)) / len(rows),
            size=(n_voxels_source, n_voxels_target),
        ).to(device).to_sparse_csr()

    fugw = FUGWSparse()
    fugw.fit(
        source_features_train,
        target_features_train,
        source_geometry_embedding=source_embeddings,
        target_geometry_embedding=target_embeddings,
        init_plan=init_plan,
        device=device,
    )

    # Use trained model to transport new features
    source_features_test = np.random.rand(n_features_test, n_voxels_source)
    target_features_test = np.random.rand(n_features_test, n_voxels_target)

    transformed_data = fugw.transform(source_features_test, device=device)
    assert transformed_data.shape == target_features_test.shape

    # Compute score
    # s = fugw.score(source_features_test, target_features_test)
    # assert isinstance(s, int) or isinstance(s, float)


# TODO: at some point, it would be nice that this test
# passes so that our model really is a Scikit learn transformer
# def test_fugw_sklearn_transform_api():
#     fugw = FUGW()
#     check_estimator(fugw)
