import pickle

from itertools import product
from tempfile import TemporaryDirectory

import numpy as np
import pytest
import torch

from fugw.mappings import FUGW
from fugw.utils import (
    _init_mock_distribution,
    _make_tensor,
    init_plan_dense,
    load_mapping,
    save_mapping,
)


def test__make_tensor_cast_type():
    int_type = torch.int32
    float_type = torch.float32

    a = np.array([1, 2, 3], dtype=np.int32)
    t = _make_tensor(a)
    assert t.dtype == int_type

    a = np.array([1, 2, 3], dtype=np.int64)
    t = _make_tensor(a)
    assert t.dtype == int_type

    a = np.array([1, 2, 3], dtype=np.float32)
    t = _make_tensor(a)
    assert t.dtype == float_type

    a = np.array([1, 2, 3], dtype=np.float64)
    t = _make_tensor(a)
    assert t.dtype == float_type

    a = torch.tensor([1, 2, 3], dtype=torch.int32)
    t = _make_tensor(a)
    assert t.dtype == int_type

    a = torch.tensor([1, 2, 3], dtype=torch.int64)
    t = _make_tensor(a)
    assert t.dtype == int_type

    a = torch.tensor([1, 2, 3], dtype=torch.float32)
    t = _make_tensor(a)
    assert t.dtype == float_type

    a = torch.tensor([1, 2, 3], dtype=torch.float64)
    t = _make_tensor(a)
    assert t.dtype == float_type


def test__make_tensor_preserve_type():
    a = np.array([1, 2, 3], dtype=np.int32)
    t = _make_tensor(a, dtype=torch.int64)
    assert t.dtype == torch.int64

    a = np.array([1, 2, 3], dtype=np.float32)
    t = _make_tensor(a, dtype=torch.float64)
    assert t.dtype == torch.float64

    a = torch.tensor([1, 2, 3], dtype=torch.int32)
    t = _make_tensor(a, dtype=torch.int64)
    assert t.dtype == torch.int64

    a = torch.tensor([1, 2, 3], dtype=torch.float32)
    t = _make_tensor(a, dtype=torch.float64)
    assert t.dtype == torch.float64


np.random.seed(0)
torch.manual_seed(0)

n_voxels_source = 105
n_voxels_target = 95
n_features_train = 10
n_features_test = 5

return_numpys = [True, False]

devices = [torch.device("cpu")]
if torch.cuda.is_available():
    devices.append(torch.device("cuda:0"))

solvers = ["sinkhorn", "mm", "ibpp"]


@pytest.mark.parametrize(
    "device,return_numpy,solver", product(devices, return_numpys, solvers)
)
def test_saving_and_loading(device, return_numpy, solver):
    _, source_features_train, source_geometry, _ = _init_mock_distribution(
        n_features_train, n_voxels_source, return_numpy=return_numpy
    )
    _, target_features_train, target_geometry, _ = _init_mock_distribution(
        n_features_train, n_voxels_target, return_numpy=return_numpy
    )

    fugw = FUGW()
    fugw.fit(
        source_features=source_features_train,
        target_features=target_features_train,
        source_geometry=source_geometry,
        target_geometry=target_geometry,
        solver=solver,
        solver_params={
            "nits_bcd": 3,
            "ibpp_eps_base": 1e8,
        },
        device=device,
    )

    with TemporaryDirectory() as tmpdir:
        fname = tmpdir + "/mapping.pkl"

        save_mapping(fugw, fname)

        mapping_without_weights = load_mapping(fname, load_weights=False)
        assert mapping_without_weights.pi is None

        mapping_with_weights = load_mapping(fname, load_weights=True)
        assert mapping_with_weights.pi.shape == (
            n_voxels_source,
            n_voxels_target,
        )

        with open(fname, "rb") as f:
            mapping_pickle = pickle.load(f)
            assert mapping_pickle.pi is None

            weights = pickle.load(f)
            assert weights.shape == (n_voxels_source, n_voxels_target)

    with TemporaryDirectory() as tmpdir:
        fname = tmpdir + "/mapping.pkl"

        with open(fname, "wb") as f:
            pickle.dump(fugw, f)
            pickle.dump(fugw.pi, f)

        mapping_without_weights = load_mapping(fname, load_weights=False)
        assert mapping_without_weights.pi is None

        mapping_with_weights = load_mapping(fname, load_weights=True)
        assert mapping_with_weights.pi.shape == (
            n_voxels_source,
            n_voxels_target,
        )

        with open(fname, "rb") as f:
            mapping_pickle = pickle.load(f)
            assert mapping_pickle.pi is None

            weights = pickle.load(f)
            assert weights.shape == (n_voxels_source, n_voxels_target)


@pytest.mark.parametrize(
    "method", ["identity", "entropic", "permutation", "unknown"]
)
def test_init_plan(method):
    n_source = 101
    n_target = 99

    if method == "unknown":
        with pytest.raises(Exception, match="Unknown initialisation method.*"):
            init_plan_dense(n_source, n_target, method=method)
    else:
        if method == "identity":
            with pytest.raises(
                AssertionError, match="Number of source and target.*"
            ):
                init_plan_dense(n_source, n_target, method=method)

            n_source = 100
            n_target = 100

        plan = init_plan_dense(n_source, n_target, method=method)
        assert plan.shape == (n_source, n_target)
        # Check that plan satisfies marginal constraints
        assert torch.allclose(plan.sum(dim=0), torch.ones(n_target) / n_target)
        assert torch.allclose(plan.sum(dim=1), torch.ones(n_source) / n_source)
