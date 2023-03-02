import numpy as np
import torch

from fugw.utils import make_tensor


def test_make_tensor_cast_type():
    int_type = torch.int32
    float_type = torch.float32

    a = np.array([1, 2, 3], dtype=np.int32)
    t = make_tensor(a)
    assert t.dtype == int_type

    a = np.array([1, 2, 3], dtype=np.int64)
    t = make_tensor(a)
    assert t.dtype == int_type

    a = np.array([1, 2, 3], dtype=np.float32)
    t = make_tensor(a)
    assert t.dtype == float_type

    a = np.array([1, 2, 3], dtype=np.float64)
    t = make_tensor(a)
    assert t.dtype == float_type

    a = torch.tensor([1, 2, 3], dtype=torch.int32)
    t = make_tensor(a)
    assert t.dtype == int_type

    a = torch.tensor([1, 2, 3], dtype=torch.int64)
    t = make_tensor(a)
    assert t.dtype == int_type

    a = torch.tensor([1, 2, 3], dtype=torch.float32)
    t = make_tensor(a)
    assert t.dtype == float_type

    a = torch.tensor([1, 2, 3], dtype=torch.float64)
    t = make_tensor(a)
    assert t.dtype == float_type


def test_make_tensor_preserve_type():
    a = np.array([1, 2, 3], dtype=np.int32)
    t = make_tensor(a, dtype=torch.int64)
    assert t.dtype == torch.int64

    a = np.array([1, 2, 3], dtype=np.float32)
    t = make_tensor(a, dtype=torch.float64)
    assert t.dtype == torch.float64

    a = torch.tensor([1, 2, 3], dtype=torch.int32)
    t = make_tensor(a, dtype=torch.int64)
    assert t.dtype == torch.int64

    a = torch.tensor([1, 2, 3], dtype=torch.float32)
    t = make_tensor(a, dtype=torch.float64)
    assert t.dtype == torch.float64
