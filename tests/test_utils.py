import numpy as np
import torch

from fugw.utils import make_tensor


def test_make_tensor():
    a = np.array([1, 2, 3], dtype=np.int32)
    t = make_tensor(a)
    assert t.dtype == torch.int32

    a = np.array([1, 2, 3], dtype=np.int64)
    t = make_tensor(a)
    assert t.dtype == torch.int64

    a = np.array([1, 2, 3], dtype=np.float32)
    t = make_tensor(a)
    assert t.dtype == torch.float32

    a = np.array([1, 2, 3], dtype=np.float64)
    t = make_tensor(a)
    assert t.dtype == torch.float64

    a = torch.tensor([1, 2, 3], dtype=torch.int32)
    t = make_tensor(a)
    assert t.dtype == torch.int32

    a = torch.tensor([1, 2, 3], dtype=torch.int64)
    t = make_tensor(a)
    assert t.dtype == torch.int64

    a = torch.tensor([1, 2, 3], dtype=torch.float32)
    t = make_tensor(a)
    assert t.dtype == torch.float32

    a = torch.tensor([1, 2, 3], dtype=torch.float64)
    t = make_tensor(a)
    assert t.dtype == torch.float64
