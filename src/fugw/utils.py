import numpy as np
import torch
from sklearn.base import BaseEstimator, TransformerMixin


class BaseModel(BaseEstimator, TransformerMixin):
    def fit(self, source_data, target_data):
        return None

    def transform(self, source_map):
        return None


def make_tensor(x):
    if isinstance(x, np.ndarray):
        return torch.tensor(x)
    elif isinstance(x, torch.Tensor):
        return x
    else:
        raise Exception(f"Expected np.array or torch.tensor, got {type(x)}")


def low_rank_squared_l2(X, Y):
    """
    Write square Euclidean distance matrix M as exact product
    of two low rank matrices A1 and A2: M = A1 A2^t
    """
    device, dtype = X.device, X.dtype
    nx, ny = X.shape[0], Y.shape[0]

    Vx = (X**2).sum(1, keepdim=True)  # shape nx x 1
    Vy = (Y**2).sum(1, keepdim=True)  # shape ny x 1
    ones_x = torch.ones(nx, 1).to(device).to(dtype)  # shape nx x 1
    ones_y = torch.ones(ny, 1).to(device).to(dtype)  # shape ny x 1

    A1 = torch.cat([Vx, ones_x, -(2**0.5) * X], dim=1)  # shape nx x (d+2)
    A2 = torch.cat([ones_y, Vy, 2**0.5 * Y], dim=1)  # shape ny x (d+2)

    return (A1, A2)