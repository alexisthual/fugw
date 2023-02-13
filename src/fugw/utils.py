import numpy as np
import torch

from rich.console import Console

console = Console()


class BaseModel():
    def fit(self, source_data, target_data):
        return None

    def transform(self, source_map):
        return None


def make_tensor(x, device=None, dtype=torch.float32):
    if isinstance(x, np.ndarray):
        return torch.tensor(x).to(device, dtype)
    elif isinstance(x, torch.Tensor):
        return x.to(device, dtype)
    else:
        raise Exception(f"Expected np.ndarray or torch.Tensor, got {type(x)}")


def make_sparse_csr_tensor(x, device=None, dtype=torch.float32):
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


def make_csr_matrix(crow_indices, col_indices, values, size, device):
    return torch.sparse_csr_tensor(
        crow_indices.to(device),
        col_indices.to(device),
        values.to(device),
        size=size,
    )


def low_rank_squared_l2(X, Y):
    """
    Write square Euclidean distance matrix M as exact product
    of two low rank matrices A1 and A2: M = A1 A2^t
    """
    if not (isinstance(X, torch.Tensor) and isinstance(Y, torch.Tensor)):
        X = torch.Tensor(X)
        Y = torch.Tensor(Y)

    device, dtype = X.device, X.dtype

    nx, ny = X.shape[0], Y.shape[0]

    Vx = (X**2).sum(1, keepdim=True)  # shape nx x 1
    Vy = (Y**2).sum(1, keepdim=True)  # shape ny x 1
    ones_x = torch.ones(nx, 1).to(device).to(dtype)  # shape nx x 1
    ones_y = torch.ones(ny, 1).to(device).to(dtype)  # shape ny x 1

    A1 = torch.cat([Vx, ones_x, -(2**0.5) * X], dim=1)  # shape nx x (d+2)
    A2 = torch.cat([ones_y, Vy, 2**0.5 * Y], dim=1)  # shape ny x (d+2)

    return (A1, A2)
