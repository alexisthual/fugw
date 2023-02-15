import numpy as np
import torch

from rich.console import Console

console = Console()


class BaseTransformer:
    def __init__(
        self,
        alpha=0.5,
        rho=1,
        eps=1e-2,
        reg_mode="joint",
    ):
        """Init FUGW problem.

        Parameters
        ----------
        alpha: float, optional, defaults to 0.5
            Value in ]0, 1[, interpolates the relative importance of
            the Wasserstein and the Gromov-Wasserstein losses
            in the FUGW loss (see equation)
        rho: float or tuple of 2 floats, optional, defaults to 1
            Value in ]0, +inf[, controls the relative importance of
            the marginal constraints. High values force the mass of
            each point to be transported ;
            low values allow for some mass loss
        eps: float, optional, defaults to 1e-2
            Value in ]0, +inf[, controls the relative importance of
            the entropy loss
        reg_mode: "joint" or "independent", optional, defaults to "joint"
            "joint": use unbalanced-GW-like regularisation term
            "independent": use unbalanced-W-like regularisation term
        """

        self.alpha = alpha
        self.rho = rho
        self.eps = eps
        self.reg_mode = reg_mode

    def fit(self, source_data, target_data):
        return None

    def transform(self, source_map):
        return None


class BaseSolver:
    def __init__(
        self,
        nits_bcd=10,
        nits_uot=1000,
        tol_bcd=1e-7,
        tol_uot=1e-7,
        early_stopping_threshold=1e-6,
        eval_bcd=1,
        eval_uot=10,
        # dc-specific parameters
        dc_eps_base=1,
        dc_nits_sinkhorn=1,
    ):
        """Init FUGW solver.

        Parameters
        ----------
        nits_bcd: int,
            Number of block-coordinate-descent iterations to run
        nits_uot: int,
            Number of solver iteration to run at each BCD iteration
        tol_bcd: float,
            Stop the BCD procedure early if the absolute difference
            between two consecutive transport plans
            under this threshold
        tol_uot: float,
            Stop the BCD procedure early if the absolute difference
            between two consecutive transport plans
            under this threshold
        early_stopping_threshold: float,
            Stop the BCD procedure early if the FUGW loss falls
            under this threshold
        eval_bcd: int,
            During .fit(), at every eval_bcd step:
            1. compute the FUGW loss and store it in an array
            2. consider stopping early
        eval_uot: int,
            During .fit(), at every eval_uot step:
            1. consider stopping early
        dc_eps_base: int,
            Regularization parameter specific to the dc solver
        dc_nits_sinkhorn: int,
            Number of sinkhorn iterations to run
            within each uot iteration of the dc solver.
        """

        self.nits_bcd = nits_bcd
        self.nits_uot = nits_uot
        self.tol_bcd = tol_bcd
        self.tol_uot = tol_uot
        self.early_stopping_threshold = early_stopping_threshold
        self.eval_bcd = eval_bcd
        self.eval_uot = eval_uot
        self.dc_eps_base = dc_eps_base
        self.dc_nits_sinkhorn = dc_nits_sinkhorn


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
