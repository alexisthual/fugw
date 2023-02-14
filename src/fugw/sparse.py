import numpy as np
import torch
import warnings

from fugw.solvers.sparse import FUGWSparseSolver
from fugw.utils import (
    BaseModel,
    low_rank_squared_l2,
    make_sparse_csr_tensor,
    make_tensor,
)


class FUGWSparse(BaseModel):
    def __init__(
        self,
        alpha=0.5,
        rho=1,
        eps=1e-2,
        uot_solver="dc",
        reg_mode="joint",
        verbose=False,
        early_stopping_threshold=1e-6,
        **kwargs,
    ):
        # Save model arguments
        self.alpha = alpha
        self.rho = rho
        self.eps = eps
        self.uot_solver = uot_solver
        self.reg_mode = reg_mode
        self.verbose = verbose
        self.early_stopping_threshold = early_stopping_threshold

    def fit(
        self,
        source_features,
        target_features,
        source_geometry_embedding=None,
        target_geometry_embedding=None,
        source_weights=None,
        target_weights=None,
        init_plan=None,
        init_duals=None,
        device="auto",
        **kwargs,
    ):
        """
        Compute transport plan between source and target distributions
        using feature maps and geometries.
        In our case, feature maps are fMRI contrast maps,
        and geometries are that of the cortical meshes of individuals
        under study.

        Parameters
        ----------
        source_features: ndarray(n_features, n1)
            Feature maps for source subject.
            n_features is the number of contrast maps, it should
            be the same for source and target data.
            n1 is the number of nodes on the source graph, it
            can be different from n2, the number of nodes on the
            target graph.
        target_features: ndarray(n_features, n2)
            Feature maps for target subject.
        source_geometry_embedding: ndarray(n1, d), optional
            Embedding X such that norm(X_i - X_j) approximates
            the anatomical distance between vertices i and j
            of the source mesh
        target_geometry_embedding: ndarray(n2, d), optional
            Embedding X such that norm(X_i - X_j) approximates
            the anatomical distance between vertices i and j
            of the target mesh
        source_weights: ndarray(n1) or None
            Distribution weights of source nodes.
            Should sum to 1. If None, eahc node's weight
            will be set to 1 / n1.
        target_weights: ndarray(n1) or None
            Distribution weights of target nodes.
            Should sum to 1. If None, eahc node's weight
            will be set to 1 / n2.
        init_plan: torch.sparse COO or CSR matrix or None
            Torch sparse matrix whose sparsity mask will
            be that of the transport plan computed by this solver.
        device: "auto" or torch.device
            if "auto": use first available gpu if it's available,
            cpu otherwise.

        Returns
        -------
        self: FUGWSparse class object
            It comes with the following attributes:
            - pi: a fitted transport plan stored on CPU as a torch COO matrix
            - loss_steps: BCD steps at which the FUGW loss was evaluated
            - loss_: values of FUGW loss
            - loss_ent_: values of FUGW loss with entropy
        """
        if device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda", 0)
            else:
                device = torch.device("cpu")

        if isinstance(self.rho, float) or isinstance(self.rho, int):
            rho_x = self.rho
            rho_y = self.rho
        elif isinstance(self.rho, tuple) and len(self.rho) == 2:
            rho_x, rho_y = self.rho
        else:
            raise ValueError(
                "Invalid value of rho. Must be either a scalar or a tuple of"
                " two scalars."
            )

        # Set weights if they were not set by user
        if source_weights is None:
            Ws = (
                torch.ones(source_features.shape[1], device=device)
                / source_features.shape[1]
            )
        else:
            Ws = make_tensor(source_weights, device=device)

        if target_weights is None:
            Wt = (
                torch.ones(target_features.shape[1], device=device)
                / target_features.shape[1]
            )
        else:
            Wt = make_tensor(target_weights, device=device)

        # Compute distance matrix between features
        Fs = make_tensor(source_features.T, device=device)
        Ft = make_tensor(target_features.T, device=device)
        K1, K2 = low_rank_squared_l2(Fs, Ft)
        K1 = make_tensor(K1, device=device)
        K2 = make_tensor(K2, device=device)

        # Load anatomical kernels to GPU
        Gs1, Gs2 = low_rank_squared_l2(
            source_geometry_embedding, source_geometry_embedding
        )
        Gs1 = make_tensor(Gs1, device=device)
        Gs2 = make_tensor(Gs2, device=device)
        Gt1, Gt2 = low_rank_squared_l2(
            target_geometry_embedding, target_geometry_embedding
        )
        Gt1 = make_tensor(Gt1, device=device)
        Gt2 = make_tensor(Gt2, device=device)

        # Create model
        model = FUGWSparseSolver(**kwargs)

        # Check that all init_plan is valid
        if init_plan is None:
            warnings.warn(
                "Warning: init_plan is None, so this solver "
                "will compute a dense solution. However, "
                "dense solutions are much more efficiently computed "
                "by fugw.FUGW"
            )
        else:
            init_plan = make_sparse_csr_tensor(init_plan, device=device)

        # Compute transport plan
        (
            pi,
            gamma,
            duals_p,
            duals_g,
            loss_steps,
            loss_,
            loss_ent_,
        ) = model.solver(
            px=Ws,
            py=Wt,
            K=(K1, K2),
            Gs=(Gs1, Gs2),
            Gt=(Gt1, Gt2),
            alpha=self.alpha,
            rho_x=rho_x,
            rho_y=rho_y,
            eps=self.eps,
            uot_solver=self.uot_solver,
            reg_mode=self.reg_mode,
            early_stopping_threshold=self.early_stopping_threshold,
            init_plan=init_plan,
            init_duals=init_duals,
            verbose=self.verbose,
        )

        self.pi = pi.to_sparse_coo().detach().cpu()
        self.loss_steps = loss_steps
        self.loss_ = loss_
        self.loss_ent_ = loss_ent_

        # Free allocated GPU memory
        del Fs, Ft, K1, K2, Gs1, Gs2, Gt1, Gt2
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return (pi, gamma, duals_p, duals_g, loss_steps, loss_, loss_ent_)

    def transform(self, source_features, device="auto"):
        """
        Transport source feature maps using fitted OT plan.
        Use GPUs if available.

        Parameters
        ----------
        source_features: ndarray(n_samples, n1) or ndarray(n1)
            Contrast map for source subject
        device: "auto" or torch.device
            If "auto": use first available GPU if it's available,
            CPU otherwise.

        Returns
        -------
        transported_data: ndarray(n_samples, n2) or ndarray(n2)
            Contrast map transported in target subject's space
        """
        if device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda", 0)
            else:
                device = torch.device("cpu")

        if self.pi is None:
            raise Exception("Model should be fitted before calling transform")

        # Set correct type and device
        source_features_tensor = make_tensor(source_features, device=device)

        is_one_dimensional = False
        if source_features_tensor.ndim == 1:
            is_one_dimensional = True
            source_features_tensor = source_features_tensor.reshape(1, -1)
        if source_features_tensor.ndim > 2:
            raise ValueError(
                "source_features has too many dimensions:"
                f" {source_features_tensor.ndim}"
            )

        # Transform data
        transformed_data = (
            (
                torch.sparse.mm(
                    self.pi.to(device).transpose(0, 1),
                    source_features_tensor.T,
                ).to_dense()
                / (
                    torch.sparse.sum(self.pi.to(device), dim=0)
                    .to_dense()
                    .reshape(-1, 1)
                    # Add very small value to handle null rows
                    + 1e-16
                )
            )
            .T.detach()
            .cpu()
        )

        # Free allocated GPU memory
        del source_features_tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Modify returned tensor so that it matches
        # source_features's type and shape
        if isinstance(source_features, np.ndarray):
            transformed_data = transformed_data.numpy()

        if transformed_data.ndim > 1 and is_one_dimensional:
            transformed_data = transformed_data.flatten()

        return transformed_data

    def inverse_transform(self, target_features, device="auto"):
        """
        Transport target feature maps using fitted OT plan.
        Use GPUs if available.

        Parameters
        ----------
        target_features: ndarray(n_samples, n2) or ndarray(n2)
            Contrast map for target subject
        device: "auto" or torch.device
            If "auto": use first available GPU if it's available,
            CPU otherwise.

        Returns
        -------
        transported_data: ndarray(n_samples, n1) or ndarray(n1)
            Contrast map transported in target subject's space
        """
        if device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda", 0)
            else:
                device = torch.device("cpu")

        if self.pi is None:
            raise Exception("Model should be fitted before calling transform")

        # Set correct type and device
        target_features_tensor = make_tensor(target_features, device=device)

        is_one_dimensional = False
        if target_features_tensor.ndim == 1:
            is_one_dimensional = True
            target_features_tensor = target_features_tensor.reshape(1, -1)
        if target_features_tensor.ndim > 2:
            raise ValueError(
                "target_features has too many dimensions:"
                f" {target_features_tensor.ndim}"
            )

        # Transform data
        transformed_data = (
            (
                torch.sparse.mm(
                    self.pi.to(device),
                    target_features_tensor.T,
                ).to_dense()
                / (
                    torch.sparse.sum(self.pi.to(device), dim=1)
                    .to_dense()
                    .reshape(-1, 1)
                    # Add very small value to handle null rows
                    + 1e-16
                )
            )
            .T.detach()
            .cpu()
        )

        # Free allocated GPU memory
        del target_features_tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Modify returned tensor so that it matches
        # target_features's type and shape
        if isinstance(target_features, np.ndarray):
            transformed_data = transformed_data.numpy()

        if transformed_data.ndim > 1 and is_one_dimensional:
            transformed_data = transformed_data.flatten()

        return transformed_data
