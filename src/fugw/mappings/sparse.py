import numpy as np
import torch
import warnings

from fugw.mappings.utils import BaseMapping
from fugw.solvers.sparse import FUGWSparseSolver
from fugw.utils import (
    low_rank_squared_l2,
    make_sparse_csr_tensor,
    make_tensor,
)


class FUGWSparse(BaseMapping):
    """Class computing sparse transport plans"""

    def fit(
        self,
        source_features=None,
        target_features=None,
        source_geometry_embedding=None,
        target_geometry_embedding=None,
        source_weights=None,
        target_weights=None,
        init_plan=None,
        init_duals=None,
        solver="mm",
        solver_params={},
        device="auto",
        verbose=False,
    ):
        """
        Compute transport plan between source and target distributions
        using feature maps and geometries.
        In our case, feature maps are fMRI contrast maps,
        and geometries are that of the cortical meshes of individuals
        under study.

        Parameters
        ----------
        source_features: ndarray(n_features, n), optional
            Feature maps for source subject.
            n_features is the number of contrast maps, it should
            be the same for source and target data.
            n is the number of nodes on the source graph, it
            can be different from m, the number of nodes on the
            target graph.
            **This array should be normalized**, otherwise you will
            run into computational errors.
        target_features: ndarray(n_features, m), optional
            Feature maps for target subject.
            **This array should be normalized**, otherwise you will
            run into computational errors.
        source_geometry_embedding: ndarray(n, k), optional
            Embedding X such that norm(X_i - X_j) approximates
            the anatomical distance between vertices i and j
            of the source mesh
            **This array should be normalized**, otherwise you will
            run into computational errors.
        target_geometry_embedding: ndarray(m, k), optional
            Embedding X such that norm(X_i - X_j) approximates
            the anatomical distance between vertices i and j
            of the target mesh
            **This array should be normalized**, otherwise you will
            run into computational errors.
        source_weights: ndarray(n) or None
            Distribution weights of source nodes.
            Should sum to 1. If None, each node's weight
            will be set to 1 / n.
        target_weights: ndarray(n) or None
            Distribution weights of target nodes.
            Should sum to 1. If None, each node's weight
            will be set to 1 / m.
        init_plan: torch.sparse COO or CSR matrix or None
            Torch sparse matrix whose sparsity mask will
            be that of the transport plan computed by this solver.
        solver: "mm" or "ibpp"
            Solver to use.
        solver_params: fugw.solvers.utils.BaseSolver params
            Parameters given to the solver.
        device: "auto" or torch.device
            if "auto": use first available gpu if it's available,
            cpu otherwise.
        verbose: bool, optional, defaults to False
            Log solving process.

        Returns
        -------
        self: FUGWSparse class object
        """
        if device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda", 0)
            else:
                device = torch.device("cpu")

        if isinstance(self.rho, float) or isinstance(self.rho, int):
            rho_s = self.rho
            rho_t = self.rho
        elif isinstance(self.rho, tuple) and len(self.rho) == 2:
            rho_s, rho_t = self.rho
        else:
            raise ValueError(
                "Invalid value of rho. Must be either a scalar or a tuple of"
                " two scalars."
            )

        # Set weights if they were not set by user
        if source_weights is None:
            ws = (
                torch.ones(source_features.shape[1], device=device)
                / source_features.shape[1]
            )
        else:
            ws = make_tensor(source_weights, device=device)

        if target_weights is None:
            wt = (
                torch.ones(target_features.shape[1], device=device)
                / target_features.shape[1]
            )
        else:
            wt = make_tensor(target_weights, device=device)

        # Compute distance matrix between features
        Fs = make_tensor(source_features.T, device=device)
        Ft = make_tensor(target_features.T, device=device)
        F1, F2 = low_rank_squared_l2(Fs, Ft)
        F1 = make_tensor(F1, device=device)
        F2 = make_tensor(F2, device=device)

        # Load anatomical kernels to GPU
        Ds1, Ds2 = low_rank_squared_l2(
            source_geometry_embedding, source_geometry_embedding
        )
        Ds1 = make_tensor(Ds1, device=device)
        Ds2 = make_tensor(Ds2, device=device)
        Dt1, Dt2 = low_rank_squared_l2(
            target_geometry_embedding, target_geometry_embedding
        )
        Dt1 = make_tensor(Dt1, device=device)
        Dt2 = make_tensor(Dt2, device=device)

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

        # Create model
        model = FUGWSparseSolver(**solver_params)

        # Compute transport plan
        res = model.solve(
            alpha=self.alpha,
            rho_s=rho_s,
            rho_t=rho_t,
            eps=self.eps,
            reg_mode=self.reg_mode,
            F=(F1, F2),
            Ds=(Ds1, Ds2),
            Dt=(Dt1, Dt2),
            ws=ws,
            wt=wt,
            init_plan=init_plan,
            init_duals=init_duals,
            solver=solver,
            verbose=verbose,
        )

        self.pi = res["pi"].to_sparse_coo().detach().cpu()
        self.loss_steps = res["loss_steps"]
        self.loss = res["loss"]
        self.loss_entropic = res["loss_entropic"]
        self.loss_times = res["loss_times"]

        # Free allocated GPU memory
        del Fs, Ft, F1, F2, Ds1, Ds2, Dt1, Dt2
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return self

    def transform(self, source_features, device="auto"):
        """
        Transport source feature maps using fitted OT plan.
        Use GPUs if available.

        Parameters
        ----------
        source_features: ndarray(n_samples, n) or ndarray(n)
            Contrast map for source subject
        device: "auto" or torch.device
            If "auto": use first available GPU if it's available,
            CPU otherwise.

        Returns
        -------
        transported_data: ndarray(n_samples, m) or ndarray(m)
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
        target_features: ndarray(n_samples, m) or ndarray(m)
            Contrast map for target subject
        device: "auto" or torch.device
            If "auto": use first available GPU if it's available,
            CPU otherwise.

        Returns
        -------
        transported_data: ndarray(n_samples, n) or ndarray(n)
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
