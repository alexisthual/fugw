import numpy as np
import torch
import warnings

from fugw.mappings.utils import BaseMapping
from fugw.solvers.sparse import FUGWSparseSolver
from fugw.utils import (
    _low_rank_squared_l2,
    _make_sparse_csr_tensor,
    _make_tensor,
    console,
)


class FUGWSparse(BaseMapping):
    """Class computing sparse transport plans"""

    def fit(
        self,
        source_features=None,
        target_features=None,
        source_geometry_embedding=None,
        target_geometry_embedding=None,
        source_features_val=None,
        target_features_val=None,
        source_geometry_embedding_val=None,
        target_geometry_embedding_val=None,
        source_weights=None,
        target_weights=None,
        init_plan=None,
        init_duals=None,
        solver="mm",
        solver_params={},
        callback_bcd=None,
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
        source_features_val: ndarray(n_features, n) or None
            Feature maps for source subject used for validation.
            If None, source_features will be used instead.
        target_features_val: ndarray(n_features, m) or None
            Feature maps for target subject used for validation.
            If None, target_features will be used instead.
        source_geometry_embedding_val: ndarray(n, n) or None
            Kernel matrix of anatomical distances
            between nodes of source mesh used for validation.
            If None, source_geometry will be used instead.
        target_geometry_embedding_val: ndarray(m, m) or None
            Kernel matrix of anatomical distances
            between nodes of target mesh used for validation.
            If None, target_geometry will be used instead.
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
        callback_bcd: callable or None
            Callback function called at the end of each BCD step.
            It will be called with the following arguments:

                - locals (dictionary containing all local variables)
        device: "auto" or torch.device
            if "auto": use first available gpu if it's available,
            cpu otherwise.
        verbose: bool, optional, defaults to False
            Log solving process.

        Returns
        -------
        self: FUGWSparse class object
        """
        if self.divergence == "l2" and solver != "mm":
            raise ValueError(
                "Solver must be 'mm' if divergence is 'l2'."
                " Got solver = '{}'".format(solver)
            )

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
            ws = _make_tensor(source_weights, device=device)

        if target_weights is None:
            wt = (
                torch.ones(target_features.shape[1], device=device)
                / target_features.shape[1]
            )
        else:
            wt = _make_tensor(target_weights, device=device)

        # If initial plan is provided, move it to device.
        # Convert it to sparse CSR if it's not already.
        pi_init = (
            _make_tensor(init_plan.to_sparse_csr(), device=device)
            if init_plan is not None
            else None
        )

        # Compute distance matrix between features
        Fs = _make_tensor(source_features.T, device=device)
        Ft = _make_tensor(target_features.T, device=device)
        F1, F2 = _low_rank_squared_l2(Fs, Ft)
        F1 = _make_tensor(F1, device=device)
        F2 = _make_tensor(F2, device=device)

        # Load anatomical kernels to GPU
        Ds1, Ds2 = _low_rank_squared_l2(
            source_geometry_embedding, source_geometry_embedding
        )
        Ds1 = _make_tensor(Ds1, device=device)
        Ds2 = _make_tensor(Ds2, device=device)
        Dt1, Dt2 = _low_rank_squared_l2(
            target_geometry_embedding, target_geometry_embedding
        )
        Dt1 = _make_tensor(Dt1, device=device)
        Dt2 = _make_tensor(Dt2, device=device)

        # Do the same for validation data if it was provided
        if source_features_val is not None and target_features_val is not None:
            Fs_val = _make_tensor(source_features_val.T, device=device)
            Fs_val = _make_tensor(source_features.T, device=device)
            Ft_val = _make_tensor(target_features.T, device=device)
            F1_val, F2_val = _low_rank_squared_l2(Fs_val, Ft_val)
            F1_val = _make_tensor(F1_val, device=device)
            F2_val = _make_tensor(F2_val, device=device)

        elif source_features_val is not None and target_features_val is None:
            raise ValueError(
                "Source features validation data provided but not target"
                " features validation data."
            )

        elif source_features_val is None and target_features_val is not None:
            raise ValueError(
                "Target features validation data provided but not source"
                " features validation data."
            )

        else:
            F1_val, F2_val = None, None

            # Raise warning if validation feature maps are not provided
            if verbose:
                console.log(
                    "Validation data for feature maps is not provided."
                    " Using training data instead."
                )

        if (
            source_geometry_embedding_val is not None
            and target_geometry_embedding_val is not None
        ):
            Ds1_val, Ds2_val = _low_rank_squared_l2(
                source_geometry_embedding_val, source_geometry_embedding_val
            )
            Ds1_val = _make_tensor(Ds1, device=device)
            Ds2_val = _make_tensor(Ds2, device=device)
            Dt1_val, Dt2_val = _low_rank_squared_l2(
                target_geometry_embedding, target_geometry_embedding
            )
            Dt1_val = _make_tensor(Dt1_val, device=device)
            Dt2_val = _make_tensor(Dt2_val, device=device)

        elif (
            source_geometry_embedding_val is not None
            and target_geometry_embedding_val is None
        ):
            raise ValueError(
                "Source geometry validation data provided but not target"
                " geometry validation data."
            )

        elif (
            source_geometry_embedding_val is None
            and target_geometry_embedding_val is not None
        ):
            raise ValueError(
                "Target geometry validation data provided but not source"
                " geometry validation data."
            )

        else:
            Ds1_val, Ds2_val = Ds1, Ds2
            Dt1_val, Dt2_val = Dt1, Dt2

            # Raise warning if validation anatomical kernels are not provided
            if verbose:
                console.log(
                    "Validation data for anatomical kernels is not provided."
                    " Using training data instead."
                )

        # Check that all init_plan is valid
        if init_plan is None:
            warnings.warn(
                "Warning: init_plan is None, so this solver "
                "will compute a dense solution. However, "
                "dense solutions are much more efficiently computed "
                "by fugw.FUGW"
            )
        else:
            init_plan = _make_sparse_csr_tensor(init_plan, device=device)

        # Create model
        model = FUGWSparseSolver(**solver_params)

        # Compute transport plan
        res = model.solve(
            alpha=self.alpha,
            rho_s=rho_s,
            rho_t=rho_t,
            eps=self.eps,
            reg_mode=self.reg_mode,
            divergence=self.divergence,
            F=(F1, F2),
            Ds=(Ds1, Ds2),
            Dt=(Dt1, Dt2),
            F_val=(F1_val, F2_val),
            Ds_val=(Ds1_val, Ds2_val),
            Dt_val=(Dt1_val, Dt2_val),
            ws=ws,
            wt=wt,
            init_plan=pi_init,
            init_duals=init_duals,
            solver=solver,
            callback_bcd=callback_bcd,
            verbose=verbose,
        )

        self.pi = res["pi"].to_sparse_coo().detach().cpu()
        self.loss = res["loss"]
        self.loss_val = res["loss_val"]
        self.loss_steps = res["loss_steps"]
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
        source_features_tensor = _make_tensor(source_features, device=device)

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
        target_features_tensor = _make_tensor(target_features, device=device)

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
