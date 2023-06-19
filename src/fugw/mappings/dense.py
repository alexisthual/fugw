import numpy as np
import torch

from fugw.solvers.dense import FUGWSolver
from fugw.mappings.utils import BaseMapping, console
from fugw.utils import _make_tensor, init_plan_dense


class FUGW(BaseMapping):
    """Class computing dense transport plans."""

    def fit(
        self,
        source_features=None,
        target_features=None,
        source_geometry=None,
        target_geometry=None,
        source_features_val=None,
        target_features_val=None,
        source_geometry_val=None,
        target_geometry_val=None,
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
        source_geometry: ndarray(n, n)
            Kernel matrix of anatomical distances
            between nodes of source mesh
            **This array should be normalized**, otherwise you will
            run into computational errors.
        target_geometry: ndarray(m, m)
            Kernel matrix of anatomical distances
            between nodes of target mesh
            **This array should be normalized**, otherwise you will
            run into computational errors.
        source_features_val: ndarray(n_features, n) or None
            Feature maps for source subject used for validation.
            If None, source_features will be used instead.
        target_features_val: ndarray(n_features, m) or None
            Feature maps for target subject used for validation.
            If None, target_features will be used instead.
        source_geometry_val: ndarray(n, n) or None
            Kernel matrix of anatomical distances
            between nodes of source mesh used for validation.
            If None, source_geometry will be used instead.
        target_geometry_val: ndarray(m, m) or None
            Kernel matrix of anatomical distances
            between nodes of target mesh used for validation.
            If None, target_geometry will be used instead.
        source_weights: ndarray(n) or None
            Distribution weights of source nodes.
            Should sum to 1. If None, eahc node's weight
            will be set to 1 / n.
        target_weights: ndarray(n) or None
            Distribution weights of target nodes.
            Should sum to 1. If None, eahc node's weight
            will be set to 1 / m.
        init_plan: ndarray(n, m) or None
            Transport plan to use at initialisation.
            If None, an entropic initialization will be used.
        init_duals: tuple of [ndarray(n), ndarray(m)] or None
            Dual potentials to use at initialisation.
        solver: "sinkhorn" or "mm" or "ibpp"
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
        self: FUGW class object
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
        # Otherwise, initialize it with entropic initialization
        pi_init = (
            _make_tensor(init_plan, device=device)
            if init_plan is not None
            else _make_tensor(
                init_plan_dense(
                    source_features.shape[1],
                    target_features.shape[1],
                    weights_source=ws,
                    weights_target=wt,
                    method="entropic",
                ),
                device=device,
            )
        )

        # Compute distance matrix between features
        Fs = _make_tensor(source_features.T, device=device)
        Ft = _make_tensor(target_features.T, device=device)
        F = torch.cdist(Fs, Ft, p=2) ** 2

        # Load anatomical kernels to GPU
        Ds = _make_tensor(source_geometry, device=device)
        Dt = _make_tensor(target_geometry, device=device)

        # Do the same for validation data if it was provided
        if source_features_val is not None and target_features_val is not None:
            Fs_val = _make_tensor(source_features_val.T, device=device)
            Ft_val = _make_tensor(target_features_val.T, device=device)
            F_val = torch.cdist(Fs_val, Ft_val, p=2) ** 2

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
            F_val = None

            # Raise warning if validation feature maps are not provided
            if verbose:
                console.log(
                    "Validation data for feature maps is not provided."
                    " Using training data instead."
                )

        if source_geometry_val is not None and target_geometry_val is not None:
            Ds_val = _make_tensor(source_geometry_val, device=device)
            Dt_val = _make_tensor(target_geometry_val, device=device)

        elif source_geometry_val is not None and target_geometry_val is None:
            raise ValueError(
                "Source geometry validation data provided but not target"
                " geometry validation data."
            )

        elif source_geometry_val is None and target_geometry_val is not None:
            raise ValueError(
                "Target geometry validation data provided but not source"
                " geometry validation data."
            )

        else:
            Ds_val = None
            Dt_val = None

            # Raise warning if validation anatomical kernelsare not provided
            if verbose:
                console.log(
                    "Validation data for anatomical kernels is not provided."
                    " Using training data instead."
                )

        # Create model
        model = FUGWSolver(**solver_params)

        # Compute transport plan
        res = model.solve(
            alpha=self.alpha,
            rho_s=rho_s,
            rho_t=rho_t,
            eps=self.eps,
            reg_mode=self.reg_mode,
            divergence=self.divergence,
            F=F,
            Ds=Ds,
            Dt=Dt,
            F_val=F_val,
            Ds_val=Ds_val,
            Dt_val=Dt_val,
            ws=ws,
            wt=wt,
            init_plan=pi_init,
            init_duals=init_duals,
            solver=solver,
            callback_bcd=callback_bcd,
            verbose=verbose,
        )

        # Store variables of interest in model
        self.pi = res["pi"].detach().cpu()
        self.loss = res["loss"]
        self.loss_steps = res["loss_steps"]
        self.loss_times = res["loss_times"]
        self.loss_val = res["loss_val"]

        # Free allocated GPU memory
        del Fs, Ft, F, Ds, Dt
        if source_features_val is not None:
            del Fs_val, Ft_val, F_val, Ds_val, Dt_val
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

        is_one_dimensional = False
        if source_features.ndim == 1:
            is_one_dimensional = True
            source_features = source_features.reshape(1, -1)
        if source_features.ndim > 2:
            raise ValueError(
                "source_features has too many dimensions:"
                f" {source_features.ndim}"
            )

        # Move data to device if need be
        pi = _make_tensor(self.pi, device=device)
        source_features_tensor = _make_tensor(source_features, device=device)

        # Transform data
        transformed_data = (
            (pi.T @ source_features_tensor.T / pi.sum(dim=0).reshape(-1, 1))
            .T.detach()
            .cpu()
        )

        # Free allocated GPU memory
        del pi, source_features_tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Modify returned tensor so that it matches
        # source_features's shape and python type
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
            Contrast map transported in source subject's space
        """
        if device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda", 0)
            else:
                device = torch.device("cpu")

        if self.pi is None:
            raise Exception("Model should be fitted before calling transform")

        is_one_dimensional = False
        if target_features.ndim == 1:
            is_one_dimensional = True
            target_features = target_features.reshape(1, -1)
        if target_features.ndim > 2:
            raise ValueError(
                "target_features has too many dimensions:"
                f" {target_features.ndim}"
            )

        # Move data to device if need be
        pi = _make_tensor(self.pi, device=device)
        target_features_tensor = _make_tensor(target_features, device=device)

        # Transform data
        transformed_data = (
            (pi @ target_features_tensor.T / pi.sum(dim=1).reshape(-1, 1))
            .T.detach()
            .cpu()
        )

        # Free allocated GPU memory
        del pi, target_features_tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Modify returned tensor so that it matches
        # target_features's  shape and python type
        if isinstance(target_features, np.ndarray):
            transformed_data = transformed_data.numpy()

        if transformed_data.ndim > 1 and is_one_dimensional:
            transformed_data = transformed_data.flatten()

        return transformed_data
