import numpy as np
import torch

from fugw.solvers.dense import FUGWSolver
from fugw.mappings.utils import BaseMapping
from fugw.utils import make_tensor


class FUGW(BaseMapping):
    """Class computing dense transport plans"""

    def fit(
        self,
        source_features=None,
        target_features=None,
        source_geometry=None,
        target_geometry=None,
        source_weights=None,
        target_weights=None,
        init_plan=None,
        init_duals=None,
        uot_solver="sinkhorn",
        device="auto",
        verbose=False,
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
        init_duals: tuple of [ndarray(n), ndarray(m)] or None
            Dual potentials to use at initialisation.
        uot_solver: "sinkhorn" or "mm" or "ibpp"
            Solver to use.
        device: "auto" or torch.device
            if "auto": use first available gpu if it's available,
            cpu otherwise.
        verbose: bool, optional, defaults to False
            Log solving process.

        Returns
        -------
        self: FUGW class object
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
        F = torch.cdist(Fs, Ft, p=2) ** 2

        # Load anatomical kernels to GPU
        Ds = make_tensor(source_geometry, device=device)
        Dt = make_tensor(target_geometry, device=device)

        # Create model
        model = FUGWSolver(**kwargs)

        # Compute transport plan
        res = model.solve(
            alpha=self.alpha,
            rho_s=rho_s,
            rho_t=rho_t,
            eps=self.eps,
            reg_mode=self.reg_mode,
            F=F,
            Ds=Ds,
            Dt=Dt,
            ws=ws,
            wt=wt,
            init_plan=init_plan,
            init_duals=init_duals,
            uot_solver=uot_solver,
            verbose=verbose,
        )

        # Store variables of interest in model
        self.pi = res["pi"].detach().cpu()
        self.loss_steps = res["loss_steps"]
        self.loss = res["loss"]
        self.loss_entropic = res["loss_entropic"]
        self.loss_times = res["loss_times"]

        # Free allocated GPU memory
        del Fs, Ft, F, Ds, Dt
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
        pi = make_tensor(self.pi, device=device)
        source_features_tensor = make_tensor(source_features, device=device)

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
        pi = make_tensor(self.pi, device=device)
        target_features_tensor = make_tensor(target_features, device=device)

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
