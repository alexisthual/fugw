import torch

from fugw.mappings.dense import FUGW
from fugw.utils import _make_tensor, console


class FUGWBarycenter:
    """FUGW barycenters"""

    def __init__(
        self,
        alpha=0.5,
        rho=1,
        eps=1e-2,
        reg_mode="joint",
        force_psd=False,
        learn_geometry=False,
    ):
        # Save model arguments
        self.alpha = alpha
        self.rho = rho
        self.eps = eps
        self.reg_mode = reg_mode
        self.force_psd = force_psd
        self.learn_geometry = learn_geometry

    @staticmethod
    def update_barycenter_geometry(
        plans_, weights_, geometry_, force_psd, device
    ):
        barycenter_geometry = 0
        # pi_samp, pi_feat: both of size (ns, n)
        for i, (plans, weights) in enumerate(zip(plans_, weights_)):
            if len(geometry_) == 1 and len(weights_) > 1:
                C = _make_tensor(geometry_[0], device=device)
            else:
                C = _make_tensor(geometry_[i], device=device)

            pi_samp, pi_feat = plans
            pi1_samp, pi1_feat = pi_samp.sum(0), pi_feat.sum(0)

            if force_psd:
                if isinstance(C, tuple):
                    C1, C2 = C
                    term = pi_samp.T @ (C1 @ (C2.T @ pi_samp)) / (
                        pi1_samp[:, None] * pi1_samp[None, :]
                    ) + pi_feat.T @ (C1 @ (C2.T @ pi_feat)) / (
                        pi1_feat[:, None] * pi1_feat[None, :]
                    )
                elif torch.is_tensor(C):
                    term = pi_samp.T @ C @ pi_samp / (
                        pi1_samp[:, None] * pi1_samp[None, :]
                    ) + pi_feat.T @ C @ pi_feat / (
                        pi1_feat[:, None] * pi1_feat[None, :]
                    )
                term = term / 2

            else:
                if isinstance(C, tuple):
                    C1, C2 = C
                    term = (
                        pi_samp.T
                        @ (C1 @ (C2.T @ pi_feat))
                        / (pi1_samp[:, None] * pi1_feat[None, :])
                    )  # shape (n, n)
                elif torch.is_tensor(C):
                    term = (
                        pi_samp.T
                        @ C
                        @ pi_feat
                        / (pi1_samp[:, None] * pi1_feat[None, :])
                    )  # shape (n, n)

            w = _make_tensor(weights, device=device)
            barycenter_geometry = (
                barycenter_geometry + w * term
            )  # shape (n, n)

        return barycenter_geometry

    @staticmethod
    def update_barycenter_features(plans, weights_list, features_list, device):
        for i, (pi, weights, features) in enumerate(
            zip(plans, weights_list, features_list)
        ):
            w = _make_tensor(weights, device=device)
            f = _make_tensor(features, device=device)
            if features is not None:
                acc = w * pi.T @ f.T / (pi.sum(0).reshape(-1, 1) + 1e-16)

                if i == 0:
                    barycenter_features = acc
                else:
                    barycenter_features += acc

        # Normalize barycenter features
        min_val = barycenter_features.min(dim=0, keepdim=True).values
        max_val = barycenter_features.max(dim=0, keepdim=True).values
        barycenter_features = (
            2 * (barycenter_features - min_val) / (max_val - min_val) - 1
        )
        return barycenter_features.T

    @staticmethod
    def get_dim(C):
        if isinstance(C, tuple):
            return C[0].shape[0]
        elif torch.is_tensor(C):
            return C.shape[0]

    @staticmethod
    def get_device_dtype(C):
        if isinstance(C, tuple):
            return C[0].device, C[0].dtype
        elif torch.is_tensor(C):
            return C.device, C.dtype

    def compute_all_ot_plans(
        self,
        plans,
        duals,
        weights_list,
        features_list,
        geometry_list,
        barycenter_weights,
        barycenter_features,
        barycenter_geometry,
        solver,
        solver_params,
        callback_barycenter,
        device,
        verbose,
    ):
        new_plans = []
        new_losses = []

        for i, (features, weights) in enumerate(
            zip(features_list, weights_list)
        ):
            if verbose:
                console.log(f"Updating mapping {i + 1} / {len(weights_list)}")

            if len(geometry_list) == 1 and len(weights_list) > 1:
                G = geometry_list[0]
            else:
                G = geometry_list[i]

            mapping = FUGW(
                alpha=self.alpha,
                rho=self.rho,
                eps=self.eps,
                reg_mode=self.reg_mode,
            )

            mapping.fit(
                source_features=features,
                target_features=barycenter_features,
                source_geometry=G,
                target_geometry=barycenter_geometry,
                source_weights=weights,
                target_weights=barycenter_weights,
                init_plan=plans[i] if plans is not None else None,
                init_duals=duals[i] if duals is not None else None,
                solver=solver,
                solver_params=solver_params,
                device=device,
                verbose=verbose,
            )

            new_plans.append(mapping.pi)
            new_losses.append(
                (
                    mapping.loss,
                    mapping.loss_steps,
                    mapping.loss_times,
                )
            )

        return new_plans, new_losses

    def fit(
        self,
        weights_list,
        features_list,
        geometry_list,
        barycenter_size=None,
        init_barycenter_weights=None,
        init_barycenter_features=None,
        init_barycenter_geometry=None,
        solver="sinkhorn",
        solver_params={},
        nits_barycenter=5,
        device="auto",
        callback_barycenter=None,
        verbose=False,
    ):
        """Compute barycentric features and geometry
        minimizing FUGW loss to list of distributions given as input.
        In this documentation, we refer to a single distribution as
        an a subject's or an individual's distribution.

        Parameters
        ----------
        weights_list (list of np.array): List of weights. Different individuals
            can have weights with different sizes.
        features_list (list of np.array): List of features. Individuals should
            have the same number of features n_features.
        geometry_list (list of np.array or np.array): List of kernel matrices
            or just one kernel matrix if it's shared across individuals
            barycenter_size (int, optional): Size of computed
            barycentric features and geometry. Defaults to None.
        init_barycenter_weights (np.array, optional): Distribution weights
            of barycentric points. If None, points will have uniform
            weights. Defaults to None.
        init_barycenter_features (np.array, optional): np.array of size
            (barycenter_size, n_features). Defaults to None.
        init_barycenter_geometry (np.array, optional): np.array of size
            (barycenter_size, barycenter_size). Defaults to None.
        device: "auto" or torch.device
            if "auto": use first available gpu if it's available,
            cpu otherwise.
        callback_barycenter: callable or None
            Callback function called at the end of each barycenter step.
            It will be called with the following arguments:

                - locals (dictionary containing all local variables)

        Returns
        -------
        barycenter_weights: np.array of size (barycenter_size)
        barycenter_features: np.array of size (barycenter_size, n_features)
        barycenter_geometry: np.array of size
            (barycenter_size, barycenter_size)
        plans: list of arrays
        duals: list of (array, array)
        losses_each_bar_step: list such that l[s][i]
            is a tuple containing:
                - loss
                - loss_steps
                - loss_times
            for individual i at barycenter computation step s
        """
        if device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda", 0)
            else:
                device = torch.device("cpu")

        if barycenter_size is None:
            barycenter_size = weights_list[0].shape[0]

        # Initialize barycenter weights, features and geometry
        if init_barycenter_weights is None:
            barycenter_weights = (
                torch.ones(barycenter_size) / barycenter_size
            ).to(device)
        else:
            barycenter_weights = _make_tensor(
                init_barycenter_weights, device=device
            )

        if init_barycenter_features is None:
            barycenter_features = torch.ones(
                (features_list[0].shape[0], barycenter_size)
            ).to(device)
            barycenter_features = barycenter_features / torch.norm(
                barycenter_features, dim=1
            ).reshape(-1, 1)
        else:
            barycenter_features = _make_tensor(
                init_barycenter_features, device=device
            )

        if init_barycenter_geometry is None and self.learn_geometry is False:
            raise ValueError(
                "In the fixed support case, init_barycenter_geometry must be"
                " provided."
            )
        elif init_barycenter_geometry is None and self.learn_geometry is True:
            barycenter_geometry = (
                torch.ones((barycenter_size, barycenter_size)).to(device)
                / barycenter_size
            )
        else:
            barycenter_geometry = _make_tensor(
                init_barycenter_geometry, device=device
            )

        plans = None
        duals = None
        losses_each_bar_step = []

        for idx in range(nits_barycenter):
            if verbose:
                console.log(
                    f"Barycenter iterations {idx + 1} / {nits_barycenter}"
                )
            # Transport all elements
            plans, losses = self.compute_all_ot_plans(
                plans,
                duals,
                weights_list,
                features_list,
                geometry_list,
                barycenter_weights,
                barycenter_features,
                barycenter_geometry,
                solver,
                solver_params,
                callback_barycenter,
                device,
                verbose,
            )

            losses_each_bar_step.append(losses)

            # Update barycenter features and geometry
            barycenter_features = self.update_barycenter_features(
                plans, weights_list, features_list, device
            )
            if self.learn_geometry:
                barycenter_geometry = self.update_barycenter_geometry(
                    plans, weights_list, geometry_list, self.force_psd, device
                )

            if callback_barycenter is not None:
                callback_barycenter(locals())

        return (
            barycenter_weights,
            barycenter_features,
            barycenter_geometry,
            plans,
            duals,
            losses_each_bar_step,
        )
