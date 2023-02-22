import torch

from fugw.transformers.dense import FUGW
from fugw.transformers.utils import make_tensor


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
    def update_barycenter_geometry(plans_, weights_, geometry_, force_psd, device):
        barycenter_geometry = 0
        # pi_samp, pi_feat: both of size (ns, n)
        for i, (plans, weights) in enumerate(zip(plans_, weights_)):
            if len(geometry_) == 1 and len(weights_) > 1:
                C = make_tensor(geometry_[0], device=device)
            else:
                C = make_tensor(geometry_[i], device=device)

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

            w = make_tensor(weights, device=device)
            barycenter_geometry = barycenter_geometry + w * term  # shape (n, n)

        return barycenter_geometry

    @staticmethod
    def update_barycenter_features(plans_, weights_, features_, device):
        barycenter_features = 0
        for i, ((pi_samp, pi_feat), weights, features) in enumerate(
            zip(plans_, weights_, features_)
        ):
            w = make_tensor(weights, device=device)
            f = make_tensor(features, device=device)
            pi_sum = pi_samp + pi_feat
            if features is not None:
                acc = w * pi_sum.T @ f.T / pi_sum.sum(0).reshape(-1, 1)

                if i == 0:
                    barycenter_features = acc
                else:
                    barycenter_features += acc

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
        plans_,
        duals_,
        weights_,
        features_,
        geometry_,
        barycenter_weights,
        barycenter_features,
        barycenter_geometry,
        uot_solver,
        early_stopping_threshold,
        device,
        verbose,
    ):
        new_plans_ = []
        new_duals_ = []
        new_losses_ = []

        for i, (features, weights) in enumerate(zip(features_, weights_)):
            if len(geometry_) == 1 and len(weights_) > 1:
                G = geometry_[0]
            else:
                G = geometry_[i]

            fugw = FUGW(
                alpha=self.alpha,
                rho=self.rho,
                eps=self.eps,
                reg_mode=self.reg_mode,
            )

            (
                pi,
                gamma,
                dual_pi,
                dual_gamma,
                loss_steps,
                loss,
                loss_ent,
            ) = fugw.fit(
                features,
                barycenter_features,
                source_geometry=G,
                target_geometry=barycenter_geometry,
                source_weights=weights,
                target_weights=barycenter_weights,
                uot_solver=uot_solver,
                # TODO: check if 2 plans couldn't be used instead of just one
                init_plan=plans_[i][0] if plans_ is not None else None,
                init_duals=duals_[i][0] if duals_ is not None else None,
                early_stopping_threshold=early_stopping_threshold,
                device=device,
                verbose=verbose,
            )

            new_plans_.append((pi, gamma))
            new_duals_.append((dual_pi, dual_gamma))
            new_losses_.append((loss_steps, loss, loss_ent))

        return new_plans_, new_duals_, new_losses_

    def fit(
        self,
        weights_,
        features_,
        geometry_,
        barycenter_size=None,
        init_barycenter_weights=None,
        init_barycenter_features=None,
        init_barycenter_geometry=None,
        uot_solver="sinkhorn",
        nits_barycenter=5,
        early_stopping_threshold=1e-6,
        device="auto",
        verbose=False,
    ):
        """Compute barycentric features and geometry
        minimizing FUGW loss to list of distributions given as input.
        In this documentation, we refer to a single distribution as
        an a subject's or an individual's distribution.

        Parameters
        ----------
        weights_ (list of np.array): List of weights. Different individuals
            can have weights with different sizes.
        features_ (list of np.array): List of features. Individuals should
            have the same number of features n_features.
        geometry_ (list of np.array or np.array): List of kernel matrices
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

        Returns
        -------
        barycenter_weights: np.array of size (barycenter_size)
        barycenter_features: np.array of size (barycenter_size, n_features)
        barycenter_geometry: np.array of size
            (barycenter_size, barycenter_size)
        plans_: list of (np.array, np.array)
        duals_: list of (np.array, np.array)
        losses_each_bar_step: list such that l[s][i]
            is a tuple containing (loss_steps, loss, loss_ent)
            for individual i at barycenter computation step s
        """
        if device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda", 0)
            else:
                device = torch.device("cpu")

        if barycenter_size is None:
            barycenter_size = weights_[0].shape[0]

        # Initialize barycenter weights, features and geometry
        if init_barycenter_weights is None:
            barycenter_weights = (torch.ones(barycenter_size) / barycenter_size).to(
                device
            )
        else:
            barycenter_weights = make_tensor(init_barycenter_weights, device=device)

        if init_barycenter_features is None:
            barycenter_features = torch.ones(
                (features_[0].shape[0], barycenter_size)
            ).to(device)
            barycenter_features = barycenter_features / torch.norm(
                barycenter_features, dim=1
            ).reshape(-1, 1)
        else:
            barycenter_features = make_tensor(init_barycenter_features, device=device)

        if init_barycenter_geometry is None:
            barycenter_geometry = (
                torch.ones((barycenter_size, barycenter_size)).to(device)
                / barycenter_size
            )
        else:
            barycenter_geometry = make_tensor(init_barycenter_geometry, device=device)

        plans_ = None
        duals_ = None
        losses_each_bar_step = []

        for _ in range(nits_barycenter):
            # Transport all elements
            plans_, duals_, losses_ = self.compute_all_ot_plans(
                plans_,
                duals_,
                weights_,
                features_,
                geometry_,
                barycenter_weights,
                barycenter_features,
                barycenter_geometry,
                uot_solver,
                early_stopping_threshold,
                device,
                verbose,
            )

            losses_each_bar_step.append(losses_)

            # Update barycenter features and geometry
            barycenter_features = self.update_barycenter_features(
                plans_, weights_, features_, device
            )
            if self.learn_geometry:
                barycenter_geometry = self.update_barycenter_geometry(
                    plans_, weights_, geometry_, self.force_psd, device
                )

        return (
            barycenter_weights,
            barycenter_features,
            barycenter_geometry,
            plans_,
            duals_,
            losses_each_bar_step,
        )
