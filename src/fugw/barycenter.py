import torch

from fugw import FUGW
from fugw.utils import make_tensor


class FUGWBarycenter:
    def __init__(
        self,
        alpha=0.5,
        rho=1,
        eps=1e-2,
        uot_solver="sinkhorn",
        reg_mode="joint",
        early_stopping_threshold=1e-6,
        nits_barycenter=5,
        force_psd=False,
        learn_geometry=False,
        verbose=False,
        **kwargs,
    ):
        # Save model arguments
        self.alpha = alpha
        self.rho = rho
        self.eps = eps
        self.uot_solver = uot_solver
        self.reg_mode = reg_mode
        self.early_stopping_threshold = early_stopping_threshold
        self.nits_barycenter = nits_barycenter
        self.force_psd = force_psd
        self.learn_geometry = learn_geometry
        self.verbose = verbose

    @staticmethod
    def update_barycenter_geometry(
        plans_, weights_, geometry_, force_psd=False
    ):
        # Check cuda availability
        use_cuda = torch.cuda.is_available()
        dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

        barycenter_geometry = 0
        # pi_samp, pi_feat: both of size (ns, n)
        for i, (plans, weights) in enumerate(zip(plans_, weights_)):
            if len(geometry_) == 1 and len(weights_) > 1:
                C = make_tensor(geometry_[0]).type(dtype)
            else:
                C = make_tensor(geometry_[i]).type(dtype)

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

            w = make_tensor(weights)
            barycenter_geometry = (
                barycenter_geometry + w * term
            )  # shape (n, n)

        return barycenter_geometry

    @staticmethod
    def update_barycenter_features(plans_, weights_, features_):
        barycenter_features = 0
        for i, ((pi_samp, pi_feat), weights, features) in enumerate(
            zip(plans_, weights_, features_)
        ):
            w = make_tensor(weights)
            f = make_tensor(features)
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
    ):
        new_plans_ = []
        new_duals_ = []
        new_costs_ = []

        for i, (features, weights) in enumerate(zip(features_, weights_)):
            if len(geometry_) == 1 and len(weights_) > 1:
                G = geometry_[0]
            else:
                G = geometry_[i]

            fugw = FUGW(
                alpha=self.alpha,
                rho=self.rho,
                eps=self.eps,
                uot_solver=self.uot_solver,
                reg_mode=self.reg_mode,
                verbose=self.verbose,
                early_stopping_threshold=self.early_stopping_threshold,
            )

            pi, gamma, dual_pi, dual_gamma, loss, loss_ent = fugw.fit(
                features,
                barycenter_features,
                source_geometry=G,
                target_geometry=barycenter_geometry,
                source_weights=weights,
                target_weights=barycenter_weights,
                # TODO: check if 2 plans couldn't be used instead of just one
                # in following line
                init_plan=plans_[i][0] if plans_ is not None else None,
                init_duals=duals_[i][0] if duals_ is not None else None,
                return_plans_only=False,
            )

            new_plans_.append((pi, gamma))
            new_duals_.append((dual_pi, dual_gamma))
            new_costs_.append((loss[-1], loss_ent[-1]))

        return new_plans_, new_duals_, new_costs_

    def fit(
        self,
        weights_,
        features_,
        geometry_,
        barycenter_size=None,
        init_barycenter_weights=None,
        init_barycenter_features=None,
        init_barycenter_geometry=None,
        return_barycenter_only=True,
    ):
        # Check cuda availability
        use_cuda = torch.cuda.is_available()
        dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

        log_cost = []

        if barycenter_size is None:
            barycenter_size = weights_[0].shape[0]

        # Initialize barycenter weights, features and geometry
        if init_barycenter_weights is None:
            barycenter_weights = (
                torch.ones(barycenter_size).type(dtype) / barycenter_size
            )
        else:
            barycenter_weights = make_tensor(init_barycenter_weights).type(
                dtype
            )

        if init_barycenter_features is None:
            barycenter_features = torch.ones(
                (features_[0].shape[0], barycenter_size)
            ).type(dtype)
        else:
            barycenter_features = make_tensor(init_barycenter_features).type(
                dtype
            )

        if init_barycenter_geometry is None:
            barycenter_geometry = torch.ones(
                (barycenter_size, barycenter_size)
            ).type(dtype)
        else:
            barycenter_geometry = make_tensor(init_barycenter_geometry).type(
                dtype
            )

        plans_ = None
        duals_ = None
        # TODO: store and return cost
        # costs_ = None

        for step in range(self.nits_barycenter):
            # Transport all elements
            plans_, duals_, costs_ = self.compute_all_ot_plans(
                plans_,
                duals_,
                weights_,
                features_,
                geometry_,
                barycenter_weights,
                barycenter_features,
                barycenter_geometry,
            )

            # Update barycenter features and geometry
            barycenter_features = self.update_barycenter_features(
                plans_, weights_, features_
            )
            if self.learn_geometry:
                barycenter_geometry = self.update_barycenter_geometry(
                    plans_, weights_, geometry_, self.force_psd
                )

            # Compute cost
            # print(costs_)
            # cost = sum(costs[0] * w for (costs, w) in zip(costs_, weights_))
            # print(cost)

            # if self.verbose:
            #     print(f"Cost at iteration {step+1} = {cost}")

            # log_cost.append(cost)
            # print(log_cost)
            # if abs(log_cost[-2] - log_cost[-1]) < self.tol_bary:
            #     break

        if return_barycenter_only:
            return barycenter_features, barycenter_geometry
        else:
            return (barycenter_features, barycenter_geometry, plans_, log_cost)
