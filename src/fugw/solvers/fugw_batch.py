import torch
import numpy as np
from tqdm import tqdm
from functools import partial


class BatchFUGWSolver:
    def __init__(
        self, nits=100, nits_sinkhorn=100, tol=1e-7, tol_sinkhorn=1e-7
    ):
        self.nits = nits
        self.nits_sinkhorn = nits_sinkhorn
        self.tol = tol
        self.tol_sinkhorn = tol_sinkhorn

    @staticmethod
    def get_factored_distance(X, Y):
        """
        Write square Euclidean distance matrix as product of two low rank matrices.
        """
        device, dtype = X.device, X.dtype
        nx, ny = X.shape[0], Y.shape[0]

        Vx = (X ** 2).sum(1, keepdim=True)  # shape nx x 1
        Vy = (Y ** 2).sum(1, keepdim=True)  # shape ny x 1
        ones_x = torch.ones(nx, 1).to(device).to(dtype)  # shape nx x 1
        ones_y = torch.ones(ny, 1).to(device).to(dtype)  # shape ny x 1

        D1 = torch.cat(
            [Vx, ones_x, -(2 ** 0.5) * X], dim=1
        )  # shape nx x (d+2)
        D2 = torch.cat([ones_y, Vy, 2 ** 0.5 * Y], dim=1)  # shape ny x (d+2)

        return (D1, D2)

    @staticmethod
    def get_tensor_cost(
        local_ugw_cost, data_const, alpha, position, idx, batch_size
    ):
        Cx_vec, Cy_vec, bary_centre, scalar = local_ugw_cost
        A1, _, B1, _, D1, D2 = data_const

        if position == 1:  # shape batchsize1 x ny
            if alpha == 0:
                cost = 0.5 * D1[idx : idx + batch_size, :] @ D2.T
            elif alpha == 1:
                cost = (
                    Cx_vec[idx : idx + batch_size, None]
                    + Cy_vec[None, :]
                    - 2 * A1[idx : idx + batch_size, :] @ bary_centre @ B1.T
                )
            else:
                cost = (
                    alpha
                    * (
                        Cx_vec[idx : idx + batch_size, None]
                        + Cy_vec[None, :]
                        - 2
                        * A1[idx : idx + batch_size, :]
                        @ bary_centre
                        @ B1.T
                    )
                    + (1 - alpha) / 2 * D1[idx : idx + batch_size, :] @ D2.T
                )

        elif position == 2:  # shape nx x batchsize2
            if alpha == 0:
                cost = 0.5 * D1 @ D2[idx : idx + batch_size, :].T
            elif alpha == 1:
                cost = (
                    Cx_vec[:, None]
                    + Cy_vec[None, idx : idx + batch_size]
                    - 2 * A1 @ bary_centre @ B1[idx : idx + batch_size, :].T
                )
            else:
                cost = (
                    alpha
                    * (
                        Cx_vec[:, None]
                        + Cy_vec[None, idx : idx + batch_size]
                        - 2
                        * A1
                        @ bary_centre
                        @ B1[idx : idx + batch_size, :].T
                    )
                    + (1 - alpha) / 2 * D1 @ D2[idx : idx + batch_size, :].T
                )

        cost += scalar

        return cost

    def get_kl_of_vector(self, p, log_q, mode_kl):
        """
        Calculate the KL between two vectors.
        """
        kl = torch.nan_to_num(
            p * (p.log() - log_q), nan=0.0, posinf=0.0, neginf=0.0
        ).sum()
        if mode_kl == "exact":
            kl += log_q.exp().sum() - p.sum()
        return kl

    def get_kl_of_coupling(
        self, dict_plan, eps, log_p, data_const, alpha, mode_kl
    ):
        """
        Calculate the KL between two matrices.
        """
        marg_1, marg_2 = dict_plan["marginals"]
        u, v = dict_plan["new_duals"]
        local_ugw_cost = dict_plan["local_ugw_cost"]

        log_a, log_b = log_p
        bs1, r1 = self.sample_batch[0]

        s1 = u + log_a
        s2 = v + log_b
        kl = -(marg_1 * log_a).sum() - (marg_2 * log_b).sum()
        for i in r1:
            cost = self.get_tensor_cost(
                local_ugw_cost, data_const, alpha, 1, i, bs1
            )  # shape bs1 x ny
            log_plan = s1[i : i + bs1, None] + s2[None, :] - cost / eps
            kl += (log_plan * log_plan.exp()).sum()  # entropy

        if mode_kl == "exact":
            mass_ab = log_a.exp().sum() * log_b.exp().sum()
            mass = marg_1.sum()
            kl += mass_ab - mass

        return kl

    def sinkhorn(self, dict_plan, rho1, rho2, eps, log_p, data_const, alpha):
        tau1 = (
            1
            if (torch.isnan(rho1) or torch.isinf(rho1))
            else rho1 / (rho1 + eps)
        )
        tau2 = (
            1
            if (torch.isnan(rho2) or torch.isinf(rho2))
            else rho2 / (rho2 + eps)
        )

        log_a, log_b = log_p
        u, v = dict_plan["duals"]
        local_ugw_cost = dict_plan["local_ugw_cost"]

        (bs1, r1), (bs2, r2) = self.sample_batch

        def get_lse(dual, pos, idx, bs):
            with torch.no_grad():
                cost = self.get_tensor_cost(
                    local_ugw_cost, data_const, alpha, pos, idx, bs
                )
                if pos == 2:
                    lse = torch.logsumexp(
                        (dual + log_a)[:, None] - cost / eps, dim=0
                    )
                elif pos == 1:
                    lse = torch.logsumexp(
                        (dual + log_b)[None, :] - cost / eps, dim=1
                    )

            return lse

        for _ in tqdm(range(self.nits_sinkhorn)):
            u_prev = u.clone()
            if rho2 == 0:  # semi-relaxed
                v = torch.zeros_like(v)
            else:
                v = -tau2 * torch.cat(tuple(get_lse(u, 2, j, bs2) for j in r2))
                # for j in r2:
                #     cost = self.get_tensor_cost(
                #         local_ugw_cost, data_const, alpha, 2, j, bs2
                #     )  # shape nx x bs2
                #     v[j: j + bs2] = -tau2 * torch.logsumexp(
                #         (u + log_a)[:, None] - cost / eps, dim=0
                #     )

            if rho1 == 0:  # semi-relaxed
                u = torch.zeros_like(u)
            else:
                u = -tau1 * torch.cat(tuple(get_lse(v, 1, i, bs1) for i in r1))

                # for i in r1:
                #     cost = self.get_tensor_cost(
                #         local_ugw_cost, data_const, alpha, 1, i, bs1
                #     )  # shape bs1 x ny
                #     u[i: i + bs1] = -tau1 * torch.logsumexp(
                #         (v + log_b)[None, :] - cost / eps, dim=1
                #     )

            if (u - u_prev).abs().max().item() < self.tol_sinkhorn:
                break

        s1 = u + log_a
        s2 = v + log_b

        marg_1 = torch.zeros_like(u)
        for i in r1:
            cost = self.get_tensor_cost(
                local_ugw_cost, data_const, alpha, 1, i, bs1
            )  # shape bs1 x ny
            marg_1[i : i + bs1] = (
                (s1[i : i + bs1, None] + s2[None, :] - cost / eps)
                .exp()
                .sum(dim=1)
            )

        marg_2 = torch.zeros_like(v)
        for j in r2:
            cost = self.get_tensor_cost(
                local_ugw_cost, data_const, alpha, 2, j, bs2
            )  # size nx x bs2
            marg_2[j : j + bs2] = (
                (s1[:, None] + s2[None, j : j + bs2] - cost / eps)
                .exp()
                .sum(dim=0)
            )

        dict_plan["duals"] = (u, v)
        dict_plan["marginals"] = (marg_1, marg_2)

        return dict_plan

    def _get_coupling(self, dict_plan, log_p, data_const, eps, alpha):

        u, v = dict_plan["new_duals"]
        local_ugw_cost = dict_plan["local_ugw_cost"]
        log_a, log_b = log_p
        bs1, r1 = self.sample_batch[0]

        device, dtype = u.device, u.dtype
        coupling = torch.zeros((u.shape[0], v.shape[0])).to(device).to(dtype)

        s1 = u + log_a
        s2 = v + log_b
        for i in r1:
            cost = self.get_tensor_cost(
                local_ugw_cost, data_const, alpha, 1, i, bs1
            )
            log_plan = s1[i : i + bs1, None] + s2[None, :] - cost / eps
            coupling[i : i + bs1, :] = log_plan.exp()

        return coupling

    #######################################
    # Begin calculation of local_ugw_cost #

    def get_loss_scalar(
        self, dict_plan, rho1, rho2, eps, log_p, data_const, alpha, reg_mode
    ):
        marg_1, marg_2 = dict_plan["marginals"]
        log_a, log_b = log_p

        scalar = 0.0
        if alpha != 0:
            if rho1 != np.inf and rho1 != 0:
                scalar += (
                    rho1
                    * alpha
                    * self.get_kl_of_vector(marg_1, log_a, "approx")
                )  # approx KL term
            if rho2 != np.inf and rho2 != 0:
                scalar += (
                    rho2
                    * alpha
                    * self.get_kl_of_vector(marg_2, log_b, "approx")
                )  # approx KL term

        if reg_mode == "joint":
            scalar += eps * self.get_kl_of_coupling(
                dict_plan, eps, log_p, data_const, alpha, "exact"
            )

        return scalar

    def get_C_vec(self, marginals, data_const):
        marg_1, marg_2 = marginals
        A1, A2, B1, B2, _, _ = data_const
        (bs1, r1), (bs2, r2) = self.sample_batch

        Cx_vec = torch.zeros_like(marg_1)
        for i in r1:
            Cx_vec[i : i + bs1] = (A1[i : i + bs1, :] @ A2.T) ** 2 @ marg_1

        Cy_vec = torch.zeros_like(marg_2)
        for j in r2:
            Cy_vec[j : j + bs2] = (B1[j : j + bs2, :] @ B2.T) ** 2 @ marg_2

        return Cx_vec, Cy_vec

    def get_bary_centre_intermediate(
        self, duals, log_p, local_ugw_cost, data_const, eps, alpha
    ):
        u, v = duals
        log_a, log_b = log_p
        bs1, r1 = self.sample_batch[0]

        _, A2, _, B2, _, _ = data_const
        na, db = A2.shape[0], B2.shape[1]
        device, dtype = A2.device, A2.dtype
        bary_centre = (
            torch.zeros((na, db)).to(device).to(dtype)
        )  # shape na x db

        s1 = u + log_a
        s2 = v + log_b
        for i in r1:
            cost = self.get_tensor_cost(
                local_ugw_cost, data_const, alpha, 1, i, bs1
            )
            log_plan = s1[i : i + bs1, None] + s2[None, :] - cost / eps
            bary_centre[i : i + bs1, :] = log_plan.exp() @ B2

        bary_centre = A2.T @ bary_centre  # shape da x db

        return bary_centre

    def get_local_ugw_cost(
        self, dict_plan, data_const, alpha, log_p, rho1, rho2, eps, reg_mode
    ):
        if alpha == 0:
            Cx_vec, Cy_vec, bary_centre = 0.0, 0.0, 0.0
        else:
            marginals = dict_plan["marginals"]
            duals = dict_plan["new_duals"]
            local_ugw_cost = dict_plan["local_ugw_cost"]
            Cx_vec, Cy_vec = self.get_C_vec(marginals, data_const)
            bary_centre = self.get_bary_centre_intermediate(
                duals, log_p, local_ugw_cost, data_const, eps, alpha
            )

        scalar = self.get_loss_scalar(
            dict_plan, rho1, rho2, eps, log_p, data_const, alpha, reg_mode
        )
        local_ugw_cost = (Cx_vec, Cy_vec, bary_centre, scalar)

        return local_ugw_cost

    # End calculation of local_ugw_cost #
    ######################################

    def get_fugw_cost(
        self, dict_pi, dict_gamma, data_const, log_p, rho, alpha, eps, reg_mode
    ):
        """
        Calculate the objective function of FUGW.
        """

        def f(mass, marg, axis):
            if axis == 1:
                value = (
                    alpha * rho1 * mass + (1 - alpha) / 2 * rho3
                ) * self.get_kl_of_vector(marg, log_a, "exact")
            elif axis == 2:
                value = (
                    alpha * rho2 * mass + (1 - alpha) / 2 * rho4
                ) * self.get_kl_of_vector(marg, log_b, "exact")
            return value

        log_a, log_b = log_p
        rho1, rho2, rho3, rho4 = rho
        (bs1, r1) = self.sample_batch[0]

        ug, vg = dict_gamma["new_duals"]
        local_ugw_cost_gamma = dict_gamma["local_ugw_cost"]
        s1_gamma = ug + log_a
        s2_gamma = vg + log_b

        if alpha != 1:
            _, _, _, _, D1, D2 = data_const
            up, vp = dict_pi["new_duals"]
            local_ugw_cost_pi = dict_pi["local_ugw_cost"]
            s1_pi = up + log_a
            s2_pi = vp + log_b

        # gw and OT terms
        fugw_cost = 0.0
        for i in r1:
            cost_gamma = self.get_tensor_cost(
                local_ugw_cost_gamma, data_const, alpha, 1, i, bs1
            )
            log_gamma = (
                s1_gamma[i : i + bs1, None]
                + s2_gamma[None, :]
                - cost_gamma / eps
            )
            true_cost = (
                cost_gamma - local_ugw_cost_gamma[-1]
            )  # Need to remove scalar term
            fugw_cost += (true_cost * log_gamma.exp()).sum()

            if alpha != 1:
                cost_pi = self.get_tensor_cost(
                    local_ugw_cost_pi, data_const, alpha, 1, i, bs1
                )
                log_pi = (
                    s1_pi[i : i + bs1, None] + s2_pi[None, :] - cost_pi / eps
                )
                fugw_cost += (
                    (1 - alpha)
                    / 2
                    * ((D1[i : i + bs1, :] @ D2.T) * log_pi.exp()).sum()
                )

        # KL terms
        marginal_pi = dict_pi["marginals"]
        marginal_gamma = dict_gamma["marginals"]
        mass_pi = marginal_pi[0].sum()
        mass_gamma = marginal_gamma[0].sum()

        if rho1 != np.inf and rho3 != np.inf:
            fugw_cost += (
                f(mass_gamma, marginal_pi[0], 1)
                + f(mass_pi, marginal_gamma[0], 1)
                + alpha
                * rho1
                * (mass_pi - log_a.exp().sum())
                * (mass_gamma - log_a.exp().sum())
            )
        if rho2 != np.inf and rho4 != np.inf:
            fugw_cost += (
                f(mass_gamma, marginal_pi[1], 2)
                + f(mass_pi, marginal_gamma[1], 2)
                + alpha
                * rho2
                * (mass_pi - log_b.exp().sum())
                * (mass_gamma - log_b.exp().sum())
            )

        # entropic term
        ent_fugw_cost = fugw_cost.detach().clone()
        kl_pi = self.get_kl_of_coupling(
            dict_pi, eps, log_p, data_const, alpha, "exact"
        )
        kl_gamma = self.get_kl_of_coupling(
            dict_gamma, eps, log_p, data_const, alpha, "exact"
        )

        if reg_mode == "joint":
            mass_ab = log_a.exp().sum() * log_b.exp().sum()
            ent_fugw_cost += eps * (
                mass_pi * kl_gamma
                + mass_gamma * kl_pi
                + (mass_pi - mass_ab) * (mass_gamma - mass_ab)
            )
        elif reg_mode == "independent":
            ent_fugw_cost += eps * (kl_pi + kl_gamma)

        return fugw_cost, ent_fugw_cost

    def _get_barycentre(
        self, data_target, dict_plan, log_p, data_const, eps, alpha
    ):
        """
        Calculate the barycentre corresponding to the target data.
        """
        u, v = dict_plan["new_duals"]
        local_ugw_cost = dict_plan["local_ugw_cost"]
        marg_1 = dict_plan["marginals"][0]
        log_a, log_b = log_p
        bs1, r1 = self.sample_batch[0]

        device, dtype = u.device, u.dtype
        barycentre = (
            torch.zeros((u.shape[0], data_target.shape[1]))
            .to(device)
            .to(dtype)
        )  # shape n1 x d2

        s1 = u + log_a
        s2 = v + log_b
        for i in r1:
            cost = self.get_tensor_cost(
                local_ugw_cost, data_const, alpha, 1, i, bs1
            )
            log_plan = s1[i : i + bs1, None] + s2[None, :] - cost / eps
            barycentre[i : i + bs1, :] = log_plan.exp() @ data_target

        barycentre = barycentre / marg_1.reshape(-1, 1)

        return barycentre

    def solver(
        self,
        X,
        Y,
        D,
        px=None,
        py=None,
        rho=(1e-1, None, None, None),
        eps=1e-2,
        alpha=1,
        reg_mode="joint",
        verbose=False,
        verbose_ot=False,
        save_freq=1,
        max_tensor_size=1e9,
        return_coupling=False,
    ):
        """
        To recover:
        - Ent-LB-UGW: alpha = 1, mode = "joint", rho1 != infty, rho2 != infty. No need to care about rho3 and rho4.
        - EGW: alpha = 1, mode = "independent", rho1 = rho2 = infty. No need to care about rho3 and rho4.
        - Ent-FGW: 0 < alpha < 1, D != None, mode = "independent", rho1 = rho2 = infty (so rho3 = rho4 = infty)
        - Ent-semi-relaxed GW: alpha = 1, mode = "independent", (rho1 = 0, rho2 = infty), or (rho1 = infty, rho2 = 0).
        No need to care about rho3 and rho4.
        - Ent-semi-relaxed FGW: 0 < alpha < 1, mode = "independent", (rho1 = rho3 = 0, rho2 = rho4 = infty),
        or (rho1 = rho3 = infty, rho2 = rho4 = 0).
        - Ent-UOT: alpha = 0, mode = "independent", D != None, rho1 != infty, rho2 != infty, rho3 != infty, rho4 != infty.

        Parameters
        ----------
        X: source matrix of size nx x dx
        Y: target matrix of size ny x dy
        D: tuple of attribute matrices (attr_X, attr_Y) of size (nx x d) and (ny x d)
        px: tuple of 2 vectors of length (n1, d1). Measures assigned on rows and columns of X.
        py: tuple of 2 vectors of length (n2, d2). Measures assigned on rows and columns of Y.
        rho: tuple of 4 relaxation parameters for UGW and UOT.
        eps: regularisation parameter for entropic approximation.
        alpha: between 0 and 1. Interpolation parameter for fused UGW.
        reg_mode:
            reg_mode = "joint": use UGW-like regularisation term
            reg_mode = "independent": use COOT-like regularisation
        init_n: matrix of size n1 x n2 if not None. Initialisation matrix for sample coupling.
        log: True if the loss is recorded, False otherwise.
        verbose: if True then print the recorded loss.
        save_freq: The multiplier of iteration at which the loss is calculated. For example, if save_freq = 10, then the
                    loss is calculated at iteration 10, 20, 30, etc...

        Returns
        ----------
        pi: matrix of size n1 x n2. Sample matrix.
        gamma: matrix of size d1 x d2. Feature matrix.
        log_cost: if log is True, return a list of loss (without taking into account the regularisation term).
        log_ent_cost: if log is True, return a list of entropic loss.
        """

        attr_X, attr_Y = D

        # Adjust rho in some settings.
        rho1, rho2, rho3, rho4 = rho
        if rho1 is None:
            rho1 = np.inf
        if rho2 is None:
            rho2 = np.inf

        if attr_X is None or attr_Y is None or alpha == 1:
            print("NO fused")
            rho3, rho4 = 0, 0
            alpha = 1
        else:
            print("Fused")
            if rho3 is None:
                rho3 = rho1
            if rho4 is None:
                rho4 = rho2
            if rho3 == np.inf or rho1 == np.inf:
                rho1, rho3 = np.inf, np.inf
            if rho4 == np.inf or rho2 == np.inf:
                rho2, rho4 = np.inf, np.inf
        rho = (rho1, rho2, rho3, rho4)  # update rho

        nx, ny = X.shape[0], Y.shape[0]
        device, dtype = X.device, X.dtype

        batch_size_x, batch_size_y = (
            min(int(max_tensor_size / ny), nx),
            min(int(max_tensor_size / nx), ny),
        )
        range_x, range_y = (
            range(0, nx, batch_size_x),
            range(0, ny, batch_size_y),
        )
        range_x = tqdm(range_x) if verbose_ot else range_x
        range_y = tqdm(range_y) if verbose_ot else range_y
        self.sample_batch = ((batch_size_x, range_x), (batch_size_y, range_y))

        if px is None:
            px = torch.ones(nx).to(device).to(dtype) / nx
        if py is None:
            py = torch.ones(ny).to(device).to(dtype) / ny

        log_p = (px.log(), py.log())

        if alpha == 0:
            A1, A2, B1, B2 = 0.0, 0.0, 0.0, 0.0
        else:
            A1, A2 = self.get_factored_distance(X, X)
            B1, B2 = self.get_factored_distance(Y, Y)

        if alpha == 1:
            D1, D2 = 0.0, 0.0
        else:
            D1, D2 = self.get_factored_distance(attr_X, attr_Y)

        data_const = (A1, A2, B1, B2, D1, D2)

        # must have equal mass because they are marginals
        init_marginals = (
            px * (py.sum() / px.sum()).sqrt(),
            py * (px.sum() / py.sum()).sqrt(),
        )

        if alpha == 0:
            Cx_vec, Cy_vec, bary_centre = 0.0, 0.0, 0.0
        else:
            na, db = A2.shape[0], B2.shape[1]
            bary_centre = (
                torch.zeros((na, db)).to(device).to(dtype)
            )  # shape na x db
            for i in range_x:
                # shape batchsize x db
                bary_centre[i : i + batch_size_x, :] = (
                    px[i : i + batch_size_x, None]
                    * py[None, :]
                    / (px.sum() * py.sum()).sqrt()
                ) @ B2
            bary_centre = A2.T @ bary_centre  # shape da x db
            Cx_vec, Cy_vec = self.get_C_vec(init_marginals, data_const)

        dict_pi = {
            "local_ugw_cost": (Cx_vec, Cy_vec, bary_centre, 0.0),
            "duals": (torch.zeros_like(px), torch.zeros_like(py)),
            "marginals": init_marginals,
            "new_duals": (torch.zeros_like(px), torch.zeros_like(py)),
        }

        dict_gamma = {
            "local_ugw_cost": (Cx_vec, Cy_vec, bary_centre, 0.0),
            "duals": (torch.zeros_like(px), torch.zeros_like(py)),
            "marginals": init_marginals,
            "new_duals": (torch.zeros_like(px), torch.zeros_like(py)),
        }

        mass_gamma = (px.sum() * py.sum()).sqrt()

        const1 = alpha * rho1
        const2 = alpha * rho2
        const3 = (1 - alpha) * rho3 / 2
        const4 = (1 - alpha) * rho4 / 2

        log_cost = [np.inf]
        log_ent_cost = [np.inf]
        i = 1
        err = np.inf

        while (err > self.tol) and (i <= self.nits):

            # Update pi
            dict_pi["local_ugw_cost"] = self.get_local_ugw_cost(
                dict_gamma, data_const, alpha, log_p, rho1, rho2, eps, reg_mode
            )

            r13 = const1 * mass_gamma + const3
            r24 = const2 * mass_gamma + const4
            new_eps = mass_gamma * eps if reg_mode == "joint" else eps
            dict_pi = self.sinkhorn(
                dict_pi, r13, r24, new_eps, log_p, data_const, alpha
            )

            # rescale pi: marginals, duals, mass
            mass_pi = dict_pi["marginals"][0].sum()

            rescale_const = (mass_gamma / mass_pi).sqrt()
            dict_pi["marginals"] = (
                dict_pi["marginals"][0] * rescale_const,
                dict_pi["marginals"][1] * rescale_const,
            )
            dict_pi["new_duals"] = (
                dict_pi["duals"][0] + rescale_const.log(),
                dict_pi["duals"][1],
            )
            mass_pi *= rescale_const

            # Update gamma
            dict_gamma["local_ugw_cost"] = self.get_local_ugw_cost(
                dict_pi, data_const, alpha, log_p, rho1, rho2, eps, reg_mode
            )
            r13 = const1 * mass_pi + const3
            r24 = const2 * mass_pi + const4
            new_eps = mass_pi * eps if reg_mode == "joint" else eps
            dict_gamma = self.sinkhorn(
                dict_gamma, r13, r24, new_eps, log_p, data_const, alpha
            )

            # rescale gamma: marginals, duals, mass
            mass_gamma = dict_gamma["marginals"][0].sum()
            rescale_const = (mass_pi / mass_gamma).sqrt()

            dict_gamma["marginals"] = (
                dict_gamma["marginals"][0] * rescale_const,
                dict_gamma["marginals"][1] * rescale_const,
            )
            dict_gamma["new_duals"] = (
                dict_gamma["duals"][0] + rescale_const.log(),
                dict_gamma["duals"][1],
            )
            mass_gamma *= rescale_const

            # UPDATE ERROR
            if i % save_freq == 0:
                cost, ent_cost = self.get_fugw_cost(
                    dict_pi,
                    dict_gamma,
                    data_const,
                    log_p,
                    rho,
                    alpha,
                    eps,
                    reg_mode,
                )
                log_cost.append(cost.item())
                log_ent_cost.append(ent_cost.item())
                err = abs(log_ent_cost[-2] - log_ent_cost[-1])

                if verbose:
                    print("Cost at iteration {}: {}".format(i, cost.item()))

            i += 1

        if return_coupling:
            self.get_coupling = partial(
                self._get_coupling,
                log_p=log_p,
                data_const=data_const,
                eps=eps,
                alpha=alpha,
            )
        self.get_barycentre = partial(
            self._get_barycentre,
            dict_plan=dict_pi,
            log_p=log_p,
            data_const=data_const,
            eps=eps,
            alpha=alpha,
        )

        return dict_pi, dict_gamma, log_cost[1:], log_ent_cost[1:]

    # def faster_solver(self, X, Y, D, px=(None, None), py=(None, None), rho=(1e-1, None, None, None), eps=1e-2, alpha=1, reg_mode="joint",
    #                   init_n=None, eps_step=10, init_eps=1, verbose=False, save_freq=10):

    #     if eps >= init_eps:
    #         return self.solver(X, Y, D, px, py, rho, eps, alpha, reg_mode, init_n, verbose, save_freq)

    #     else:
    #         pi_init = init_n

    #         nits = self.nits
    #         nits_sinkhorn = self.nits_sinkhorn

    #         self.nits = 10
    #         self.nits_sinkhorn = 10

    #         while (init_eps > eps):
    #             pi_init, _ = self.solver(
    #                 X, Y, D, px, py, rho, init_eps, alpha, reg_mode, pi_init)
    #             init_eps /= eps_step

    #         self.nits = nits
    #         self.nits_sinkhorn = nits_sinkhorn

    #         return self.solver(X, Y, D, px, py, rho, eps, alpha, reg_mode, pi_init, verbose, save_freq)


# if __name__ == "__main__":

#     use_cuda = torch.cuda.is_available()
#     device = torch.device("cuda:0" if use_cuda else "cpu")
#     torch.backends.cudnn.benchmark = True

#     nx = 2087
#     dx = 3
#     ny = 1561
#     dy = 2
#     d = 5

#     x = torch.rand(nx, dx).to(device)
#     y = torch.rand(ny, dy).to(device)
#     attr_X = torch.rand(nx, d).to(device)
#     attr_Y = torch.rand(ny, d).to(device)

#     rho1 = 1e-4
#     rho2 = 1e-5
#     rho3 = 1e-2
#     rho4 = 1e-3
#     mode = "independent"  # only support for this mode only, DO NOT try mode="joint"
#     rho = (rho1, rho2, rho3, rho4)
#     eps = 1e-3
#     alpha = 0.2

#     fugw_batch = BatchFUGW(
#         nits=100, nits_sinkhorn=200, tol=1e-5, tol_sinkhorn=1e-7
#     )
#     (
#         dict_pi,
#         dict_gamma,
#         log_cost_batch,
#         log_ent_cost_batch,
#     ) = fugw_batch.solver(
#         X=x,
#         Y=y,
#         D=(attr_X, attr_Y),
#         rho=rho,
#         eps=eps,
#         alpha=alpha,
#         reg_mode=mode,
#         verbose=True,
#         verbose_ot=False,
#         save_freq=1,
#         max_tensor_size=1e5,
#         return_coupling=True,
#     )
