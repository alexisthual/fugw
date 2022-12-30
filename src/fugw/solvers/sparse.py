from functools import partial

import numpy as np
import torch
from ot.gromov import fused_gromov_wasserstein as fgw
from ot.gromov import gromov_wasserstein as gw

from .utils import (
    batch_elementwise_prod_and_sum,
    compute_approx_kl,
    compute_approx_kl_sparse,
    compute_kl_sparse,
    compute_quad_kl,
    compute_quad_kl_sparse,
    elementwise_prod_fact_sparse,
    solver_dc_sparse,
    solver_mm_sparse,
)


class FUGWSparseSolver:
    def __init__(
        self,
        nits_bcd=50,
        nits_uot=1000,
        tol_bcd=1e-7,
        tol_uot=1e-7,
        eval_bcd=2,
        eval_uot=10,
        **kwargs,
    ):
        """
        write me
        """

        self.nits_bcd = nits_bcd
        self.nits_uot = nits_uot
        self.tol_bcd = tol_bcd
        self.tol_uot = tol_uot
        self.eval_bcd = eval_bcd
        self.eval_uot = eval_uot

    def local_cost(self, pi, transpose, data_const, tuple_p, hyperparams):
        """
        Returns local cost matrix.
        """

        rho_x, rho_y, eps, alpha, reg_mode = hyperparams
        px, py, pxy = tuple_p
        # X_sqr, Y_sqr, X, Y, D = data_const
        (
            (Gs_sqr_1, Gs_sqr_2),
            (Gt_sqr_1, Gt_sqr_2),
            (Gs1, Gs2),
            (Gt1, Gt2),
            (K1, K2),
        ) = data_const
        if transpose:
            # X_sqr, Y_sqr, X, Y = X_sqr.T, Y_sqr.T, X.T, Y.T
            Gs_sqr_1, Gs_sqr_2 = Gs_sqr_2, Gs_sqr_1
            Gt_sqr_1, Gt_sqr_2 = Gt_sqr_2, Gt_sqr_1
            Gs1, Gs2 = Gs2, Gs1
            Gt1, Gt2 = Gt2, Gt1

        pi1, pi2 = (
            torch.sparse.sum(pi, 1).to_dense(),
            torch.sparse.sum(pi, 0).to_dense(),
        )

        cost_const = 0
        rows, cols = pi._indices()

        if reg_mode == "joint":
            cost_const += eps * compute_approx_kl_sparse(pi, pxy)

        # or when cost is balanced
        if rho_x != float("inf") and rho_x != 0:
            cost_const += rho_x * compute_approx_kl(pi1, px)
        if rho_y != float("inf") and rho_y != 0:
            cost_const += rho_y * compute_approx_kl(pi2, py)

        # avoid unnecessary calculation of UGW when alpha = 0
        cost_values = torch.zeros_like(pi._values())
        if alpha != 1 and K1 is not None and K2 is not None:
            # cost = cost + (1 - alpha) * D / 2
            # cost_values = (K1[rows, :] * K2[cols, :]).sum(1)
            cost_values = batch_elementwise_prod_and_sum(K1, K2, rows, cols, 1)
            cost_values *= (1 - alpha) / 2

        # or UOT when alpha = 1
        if alpha != 0:
            # A = X_sqr @ pi1
            # B = Y_sqr @ pi2
            # gw_cost = A[:, None] + B[None, :] - 2 * X @ pi @ Y.T
            A = Gs_sqr_1 @ (Gs_sqr_2.T @ pi1)
            B = Gt_sqr_1 @ (Gt_sqr_2.T @ pi2)
            C1, C2 = Gs1, ((Gs2.T @ torch.sparse.mm(pi, Gt2)) @ Gt1.T).T

            gw_cost_values = A[rows] + B[cols]
            # gw_cost_values -= 2 * (C1[rows, :] * C2[cols, :]).sum(1)
            gw_cost_values -= 2 * batch_elementwise_prod_and_sum(
                C1, C2, rows, cols, 1
            )

            cost_values += alpha * gw_cost_values

        cost = torch.sparse_coo_tensor(
            pi._indices(), cost_values + cost_const, pi.size()
        )
        return cost

    def fugw_cost(self, pi, gamma, data_const, tuple_p, hyperparams):
        """
        Returns scalar fugw cost.
        """

        rho_x, rho_y, eps, alpha, reg_mode = hyperparams
        px, py, pxy = tuple_p
        (
            (Gs_sqr_1, Gs_sqr_2),
            (Gt_sqr_1, Gt_sqr_2),
            (Gs1, Gs2),
            (Gt1, Gt2),
            (K1, K2),
        ) = data_const

        pi1, pi2 = (
            torch.sparse.sum(pi, 1).to_dense(),
            torch.sparse.sum(pi, 0).to_dense(),
        )
        gamma1, gamma2 = (
            torch.sparse.sum(gamma, 1).to_dense(),
            torch.sparse.sum(gamma, 0).to_dense(),
        )

        cost = 0

        if alpha != 1 and K1 is not None and K2 is not None:
            # w_cost = (D * pi).sum() + (D * gamma).sum()
            w_cost = torch.sparse.sum(
                elementwise_prod_fact_sparse(K1, K2, pi + gamma)
            )
            cost = cost + (1 - alpha) * w_cost / 2

        if alpha != 0:
            A = (Gs_sqr_1 @ (Gs_sqr_2.T @ gamma1)).dot(pi1)
            B = (Gt_sqr_1 @ (Gt_sqr_2.T @ gamma2)).dot(pi2)
            C = elementwise_prod_fact_sparse(
                Gs1, ((Gs2.T @ torch.sparse.mm(gamma, Gt2)) @ Gt1.T).T, pi
            )
            gw_cost = A + B - 2 * torch.sparse.sum(C)
            cost = cost + alpha * gw_cost

        if rho_x != float("inf") and rho_x != 0:
            cost = cost + rho_x * compute_quad_kl(pi1, gamma1, px, px)
        if rho_y != float("inf") and rho_y != 0:
            cost = cost + rho_y * compute_quad_kl(pi2, gamma2, py, py)

        if reg_mode == "joint":
            entropic_cost = cost + eps * compute_quad_kl_sparse(
                pi, gamma, pxy, pxy
            )
        elif reg_mode == "independent":
            entropic_cost = (
                cost
                + eps * compute_kl_sparse(pi, pxy)
                + eps * compute_kl_sparse(gamma, pxy)
            )

        return cost.item(), entropic_cost.item()

    def project_on_target_domain(self, Xt, pi):
        """
        Project the SOURCE data on the TARGET space via the formula:
        proj = diag(1 / P_{# 1}) P Xt (need to be typed in latex).

        Parameters
        ----------
        Xt: target data of size nt x dt,
            NOT to be confused with the target distance matrix.
        pi: optimal plan of size ns x nt.

        Returns
        -------
        Projection of size ns x dt
        """

        projection = torch.sparse.mm(pi, Xt) / torch.sparse.sum(
            pi, 1
        ).to_dense().reshape(-1, 1)

        return projection

    def project_on_source_domain(self, Xs, pi):
        """
        Project the TARGET data on the SOURCE space via the formula:
        proj = diag(1 / P1_{# 2}) P.T Xs (need to be typed in latex).

        Parameters
        ----------
        Xs: source data of size ns x ds,
            NOT to be confused with the source distance matrix.
        pi: optimal plan of size ns x nt.

        Returns
        -------
        Projection of size nt x ds
        """

        projection = torch.sparse.mm(
            pi.transpose(1, 0), Xs
        ) / torch.sparse.sum(pi, 0).to_dense().reshape(-1, 1)

        return projection

    def solver_fugw(
        self,
        Gs,
        Gt,
        px=None,
        py=None,
        K=None,
        alpha=1,
        rho_x=float("inf"),
        rho_y=float("inf"),
        eps=1e-2,
        uot_solver="dc",
        reg_mode="joint",
        init_plan=None,
        init_duals=None,
        return_plans_only=True,
        verbose=False,
        early_stopping_threshold=1e-6,
        dc_eps_base=1,
        dc_nits_sinkhorn=1,
    ):
        """
        Parameters for mode:
        - Ent-LB-UGW: alpha = 1, mode = "joint", rho_x != inf, rho_y != inf.
            No need to care about rho3 and rho4.
        - EGW: alpha = 1, mode = "independent", rho_x = rho_y = inf.
            No need to care about rho3 and rho4.
        - Ent-FGW: 0 < alpha < 1, D != None, mode = "independent",
            rho_x = rho_y = inf (so rho3 = rho4 = inf)
        - Ent-semi-relaxed GW: alpha = 1, mode = "independent",
            (rho_x = 0, rho_y = inf), or (rho_x = inf, rho_y = 0).
            No need to care about rho3 and rho4.
        - Ent-semi-relaxed FGW: 0 < alpha < 1, mode = "independent",
            (rho_x = rho3 = 0, rho_y = rho4 = inf),
            or (rho_x = rho3 = inf, rho_y = rho4 = 0).
        - Ent-UOT: alpha = 0, mode = "independent", D != None,
            rho_x != inf, rho_y != inf, rho3 != inf, rho4 != inf.

        Parameters
        ----------
        X: matrix of size n1 x n1
        Y: matrix of size n2 x n2
        D: matrix of size n1 x n2. Feature matrix, in case of fused GW
        px: tuple of 2 vectors of length (n1, d1).
            Measures assigned on rows and columns of X.
        py: tuple of 2 vectors of length (n2, d2).
            Measures assigned on rows and columns of Y.
        rho: tuple of 4 relaxation parameters for UGW and UOT.
        eps: regularisation parameter for entropic approximation.
        alpha: between 0 and 1. Interpolation parameter for fused UGW.
        reg_mode:
            reg_mode="joint": use UGW-like regularisation term
            reg_mode = "independent": use COOT-like regularisation
        init_n: matrix of size n1 x n2 if not None.
            Initialisation matrix for sample coupling.
        return_plans_only: if False, return duals and loss as well.
        verbose: if True then print the recorded loss.

        Returns
        -------
        pi: matrix of size n1 x n2. Sample matrix.
        gamma: matrix of size d1 x d2. Feature matrix.
        log_cost: if log is True, return a list of loss
            (without taking into account the regularisation term).
        log_ent_cost: if log is True, return a list of entropic loss.
        """

        # sanity check
        if uot_solver == "mm" and (
            rho_x == float("inf") or rho_y == float("inf")
        ):
            uot_solver = "dc"
        if uot_solver == "sinkhorn" and eps == 0:
            uot_solver = "dc"

        nx, ny = Gs[0].shape[0], Gt[0].shape[0]
        device, dtype = Gs[0].device, Gs[0].dtype

        # constant data variables
        # Gs_sqr = Gs**2
        # Gt_sqr = Gt**2
        Gs1, Gs2 = Gs
        Gs_sqr = (
            torch.einsum("ij,il->ijl", Gs1, Gs1).reshape(
                Gs1.shape[0], Gs1.shape[1] ** 2
            ),
            torch.einsum("ij,il->ijl", Gs2, Gs2).reshape(
                Gs2.shape[0], Gs2.shape[1] ** 2
            ),
        )

        Gt1, Gt2 = Gt
        Gt_sqr = (
            torch.einsum("ij,il->ijl", Gt1, Gt1).reshape(
                Gt1.shape[0], Gt1.shape[1] ** 2
            ),
            torch.einsum("ij,il->ijl", Gt2, Gt2).reshape(
                Gt2.shape[0], Gt2.shape[1] ** 2
            ),
        )

        if alpha == 1 or K[0] is None or K[1] is None:
            alpha, K = 1, (None, None)

        # initialise coupling and dual vectors
        if init_plan is not None:
            pi = init_plan
            gamma = pi
        else:
            # TODO: Ending up here should raise a warning
            # because computations are done using sparse matrices
            # but they are actually dense
            pi = torch.sparse_coo_tensor(
                torch.from_numpy(
                    np.array(
                        [
                            np.tile(np.arange(nx), ny),
                            np.repeat(np.arange(ny), nx),
                        ]
                    )
                ).type(dtype),
                torch.from_numpy(np.ones(nx * ny) / (nx * ny)).type(dtype),
                (nx, ny),
            )
            gamma = pi

        # measures on rows and columns
        if px is None:
            px = torch.ones(nx).to(device).to(dtype) / nx
        if py is None:
            py = torch.ones(ny).to(device).to(dtype) / ny

        # pxy = px[:, None] * py[None, :]
        rows, cols = pi._indices()
        pxy_values = px[rows] * py[cols]
        pxy = torch.sparse_coo_tensor(pi._indices(), pxy_values, pi.size())

        # if uot_solver == "sinkhorn":
        #     if init_duals is None:
        #         duals_p = (
        #             torch.zeros_like(px),
        #             torch.zeros_like(py),
        #         )  # shape n1, n2
        #     else:
        #         duals_p = init_duals
        #     duals_g = duals_p
        if uot_solver == "mm":
            duals_p, duals_g = None, None
        elif uot_solver == "dc":
            if init_duals is None:
                duals_p = (
                    torch.ones_like(px),
                    torch.ones_like(py),
                )  # shape n1, n2
            else:
                duals_p = init_duals
            duals_g = duals_p

        compute_local_cost = partial(
            self.local_cost,
            data_const=(Gs_sqr, Gt_sqr, Gs, Gt, K),
            tuple_p=(px, py, pxy),
            hyperparams=(rho_x, rho_y, eps, alpha, reg_mode),
        )

        compute_fugw_cost = partial(
            self.fugw_cost,
            data_const=(Gs_sqr, Gt_sqr, Gs, Gt, K),
            tuple_p=(px, py, pxy),
            hyperparams=(rho_x, rho_y, eps, alpha, reg_mode),
        )

        # self_solver_scaling = partial(
        #     solver_scaling,
        #     tuple_pxy=(px.log(), py.log(), pxy),
        #     train_params=(self.nits_uot, self.tol_uot, self.eval_uot),
        # )

        self_solver_mm = partial(
            solver_mm_sparse,
            tuple_pxy=(px, py),
            train_params=(self.nits_uot, self.tol_uot, self.eval_uot),
        )

        self_solver_dc = partial(
            solver_dc_sparse,
            tuple_pxy=(px, py, pxy),
            train_params=(
                self.nits_uot,
                dc_nits_sinkhorn,
                dc_eps_base,
                self.tol_uot,
                self.eval_uot,
            ),
            verbose=verbose,
        )

        # initialise log
        log_cost = []
        log_ent_cost = [float("inf")]
        idx = 0
        err = self.tol_bcd + 1e-3

        while (err > self.tol_bcd) and (idx <= self.nits_bcd):
            pi_prev = pi.detach().clone()

            # Update gamma (feature coupling)
            mp = torch.sparse.sum(pi)
            new_rho_x = rho_x * mp
            new_rho_y = rho_y * mp
            new_eps = mp * eps if reg_mode == "joint" else eps
            uot_params = (new_rho_x, new_rho_y, new_eps)

            Tg = compute_local_cost(pi, transpose=True)  # size d1 x d2
            # if uot_solver == "sinkhorn":
            #     duals_g, gamma = self_solver_scaling(Tg, duals_g, uot_params)
            if uot_solver == "mm":
                gamma = self_solver_mm(Tg, gamma, uot_params)
            if uot_solver == "dc":
                duals_g, gamma = self_solver_dc(Tg, gamma, duals_g, uot_params)
            gamma = (
                mp / torch.sparse.sum(gamma)
            ).sqrt() * gamma  # shape d1 x d2

            # Update pi (sample coupling)
            mg = torch.sparse.sum(gamma)
            new_rho_x = rho_x * mg
            new_rho_y = rho_y * mg
            new_eps = mp * eps if reg_mode == "joint" else eps
            uot_params = (new_rho_x, new_rho_y, new_eps)

            Tp = compute_local_cost(gamma, transpose=False)  # size n1 x n2
            # if uot_solver == "sinkhorn":
            #     duals_p, pi = self_solver_scaling(Tp, duals_p, uot_params)
            if uot_solver == "mm":
                pi = self_solver_mm(Tp, pi, uot_params)
            elif uot_solver == "dc":
                duals_p, pi = self_solver_dc(Tp, pi, duals_p, uot_params)
            pi = (mg / torch.sparse.sum(pi)).sqrt() * pi  # shape n1 x n2

            # Update error
            err = (pi._values() - pi_prev._values()).abs().sum().item()
            if idx % self.eval_bcd == 0:
                cost, ent_cost = compute_fugw_cost(pi, gamma)

                log_cost.append(cost)
                log_ent_cost.append(ent_cost)

                if verbose:
                    print("Cost at iteration {}: {}".format(idx + 1, ent_cost))

                if (
                    abs(log_ent_cost[-2] - log_ent_cost[-1])
                    < early_stopping_threshold
                ):
                    break

            idx += 1

        if pi.isnan().any() or gamma.isnan().any():
            print("There is NaN in coupling")

        if return_plans_only:
            return pi, gamma
        else:
            return pi, gamma, duals_p, duals_g, log_cost, log_ent_cost

    def solver_fgw(
        self,
        X,
        Y,
        px=None,
        py=None,
        D=None,
        alpha=1,
        init_plan=None,
        verbose=True,
        **kwargs,
    ):
        nx, ny = X.shape[0], Y.shape[0]
        device, dtype = X.device, X.dtype

        # measures on rows and columns
        if px is None:
            px = torch.ones(nx).to(device).to(dtype) / nx
        if py is None:
            py = torch.ones(ny).to(device).to(dtype) / ny

        if alpha == 1 or D is None:
            pi, dict_log = gw(
                C1=X,
                C2=Y,
                p=px,
                q=py,
                log=True,
                G0=init_plan,
                verbose=verbose,
                **kwargs,
            )
            loss = dict_log["gw_dist"]
        else:
            pi, dict_log = fgw(
                M=D,
                C1=X,
                C2=Y,
                p=px,
                q=py,
                alpha=alpha,
                G0=init_plan,
                log=True,
                verbose=verbose,
                **kwargs,
            )
            loss = dict_log["fgw_dist"]

        duals = (dict_log["u"], dict_log["v"])

        return pi, duals, loss

    def solver(
        self,
        Gs,
        Gt,
        px=None,
        py=None,
        K=None,
        alpha=1,
        rho_x=float("inf"),
        rho_y=float("inf"),
        eps=1e-2,
        uot_solver="dc",
        reg_mode="joint",
        init_plan=None,
        init_duals=None,
        return_plans_only=True,
        verbose=False,
        early_stopping_threshold=1e-6,
        dc_eps_base=1,
        dc_nits_sinkhorn=1,
        **gw_kwargs,
    ):
        # if rho_x == float("inf") and rho_y == float("inf") and eps == 0:
        #     pi, duals, loss = self.solver_fgw(
        #         X, Y, px, py, D, alpha, init_plan, verbose, **gw_kwargs
        #     )
        #     if return_plans_only:
        #         return pi, pi
        #     else:
        #         return pi, pi, duals, duals, loss, loss
        # elif eps == 0 and (
        #     (rho_x == 0 and rho_y == float("inf"))
        #     or (rho_x == 0 and rho_y == float("inf"))
        # ):
        #     raise ValueError(
        #         "Invalid rho and eps. Unregularized semi-relaxed GW is not"
        #         " supported."
        #     )
        # else:
        return self.solver_fugw(
            Gs,
            Gt,
            px=px,
            py=py,
            K=K,
            alpha=alpha,
            rho_x=rho_x,
            rho_y=rho_y,
            eps=eps,
            uot_solver=uot_solver,
            reg_mode=reg_mode,
            early_stopping_threshold=early_stopping_threshold,
            dc_eps_base=dc_eps_base,
            dc_nits_sinkhorn=dc_nits_sinkhorn,
            init_plan=init_plan,
            init_duals=init_duals,
            return_plans_only=return_plans_only,
            verbose=verbose,
        )
