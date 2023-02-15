from functools import partial

import torch

from fugw.utils import BaseSolver, console
from .utils import (
    compute_approx_kl,
    compute_kl,
    compute_quad_kl,
    solver_dc,
    solver_mm,
    solver_sinkhorn,
)


class FUGWSolver(BaseSolver):
    def local_biconvex_cost(
        self, pi, transpose, data_const, tuple_weights, hyperparams
    ):
        """
        Before each block coordinate descent (BCD) step,
        the local cost matrix is updated.
        This local cost is a matrix of size (n, m)
        which evaluates the cost between every pair of points
        of the source and target distributions.
        Then, we run a BCD (sinkhorn, dc or mm) step
        which makes use of this cost to update the transport plans.
        """

        rho_s, rho_t, eps, alpha, reg_mode = hyperparams
        ws, wt, ws_dot_wt = tuple_weights
        X_sqr, Y_sqr, X, Y, D = data_const
        if transpose:
            X_sqr, Y_sqr, X, Y = X_sqr.T, Y_sqr.T, X.T, Y.T

        pi1, pi2 = pi.sum(1), pi.sum(0)

        cost = torch.zeros_like(D)

        # Avoid unnecessary calculation of UGW when alpha = 0
        if alpha != 1 and D is not None:
            wasserstein_cost = D
            cost += (1 - alpha) / 2 * wasserstein_cost

        # or UOT when alpha = 1
        if alpha != 0:
            A = X_sqr @ pi1
            B = Y_sqr @ pi2
            gromov_wasserstein_cost = (
                A[:, None] + B[None, :] - 2 * X @ pi @ Y.T
            )

            cost += alpha * gromov_wasserstein_cost

        # or when cost is balanced
        if rho_s != float("inf") and rho_s != 0:
            marginal_cost_dim1 = compute_approx_kl(pi1, ws)
            cost += rho_s * marginal_cost_dim1
        if rho_t != float("inf") and rho_t != 0:
            marginal_cost_dim2 = compute_approx_kl(pi2, wt)
            cost += rho_t * marginal_cost_dim2

        if reg_mode == "joint":
            entropic_cost = compute_approx_kl(pi, ws_dot_wt)
            cost += eps * entropic_cost

        return cost

    def fugw_loss(self, pi, gamma, data_const, tuple_weights, hyperparams):
        """
        Returns scalar fugw loss, which is a combination of:
        - a Wasserstein loss on features
        - a Gromow-Wasserstein loss on geometries
        - marginal constraints on the computed OT plan
        - an entropic regularisation
        """

        rho_s, rho_t, eps, alpha, reg_mode = hyperparams
        ws, wt, ws_dot_wt = tuple_weights
        X_sqr, Y_sqr, X, Y, D = data_const

        pi1, pi2 = pi.sum(1), pi.sum(0)
        gamma1, gamma2 = gamma.sum(1), gamma.sum(0)

        loss = 0

        if alpha != 1 and D is not None:
            wasserstein_loss = (D * pi).sum() + (D * gamma).sum()
            loss += (1 - alpha) / 2 * wasserstein_loss

        if alpha != 0:
            A = (X_sqr @ gamma1).dot(pi1)
            B = (Y_sqr @ gamma2).dot(pi2)
            C = (X @ gamma @ Y.T) * pi
            gromov_wasserstein_loss = A + B - 2 * C.sum()
            loss += alpha * gromov_wasserstein_loss

        if rho_s != float("inf") and rho_s != 0:
            marginal_constraint_dim1 = compute_quad_kl(pi1, gamma1, ws, ws)
            loss += rho_s * marginal_constraint_dim1
        if rho_t != float("inf") and rho_t != 0:
            marginal_constraint_dim2 = compute_quad_kl(pi2, gamma2, wt, wt)
            loss += rho_t * marginal_constraint_dim2

        if reg_mode == "joint":
            entropic_regularization = compute_quad_kl(
                pi, gamma, ws_dot_wt, ws_dot_wt
            )
        elif reg_mode == "independent":
            entropic_regularization = compute_kl(pi, ws_dot_wt) + compute_kl(
                gamma, ws_dot_wt
            )

        entropic_loss = loss + eps * entropic_regularization

        return loss.item(), entropic_loss.item()

    def solve(
        self,
        alpha=0.5,
        rho_s=1,
        rho_t=1,
        eps=1e-2,
        reg_mode="joint",
        F=None,
        Ds=None,
        Dt=None,
        ws=None,
        wt=None,
        init_plan=None,
        init_duals=None,
        uot_solver="sinkhorn",
        verbose=False,
    ):
        """
        Function running the BCD iterations.

        Parameters
        ----------
        alpha: float
        rho_s: float
        rho_t: float
        eps: float
        reg_mode: str
        F: matrix of size n x m.
            Kernel matrix between the source and target features.
        Ds: matrix of size n x n
        Dt: matrix of size m x m
        ws: ndarray(n), None
            Measures assigned to source points.
        wt: ndarray(m), None
            Measures assigned to target points.
        init_plan: matrix of size n x m if not None.
            Initialisation matrix for coupling.
        init_duals: tuple or None
            Initialisation duals for coupling.
        uot_solver: "sinkhorn", "mm", "dc"
            Solver to use.
        verbose: bool, optional, defaults to False
            Log solving process.

        Returns
        -------
        pi: matrix of size n x m. Sample matrix.
        gamma: matrix of size d1 x d2. Feature matrix.
        log_cost: if log is True, return a list of loss
            (without taking into account the regularisation term).
        log_ent_cost: if log is True, return a list of entropic loss.
        """

        if rho_s == float("inf") and rho_t == float("inf") and eps == 0:
            raise ValueError(
                "This package does not handle balanced cases "
                "(ie infinite values of rho). "
                "You should have a look at POT (https://pythonot.github.io) "
                "and in particular at ot.gromov.fused_gromov_wasserstein "
                "(https://pythonot.github.io/gen_modules/ot.gromov.html"
                "#ot.gromov.fused_gromov_wasserstein)"
            )
        elif eps == 0 and (
            (rho_s == 0 and rho_t == float("inf"))
            or (rho_s == 0 and rho_t == float("inf"))
        ):
            raise ValueError(
                "Invalid rho and eps. Unregularized semi-relaxed GW is not "
                "supported."
            )

        # sanity check
        if uot_solver == "mm" and (
            rho_s == float("inf") or rho_t == float("inf")
        ):
            uot_solver = "dc"
        if uot_solver == "sinkhorn" and eps == 0:
            uot_solver = "dc"

        device, dtype = Ds.device, Ds.dtype

        # constant data variables
        Ds_sqr = Ds**2
        Dt_sqr = Dt**2

        if alpha == 1 or F is None:
            alpha = 1
            F = None

        # measures on rows and columns
        if ws is None:
            n = Ds.shape[0]
            ws = torch.ones(n).to(device).to(dtype) / n
        if wt is None:
            m = Dt.shape[0]
            wt = torch.ones(m).to(device).to(dtype) / m
        ws_dot_wt = ws[:, None] * wt[None, :]

        # initialise coupling and dual vectors
        pi = ws_dot_wt if init_plan is None else init_plan
        gamma = pi

        if uot_solver == "sinkhorn":
            if init_duals is None:
                duals_p = (
                    torch.zeros_like(ws),
                    torch.zeros_like(wt),
                )
            else:
                duals_p = init_duals
            duals_g = duals_p
        elif uot_solver == "mm":
            duals_p, duals_g = None, None
        elif uot_solver == "dc":
            if init_duals is None:
                duals_p = (
                    torch.ones_like(ws),
                    torch.ones_like(wt),
                )
            else:
                duals_p = init_duals
            duals_g = duals_p

        compute_local_biconvex_cost = partial(
            self.local_biconvex_cost,
            data_const=(Ds_sqr, Dt_sqr, Ds, Dt, F),
            tuple_weights=(ws, wt, ws_dot_wt),
            hyperparams=(rho_s, rho_t, eps, alpha, reg_mode),
        )

        compute_fugw_loss = partial(
            self.fugw_loss,
            data_const=(Ds_sqr, Dt_sqr, Ds, Dt, F),
            tuple_weights=(ws, wt, ws_dot_wt),
            hyperparams=(rho_s, rho_t, eps, alpha, reg_mode),
        )

        self_solver_sinkhorn = partial(
            solver_sinkhorn,
            tuple_weights=(ws, wt, ws_dot_wt),
            train_params=(self.nits_uot, self.tol_uot, self.eval_uot),
        )

        self_solver_mm = partial(
            solver_mm,
            tuple_weights=(ws, wt),
            train_params=(self.nits_uot, self.tol_uot, self.eval_uot),
        )

        self_solver_dc = partial(
            solver_dc,
            tuple_weights=(ws, wt, ws_dot_wt),
            train_params=(
                self.nits_uot,
                self.dc_nits_sinkhorn,
                self.dc_eps_base,
                self.tol_uot,
                self.eval_uot,
            ),
            verbose=verbose,
        )

        # Initialize loss
        loss_steps = []
        loss_ = []
        loss_ent_ = []
        idx = 0
        err = None

        while (err is None or err > self.tol_bcd) and (idx <= self.nits_bcd):
            pi_prev = pi.detach().clone()

            # Update gamma
            mp = pi.sum()
            new_rho_s = rho_s * mp
            new_rho_t = rho_t * mp
            new_eps = mp * eps if reg_mode == "joint" else eps
            uot_params = (new_rho_s, new_rho_t, new_eps)

            cost_gamma = compute_local_biconvex_cost(pi, transpose=True)
            if uot_solver == "sinkhorn":
                duals_g, gamma = self_solver_sinkhorn(
                    cost_gamma, duals_g, uot_params
                )
            elif uot_solver == "mm":
                gamma = self_solver_mm(cost_gamma, gamma, uot_params)
            if uot_solver == "dc":
                duals_g, gamma = self_solver_dc(
                    cost_gamma, gamma, duals_g, uot_params
                )
            gamma = (mp / gamma.sum()).sqrt() * gamma

            # Update pi
            mg = gamma.sum()
            new_rho_s = rho_s * mg
            new_rho_t = rho_t * mg
            new_eps = mp * eps if reg_mode == "joint" else eps
            uot_params = (new_rho_s, new_rho_t, new_eps)

            cost_pi = compute_local_biconvex_cost(gamma, transpose=False)
            if uot_solver == "sinkhorn":
                duals_p, pi = self_solver_sinkhorn(
                    cost_pi, duals_p, uot_params
                )
            elif uot_solver == "mm":
                pi = self_solver_mm(cost_pi, pi, uot_params)
            elif uot_solver == "dc":
                duals_p, pi = self_solver_dc(cost_pi, pi, duals_p, uot_params)
            pi = (mg / pi.sum()).sqrt() * pi

            # Update error
            err = (pi - pi_prev).abs().sum().item()
            if idx % self.eval_bcd == 0:
                loss, loss_ent = compute_fugw_loss(pi, gamma)

                loss_steps.append(idx)
                loss_.append(loss)
                loss_ent_.append(loss_ent)

                if verbose:
                    console.log(
                        f"BCD step {idx}/{self.nits_bcd}\t"
                        f"FUGW loss:\t{loss} (base)\t{loss_ent} (entropic)"
                    )

                if (
                    len(loss_ent_) >= 2
                    and abs(loss_ent_[-2] - loss_ent_[-1])
                    < self.early_stopping_threshold
                ):
                    break

            idx += 1

        if pi.isnan().any() or gamma.isnan().any():
            console.log("There is NaN in coupling")

        return pi, gamma, duals_p, duals_g, loss_steps, loss_, loss_ent_
