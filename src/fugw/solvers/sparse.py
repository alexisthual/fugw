from functools import partial

import numpy as np
import torch

from fugw.utils import console, make_csr_matrix
from .utils import (
    batch_elementwise_prod_and_sum,
    compute_approx_kl,
    compute_approx_kl_sparse,
    compute_kl_sparse,
    compute_quad_kl,
    compute_quad_kl_sparse,
    crow_indices_to_row_indices,
    csr_sum,
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
        eval_bcd=1,
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
        Before each block coordinate descent (BCD) step,
        the local cost matrix is updated.
        This local cost is a matrix of size (n1, n2)
        which evaluates the cost between every pair of points
        of the source and target distributions.
        Then, we run a BCD (sinkhorn, dc or mm) step
        which makes use of this cost to update the transport plans.
        """

        rho_x, rho_y, eps, alpha, reg_mode = hyperparams
        px, py, pxy = tuple_p
        device = px.device

        (
            (Gs_sqr_1, Gs_sqr_2),
            (Gt_sqr_1, Gt_sqr_2),
            (Gs1, Gs2),
            (Gt1, Gt2),
            (K1, K2),
        ) = data_const
        if transpose:
            Gs_sqr_1, Gs_sqr_2 = Gs_sqr_2, Gs_sqr_1
            Gt_sqr_1, Gt_sqr_2 = Gt_sqr_2, Gt_sqr_1
            Gs1, Gs2 = Gs2, Gs1
            Gt1, Gt2 = Gt2, Gt1

        pi1, pi2 = (
            csr_sum(pi, dim=1),
            csr_sum(pi, dim=0),
        )

        # Avoid unnecessary calculation of UGW when alpha = 0
        # row_indices, col_indices = pi._indices()
        crow_indices, col_indices = (
            pi.crow_indices(),
            pi.col_indices(),
        )
        row_indices = crow_indices_to_row_indices(crow_indices)
        cost_values = torch.zeros_like(pi.values()).to(device)

        if alpha != 1 and K1 is not None and K2 is not None:
            wasserstein_cost_values = batch_elementwise_prod_and_sum(
                K1, K2, row_indices, col_indices, 1
            )
            cost_values += (1 - alpha) / 2 * wasserstein_cost_values

        # or UOT when alpha = 1
        if alpha != 0:
            A = Gs_sqr_1 @ (Gs_sqr_2.T @ pi1)
            B = Gt_sqr_1 @ (Gt_sqr_2.T @ pi2)
            C1, C2 = Gs1, ((Gs2.T @ torch.sparse.mm(pi, Gt2)) @ Gt1.T).T

            gromov_wasserstein_cost_values = (
                A[row_indices]
                + B[col_indices]
                - 2
                * batch_elementwise_prod_and_sum(
                    C1, C2, row_indices, col_indices, 1
                )
            )

            cost_values += alpha * gromov_wasserstein_cost_values

        # or when cost is balanced
        if rho_x != float("inf") and rho_x != 0:
            marginal_cost_dim1 = compute_approx_kl(pi1, px)
            cost_values += rho_x * marginal_cost_dim1
        if rho_y != float("inf") and rho_y != 0:
            marginal_cost_dim2 = compute_approx_kl(pi2, py)
            cost_values += rho_y * marginal_cost_dim2

        if reg_mode == "joint":
            cost_values += eps * compute_approx_kl_sparse(pi, pxy)

        cost = torch.sparse_csr_tensor(
            crow_indices, col_indices, cost_values, size=pi.size()
        )

        return cost

    def fugw_loss(self, pi, gamma, data_const, tuple_p, hyperparams):
        """
        Returns a scalar which is a lower bound on the fugw loss.
        This lower bound is a combination of:
        - a Wasserstein loss on features
        - a Gromow-Wasserstein loss on geometries
        - marginal constraints on the computed OT plan
        - an entropic regularization
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
            csr_sum(pi, dim=1),
            csr_sum(pi, dim=0),
        )
        gamma1, gamma2 = (
            csr_sum(gamma, dim=1),
            csr_sum(gamma, dim=0),
        )

        loss = 0

        if alpha != 1 and K1 is not None and K2 is not None:
            # TODO: warning, torch sparse transforms LongTensor
            # into IntTensor for crow and col indices
            # when adding 2 CSR matrices together.
            # This could become problematic for sparse matrices
            # with more non-null elements than an int can store.
            wasserstein_loss = csr_sum(
                elementwise_prod_fact_sparse(K1, K2, pi + gamma)
            )
            loss += (1 - alpha) / 2 * wasserstein_loss

        if alpha != 0:
            A = (Gs_sqr_1 @ (Gs_sqr_2.T @ gamma1)).dot(pi1)
            B = (Gt_sqr_1 @ (Gt_sqr_2.T @ gamma2)).dot(pi2)
            C = elementwise_prod_fact_sparse(
                Gs1, ((Gs2.T @ torch.sparse.mm(gamma, Gt2)) @ Gt1.T).T, pi
            )
            gromov_wasserstein_loss = A + B - 2 * csr_sum(C)
            loss += alpha * gromov_wasserstein_loss

        if rho_x != float("inf") and rho_x != 0:
            marginal_constraint_dim1 = compute_quad_kl(pi1, gamma1, px, px)
            loss += rho_x * marginal_constraint_dim1
        if rho_y != float("inf") and rho_y != 0:
            marginal_constraint_dim2 = compute_quad_kl(pi2, gamma2, py, py)
            loss += rho_y * marginal_constraint_dim2

        if reg_mode == "joint":
            entropic_regularization = compute_quad_kl_sparse(
                pi, gamma, pxy, pxy
            )
        elif reg_mode == "independent":
            entropic_regularization = compute_kl_sparse(
                pi, pxy
            ) + compute_kl_sparse(gamma, pxy)

        entropic_loss = loss + eps * entropic_regularization

        return loss.item(), entropic_loss.item()

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

        projection = torch.sparse.mm(pi, Xt) / csr_sum(pi, dim=1).reshape(
            -1, 1
        )

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

        projection = torch.sparse.mm(pi.transpose(0, 1), Xs) / csr_sum(
            pi, dim=0
        ).reshape(-1, 1)

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
            pi = (
                torch.sparse_coo_tensor(
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
                .to(device)
                .to_sparse_csr()
            )
            gamma = pi

        # measures on rows and columns
        if px is None:
            px = torch.ones(nx).to(device).to(dtype) / nx
        if py is None:
            py = torch.ones(ny).to(device).to(dtype) / ny

        crow_indices, col_indices = pi.crow_indices(), pi.col_indices()
        row_indices = crow_indices_to_row_indices(crow_indices)
        pxy_values = px[row_indices] * py[col_indices]

        pxy = make_csr_matrix(
            crow_indices, col_indices, pxy_values, pi.size(), device
        )

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

        compute_fugw_loss = partial(
            self.fugw_loss,
            data_const=(Gs_sqr, Gt_sqr, Gs, Gt, K),
            tuple_p=(px, py, pxy),
            hyperparams=(rho_x, rho_y, eps, alpha, reg_mode),
        )

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

        # Initialise loss
        loss_steps = []
        loss_ = []
        loss_ent_ = []
        idx = 0
        err = self.tol_bcd + 1e-3

        while (err > self.tol_bcd) and (idx <= self.nits_bcd):
            pi_prev = pi.detach().clone()

            # Update gamma (feature coupling)
            mp = csr_sum(pi)
            new_rho_x = rho_x * mp
            new_rho_y = rho_y * mp
            new_eps = mp * eps if reg_mode == "joint" else eps
            uot_params = (new_rho_x, new_rho_y, new_eps)

            Tg = compute_local_cost(pi, transpose=True)  # size d1 x d2
            if uot_solver == "mm":
                gamma = self_solver_mm(Tg, gamma, uot_params)
            if uot_solver == "dc":
                duals_g, gamma = self_solver_dc(Tg, gamma, duals_g, uot_params)

            gamma_scaling_factor = (mp / csr_sum(gamma)).sqrt()
            gamma = torch.sparse_csr_tensor(
                gamma.crow_indices(),
                gamma.col_indices(),
                gamma.values() * gamma_scaling_factor,
                size=gamma.size(),
                device=device,
            )

            # Update pi (sample coupling)
            mg = csr_sum(gamma)
            new_rho_x = rho_x * mg
            new_rho_y = rho_y * mg
            new_eps = mp * eps if reg_mode == "joint" else eps
            uot_params = (new_rho_x, new_rho_y, new_eps)

            Tp = compute_local_cost(gamma, transpose=False)  # size n1 x n2
            if uot_solver == "mm":
                pi = self_solver_mm(Tp, pi, uot_params)
            elif uot_solver == "dc":
                duals_p, pi = self_solver_dc(Tp, pi, duals_p, uot_params)

            pi_scaling_factor = (mg / csr_sum(pi)).sqrt()
            pi = torch.sparse_csr_tensor(
                pi.crow_indices(),
                pi.col_indices(),
                pi.values() * pi_scaling_factor,
                size=pi.size(),
                device=device,
            )

            # Update error
            err = (pi.values() - pi_prev.values()).abs().sum().item()
            if idx % self.eval_bcd == 0:
                loss, loss_ent = compute_fugw_loss(pi, gamma)

                loss_steps.append(idx)
                loss_.append(loss)
                loss_ent_.append(loss_ent)

                if verbose:
                    console.log(
                        f"FUGW loss at BCD step {idx}:\t{loss}\t{loss_ent}"
                    )

                if (
                    len(loss_ent_) >= 2
                    and abs(loss_ent_[-2] - loss_ent_[-1])
                    < early_stopping_threshold
                ):
                    break

            idx += 1

        if pi.values().isnan().any() or gamma.values().isnan().any():
            console.log("There is NaN in coupling")

        return pi, gamma, duals_p, duals_g, loss_steps, loss_, loss_ent_

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
        verbose=False,
        early_stopping_threshold=1e-6,
        dc_eps_base=1,
        dc_nits_sinkhorn=1,
        **gw_kwargs,
    ):
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
            verbose=verbose,
        )
