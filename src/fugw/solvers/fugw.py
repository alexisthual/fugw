import torch
import numpy as np

# require torch >= 1.9

"""
Code adapted from: https://github.com/thibsej/unbalanced_gromov_wasserstein
"""


class FUGWSolver:
    def __init__(
        self,
        nits=100,
        nits_sinkhorn=100,
        tol=1e-7,
        tol_sinkhorn=1e-7,
        **kwargs
    ):
        self.nits = nits
        self.nits_sinkhorn = nits_sinkhorn
        self.tol = tol
        self.tol_sinkhorn = tol_sinkhorn

    def sinkhorn(self, T, u, v, tuple_ab, rho1, rho2, eps):

        a, b, ab = tuple_ab
        tau1 = 1 if torch.isinf(rho1) else rho1 / (rho1 + eps)
        tau2 = 1 if torch.isinf(rho2) else rho2 / (rho2 + eps)

        for _ in range(self.nits_sinkhorn):
            u_prev = u.detach().clone()
            if rho2 == 0:
                v = torch.zeros_like(v)
            else:
                v = -tau2 * ((u + a.log())[:, None] - T / eps).logsumexp(dim=0)

            if rho1 == 0:
                u = torch.zeros_like(u)
            else:
                u = -tau1 * ((v + b.log())[None, :] - T / eps).logsumexp(dim=1)

            if (u - u_prev).abs().max().item() < self.tol_sinkhorn:
                break

        pi = ab * (u[:, None] + v[None, :] - T / eps).exp()

        return u, v, pi

    def approx_kl(self, p, q):
        # By convention: 0 log 0 = 0
        return torch.nan_to_num(
            p * (p / q).log(), nan=0.0, posinf=0.0, neginf=0.0
        ).sum()

    def kl(self, p, q):
        return self.approx_kl(p, q) - p.sum() + q.sum()

    def quad_kl(self, mu, nu, alpha, beta):
        """
        Calculate the KL divergence between two product measures:
        KL(mu otimes nu, alpha otimes beta) =
        m_mu * KL(nu, beta)
        + m_nu * KL(mu, alpha)
        + (m_mu - m_alpha) * (m_nu - m_beta)

        Parameters
        ----------
        mu: vector or matrix
        nu: vector or matrix
        alpha: vector or matrix with the same size as mu
        beta: vector or matrix with the same size as nu

        Returns
        ----------
        KL divergence between two product measures
        """

        m_mu = mu.sum()
        m_nu = nu.sum()
        m_alpha = alpha.sum()
        m_beta = beta.sum()
        const = (m_mu - m_alpha) * (m_nu - m_beta)
        kl = m_nu * self.kl(mu, alpha) + m_mu * self.kl(nu, beta) + const

        return kl

    def compute_local_cost(self, data_const, pi, tuple_ab, hyperparams):
        """ """

        rho, eps, alpha = hyperparams
        rho1, rho2, rho3, rho4 = rho
        a, b, ab = tuple_ab
        X_sqr, Y_sqr, X, Y, D = data_const

        cost = 0
        if self.reg_mode == "joint":
            cost = cost + eps * self.approx_kl(pi, ab)

        # avoid unnecessary calculation of UGW when alpha = 0
        # or UOT when alpha = 1
        if alpha != 1 and D is not None:
            cost = cost + (1 - alpha) * D / 2

        if alpha != 0:
            pi1, pi2 = pi.sum(1), pi.sum(0)
            A = X_sqr @ pi1
            B = Y_sqr @ pi2
            gw_cost = A[:, None] + B[None, :] - 2 * X @ pi @ Y.T

            if rho1 != np.inf and rho1 != 0 and rho3 != np.inf:
                gw_cost = gw_cost + rho1 * self.approx_kl(pi1, a)
            if rho2 != np.inf and rho2 != 0 and rho4 != np.inf:
                gw_cost = gw_cost + rho2 * self.approx_kl(pi2, b)

            cost = cost + alpha * gw_cost

        return cost

    def ucoot_cost(
        self, data_const, tuple_ab1, tuple_ab2, pi, gamma, hyperparams
    ):
        rho, eps, alpha = hyperparams
        rho1, rho2, rho3, rho4 = rho
        a1, b1, ab1 = tuple_ab1
        a2, b2, ab2 = tuple_ab2
        X_sqr, Y_sqr, X, Y, D = data_const

        pi1, pi2 = pi.sum(1), pi.sum(0)
        gamma1, gamma2 = gamma.sum(1), gamma.sum(0)

        cost = 0

        if alpha != 0:
            A = (X_sqr @ gamma1).dot(pi1)
            B = (Y_sqr @ gamma2).dot(pi2)
            C = (X @ gamma @ Y.T) * pi
            gw_cost = A + B - 2 * C.sum()

            if rho1 != np.inf and rho1 != 0:
                gw_cost += rho1 * self.quad_kl(pi1, gamma1, a1, a2)
            if rho2 != np.inf and rho2 != 0:
                gw_cost += rho2 * self.quad_kl(pi2, gamma2, b1, b2)

            cost = cost + alpha * gw_cost

        if alpha != 1:
            uot_cost = 0
            if D is not None:
                uot_cost += (D * pi).sum() + (D * gamma).sum()
            if rho3 != np.inf and rho3 != 0:
                uot_cost += rho3 * self.kl(pi1, a1) + rho3 * self.kl(
                    gamma1, a2
                )
            if rho4 != np.inf and rho4 != 0:
                uot_cost += rho4 * self.kl(pi2, b1) + rho4 * self.kl(
                    gamma2, b2
                )

            cost = cost + (1 - alpha) * uot_cost / 2

        if self.reg_mode == "joint":
            ent_cost = cost + eps * self.quad_kl(pi, gamma, ab1, ab2)
        elif self.reg_mode == "independent":
            ent_cost = (
                cost + eps * self.kl(pi, ab1) + eps * self.kl(gamma, ab2)
            )

        return cost, ent_cost

    def get_barycentre(self, Xt, sample_coupling):
        """
        Calculate the barycentre by the following formula:
        diag(1 / P1_{n_2}) P Xt
        (need to be typed in latex).

        Parameters
        ----------
        Xt: target data of size n2 x d2,
            NOT to be confused with the target distance matrix.
        sample_coupling: optimal plan of size n1 x n2.

        Returns
        -------
        Barycentre of size n1 x d2
        """

        barycentre = (
            sample_coupling @ Xt / sample_coupling.sum(1).reshape(-1, 1)
        )

        return barycentre

    def solver(
        self,
        X,
        Y,
        D=None,
        px=(None, None),
        py=(None, None),
        rho=(float("inf"), float("inf"), 0, 0),
        eps=1e-2,
        alpha=1,
        reg_mode="joint",
        init_n=None,
        log=False,
        verbose=False,
        early_stopping_threshold=1e-6,
        save_freq=10,
    ):
        """
        Parameters for mode:
        - Ent-LB-UGW: alpha = 1, mode = "joint", rho1 != infty, rho2 != infty.
            No need to care about rho3 and rho4.
        - EGW: alpha = 1, mode = "independent", rho1 = rho2 = infty.
            No need to care about rho3 and rho4.
        - Ent-FGW: 0 < alpha < 1, D != None, mode = "independent",
            rho1 = rho2 = infty (so rho3 = rho4 = infty)
        - Ent-semi-relaxed GW: alpha = 1, mode = "independent",
            (rho1 = 0, rho2 = infty), or (rho1 = infty, rho2 = 0).
            No need to care about rho3 and rho4.
        - Ent-semi-relaxed FGW: 0 < alpha < 1, mode = "independent",
            (rho1 = rho3 = 0, rho2 = rho4 = infty),
            or (rho1 = rho3 = infty, rho2 = rho4 = 0).
        - Ent-UOT: alpha = 0, mode = "independent", D != None,
            rho1 != infty, rho2 != infty, rho3 != infty, rho4 != infty.

        Parameters
        ----------
        X: matrix of size n1 x d1
        Y: matrix of size n2 x d2
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
        log: True if the loss is recorded, False otherwise.
        verbose: if True then print the recorded loss.
        save_freq: The multiplier of iteration at which the loss is calculated.
            For example, if save_freq = 10, then the
            loss is calculated at iteration 10, 20, 30, etc...

        Returns
        -------
        pi: matrix of size n1 x n2. Sample matrix.
        gamma: matrix of size d1 x d2. Feature matrix.
        log_cost: if log is True, return a list of loss
            (without taking into account the regularisation term).
        log_ent_cost: if log is True, return a list of entropic loss.
        """

        self.reg_mode = reg_mode

        n1, d1 = X.shape
        n2, d2 = Y.shape
        device, dtype = X.device, X.dtype

        # hyper-parameters
        rho1, rho2, rho3, rho4 = rho

        if rho3 == np.inf or rho1 == np.inf:
            rho1, rho3 = np.inf, np.inf
        if rho4 == np.inf or rho2 == np.inf:
            rho2, rho4 = np.inf, np.inf

        rho = (rho1, rho2, rho3, rho4)
        hyperparams = (rho, eps, alpha)

        const1 = alpha * rho1
        const2 = alpha * rho2
        const3 = (1 - alpha) * rho3 / 2
        const4 = (1 - alpha) * rho4 / 2

        # measures on rows and columns
        a1, a2 = px
        b1, b2 = py

        if a1 is None:
            a1 = torch.ones(n1).to(device).to(dtype) / n1
        if a2 is None:
            a2 = torch.ones(d1).to(device).to(dtype) / d1
        if b1 is None:
            b1 = torch.ones(n2).to(device).to(dtype) / n2
        if b2 is None:
            b2 = torch.ones(d2).to(device).to(dtype) / d2
        ab1 = a1[:, None] * b1[None, :]
        ab2 = a2[:, None] * b2[None, :]

        tuple_ab1 = (a1, b1, ab1)
        tuple_ab2 = (a2, b2, ab2)

        # constant data variables
        X_sqr = X ** 2
        Y_sqr = Y ** 2
        data_const = (X_sqr, Y_sqr, X, Y, D)
        data_const_T = (X_sqr.T, Y_sqr.T, X.T, Y.T, D)

        # initialise coupling and dual vectors
        if init_n is None:
            pi = a1[:, None] * b1[None, :]  # size n1 x n2
        else:
            pi = init_n

        up, vp = torch.zeros_like(a1), torch.zeros_like(b1)  # shape n1, n2
        ug, vg = torch.zeros_like(a2), torch.zeros_like(b2)  # shape d1, d2

        # initialise log
        log_cost = []
        log_ent_cost = []
        i = 1
        err = self.tol + 1e-3

        while (err > self.tol) and (i <= self.nits):
            pi_prev = pi.detach().clone()

            # Update gamma (feature coupling)
            mp = pi.sum()
            r13 = const1 * mp + const3
            r24 = const2 * mp + const4
            new_eps = mp * eps if reg_mode == "joint" else eps

            Tg = self.compute_local_cost(
                data_const_T, pi, tuple_ab1, hyperparams
            )  # size d1 x d2
            ug, vg, gamma = self.sinkhorn(
                Tg, ug, vg, tuple_ab2, r13, r24, new_eps
            )
            gamma = (mp / gamma.sum()).sqrt() * gamma  # shape d1 x d2

            # Update pi (sample coupling)
            mg = gamma.sum()
            r13 = const1 * mg + const3
            r24 = const2 * mg + const4
            new_eps = mp * eps if reg_mode == "joint" else eps

            Tp = self.compute_local_cost(
                data_const, gamma, tuple_ab2, hyperparams
            )  # size n1 x n2
            up, vp, pi = self.sinkhorn(
                Tp, up, vp, tuple_ab1, r13, r24, new_eps
            )
            pi = (mg / pi.sum()).sqrt() * pi  # shape n1 x n2

            # Update error
            err = (pi - pi_prev).abs().sum().item()
            if log and (i % save_freq == 0):
                cost, ent_cost = self.ucoot_cost(
                    data_const, tuple_ab1, tuple_ab2, pi, gamma, hyperparams
                )

                log_cost.append(cost.item())
                log_ent_cost.append(ent_cost.item())
                if (
                    len(log_ent_cost) >= 2
                    and abs(log_ent_cost[-2] - log_ent_cost[-1])
                    < early_stopping_threshold
                ):
                    break

                if verbose:
                    print("Cost at iteration {}: {}".format(i, cost.item()))

            i += 1

        if pi.isnan().any() or gamma.isnan().any():
            print("There is NaN in coupling")

        if log:
            return pi, gamma, log_cost, log_ent_cost
        else:
            return pi, gamma

    def faster_solver(
        self,
        X,
        Y,
        D=None,
        px=(None, None),
        py=(None, None),
        rho=(1e-1, None, None, None),
        eps=1e-2,
        alpha=1,
        reg_mode="joint",
        init_n=None,
        log=False,
        verbose=False,
        early_stopping_threshold=1e-6,
        save_freq=10,
        eps_step=10,
        init_eps=1,
    ):
        """
        Solver with warm start for small epsilon.
        """

        if eps < init_eps:

            nits = self.nits
            nits_sinkhorn = self.nits_sinkhorn

            self.nits = 10
            self.nits_sinkhorn = 10

            while init_eps > eps:
                init_n, _ = self.solver(
                    X,
                    Y,
                    D,
                    px,
                    py,
                    rho,
                    init_eps,
                    alpha,
                    reg_mode,
                    init_n,
                    log=False,
                )
                init_eps /= eps_step

            self.nits = nits
            self.nits_sinkhorn = nits_sinkhorn

        return self.solver(
            X,
            Y,
            D,
            px,
            py,
            rho,
            eps,
            alpha,
            reg_mode,
            init_n,
            log,
            verbose,
            early_stopping_threshold,
            save_freq,
        )
