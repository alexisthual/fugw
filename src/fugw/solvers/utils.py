import torch
# require torch >= 1.9

def solver_scaling(cost, init_duals, params, tuple_ab, train_params):
    """
    Scaling algo    
    """

    a, b, ab = tuple_ab
    u, v = init_duals
    rho1, rho2, eps = params
    niters, tol, eval_freq = train_params

    tau1 = 1 if torch.isinf(rho1) else rho1 / (rho1 + eps)
    tau2 = 1 if torch.isinf(rho2) else rho2 / (rho2 + eps)

    for idx in range(niters):
        u_prev = u.detach().clone()
        if rho2 == 0:
            v = torch.zeros_like(v)
        else:
            v = -tau2 * ((u + a.log())[:, None] - cost / eps).logsumexp(dim=0)

        if rho1 == 0:
            u = torch.zeros_like(u)
        else:
            u = -tau1 * ((v + b.log())[None, :] - cost / eps).logsumexp(dim=1)

        if idx % eval_freq == 0 and (u - u_prev).abs().max().item() < tol:
            break

    pi = ab * (u[:, None] + v[None, :] - cost / eps).exp()

    return (u, v), pi

def solver_mm(cost, init_pi, params, tuple_ab, train_params):
    """
    Solve (entropic) UOT using the max-min algorithm.

    Allow epsilon to be 0 but rho_x and rho_y can't be infinity.

    Note that if the parameters are small so that numerically, the exponential of 
    negative cost will contain zeros and this serves as sparsification of the optimal plan. 

    If the parameters are large, then the resulting optimal plan is more dense than the one 
    obtained from Sinkhorn algo. 
    But all parameters should not be too small, otherwise the kernel will contain too many zeros 
    and consequently, the optimal plan will contain NaN (because the Kronecker sum of two marginals 
    will eventually contain zeros, and divided by zero will result in undesirable result).
    """

    niters, tol, eval_freq = train_params
    a, b = tuple_ab
    rho1, rho2, eps = params

    sum_param = rho1 + rho2 + eps
    tau1, tau2, reg = rho1 / sum_param, rho2 / sum_param, eps / sum_param
    K = a[:,None]**(tau1 + reg) * b[None,:]**(tau2 + reg) * (- cost / sum_param).exp()

    m1, m2, pi = init_pi.sum(1), init_pi.sum(0), init_pi

    for idx in range(niters):
        m1_old, m2_old = m1.detach().clone(), m2.detach().clone()
        pi = pi**(tau1 + tau2) / (m1[:,None]**tau1 * m2[None,:]**tau2) * K
        m1, m2 = pi.sum(1), pi.sum(0)

        if (idx % eval_freq == 0) and \
            max((m1 - m1_old).abs().max(), (m2 - m2_old).abs().max()) < tol:
            break
    
    return pi

def compute_approx_kl(p, q):
    # By convention: 0 log 0 = 0
    entropy = torch.nan_to_num(p * (p / q).log(), nan=0.0, posinf=0.0, neginf=0.0).sum()
    return entropy

def compute_kl(p, q):
    return compute_approx_kl(p, q) - p.sum() + q.sum()

def compute_quad_kl(mu, nu, alpha, beta):
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
    kl = m_nu * compute_kl(mu, alpha) + m_mu * compute_kl(nu, beta) + const

    return kl