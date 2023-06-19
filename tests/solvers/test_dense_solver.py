from itertools import product

import numpy as np
import pytest
import torch

from fugw.solvers import FUGWSolver


callbacks = [None, lambda x: x["gamma"]]


@pytest.mark.parametrize(
    "solver,callback", product(["sinkhorn", "mm", "ibpp"], callbacks)
)
def test_dense_solvers(solver, callback):
    torch.manual_seed(0)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    ns = 104
    ds = 3
    nt = 151
    dt = 7
    nf = 10

    source_features = torch.rand(ns, nf).to(device)
    target_features = torch.rand(nt, nf).to(device)
    source_embeddings = torch.rand(ns, ds).to(device)
    target_embeddings = torch.rand(nt, dt).to(device)

    F = torch.cdist(source_features, target_features)
    Ds = torch.cdist(source_embeddings, source_embeddings)
    Dt = torch.cdist(target_embeddings, target_embeddings)

    Ds_normalized = Ds / Ds.max()
    Dt_normalized = Dt / Dt.max()
    F_normalized = F / F.max()

    nits_bcd = 100
    eval_bcd = 2
    fugw = FUGWSolver(
        nits_bcd=nits_bcd,
        nits_uot=1000,
        tol_bcd=1e-7,
        tol_uot=1e-7,
        tol_loss=1e-5,
        eval_bcd=eval_bcd,
        eval_uot=10,
        ibpp_eps_base=1e2,
    )

    alpha = 0.8
    rho_s = 2
    rho_t = 3
    eps = 0.02

    res = fugw.solve(
        alpha=alpha,
        rho_s=rho_s,
        rho_t=rho_t,
        eps=eps,
        reg_mode="independent",
        F=F_normalized,
        Ds=Ds_normalized,
        Dt=Dt_normalized,
        init_plan=None,
        solver=solver,
        callback_bcd=callback,
        verbose=True,
    )

    pi = res["pi"]
    gamma = res["gamma"]
    duals_pi = res["duals_pi"]
    duals_gamma = res["duals_gamma"]
    loss = res["loss"]
    loss_steps = res["loss_steps"]
    loss_times = res["loss_times"]

    assert pi.shape == (ns, nt)
    assert gamma.shape == (ns, nt)

    if solver == "mm":
        assert duals_pi is None
        assert duals_gamma is None
    else:
        assert len(duals_pi) == 2
        assert duals_pi[0].shape == (ns,)
        assert duals_pi[1].shape == (nt,)
        assert len(duals_gamma) == 2
        assert duals_gamma[0].shape == (ns,)
        assert duals_gamma[1].shape == (nt,)

    assert len(loss_steps) - 1 <= nits_bcd // eval_bcd + 1
    assert len(loss_times) == len(loss_steps)
    for key in [
        "wasserstein",
        "gromov_wasserstein",
        "marginal_constraint_dim1",
        "marginal_constraint_dim2",
        "regularization",
        "total",
    ]:
        assert len(loss[key]) == len(loss_steps)

    # Check that weighted components sum to total loss
    components = [
        ((1 - alpha), "wasserstein"),
        (alpha, "gromov_wasserstein"),
        (rho_s, "marginal_constraint_dim1"),
        (rho_t, "marginal_constraint_dim2"),
        (eps, "regularization"),
    ]
    np.testing.assert_allclose(
        loss["total"],
        np.sum(
            np.stack([c * np.array(loss[k]) for (c, k) in components]), axis=0
        ),
        rtol=1e-4,
    )

    # Loss should decrease
    assert np.all(
        np.sign(np.array(loss["total"][1:]) - np.array(loss["total"][:-1]))
        == -1
    )


@pytest.mark.parametrize("reg_mode", ["independent", "joint"])
def test_dense_solvers_l2(reg_mode):
    torch.manual_seed(0)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    ns = 204
    ds = 3
    nf = 10

    source_features = torch.rand(ns, nf).to(device)
    target_features = source_features
    source_embeddings = torch.rand(ns, ds).to(device)
    target_embeddings = source_embeddings

    F = torch.cdist(source_features, target_features)
    Ds = torch.cdist(source_embeddings, source_embeddings)
    Dt = torch.cdist(target_embeddings, target_embeddings)

    Ds_normalized = Ds / Ds.max()
    Dt_normalized = Dt / Dt.max()
    F_normalized = F / F.max()

    nits_bcd = 100
    eval_bcd = 2
    fugw = FUGWSolver(
        nits_bcd=nits_bcd,
        nits_uot=1000,
        tol_bcd=1e-7,
        tol_uot=1e-7,
        tol_loss=1e-5,
        eval_bcd=eval_bcd,
        eval_uot=10,
    )

    res = fugw.solve(
        alpha=0.8,
        rho_s=1e4,
        rho_t=1e4,
        eps=0.02,
        divergence="l2",
        reg_mode=reg_mode,
        F=F_normalized,
        Ds=Ds_normalized,
        Dt=Dt_normalized,
        solver="mm",
        verbose=True,
    )

    pi = res["pi"]
    gamma = res["gamma"]
    duals_pi = res["duals_pi"]
    duals_gamma = res["duals_gamma"]
    loss = res["loss"]
    loss_steps = res["loss_steps"]
    loss_times = res["loss_times"]

    assert pi.shape == (ns, ns)
    assert gamma.shape == (ns, ns)

    assert duals_pi is None
    assert duals_gamma is None

    assert len(loss_steps) - 1 <= nits_bcd // eval_bcd + 1
    assert len(loss_times) == len(loss_steps)
    for key in [
        "wasserstein",
        "gromov_wasserstein",
        "marginal_constraint_dim1",
        "marginal_constraint_dim2",
        "regularization",
        "total",
    ]:
        assert len(loss[key]) == len(loss_steps)
    # Loss should decrease
    # assert np.all(np.sign(np.array(loss[1:]) - np.array(loss[:-1])) == -1)

    # Check if we can recover ground truth optimal plan (identity matrix)
    pi_true = np.eye(ns, ns) / ns
    pi_np = pi.cpu().detach().numpy()
    gamma_np = gamma.cpu().detach().numpy()
    np.testing.assert_allclose(pi_true, pi_np, atol=1e-04)
    np.testing.assert_allclose(pi_true, gamma_np, atol=1e-04)


@pytest.mark.parametrize(
    "validation", ["None", "features", "geometries", "Both"]
)
def test_validation_solver(validation):
    torch.manual_seed(0)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    ns = 104
    ds = 3
    nt = 151
    dt = 7
    nf = 10

    source_features = torch.rand(ns, nf).to(device)
    target_features = torch.rand(nt, nf).to(device)
    source_embeddings = torch.rand(ns, ds).to(device)
    target_embeddings = torch.rand(nt, dt).to(device)

    F = torch.cdist(source_features, target_features)
    Ds = torch.cdist(source_embeddings, source_embeddings)
    Dt = torch.cdist(target_embeddings, target_embeddings)

    Ds_normalized = Ds / Ds.max()
    Dt_normalized = Dt / Dt.max()
    F_normalized = F / F.max()

    if validation == "None":
        F_val = None
        Ds_val = None
        Dt_val = None

    elif validation == "features":
        source_features_val = torch.rand(ns, nf).to(device)
        target_features_val = torch.rand(nt, nf).to(device)
        F_val = torch.cdist(source_features_val, target_features_val)
        Ds_val = None
        Dt_val = None

    elif validation == "geometries":
        source_embeddings_val = torch.rand(ns, ds).to(device)
        target_embeddings_val = torch.rand(nt, dt).to(device)
        F_val = None
        Ds_val = torch.cdist(source_embeddings_val, source_embeddings_val)
        Dt_val = torch.cdist(target_embeddings_val, target_embeddings_val)

    elif validation == "Both":
        source_features_val = torch.rand(ns, nf).to(device)
        target_features_val = torch.rand(nt, nf).to(device)
        F_val = torch.cdist(source_features_val, target_features_val)
        source_embeddings_val = torch.rand(ns, ds).to(device)
        target_embeddings_val = torch.rand(nt, dt).to(device)
        Ds_val = torch.cdist(source_embeddings_val, source_embeddings_val)
        Dt_val = torch.cdist(target_embeddings_val, target_embeddings_val)

    nits_bcd = 100
    eval_bcd = 2
    fugw = FUGWSolver(
        nits_bcd=nits_bcd,
        nits_uot=1000,
        tol_bcd=1e-7,
        tol_uot=1e-7,
        tol_loss=1e-5,
        eval_bcd=eval_bcd,
        eval_uot=10,
        ibpp_eps_base=1e2,
    )

    res = fugw.solve(
        alpha=0.8,
        rho_s=2,
        rho_t=3,
        eps=0.02,
        reg_mode="independent",
        F=F_normalized,
        Ds=Ds_normalized,
        Dt=Dt_normalized,
        F_val=F_val,
        Ds_val=Ds_val,
        Dt_val=Dt_val,
        init_plan=None,
        solver="sinkhorn",
        verbose=True,
    )

    loss = res["loss"]
    loss_steps = res["loss_steps"]
    loss_val = res["loss_val"]

    if validation == "None":
        assert loss_val["total"] == loss["total"]

    for key in [
        "wasserstein",
        "gromov_wasserstein",
        "marginal_constraint_dim1",
        "marginal_constraint_dim2",
        "regularization",
        "total",
    ]:
        assert len(loss_val[key]) == len(loss_steps)


def test_convergence_criteria_existence():
    with pytest.raises(
        ValueError, match="At least one of .* must be provided"
    ):
        FUGWSolver(
            nits_bcd=None,
            tol_bcd=None,
            tol_loss=None,
            nits_uot=1,
            tol_uot=1,
        )

    with pytest.raises(
        ValueError, match="At least one of .* must be provided"
    ):
        FUGWSolver(
            nits_bcd=1,
            tol_bcd=1,
            tol_loss=1,
            nits_uot=None,
            tol_uot=None,
        )

    try:
        FUGWSolver(
            nits_bcd=1,
            tol_bcd=None,
            tol_loss=None,
            nits_uot=1,
            tol_uot=None,
        )
    except Exception as e:
        assert False, f"Solver should not raise exception {e}"

    try:
        FUGWSolver(
            nits_bcd=None,
            tol_bcd=1,
            tol_loss=None,
            nits_uot=None,
            tol_uot=1,
        )
    except Exception as e:
        assert False, f"Solver should not raise exception {e}"
