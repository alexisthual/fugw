from itertools import product

import numpy as np
import pytest
import torch

from fugw.solvers import FUGWSparseSolver
from fugw.utils import _low_rank_squared_l2


devices = [torch.device("cpu")]
if torch.cuda.is_available():
    devices.append(torch.device("cuda:0"))

callbacks = [None, lambda x: x["gamma"]]


# TODO: need to test sinkhorn
@pytest.mark.parametrize(
    "solver,device,callback",
    product(["sinkhorn", "mm", "ibpp"], devices, callbacks),
)
def test_sparse_solvers(solver, device, callback):
    torch.manual_seed(1)
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

    F = _low_rank_squared_l2(source_features, target_features)
    Ds = _low_rank_squared_l2(source_embeddings, source_embeddings)
    Dt = _low_rank_squared_l2(target_embeddings, target_embeddings)

    F_norm = (F[0] @ F[1].T).max()
    Ds_norm = (Ds[0] @ Ds[1].T).max()
    Dt_norm = (Dt[0] @ Dt[1].T).max()

    F_normalized = (F[0] / F_norm, F[1] / F_norm)
    Ds_normalized = (Ds[0] / Ds_norm, Ds[1] / Ds_norm)
    Dt_normalized = (Dt[0] / Dt_norm, Dt[1] / Dt_norm)

    init_plan = (
        (torch.ones(ns, nt) / ns).to_sparse_coo().to(device).to_sparse_csr()
    )

    nits_bcd = 100
    eval_bcd = 1
    fugw = FUGWSparseSolver(
        nits_bcd=nits_bcd,
        nits_uot=1000,
        tol_bcd=1e-7,
        tol_uot=1e-7,
        tol_loss=1e-5,
        eval_bcd=eval_bcd,
        eval_uot=10,
        # Set a high value of ibpp, otherwise nans appear in coupling.
        # This will generally increase the computed fugw loss.
        ibpp_eps_base=1e2,
    )

    alpha = 0.8
    rho_s = 2
    rho_t = 3
    eps = 0.5

    res = fugw.solve(
        alpha=alpha,
        rho_s=rho_s,
        rho_t=rho_t,
        eps=eps,
        reg_mode="independent",
        F=F_normalized,
        Ds=Ds_normalized,
        Dt=Dt_normalized,
        init_plan=init_plan,
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

    assert pi.size() == (ns, nt)
    assert gamma.size() == (ns, nt)

    if solver == "mm":
        assert duals_pi is None
        assert duals_gamma is None
    elif solver == "ibpp":
        assert len(duals_pi) == 2
        assert duals_pi[0].shape == (ns,)
        assert duals_pi[1].shape == (nt,)

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


@pytest.mark.parametrize(
    "validation,device",
    product(["None", "features", "geometries", "Both"], devices),
)
def test_sparse_validation_solver(validation, device):
    torch.manual_seed(1)
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
    source_features_val = torch.rand(ns, nf).to(device)
    target_features_val = torch.rand(nt, nf).to(device)
    source_embeddings_val = torch.rand(ns, ds).to(device)
    target_embeddings_val = torch.rand(nt, dt).to(device)

    F = _low_rank_squared_l2(source_features, target_features)
    Ds = _low_rank_squared_l2(source_embeddings, source_embeddings)
    Dt = _low_rank_squared_l2(target_embeddings, target_embeddings)

    F_norm = (F[0] @ F[1].T).max()
    Ds_norm = (Ds[0] @ Ds[1].T).max()
    Dt_norm = (Dt[0] @ Dt[1].T).max()

    F_normalized = (F[0] / F_norm, F[1] / F_norm)
    Ds_normalized = (Ds[0] / Ds_norm, Ds[1] / Ds_norm)
    Dt_normalized = (Dt[0] / Dt_norm, Dt[1] / Dt_norm)

    if validation == "None":
        F_val_normalized = None, None
        Ds_val_normalized = None, None
        Dt_val_normalized = None, None

    elif validation == "features":
        F_val = _low_rank_squared_l2(source_features_val, target_features_val)
        F_norm_val = (F_val[0] @ F_val[1].T).max()
        F_val_normalized = (F_val[0] / F_norm, F_val[1] / F_norm_val)
        Ds_val_normalized = None, None
        Dt_val_normalized = None, None

    elif validation == "geometries":
        F_val_normalized = None, None
        Ds_val = _low_rank_squared_l2(
            source_embeddings_val, source_embeddings_val
        )
        Dt_val = _low_rank_squared_l2(
            target_embeddings_val, target_embeddings_val
        )
        Ds_norm_val = (Ds_val[0] @ Ds_val[1].T).max()
        Dt_norm_val = (Dt_val[0] @ Dt_val[1].T).max()
        Ds_val_normalized = (Ds_val[0] / Ds_norm_val, Ds_val[1] / Ds_norm_val)
        Dt_val_normalized = (Dt_val[0] / Dt_norm_val, Dt_val[1] / Dt_norm_val)

    elif validation == "Both":
        F_val = _low_rank_squared_l2(source_features_val, target_features_val)
        F_norm_val = (F_val[0] @ F_val[1].T).max()
        F_val_normalized = (F_val[0] / F_norm, F_val[1] / F_norm_val)
        Ds_val = _low_rank_squared_l2(
            source_embeddings_val, source_embeddings_val
        )
        Dt_val = _low_rank_squared_l2(
            target_embeddings_val, target_embeddings_val
        )
        Ds_norm_val = (Ds_val[0] @ Ds_val[1].T).max()
        Dt_norm_val = (Dt_val[0] @ Dt_val[1].T).max()
        Ds_val_normalized = (Ds_val[0] / Ds_norm_val, Ds_val[1] / Ds_norm_val)
        Dt_val_normalized = (Dt_val[0] / Dt_norm_val, Dt_val[1] / Dt_norm_val)

    init_plan = (
        (torch.ones(ns, nt) / ns).to_sparse_coo().to(device).to_sparse_csr()
    )

    nits_bcd = 10
    eval_bcd = 1
    fugw = FUGWSparseSolver(
        nits_bcd=nits_bcd,
        nits_uot=1000,
        tol_bcd=1e-7,
        tol_uot=1e-7,
        tol_loss=1e-5,
        eval_bcd=eval_bcd,
        eval_uot=10,
        # Set a high value of ibpp, otherwise nans appear in coupling.
        # This will generally increase the computed fugw loss.
        ibpp_eps_base=1e2,
    )

    res = fugw.solve(
        alpha=0.2,
        rho_s=2,
        rho_t=3,
        eps=0.02,
        reg_mode="independent",
        F=F_normalized,
        Ds=Ds_normalized,
        Dt=Dt_normalized,
        F_val=F_val_normalized,
        Ds_val=Ds_val_normalized,
        Dt_val=Dt_val_normalized,
        init_plan=init_plan,
        solver="mm",
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
        FUGWSparseSolver(
            nits_bcd=None,
            tol_bcd=None,
            tol_loss=None,
            nits_uot=1,
            tol_uot=1,
        )

    with pytest.raises(
        ValueError, match="At least one of .* must be provided"
    ):
        FUGWSparseSolver(
            nits_bcd=1,
            tol_bcd=1,
            tol_loss=1,
            nits_uot=None,
            tol_uot=None,
        )

    try:
        FUGWSparseSolver(
            nits_bcd=1,
            tol_bcd=None,
            tol_loss=None,
            nits_uot=1,
            tol_uot=None,
        )
    except Exception as e:
        assert False, f"Solver should not raise exception {e}"

    try:
        FUGWSparseSolver(
            nits_bcd=None,
            tol_bcd=1,
            tol_loss=None,
            nits_uot=None,
            tol_uot=1,
        )
    except Exception as e:
        assert False, f"Solver should not raise exception {e}"
