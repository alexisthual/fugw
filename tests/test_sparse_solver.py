from itertools import product

import pytest
import torch

from fugw.solvers.sparse import FUGWSparseSolver
from fugw.utils import low_rank_squared_l2


devices = [torch.device("cpu")]
if torch.cuda.is_available():
    devices.append(torch.device("cuda:0"))


@pytest.mark.parametrize(
    "uot_solver,device",
    product(["mm", "dc"], devices),
)
def test_solvers(uot_solver, device):
    torch.manual_seed(0)
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

    F = low_rank_squared_l2(source_features, target_features)
    Ds = low_rank_squared_l2(source_embeddings, source_embeddings)
    Dt = low_rank_squared_l2(target_embeddings, target_embeddings)

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
        early_stopping_threshold=1e-5,
        eval_bcd=eval_bcd,
        eval_uot=10,
        # Set a high value of dc, otherwise nans appear in coupling.
        # This will generally increase the computed fugw loss.
        dc_eps_base=1e2,
    )

    pi, gamma, duals_pi, duals_gamma, loss_steps, loss, loss_ent = fugw.solve(
        alpha=0.2,
        rho_s=2,
        rho_t=3,
        eps=0.02,
        reg_mode="independent",
        F=F_normalized,
        Ds=Ds_normalized,
        Dt=Dt_normalized,
        init_plan=init_plan,
        uot_solver=uot_solver,
        verbose=True,
    )

    assert pi.size() == (ns, nt)
    assert gamma.size() == (ns, nt)

    if uot_solver == "mm":
        assert duals_pi is None
        assert duals_gamma is None
    elif uot_solver == "dc":
        assert len(duals_pi) == 2
        assert duals_pi[0].shape == (ns,)
        assert duals_pi[1].shape == (nt,)

    assert len(loss_steps) <= nits_bcd // eval_bcd + 1
    assert len(loss_steps) == len(loss)
    assert len(loss) == len(loss_ent)
