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
    torch.manual_seed(100)
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

    Gs = low_rank_squared_l2(source_embeddings, source_embeddings)
    Gt = low_rank_squared_l2(target_embeddings, target_embeddings)
    K = low_rank_squared_l2(source_features, target_features)

    Gs_norm = (Gs[0] @ Gs[1].T).max()
    Gt_norm = (Gt[0] @ Gt[1].T).max()
    K_norm = (K[0] @ K[1].T).max()

    Gs_normalized = (Gs[0] / Gs_norm, Gs[1] / Gs_norm)
    Gt_normalized = (Gt[0] / Gt_norm, Gt[1] / Gt_norm)
    K_normalized = (K[0] / K_norm, K[1] / K_norm)

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
        eval_bcd=eval_bcd,
        eval_uot=10,
    )

    pi, gamma, duals_pi, duals_gamma, loss_steps, loss, loss_ent = fugw.solver(
        Gs=Gs_normalized,
        Gt=Gt_normalized,
        K=K_normalized,
        alpha=0.2,
        rho_x=2,
        rho_y=3,
        eps=0.02,
        uot_solver=uot_solver,
        reg_mode="independent",
        init_plan=init_plan,
        verbose=True,
        early_stopping_threshold=1e-5,
        # Set a high value of dc, otherwise nans appear in coupling.
        # This will generally increase the computed fugw loss.
        dc_eps_base=1e2,
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
