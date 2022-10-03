import numpy as np
from scipy.stats import pearsonr
import torch


from fugw.solvers.fugw import FUGWSolver
from fugw.utils import BaseModel


class FUGW(BaseModel):
    def __init__(
        self,
        px=None,
        py=None,
        alpha=1,
        rho=20, 
        eps=1e-2,
        uot_solver="sinkhorn",
        reg_mode="joint",
        init_plan=None,
        init_duals=None,
        verbose=False,
        early_stopping_threshold=1e-6,
        **kwargs,
    ):
        # Save model arguments
        self.px = px
        self.py = py
        self.alpha = alpha
        self.rho = rho
        self.eps = eps
        self.uot_solver = uot_solver
        self.reg_mode = reg_mode
        self.init_plan = init_plan
        self.init_duals = init_duals
        self.verbose = verbose
        self.early_stopping_threshold = early_stopping_threshold

    def fit(
        self,
        source_data,
        target_data,
        source_kernel=None,
        target_kernel=None,
        **kwargs,
    ):
        """
        Compute transport plan between source and target individuals
        using functional contrast maps and anatomies.

        Parameters
        ----------
        source_data: ndarray(n_samples, n1)
            Contrast maps for source subject.
            n_samples is the number of contrast maps, it should
            be the same for source and target data.
            n1 is the number of nodes on the source graph, it
            can be different from n2, the number of nodes on the
            target graph.
        target_data: ndarray(n_samples, n2)
            Contrast maps for target subject
        source_kernel: ndarray(n1, n1)
            Kernel matrix representing anatomical similarities
            between nodes of source subject mesh
        target_kernel: ndarray(n2, n2)
            Kernel matrix representing anatomical similarities
            between nodes of target subject mesh

        Returns
        -------
        self: FUGW class object
        """
        # Check cuda availability
        use_cuda = torch.cuda.is_available()
        dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

        if isinstance(self.rho, float) or isinstance(self.rho, int):
            rho_x, rho_y = self.rho
        elif isinstance(self.rho, tuple) and len(self.rho) == 2:
            rho_x, rho_y = self.rho
        else:
            raise ValueError("Invalid value of rho. Must be either a scalar or a tuple of two scalars.")

        Fs = torch.from_numpy(source_data.T).type(dtype)
        Ft = torch.from_numpy(target_data.T).type(dtype)
        K = torch.cdist(Fs, Ft, p=2)

        # Load anatomical kernels to GPU
        # and normalize them
        Gs = torch.from_numpy(source_kernel).type(dtype)
        Gs = Gs / Gs.max()
        Gt = torch.from_numpy(target_kernel).type(dtype)
        Gt = Gt / Gt.max()

        # Create model
        model = FUGWSolver(**kwargs)

        # Compute transport plan
        res = model.solver(
            X=Gs,
            Y=Gt,
            px=self.px,
            py=self.py,
            D=K,
            alpha=self.alpha,
            rho_x=rho_x,
            rho_y=rho_y,
            eps=self.eps,
            uot_solver=self.uot_solver,
            reg_mode=self.reg_mode,
            init_plan=self.init_plan,
            init_duals=self.init_duals,
            log=False,
            verbose=self.verbose,
            early_stopping_threshold=self.early_stopping_threshold,
        )

        self.pi = res[0].detach().cpu().numpy()

        # Free allocated GPU memory
        del Fs, Ft, K, Gs, Gt
        if use_cuda:
            torch.cuda.empty_cache()

        return self

    def transform(self, source_data):
        """
        Transport source contrast map using fitted OT plan.
        Use GPUs if available.

        Parameters
        ----------
        source_data: ndarray(n_samples, n1) or ndarray(n1)
            Contrast map for source subject

        Returns
        -------
        transported_data: ndarray(n_samples, n2) or ndarray(n2)
            Contrast map transported in target subject's space
        """
        # Check cuda availability
        use_cuda = torch.cuda.is_available()
        dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

        if self.pi is None:
            raise ("Model should be fitted before calling transform")

        is_one_dimensional = False
        if source_data.ndim == 1:
            is_one_dimensional = True
            source_data = source_data.reshape(1, -1)
        if source_data.ndim > 2:
            raise ValueError(
                "source_data has too many dimensions: " f"{source_data.ndim}"
            )

        # Move data to GPU
        pi_torch = torch.from_numpy(self.pi).type(dtype)
        source_data_torch = torch.from_numpy(source_data).type(dtype)

        # Transform data
        transformed_data_torch = (
            pi_torch.T
            @ source_data_torch.T
            / pi_torch.sum(dim=0).reshape(-1, 1)
        ).T

        # Move transformed data back to CPU
        transformed_data = transformed_data_torch.detach().cpu().numpy()

        # Free allocated GPU memory
        del pi_torch, source_data_torch
        if use_cuda:
            torch.cuda.empty_cache()

        if transformed_data.ndim > 1 and is_one_dimensional:
            transformed_data = transformed_data.flatten()

        return transformed_data

    def score(self, source_data, target_data):
        """
        Transport source contrast maps using fitted OT plan
        and compute correlation with actual target contrast maps.

        Parameters
        ----------
        source_data: ndarray(n_samples, n1)
            Contrast maps for source subject
        target_data: ndarray(n_samples, n2)
            Contrast maps for target subject

        Returns
        -------
        score: float
            Mean correlation across features of self.transform(source_data)
            and target_data
        """

        transported_data = self.transform(source_data)

        score = np.mean(
            [
                pearsonr(transported_data[i, :], target_data[i, :])[0]
                for i in range(transported_data.shape[0])
            ]
        )

        return score
