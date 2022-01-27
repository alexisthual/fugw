import numpy as np
from sklearn.metrics import r2_score
import torch


from fugw.solvers.fugw import FUGWSolver
from fugw.utils import BaseModel


class FUGW(BaseModel):
    def __init__(
        self,
        rho=(10, 10, 1, 1),
        eps=1e-4,
        alpha=0.95,
        mode="independent",
        **kwargs,
    ):
        # Save model arguments
        self.rho = rho
        self.eps = eps
        self.alpha = alpha
        self.mode = mode

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
        model = FUGWSolver(
            nits=3, nits_sinkhorn=1000, tol=1e-7, tol_sinkhorn=1e-7, **kwargs
        )

        # Compute transport plan
        pi, gamma, _, _ = model.solver(
            Gs,
            Gt,
            K,
            rho=self.rho,
            eps=self.eps,
            alpha=self.alpha,
            reg_mode=self.mode,
            log=True,
            verbose=True,
            save_freq=1,
        )

        self.pi = pi.detach().cpu().numpy()

        # Free allocated GPU memory
        del Fs, Ft, K, Gs, Gt, gamma
        if use_cuda:
            torch.cuda.empty_cache()

        return self

    def transform(self, source_data):
        """
        Transport source contrast map using fitted OT plan.

        Parameters
        ----------
        source_data: ndarray(n_samples, n1)
            Contrast map for source subject

        Returns
        -------
        transported_data: ndarray(n_samples, n2)
            Contrast map transported in target subject's space
        """
        if self.pi is None:
            raise ("Model should be fitted before calling transform")

        # source = torch.from_numpy(source_data.T).type(self.dtype)
        transported_data = (self.pi.shape[1] * self.pi.T @ source_data.T).T

        # Normalized computed contrast map
        if transported_data.ndim == 1:
            transported_data = transported_data / np.linalg.norm(
                transported_data
            )
        elif transported_data.ndim == 2:
            transported_data = (
                transported_data
                / np.linalg.norm(transported_data, axis=1)[:, None]
            )
        else:
            raise ValueError(
                f"source_data has too many dimensions: {source_data.ndims}"
            )

        return transported_data

    def score(self, source_data, target_data):
        """
        Transport source contrast maps using fitted OT plan
        and compute R^2 score with actual target contrast maps.

        Parameters
        ----------
        source_data: ndarray(n_samples, n1)
            Contrast maps for source subject
        target_data: ndarray(n_samples, n2)
            Contrast maps for target subject

        Returns
        -------
        score: float
            R^2 of self.predict(source_data) with target_data
        """

        transported_data = self.transform(source_data)
        score = r2_score(transported_data, target_data)

        return score
