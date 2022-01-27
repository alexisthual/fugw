class BatchedFUGW(BaseModel):
    def __init__(
        self,
        rho=(10, 10, 1, 1),
        eps=1e-4,
        alpha=0.95,
        mode="independent",
        **kwargs,
    ):
        # Check cuda availability
        torch.cuda.is_available()
        torch.cuda.device_count()
        use_cuda = torch.cuda.is_available()
        self.dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

        # Save model arguments
        self.rho = rho
        self.eps = eps
        self.alpha = alpha
        self.mode = mode

        spec = importlib.util.spec_from_file_location(
            "module.name",
            "/storage/store2/work/athual/repo/UCOOT/UCOOT/fugw.py",
        )
        fugw = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fugw)
        self.model = fugw.FUGW(
            nits=20, nits_sinkhorn=1000, tol=1e-7, tol_sinkhorn=1e-7, **kwargs
        )

    def fit(self, source_data, target_data, source_kernel, target_kernel):
        Fs = torch.from_numpy(source_data).type(self.dtype)
        Ft = torch.from_numpy(target_data).type(self.dtype)

        As = torch.from_numpy(source_kernel).type(self.dtype)
        As_normalized = As / As.max()
        At = torch.from_numpy(target_kernel).type(self.dtype)
        At_normalized = At / At.max()

        pi, gamma, log_cost, log_ent_cost = self.fugw_batch.solver(
            X=Fs,
            Y=Ft,
            D=(As_normalized, At_normalized),
            rho=self.rho,
            eps=self.eps,
            alpha=self.alpha,
            reg_mode=self.mode,
            log=True,
            verbose=True,
            save_freq=1,
        )

        return pi

    def predict(self, source_data):
        pass
