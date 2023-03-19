class BaseMapping:
    def __init__(
        self,
        alpha=0.5,
        rho=1,
        eps=1e-2,
        reg_mode="joint",
        divergence="KL",
    ):
        """Init FUGW problem.

        Parameters
        ----------
        alpha: float, optional, defaults to 0.5
            Value in ]0, 1[, interpolates the relative importance of
            the Wasserstein and the Gromov-Wasserstein losses
            in the FUGW loss (see equation)
        rho: float or tuple of 2 floats, optional, defaults to 1
            Value in ]0, +inf[, controls the relative importance of
            the marginal constraints. High values force the mass of
            each point to be transported ;
            low values allow for some mass loss
        eps: float, optional, defaults to 1e-2
            Value in ]0, +inf[, controls the relative importance of
            the entropy loss
        reg_mode: "joint" or "independent", optional, defaults to "joint"
            "joint": use unbalanced-GW-like regularisation term
            "independent": use unbalanced-W-like regularisation term
        divergence: "KL" or "L2", optional, defaults to "KL"
        """

        self.alpha = alpha
        self.rho = rho
        self.eps = eps
        self.reg_mode = reg_mode
        self.divergence = divergence

        self.pi = None

        self.loss_steps = []
        self.loss = []
        self.loss_entropic = []
        self.loss_times = []

    def fit(self):
        return None

    def inverse_transform(self, target_data):
        return None

    def transform(self, source_data):
        return None
