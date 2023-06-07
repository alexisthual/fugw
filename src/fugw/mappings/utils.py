from copy import deepcopy

from fugw.utils import console


class BaseMapping:
    def __init__(
        self,
        alpha=0.5,
        rho=1,
        eps=1e-2,
        reg_mode="joint",
        divergence="kl",
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
        divergence: string, optional
            What divergence to use for the marginal contraints
            and regularization. Can be "kl" or "l2".
            Defaults to "kl".

        Attributes
        ----------
        alpha: float
        rho: float
        eps: float
        reg_mode: "joint" or "independent"
        pi: numpy.ndarray or None
            Transport plan computed with ``.fit()``
        loss: dict of lists
            Dictionary containing the training loss and its unweighted
            components for each step of the block-coordinate-descent
            for which the FUGW loss was evaluated.
            Keys are: "wasserstein", "gromov_wasserstein",
            "marginal_constraint_dim1", "marginal_constraint_dim2",
            "regularization", "total".
            Values are float or None.
        loss_val: dict of lists
            Dictionary containing the validation loss and its unweighted
            components for each step of the block-coordinate-descent
            for which the FUGW loss was evaluated.
            Values are float or None.
        loss_steps: list
            BCD steps at the end of which the FUGW loss was evaluated
        loss_times: list
            Elapsed time at the end of each BCD step for which the
            FUGW loss was evaluated.
        """

        self.alpha = alpha
        self.rho = rho
        self.eps = eps
        self.reg_mode = reg_mode
        self.divergence = divergence

        self.pi = None

        self.loss = []
        self.loss_val = []
        self.loss_steps = []
        self.loss_times = []

    def fit(self):
        return None

    def inverse_transform(self, target_data):
        return None

    def transform(self, source_data):
        return None

    def __getstate__(self):
        console.log(
            "FUGW customizes pickle dumps to separate hyperparams and "
            "weights. Please check the documentation."
        )
        state = deepcopy(self.__dict__)
        state["pi"] = None
        return state

    def __setstate__(self, state):
        console.log(
            "FUGW customizes pickle dumps to separate hyperparams and "
            "weights. Please check the documentation."
        )
        self.__dict__.update(state)
        self.pi = None
