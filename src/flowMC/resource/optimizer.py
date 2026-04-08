from flowMC.resource.base import Resource
import optax
import equinox as eqx


class Optimizer(Resource):
    """Adam-W optimiser with gradient clipping for training normalising flow models.

    Attributes:
        optim (optax.GradientTransformation): The composed optax update rule.
        optim_state (optax.OptState): Optimiser state initialised from the model.
    """

    optim: optax.GradientTransformation
    optim_state: optax.OptState

    def __repr__(self):
        return "Optimizer"

    def __init__(
        self,
        model: eqx.Module,
        learning_rate: float = 1e-3,
        momentum: float = 0.9,
    ) -> None:
        """
        Args:
            model (eqx.Module): The model whose trainable parameters will be optimised.
                Used only to initialise the optimiser state.
            learning_rate (float): Adam learning rate. Defaults to 1e-3.
            momentum (float): Adam first-moment decay (``b1``). Defaults to 0.9.
        """
        self.optim = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(learning_rate=learning_rate, b1=momentum),
        )
        self.optim_state = self.optim.init(eqx.filter(model, eqx.is_inexact_array))

    def __call__(self, params, grads):
        raise NotImplementedError

    def print_parameters(self):
        raise NotImplementedError

    def save_resource(self, path: str):
        raise NotImplementedError

    def load_resource(self, path: str):
        raise NotImplementedError
