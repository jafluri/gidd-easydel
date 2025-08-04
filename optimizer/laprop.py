import typing as tp

import chex
import jax
import jax.numpy as jnp
import optax


class ScaleByLapropState(tp.NamedTuple):
    count: chex.Array  # shape=(), dtype=jnp.int32.
    mu: optax.Updates
    nu: optax.Updates


def beta_debias(beta, step):
    return 1 - (1 - beta) / (1 - beta**step)


def scale_by_laprop(
    b1: float = 0.9,
    b2: float = 0.99,
    eps: float = 1e-16,
    eps_root: float = 0.0,
    mu_dtype: tp.Optional[jax.typing.DTypeLike] = None,
) -> optax.GradientTransformation:
    mu_dtype = jax.dtypes.canonicalize_dtype(mu_dtype)

    def init_fn(params):
        mu = optax.tree.zeros_like(params, dtype=mu_dtype)  # First moment
        nu = optax.tree.zeros_like(params)  # Second moment
        return ScaleByLapropState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

    def update_fn(updates, state, params=None):
        del params
        b1_ = beta_debias(b1, state.count)
        b2_ = beta_debias(b2, state.count)

        nu = optax.tree.update_moment_per_elem_norm(updates, state.nu, b2_, 2)
        # mu = optax.tree.update_moment(updates / (jnp.sqrt(nu + eps_root) + eps), state.mu, b1_, 1)
        mu = optax.tree.update_moment(
            jax.tree.map(
                lambda g, n: None if n is None else g / (jnp.sqrt(n + eps_root) + eps),
                updates,
                nu,
                is_leaf=lambda x: x is None,
            ),
            state.mu,
            b1_,
            1,
        )
        count_inc = optax.safe_increment(state.count)

        updates = jax.tree.map(
            lambda m: None if m is None else m,
            mu,
            is_leaf=lambda x: x is None,
        )
        mu = optax.tree.cast(mu, mu_dtype)
        return updates, ScaleByLapropState(count=count_inc, mu=mu, nu=nu)

    return optax.GradientTransformation(init_fn, update_fn)


def laprop(
    learning_rate: optax.ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.99,
    eps: float = 1e-16,
    eps_root: float = 0.0,
    mu_dtype: tp.Optional[tp.Any] = None,
) -> optax.GradientTransformationExtraArgs:
    return optax.chain(
        scale_by_laprop(
            b1=b1,
            b2=b2,
            eps=eps,
            eps_root=eps_root,
            mu_dtype=mu_dtype,
        ),
        optax.scale_by_learning_rate(learning_rate),
    )


def lapropw(
    learning_rate: optax.ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.99,
    eps: float = 1e-16,
    eps_root: float = 0.0,
    mu_dtype: tp.Optional[tp.Any] = None,
    weight_decay: float = 1e-4,
    mask: tp.Optional[tp.Union[tp.Any, tp.Callable[[optax.Params], tp.Any]]] = None,
) -> optax.GradientTransformationExtraArgs:
    return optax.chain(
        scale_by_laprop(
            b1=b1,
            b2=b2,
            eps=eps,
            eps_root=eps_root,
            mu_dtype=mu_dtype,
        ),
        optax.add_decayed_weights(weight_decay, mask),
        optax.scale_by_learning_rate(learning_rate),
    )
