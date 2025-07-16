import typing as tp

import chex
import flax
import flax.struct
import jax
from eformer.pytree import auto_pytree

from easydel.infra.loss_utils import LossMetrics


@auto_pytree
class DiffusionLossMetrics(LossMetrics):
    loss: float | chex.Array | None = None
    z_loss: float | chex.Array | None = None
    weight_sum: float | chex.Array | None = None
    accuracy: float | chex.Array | None = None
    learning_rate: float | chex.Array | None = None
    max_grad_norm: flax.struct.PyTreeNode | None = None
    mean_grad_norm: flax.struct.PyTreeNode | None = None
    grad_norms: flax.struct.PyTreeNode | None = None
    chosen_rewards: jax.Array | None = None
    rejected_rewards: jax.Array | None = None
    other_metrics: tp.Mapping[str, jax.Array] | None = None
    execution_time: float | None = None