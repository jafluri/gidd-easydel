import functools
import typing as tp

import chex
import flax.nnx as nn
import jax
import optax
from eformer.escale import with_sharding_constraint
from jax.sharding import PartitionSpec

from easydel.infra.base_state import EasyDeLState
from easydel.infra.loss_utils import LossConfig, LossMetrics
from easydel.trainers.training_utils import make_assertions_and_get_sizes, minibatch_call, update_metrics, update_state_respectfully

from .loss import GiddLoss


def compute_loss(loss_fn, state, tree, minibatch) -> tuple[chex.Array, DiffusionLossMetrics]:
        input_ids = minibatch.get("input_ids", None)
        labels = minibatch.get("labels", None)
        log_snr = minibatch.get("log_snr", None)
        attention_mask = minibatch.get("attention_mask", None)
        noise_mask = minibatch.get("noise_mask", True)

        module = nn.merge(state.graphdef, tree, state.graphother)
        outputs = module(
            input_ids=input_ids,
            input_embeds=minibatch.get("input_embeds", None),
            log_snr=log_snr,
            attention_mask=attention_mask,
        )
        logits = outputs.logits

        attention_mask = minibatch.get("attention_mask", True)
        loss_mask = attention_mask & noise_mask
        if type(loss_mask) is not chex.Array:
            loss_mask = None

        loss, metrics = loss_fn(
            logits=logits,
            input_ids=input_ids,
            labels=labels,
            log_snr=log_snr,
            return_aux=True,
        )
        return loss, LossMetrics(
            loss=loss,
            other_metrics=metrics,
        )

def training_step(
    state: EasyDeLState,
    batch: tp.Mapping[str, jax.Array],
    loss_fn: GiddLoss,
    loss_config: LossConfig | None = None,
    learning_rate_fn: optax.Schedule = None,
    partition_spec: PartitionSpec | None = None,
    gradient_accumulation_steps: int = 1,
    is_training: bool = True,
) -> tuple[EasyDeLState, LossMetrics]:
    # Determine batch size, minibatch size, and enforce partition spec.
    batch_size, minibatch_size, partition_spec = make_assertions_and_get_sizes(
        batch=batch,
        gradient_accumulation_steps=gradient_accumulation_steps,
        batch_partition_spec=partition_spec,
    )
    batch = with_sharding_constraint(arr=batch, sharding=partition_spec)

    _compute_loss = functools.partial(compute_loss, loss_fn, state)

    if is_training:
        # Compute gradients and metrics across minibatches.
        gradients, metrics = minibatch_call(
            state=state,
            batch=batch,
            minibatch_size=minibatch_size,
            grad_fn=jax.value_and_grad(_compute_loss, has_aux=True),
        )
        # Update state using the computed gradients and updated metrics.
        state = update_state_respectfully(
            state=state,
            gradients=gradients,
            loss_config=loss_config,
            metrics=update_metrics(
                metrics=metrics,
                learning_rate_fn=learning_rate_fn,
                step=state.step,
                gradients=gradients,
            ),
        )
    else:
         _, metrics = _compute_loss(state, batch)

    return state, metrics

