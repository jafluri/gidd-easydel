import functools
import typing as tp

import chex
import flax.nnx as nn
import jax
import jax.profiler
import jax.numpy as jnp
import optax
import wandb
from eformer.escale import with_sharding_constraint
from jax.sharding import PartitionSpec

from easydel.infra.base_state import EasyDeLState
from easydel.infra.loss_utils import LossConfig, LossMetrics
from easydel.trainers.training_utils import make_assertions_and_get_sizes, minibatch_call, update_metrics, update_state_respectfully

from .loss import GiddLoss


def compute_loss(loss_fn, state, tree, minibatch) -> tuple[chex.Array, LossMetrics]:
    input_ids = minibatch.get("input_ids", None)
    labels = minibatch.get("labels", None)
    log_snr = minibatch.get("log_snr", None)
    attention_mask = minibatch.get("attention_mask", None)
    noise_mask = minibatch.get("noise_mask", True)

    module = nn.merge(state.graphdef, tree, state.graphother)
    outputs = module(
        input_ids=input_ids,
        inputs_embeds=minibatch.get("inputs_embeds", None),
        attention_mask=attention_mask,
        # log_snr=log_snr,
        # noise_mask=noise_mask,
        output_attentions=True,
    )
    logits = outputs.logits

    # jax.debug.breakpoint()

    # jax.debug.print("Attn entropies: {x}", x=attn_entropies)

    loss_mask = attention_mask & noise_mask

    loss, metrics = loss_fn(
        logits=logits,
        input_ids=input_ids,
        labels=labels,
        log_snr=log_snr,
        return_aux=True,
    )

    for i, (attn, logits) in enumerate(zip(outputs.attentions, outputs.attention_logits)):
        if attn is not None:
            attn_entropy = jnp.mean(jax.scipy.special.entr(attn).sum(-1))
            attn_max = jax.numpy.max(attn)
            attn_max_logit = jax.numpy.max(logits)
            attn_median_logit = jax.numpy.median(logits)
            metrics[f"attn/layer.{i}.attn_entropy"] = attn_entropy
            metrics[f"attn/layer.{i}.attn_max"] = attn_max
            metrics[f"attn/layer.{i}.max_logit"] = attn_max_logit
            metrics[f"attn/layer.{i}.median_logit"] = attn_median_logit

    # Apply mask and compute normalized loss/metrics
    if loss_mask is not True:
        # Mask the loss and all metrics
        masked_loss = loss * loss_mask
        masked_metrics = {k: v * loss_mask for k, v in metrics.items()}
        
        # Compute normalization factor
        mask_sum = loss_mask.sum()
        
        # Normalize loss and metrics by the number of valid tokens
        loss = masked_loss.sum() / jnp.maximum(mask_sum, 1.0)
        metrics = {
            k: v.sum() / jnp.maximum(mask_sum, 1.0)
            for k, v in masked_metrics.items()
        }
        metrics["num_tokens"] = mask_sum
    else:
        # No mask - compute mean directly
        loss = loss.mean()
        metrics = {k: v.mean() for k, v in metrics.items()}
        metrics["num_tokens"] = jnp.prod(jnp.array(loss.shape))

    # # No mask - compute mean directly
    # loss = loss.mean()
    # metrics = {k: v.mean() for k, v in metrics.items()}
    # metrics["num_tokens"] = jnp.prod(jnp.array(loss.shape))

    # import optax
    # loss = optax.softmax_cross_entropy_with_integer_labels(
    #     logits=logits,
    #     labels=labels,
    # ).mean()
    # metrics = {}

    # jax.lax.cond(loss < 6, jax.debug.breakpoint, lambda: None)

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
    import jax
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

        # min_grad_norm = jax.tree_util.tree_reduce(
        #     jnp.minimum,
        #     jax.tree_util.tree_map(jnp.linalg.norm, gradients),
        # )
        # jax.lax.cond(
        #     min_grad_norm < 1e-10,
        #     jax.debug.breakpoint,
        #     lambda: None,
        # )

        # def start_debugger():
        #     import debugpy
        #     debugpy.breakpoint()
        #     # pdb.set_trace()
        #     print("Debugger")
        # # open python debugger inside jax.debug
        # jax.debug.callback(
        #     start_debugger
        # )
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

    # jax.block_until_ready(state)
    # jax.profiler.save_device_memory_profile(f"memory.prof")
    # raise KeyboardInterrupt

    return state, metrics
