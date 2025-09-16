import chex
import jax
import jax.numpy as jnp
import flax.nnx as nn
import optax
from jax.sharding import PartitionSpec
from eformer.escale import PartitionAxis
from eformer.escale.partition.constraints import with_sharding_constraint

from .schedule import MixingSchedule, safe_sigmoid



class GiddLoss(nn.Module):
    def __init__(
        self,
        mixing_schedule: MixingSchedule,
        vocab_size: int,
        beta_is_div: float = 1.0,
        mask_token_id: int = -1,
        partition_axis: PartitionAxis = PartitionAxis(),
        dtype: jnp.dtype = None,
    ):
        self.mixing_schedule = mixing_schedule
        self.vocab_size = vocab_size
        self.beta_is_div = beta_is_div
        self.mask_token_id = mask_token_id
        self.partition_axis = partition_axis
        self.dtype = dtype

        self.logits_partition_spec = PartitionSpec(
            self.partition_axis.batch_axis,
            self.partition_axis.sequence_axis,
            # self.partition_axis.hidden_state_axis,
            None,
        )
        self.tokens_partition_spec = PartitionSpec(
            self.partition_axis.batch_axis,
            self.partition_axis.sequence_axis,
        )
        print("logits_partition_spec:", self.logits_partition_spec)
        print("tokens_partition_spec:", self.tokens_partition_spec)

    # def _softmax(self, x, axis=-1):
    #     x = x.astype(jnp.float32)
    #     m = jnp.max(x, axis=axis, keepdims=True)
    #     m = jax.lax.pmax(m, axis_name=self.logits_partition_spec[axis])
    #     x = x - m
    #     expx = jnp.exp(x)
    #     denom = jnp.sum(expx, axis=axis, keepdims=True)
    #     denom = lax.psum(denom, axis_name=self.logits_partition_spec[axis])
    #     return jnp.exp(x - jnp.log(denom)).astype(x.dtype)

    # def _gather_from_sharded_lastdim(self, x, ids, axis_name):
    #     local_vocab = x.shape[-1]
    #     start = local_vocab * lax.axis_index(axis_name)
    #     ids_local = ids - start
    #     mask = (ids_local >= 0) & (ids_local < local_vocab)
    #     ids_local = jnp.where(mask, ids_local, 0)
    #     vals_local = jnp.take_along_axis(x, ids_local[..., None], axis=-1).squeeze(-1)
    #     vals_local = jnp.where(mask, vals_local, 0.0)
    #     return lax.psum(vals_local, axis_name)

    def __call__(
        self,
        logits: chex.Array,
        input_ids: chex.Array,
        labels: chex.Array,
        log_snr: chex.Array,
        return_aux: bool = False,
    ) -> tuple[chex.Array, chex.Array]:
        logits = with_sharding_constraint(logits, self.logits_partition_spec)
        input_ids = with_sharding_constraint(input_ids, self.tokens_partition_spec)
        labels = with_sharding_constraint(labels, self.tokens_partition_spec)
        log_snr = with_sharding_constraint(log_snr, self.tokens_partition_spec)
        labels_one_hot = with_sharding_constraint(jax.nn.one_hot(labels, self.vocab_size, dtype=logits.dtype), self.logits_partition_spec)

        dtype = self.dtype or logits.dtype
        elbo_weights, aux = self.mixing_schedule.get_elbo_weights(log_snr, input_ids, labels, return_aux=True)
        elbo_weights = elbo_weights.clip(0, 1e6)
        loss_weights = aux["loss_weights"].clip(0, 1e3)
        elbo_weights = with_sharding_constraint(elbo_weights, self.tokens_partition_spec)
        loss_weights = with_sharding_constraint(loss_weights, self.tokens_partition_spec)
        
        logits = logits.at[..., self.mask_token_id].set(-1e6)  # Mask out the logits for the mask token.
        x_hat = nn.softmax(logits.astype(jnp.float32), axis=-1)
        log_p_t = self.mixing_schedule.marginal_log_probs(log_snr, x_hat).astype(dtype)
        log_p_t = with_sharding_constraint(log_p_t, self.logits_partition_spec)

        log_q_t = self.mixing_schedule.marginal_log_probs(log_snr, labels_one_hot)
        log_q_t = with_sharding_constraint(log_q_t, self.logits_partition_spec)

        log_p_zt = jnp.take_along_axis(log_p_t, input_ids[..., None], axis=-1).squeeze(-1).astype(jnp.float32)
        log_q_zt = jnp.take_along_axis(log_q_t, input_ids[..., None], axis=-1).squeeze(-1).astype(jnp.float32)
        log_p_zt = with_sharding_constraint(log_p_zt, self.tokens_partition_spec)
        log_q_zt = with_sharding_constraint(log_q_zt, self.tokens_partition_spec)
        ratio = jnp.exp(log_q_zt) / (jnp.exp(log_p_zt) + 1e-12)
        log_ratio = log_q_zt - log_p_zt
        is_div = ratio - log_ratio - 1

        # kl_div = optax.losses.kl_divergence_with_log_targets(log_p_t, log_q_t, axis=-1).astype(jnp.float32)
        kl_div = (jnp.exp(log_q_t) * (log_q_t - log_p_t)).sum(-1).astype(jnp.float32)
        kl_div = with_sharding_constraint(kl_div, self.tokens_partition_spec)

        loss = loss_weights * kl_div + self.beta_is_div * loss_weights * is_div

        if return_aux:
            elbo = elbo_weights * kl_div + elbo_weights * is_div
            return loss, {
                "elbo": elbo,
                "kl_loss": loss_weights * kl_div,
                "is_loss": loss_weights * is_div,
            }
        return loss
