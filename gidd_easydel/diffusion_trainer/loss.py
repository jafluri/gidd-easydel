import chex
import jax
import jax.numpy as jnp
import flax.nnx as nn
import optax

from .schedule import MixingSchedule

class GiddLoss(nn.Module):
    def __init__(
        self,
        mixing_schedule: MixingSchedule,
        vocab_size: int,
        beta_is_div: float = 1.0,
        mask_token_id: int = -1,
        dtype: jnp.dtype = None,
    ):
        self.mixing_schedule = mixing_schedule
        self.vocab_size = vocab_size
        self.beta_is_div = beta_is_div
        self.mask_token_id = mask_token_id
        self.dtype = dtype

    def __call__(
        self,
        logits: chex.Array,
        input_ids: chex.Array,
        labels: chex.Array,
        log_snr: chex.Array,
        return_aux: bool = False,
    ) -> tuple[chex.Array, chex.Array]:
        dtype = self.dtype or logits.dtype
        elbo_weights, aux = self.mixing_schedule.get_elbo_weights(log_snr, input_ids, labels, return_aux=True)
        loss_weights = aux["loss_weights"].clip(0, 1e3)
        
        logits = logits.at[..., self.mask_token_id].set(-1e6)  # Mask out the logits for the mask token.
        x_hat = nn.softmax(logits.astype(jnp.float32), axis=-1).astype(dtype)

        log_p_t = self.mixing_schedule.marginal_log_probs(log_snr, x_hat)
        log_q_t = self.mixing_schedule.marginal_log_probs_from_ids(log_snr, labels, dtype=dtype)

        log_p_zt = jnp.take_along_axis(log_p_t, input_ids[..., None], axis=-1).squeeze(-1).astype(jnp.float32)
        log_q_zt = jnp.take_along_axis(log_q_t, input_ids[..., None], axis=-1).squeeze(-1).astype(jnp.float32)
        log_ratio = log_q_zt - log_p_zt
        is_div = jnp.exp(log_ratio) - log_ratio - 1

        kl_div = optax.losses.kl_divergence_with_log_targets(log_p_t, log_q_t, axis=-1).astype(jnp.float32)

        loss = loss_weights * (kl_div + self.beta_is_div * is_div)

        if return_aux:
            elbo = elbo_weights.clip(0, 1e6) * (kl_div + is_div)
            return loss, {
                "elbo": elbo,
                "kl_loss": loss_weights * kl_div,
                "is_loss": loss_weights * is_div,
            }
        return loss
