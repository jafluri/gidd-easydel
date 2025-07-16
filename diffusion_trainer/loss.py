import chex
import jax.numpy as jnp
import flax.nnx as nn
import optax

class GiddLoss(nn.Module):
    def __init__(
        self,
        mixing_schedule: nn.Module,
        vocab_size: int,
        beta_is_div: float = 1.0,
        mask_token_id: int = -1,
    ):
        self.mixing_schedule = mixing_schedule
        self.vocab_size = vocab_size
        self.beta_is_div = beta_is_div
        self.mask_token_id = mask_token_id

    def __call__(
        self,
        logits: chex.Array,
        input_ids: chex.Array,
        labels: chex.Array,
        log_snr: chex.Array,
    ) -> tuple[chex.Array, chex.Array]:

        p_log_snr = self.mixing_schedule.p_log_snr(log_snr)
        loss_weights = self.mixing_schedule.get_loss_weights(log_snr, p_log_snr)
        elbo_weights = loss_weights / p_log_snr
        
        logits[..., self.mask_token_id] = -1e6  # Mask out the logits for the mask token.
        x_hat = nn.softmax(logits, axis=-1)

        log_p_t = self.mixing_schedule.log_probs_at(log_snr, x_hat)
        log_q_t = self.mixing_schedule.log_probs_at(
            log_snr,
            nn.one_hot(labels, self.vocab_size, dtype=logits.dtype),
        )

        log_p_zt = jnp.take_along_axis(log_p_t, input_ids[..., None], axis=-1).squeeze(-1)
        log_q_zt = jnp.take_along_axis(log_q_t, input_ids[..., None], axis=-1).squeeze(-1)
        log_ratio = log_q_zt - log_p_zt
        is_div = jnp.exp(log_ratio) - log_ratio - 1

        kl_div = optax.losses.kl_divergence_with_log_targets(log_p_t, log_q_t, axis=-1)

        loss = loss_weights * (kl_div + self.beta_is_div * is_div)
        elbo = elbo_weights * (kl_div + is_div)

        return loss, {
            "elbo": elbo,
            "kl_div": kl_div,
            "is_div": is_div,
        }
