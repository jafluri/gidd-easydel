import typing as tp
import enum
from abc import ABC, abstractmethod

import chex
import jax
import jax.numpy as jnp
import flax.nnx as nn


"""
Base classes for mixing schedules in diffusion models.
"""

def safe_sigmoid(x: chex.Array, precision=jnp.float32) -> chex.Array:
    return nn.sigmoid(x.astype(precision)).astype(x.dtype)


def safe_log(x: chex.Array, neg_infty: float = -1e6) -> chex.Array:
    return jnp.where(
        x > 0,
        jnp.log(x.clip(jnp.finfo(x.dtype).tiny)),
        neg_infty,
    )


class Priors(enum.Enum):
    UNIFORM = "uniform"
    MASKED = "masked"


class MixingRate(nn.Module, ABC):
    @property
    def min_log_snr(self) -> float:
        return self._min_log_snr

    @property
    def max_log_snr(self) -> float:
        return self._max_log_snr

    @abstractmethod
    def p_log_snr(self, log_snr: chex.Array) -> chex.Array:
        raise NotImplementedError

    @abstractmethod
    def sample_log_snr(self, key: chex.PRNGKey, shape: chex.Shape) -> chex.Array:
        raise NotImplementedError

    @abstractmethod
    def log_snr_from_time(self, time: chex.Array) -> chex.Array:
        raise NotImplementedError

    @abstractmethod
    def time_from_log_snr(self, log_snr: chex.Array) -> chex.Array:
        raise NotImplementedError

    def alpha_from_log_snr(self, log_snr: chex.Array) -> chex.Array:
        return safe_sigmoid(log_snr.clip(self.min_log_snr, self.max_log_snr))

    def log_snr_from_alpha(self, alpha: chex.Array) -> chex.Array:
        return  (safe_log(alpha) - safe_log(1 - alpha)).clip(self.min_log_snr, self.max_log_snr)


class MixingDistribution(nn.Module, ABC):
    def __init__(
        self,
        *,
        vocab_size: int = None,
        prior_distribution: Priors | chex.Array = Priors.MASKED,
        mask_token_id: int | None = None,
    ):
        self._vocab_size = vocab_size
        self._prior_distribution = prior_distribution
        self._mask_token_id = mask_token_id

    @property
    def vocab_size(self) -> int:
        """The size of the vocabulary for the mixing distribution."""
        return self._vocab_size

    @property
    def prior_distribution(self) -> Priors | chex.Array:
        """The prior distribution (pure noise) used for sampling."""
        return self._prior_distribution

    @property
    def mask_token_id(self) -> int | None:
        """The token ID used for masking in the mixing distribution. Necessary for the MASKED prior."""
        return self._mask_token_id

    @abstractmethod
    def pi_lambda(self, log_snr: chex.Array, probs: chex.Array) -> chex.Array:
        """
        Computes the mixing distribution at a given log_snr and token probabilities.
        Args:
            log_snr (shape: [...]): Per-token log. signal-to-noise ratio.
            probs (shape: [..., vocab_size]): Per-token probability distribution over vocab (one-hot encoded or similar).
        """
        raise NotImplementedError

    def pi_lambda_from_ids(self, log_snr: chex.Array, input_ids: chex.Array) -> chex.Array:
        """
        Default implementation uses `one_hot` -> `probs_from_logits`.
        May be overridden by a more efficient implementation.
        """
        probs = nn.one_hot(input_ids, self.vocab_size, dtype=log_snr.dtype)
        return self.pi_lambda(log_snr, probs)

    def pi_lambda_prime(self, log_snr: chex.Array, probs: chex.Array) -> chex.Array:
        """
        Default implementation computes the partial (element-wise) derivative
        w.r.t. log_snr using JAX autograd.
        """
        pi_prime = jax.vmap(jax.jacobian(self.pi_lambda))(log_snr.reshape(-1), probs.reshape(-1, probs.shape[-1]))
        return pi_prime.reshape(log_snr.shape + (-1,))

    def pi_lambda_prime_from_ids(self, log_snr: chex.Array, input_ids: chex.Array) -> chex.Array:
        """
        Default implementation uses `one_hot` -> `probs_from_logits`.
        May be overridden by a more efficient implementation.
        """
        probs = nn.one_hot(input_ids, self.vocab_size, dtype=log_snr.dtype)
        return self.pi_lambda_prime(log_snr, probs)


class LinearMixingRate(MixingRate):
    def __init__(
        self,
        *,
        min_log_snr: float = -10.0,
        max_log_snr: float = 10.0,
        **kwargs,
    ):
        self._min_log_snr = min_log_snr
        self._max_log_snr = max_log_snr
        self.t_min = self.time_from_log_snr(jnp.array(max_log_snr))
        self.t_max = self.time_from_log_snr(jnp.array(min_log_snr))

    def sample_log_snr(self, key: chex.PRNGKey, shape: chex.Shape) -> chex.Array:
        t = jax.random.uniform(key, shape, minval=self.t_min, maxval=self.t_max)
        log_snr = self.log_snr_from_time(t)
        return log_snr
        
    def p_log_snr(self, log_snr: chex.Array) -> chex.Array:
        sigm = safe_sigmoid(log_snr)
        return sigm * (1 - sigm)

    def log_snr_from_time(self, time: chex.Array) -> chex.Array:
        alpha = 1 - time
        return self.log_snr_from_alpha(alpha)
    
    def time_from_log_snr(self, log_snr: chex.Array) -> chex.Array:
        alpha = self.alpha_from_log_snr(log_snr)
        return 1 - alpha


class GeneralMixingDistribution(MixingDistribution):
    def __init__(
        self,
        *,
        vocab_size: int = None,
        prior_distribution: Priors | chex.Array = Priors.MASKED,
        mask_token_id: int | None = None,
        pi_lambda: tp.Callable[[chex.Array, chex.Array], chex.Array] | None = None,
        pi_lambda_from_ids: tp.Callable[[chex.Array, chex.Array], chex.Array] | None = None,
        pi_lambda_prime: tp.Callable[[chex.Array, chex.Array], chex.Array] | None = None,
        _pi_lambda_prime_from_ids: tp.Callable[[chex.Array, chex.Array], chex.Array] | None = None,
    ):
        assert pi_lambda is not None, "pi_lambda function must be provided for GeneralMixingDistribution."
        super().__init__(
            vocab_size=vocab_size,
            prior_distribution=prior_distribution,
            mask_token_id=mask_token_id,
        )
        self._pi_lambda = pi_lambda
        self._pi_lambda_from_ids = pi_lambda_from_ids
        self._pi_lambda_prime = pi_lambda_prime
        self._pi_lambda_prime_from_ids = _pi_lambda_prime_from_ids

    def pi_lambda(self, log_snr: chex.Array, probs: chex.Array) -> chex.Array:
        return self._pi_lambda(log_snr, probs)

    def pi_lambda_from_ids(self, log_snr: chex.Array, input_ids: chex.Array) -> chex.Array:
        if self._pi_lambda_from_ids is None:
            return super().pi_lambda_from_ids(log_snr, input_ids)
        return self._pi_lambda_from_ids(log_snr, input_ids)

    def pi_lambda_prime(self, log_snr: chex.Array, probs: chex.Array) -> chex.Array:
        if self._pi_lambda_prime is None:
            return super().pi_lambda_prime(log_snr, probs)
        return self._pi_lambda_prime(log_snr, probs)

    def pi_lambda_prime_from_ids(self, log_snr: chex.Array, input_ids: chex.Array) -> chex.Array:
        if self._pi_lambda_prime is None:
            return super().pi_lambda_prime_from_ids(log_snr, input_ids)
        return self._pi_lambda_prime_from_ids(log_snr, input_ids)


class HybridMixingDistribution(MixingDistribution):
    def __init__(
        self,
        *,
        vocab_size: int,
        mask_token_id: int,
        scale: float = 1.0,
        shift: float = 0.0,
        dtype: jnp.dtype = None,
    ):
        super().__init__(
            vocab_size=vocab_size,
            prior_distribution=Priors.MASKED,
            mask_token_id=mask_token_id,
        )
        self.scale = scale
        self.shift = shift
        self.dtype = dtype or jnp.float32
        self.mask_vec = nn.Variable(nn.one_hot(self.mask_token_id, self.vocab_size, dtype=dtype))
        u = jnp.full((self.vocab_size,), 1.0 / (self.vocab_size - 1), dtype=dtype)
        self.uniform_vec = nn.Variable(u.at[self.mask_token_id].set(0.0))

    def pi_lambda(self, log_snr: chex.Array, _: chex.Array | None) -> chex.Array:
        alpha = safe_sigmoid(self.scale * log_snr + self.shift)
        pi_at_logsnr = alpha[..., None].astype(self.dtype) * self.uniform_vec
        pi_at_logsnr = pi_at_logsnr.at[..., self.mask_token_id].add((1 - alpha).astype(self.dtype))
        return pi_at_logsnr

    def pi_lambda_from_ids(self, log_snr: chex.Array, _: chex.Array) -> chex.Array:
        return self.pi_lambda(log_snr, None)

    def pi_lambda_prime(self, log_snr: chex.Array, _: chex.Array) -> chex.Array:
        alpha = safe_sigmoid(self.scale * log_snr + self.shift)[..., None].astype(jnp.float32)
        alpha_prime = self.scale * alpha * (1 - alpha)
        pi_prime = alpha_prime.astype(log_snr.dtype) * (self.uniform_vec - self.mask_vec)
        return pi_prime

    def pi_lambda_prime_from_ids(self, log_snr: chex.Array, _: chex.Array) -> chex.Array:
        return self.pi_lambda_prime(log_snr, None)


class MixingSchedule(MixingRate, MixingDistribution):
    def __init__(self, rate: MixingRate, distribution: MixingDistribution):
        self._rate = rate
        self._distribution = distribution

    def marginal_probs(self, log_snr: chex.Array, probs: chex.Array) -> chex.Array:
        alpha = self.alpha_from_log_snr(log_snr)[..., None]
        return alpha.astype(probs.dtype) * probs + (1 - alpha).astype(probs.dtype) * self.pi_lambda(log_snr, probs)

    def marginal_probs_from_ids(self, log_snr: chex.Array, input_ids: chex.Array, dtype: jnp.dtype = None) -> chex.Array:
        alpha = self.alpha_from_log_snr(log_snr)
        probs = (1 - alpha[..., None]).astype(dtype) * self.pi_lambda_from_ids(log_snr, input_ids)
        probs = probs.at[(*jnp.indices(input_ids.shape), input_ids)].add(alpha.astype(dtype))  # scatter_add
        return probs

    def marginal_log_probs(self, log_snr: chex.Array, probs: chex.Array) -> chex.Array:
        marginal_probs = self.marginal_probs(log_snr, probs)
        return safe_log(marginal_probs)

    def marginal_log_probs_from_ids(self, log_snr: chex.Array, input_ids: chex.Array, dtype: jnp.dtype = None) -> chex.Array:
        marginal_probs = self.marginal_probs_from_ids(log_snr, input_ids, dtype=dtype)
        return safe_log(marginal_probs)

    def sample_marginals(self, key: chex.PRNGKey, log_snr: chex.Array, labels: chex.Array, mode: str | None = "high") -> chex.Array:
        assert log_snr.shape == labels.shape, "log_snr and labels must have the same shape."
        batch_shape = log_snr.shape

        flat_log_snr = log_snr.reshape(-1)
        flat_labels = labels.reshape(-1)
        flat_keys = jax.random.split(key, flat_log_snr.shape[0])

        def _sample_single(inputs):
            key, scalar_log_snr, token_id = inputs
            alpha = self.alpha_from_log_snr(scalar_log_snr.astype(jnp.float32))
            probs = (1 - alpha) * self.pi_lambda_from_ids(scalar_log_snr, token_id)
            probs = probs.at[token_id].add(alpha)
            log_probs = safe_log(probs)
            return jax.random.categorical(key, log_probs, axis=-1, mode=mode)

        flat_samples = jax.lax.map(_sample_single, (flat_keys, flat_log_snr, flat_labels), batch_size=128)
        return flat_samples.reshape(batch_shape)

    def sample_prior(self, key: chex.PRNGKey, shape: chex.Shape, mode: str | None = "high") -> chex.Array:
        if self.prior_distribution == Priors.MASKED:
            return jnp.full(shape, self.mask_token_id)
        elif self.prior_distribution == Priors.UNIFORM:
            return jax.random.randint(key, shape, 0, self.vocab_size)
        elif isinstance(self.prior_distribution, chex.Array):
            return jax.random.choice(key, self.vocab_size, shape=shape, p=self.prior_distribution, mode=mode)
        else:
            raise ValueError(f"Unknown prior distribution: {self.prior_distribution}")

    def get_loss_weights(self, log_snr: chex.Array, input_ids: chex.Array, labels: chex.Array) -> chex.Array:
        assert log_snr.shape == input_ids.shape == labels.shape, "log_snr, input_ids, and labels must have the same shape."
        batch_shape = log_snr.shape
        flat_log_snr = log_snr.reshape(-1)
        flat_input_ids = input_ids.reshape(-1)
        flat_labels = labels.reshape(-1)

        def _loss_weights_single(inputs):
            scalar_log_snr, input_id, label = inputs
            pi = self.pi_lambda_from_ids(scalar_log_snr, label)
            pi_prime = self.pi_lambda_prime_from_ids(scalar_log_snr, label)
            pi_at_z = pi[input_id].astype(jnp.float32)
            pi_prime_at_z = pi_prime[input_id].astype(jnp.float32)
            snr = jnp.exp(scalar_log_snr.astype(jnp.float32))
            delta_zx = (input_id == label).astype(jnp.float32)
            loss_weight = (pi_at_z - pi_prime_at_z) / (pi_at_z + snr * delta_zx)
            return loss_weight

        flat_loss_weights = jax.lax.map(_loss_weights_single, (flat_log_snr, flat_input_ids, flat_labels), batch_size=128)
        return flat_loss_weights.reshape(batch_shape)

    def get_elbo_weights(
        self,
        log_snr: chex.Array,
        input_ids: chex.Array,
        labels: chex.Array,
        return_aux: bool = False,
    ) -> chex.Array | tuple[chex.Array, dict[str, chex.Array]]:
        """
        Computes the ELBO weights for the given log_snr.
        
        Args:
            log_snr (chex.Array): log signal-to-noise ratio.
            input_ids (chex.Array): noised token IDs.
            labels (chex.Array): target token IDs.
            return_aux (bool): If True, also returns `loss_weights` and `p_log_snr`.
        
        Returns:
            chex.Array or tuple: The ELBO weights, optionally with auxiliary information.
        """
        log_snr = log_snr.astype(jnp.float32)
        loss_weights = self.get_loss_weights(log_snr, input_ids, labels)
        p_log_snr = self.p_log_snr(log_snr)
        elbo_weights = (loss_weights / p_log_snr.clip(1e-12))
        if return_aux:
            return elbo_weights, {
                "p_log_snr": p_log_snr,
                "loss_weights": loss_weights,
            }
        return elbo_weights


    """
    MixingRate methods
    """
    def p_log_snr(self, log_snr: chex.Array) -> chex.Array:
        return self._rate.p_log_snr(log_snr)

    def sample_log_snr(self, key: chex.PRNGKey, shape: chex.Shape) -> chex.Array:
        return self._rate.sample_log_snr(key, shape)

    def log_snr_from_time(self, time: chex.Array) -> chex.Array:
        return self._rate.log_snr_from_time(time)

    def time_from_log_snr(self, log_snr: chex.Array) -> chex.Array:
        return self._rate.time_from_log_snr(log_snr)

    def alpha_from_log_snr(self, log_snr: chex.Array) -> chex.Array:
        return self._rate.alpha_from_log_snr(log_snr)

    def log_snr_from_alpha(self, alpha: chex.Array) -> chex.Array:
        return self._rate.log_snr_from_alpha(alpha)

    """
    MixingDistribution methods
    """
    @property
    def vocab_size(self) -> int:
        return self._distribution.vocab_size

    @property
    def prior_distribution(self) -> Priors | chex.Array:
        return self._distribution.prior_distribution

    @property
    def mask_token_id(self) -> int | None:
        return self._distribution.mask_token_id

    def pi_lambda(self, log_snr: chex.Array, probs: chex.Array) -> chex.Array:
        return self._distribution.pi_lambda(log_snr, probs)

    def pi_lambda_from_ids(self, log_snr, input_ids):
        return self._distribution.pi_lambda_from_ids(log_snr, input_ids)

    def pi_lambda_prime(self, log_snr: chex.Array, probs: chex.Array) -> chex.Array:
        return self._distribution.pi_lambda_prime(log_snr, probs)

    def pi_lambda_prime_from_ids(self, log_snr: chex.Array, input_ids: chex.Array) -> chex.Array:
        return self._distribution.pi_lambda_prime_from_ids(log_snr, input_ids)


def create_mixing_schedule(
    rate: MixingRate | tp.Literal["linear"],
    min_log_snr: float = -10.0,
    max_log_snr: float = 10.0,
    distribution: MixingDistribution | tp.Literal["general", "hybrid"] = "hybrid",
    pi_lambda: tp.Callable[[chex.Array, chex.Array], chex.Array] | None = None,
    pi_lambda_from_ids: tp.Callable[[chex.Array, chex.Array], chex.Array] | None = None,
    pi_lambda_prime: tp.Callable[[chex.Array, chex.Array], chex.Array] | None = None,
    vocab_size: int = None,
    prior_distribution: Priors | chex.Array = Priors.MASKED,
    mask_token_id: int | None = -1,
    hybrid_scale: float = 1.0,
    hybrid_shift: float = 0.0,
    dtype: jnp.dtype = None,
) -> MixingSchedule:
    """
    Creates a mixing schedule combining a rate and a distribution.
    
    Args:
        rate (MixingRate | str):
            The mixing rate, either a MixingRate instance or "linear".
        distribution (MixingDistribution | str):
            The mixing distribution, either a MixingDistribution instance or "general" or "hybrid".
        pi_lambda (callable, optional):
            Custom function for computing the mixing distribution.
        pi_lambda_from_ids (callable, optional):
            Custom function for computing the mixing distribution from input IDs.
        pi_lambda_prime (callable, optional):
            Custom function for computing the derivative of the mixing distribution.
        vocab_size (int, optional):
            The size of the vocabulary for the mixing distribution. Must be provided for "general" or "hybrid" mixing distributions.
        prior_distribution (Priors | chex.Array, optional):
            The prior distribution used for sampling. Must be provided for "general" or "hybrid" mixing distributions.
        mask_token_id (int | None, optional):
            The token ID used for masking in the mixing distribution. Necessary for a MASKED prior distribution and for "hybrid" mixing distributions.
        hybrid_scale (float, optional):
            Scale factor for the hybrid mixing distribution.
        hybrid_shift (float, optional):
            Shift factor for the hybrid mixing distribution.
    
    Returns:
        MixingSchedule: A new mixing schedule instance.
    """
    if isinstance(rate, str):
        if rate == "linear":
            rate = LinearMixingRate(
                min_log_snr=min_log_snr,
                max_log_snr=max_log_snr,
            )
        else:
            raise ValueError(f"Unknown MixingRate type: {rate}")

    if isinstance(distribution, str):
        if vocab_size is None:
            raise ValueError("vocab_size must be provided for MixingDistribution.")
        if distribution == "general":
            if pi_lambda is None:
                raise ValueError("pi_lambda must be provided for GeneralMixingDistribution.")
            distribution = GeneralMixingDistribution(
                vocab_size=vocab_size,
                prior_distribution=prior_distribution,
                mask_token_id=mask_token_id,
                pi_lambda=pi_lambda,
                pi_lambda_from_ids=pi_lambda_from_ids,
                pi_lambda_prime=pi_lambda_prime,
            )
        elif distribution == "hybrid":
            distribution = HybridMixingDistribution(
                vocab_size=vocab_size,
                mask_token_id=mask_token_id,
                scale=hybrid_scale,
                shift=hybrid_shift,
                dtype=dtype,
            )
        else:
            raise ValueError(f"Unknown MixingDistribution type: {distribution}")
    return MixingSchedule(rate=rate, distribution=distribution)
