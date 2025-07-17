import pytest
import numpy as np
import jax
import jax.numpy as jnp
import flax.nnx as nn
from unittest.mock import Mock

from diffusion_trainer.schedule import (
    Priors, MixingRate, MixingDistribution, MixingSchedule,
    LinearMixingRate, GeneralMixingDistribution, HybridMixingDistribution,
    create_mixing_schedule
)


class TestPriors:
    """Test cases for the Priors enum."""
    
    def test_priors_values(self):
        assert Priors.UNIFORM.value == "uniform"
        assert Priors.MASKED.value == "masked"
    
    def test_priors_members(self):
        assert hasattr(Priors, "UNIFORM")
        assert hasattr(Priors, "MASKED")
        assert len(Priors) == 2


class TestMixingRate:
    """Test cases for the MixingRate abstract base class."""
    
    def test_abstract_methods(self):
        """Test that MixingRate is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            MixingRate()
    
    def test_alpha_from_log_snr(self):
        """Test alpha computation from log SNR."""
        class ConcreteMixingRate(MixingRate):
            def p_log_snr(self, log_snr):
                return log_snr
            def sample_log_snr(self, key, shape):
                return jnp.zeros(shape)
            def log_snr_from_time(self, time):
                return time
            def time_from_log_snr(self, log_snr):
                return log_snr
        
        rate = ConcreteMixingRate()
        
        # Test scalar input
        log_snr = jnp.array(0.0)
        alpha = rate.alpha_from_log_snr(log_snr)
        assert jnp.isclose(alpha, 0.5)
        
        # Test array input
        log_snr = jnp.array([-2.0, 0.0, 2.0])
        alpha = rate.alpha_from_log_snr(log_snr)
        expected = 1 / (1 + jnp.exp(-log_snr))
        assert jnp.allclose(alpha, expected)
        
        # Test extreme values
        log_snr = jnp.array([-10.0, 10.0])
        alpha = rate.alpha_from_log_snr(log_snr)
        assert alpha[0] < 0.001
        assert alpha[1] > 0.999
    
    def test_log_snr_from_alpha(self):
        """Test log SNR computation from alpha."""
        class ConcreteMixingRate(MixingRate):
            def p_log_snr(self, log_snr):
                return log_snr
            def sample_log_snr(self, key, shape):
                return jnp.zeros(shape)
            def log_snr_from_time(self, time):
                return time
            def time_from_log_snr(self, log_snr):
                return log_snr
        
        rate = ConcreteMixingRate()
        
        # Test scalar input
        alpha = jnp.array(0.5)
        log_snr = rate.log_snr_from_alpha(alpha)
        assert jnp.isclose(log_snr, 0.0)
        
        # Test array input
        alpha = jnp.array([0.1, 0.5, 0.9])
        log_snr = rate.log_snr_from_alpha(alpha)
        expected = jnp.log(alpha) - jnp.log(1 - alpha)
        assert jnp.allclose(log_snr, expected)
        
        # Test round-trip conversion
        original_log_snr = jnp.array([-2.0, 0.0, 2.0])
        alpha = rate.alpha_from_log_snr(original_log_snr)
        recovered_log_snr = rate.log_snr_from_alpha(alpha)
        assert jnp.allclose(original_log_snr, recovered_log_snr, atol=1e-6)


class TestLinearMixingRate:
    """Test cases for the LinearMixingRate class."""
    
    def test_initialization(self):
        """Test LinearMixingRate initialization."""
        rate = LinearMixingRate(min_log_snr=-5.0, max_log_snr=5.0)
        assert rate.min_log_snr == -5.0
        assert rate.max_log_snr == 5.0
        assert hasattr(rate, 't_min')
        assert hasattr(rate, 't_max')
    
    def test_default_initialization(self):
        """Test LinearMixingRate with default parameters."""
        rate = LinearMixingRate()
        assert rate.min_log_snr == -10.0
        assert rate.max_log_snr == 10.0
    
    def test_log_snr_from_time(self):
        """Test log SNR computation from time."""
        rate = LinearMixingRate()
        
        # Time 0 should give maximum log SNR
        time = jnp.array(0.0)
        alpha = 1 - time
        expected_log_snr = rate.log_snr_from_alpha(alpha)
        actual_log_snr = rate.log_snr_from_time(time)
        assert jnp.allclose(actual_log_snr, expected_log_snr)
        
        # Time 1 should give minimum log SNR
        time = jnp.array(1.0)
        alpha = 1 - time
        expected_log_snr = rate.log_snr_from_alpha(alpha)
        actual_log_snr = rate.log_snr_from_time(time)
        assert jnp.allclose(actual_log_snr, expected_log_snr)
        
        # Test array input
        time = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0])
        log_snr = rate.log_snr_from_time(time)
        assert log_snr.shape == time.shape
    
    def test_time_from_log_snr(self):
        """Test time computation from log SNR."""
        rate = LinearMixingRate()
        
        # Test scalar input
        log_snr = jnp.array(0.0)
        time = rate.time_from_log_snr(log_snr)
        alpha = rate.alpha_from_log_snr(log_snr)
        expected_time = 1 - alpha
        assert jnp.isclose(time, expected_time)
        
        # Test round-trip conversion
        original_time = jnp.array([0.1, 0.3, 0.5, 0.7, 0.9])
        log_snr = rate.log_snr_from_time(original_time)
        recovered_time = rate.time_from_log_snr(log_snr)
        assert jnp.allclose(original_time, recovered_time, atol=1e-6)
    
    def test_sample_log_snr(self, key):
        """Test log SNR sampling."""
        rate = LinearMixingRate(min_log_snr=-5.0, max_log_snr=5.0)
        
        # Test single sample
        sample = rate.sample_log_snr(key, ())
        assert sample.shape == ()
        
        # Test batch sampling
        shape = (100, 10)
        samples = rate.sample_log_snr(key, shape)
        assert samples.shape == shape
        
        # Check samples are within expected range
        alpha_samples = rate.alpha_from_log_snr(samples)
        assert jnp.all(alpha_samples >= 0)
        assert jnp.all(alpha_samples <= 1)
        
        # Test distribution properties with many samples
        large_key = jax.random.PRNGKey(123)
        large_samples = rate.sample_log_snr(large_key, (10000,))
        times = rate.time_from_log_snr(large_samples)
        assert jnp.all(times >= rate.t_min - 1e-6)
        assert jnp.all(times <= rate.t_max + 1e-6)
    
    def test_p_log_snr(self):
        """Test probability density of log SNR."""
        rate = LinearMixingRate()
        
        # Test scalar input
        log_snr = jnp.array(0.0)
        p = rate.p_log_snr(log_snr)
        sigm = nn.sigmoid(log_snr)
        expected = sigm * (1 - sigm)
        assert jnp.isclose(p, expected)
        
        # Test array input
        log_snr = jnp.array([-2.0, 0.0, 2.0])
        p = rate.p_log_snr(log_snr)
        sigm = nn.sigmoid(log_snr)
        expected = sigm * (1 - sigm)
        assert jnp.allclose(p, expected)
        
        # Test that p_log_snr is maximized at log_snr = 0
        log_snr_range = jnp.linspace(-5, 5, 100)
        p_values = rate.p_log_snr(log_snr_range)
        max_idx = jnp.argmax(p_values)
        assert jnp.abs(log_snr_range[max_idx]) < 0.1


class TestMixingDistribution:
    """Test cases for the MixingDistribution abstract base class."""
    
    def test_initialization(self, vocab_size, mask_token_id):
        """Test MixingDistribution initialization."""
        class ConcreteMixingDistribution(MixingDistribution):
            def pi_lambda(self, log_snr, probs):
                return probs
        
        # Test with MASKED prior
        dist = ConcreteMixingDistribution(
            vocab_size=vocab_size,
            prior_distribution=Priors.MASKED,
            mask_token_id=mask_token_id
        )
        assert dist.vocab_size == vocab_size
        assert dist.prior_distribution == Priors.MASKED
        assert dist.mask_token_id == mask_token_id
        
        # Test with UNIFORM prior
        dist = ConcreteMixingDistribution(
            vocab_size=vocab_size,
            prior_distribution=Priors.UNIFORM
        )
        assert dist.prior_distribution == Priors.UNIFORM
        
        # Test with custom prior distribution
        custom_prior = jnp.ones(vocab_size) / vocab_size
        dist = ConcreteMixingDistribution(
            vocab_size=vocab_size,
            prior_distribution=custom_prior
        )
        assert jnp.allclose(dist.prior_distribution, custom_prior)
    
    def test_pi_lambda_from_ids(self, vocab_size):
        """Test default implementation of pi_lambda_from_ids."""
        class ConcreteMixingDistribution(MixingDistribution):
            def pi_lambda(self, log_snr, probs):
                return probs * 0.5  # Simple test implementation
        
        dist = ConcreteMixingDistribution(vocab_size=vocab_size)
        
        # Test with single ID
        log_snr = jnp.array(0.0)
        input_ids = jnp.array(5)
        result = dist.pi_lambda_from_ids(log_snr, input_ids)
        
        expected_probs = nn.one_hot(input_ids, vocab_size) * 0.5
        assert jnp.allclose(result, expected_probs)
        
        # Test with batch of IDs
        log_snr = jnp.zeros((3,))
        input_ids = jnp.array([1, 5, 10])
        result = dist.pi_lambda_from_ids(log_snr, input_ids)
        assert result.shape == (3, vocab_size)
    
    def test_pi_lambda_prime(self, vocab_size):
        """Test default implementation of pi_lambda_prime using autograd."""
        class ConcreteMixingDistribution(MixingDistribution):
            def pi_lambda(self, log_snr, probs):
                # Use a differentiable function
                alpha = nn.sigmoid(log_snr)
                return alpha[..., None] * probs
        
        dist = ConcreteMixingDistribution(vocab_size=vocab_size)
        
        # Test with single input
        log_snr = jnp.array(0.0)
        probs = jnp.ones(vocab_size) / vocab_size
        result = dist.pi_lambda_prime(log_snr, probs)
        
        assert result.shape == (vocab_size,)
        
        # Verify derivative is correct by comparing with finite differences
        epsilon = 1e-6
        pi_plus = dist.pi_lambda(log_snr + epsilon, probs)
        pi_minus = dist.pi_lambda(log_snr - epsilon, probs)
        expected = (pi_plus - pi_minus) / (2 * epsilon)
        assert jnp.allclose(result, expected, rtol=1e-4)


class TestGeneralMixingDistribution:
    """Test cases for the GeneralMixingDistribution class."""
    
    def test_initialization_requires_pi_lambda(self, vocab_size):
        """Test that GeneralMixingDistribution requires pi_lambda."""
        with pytest.raises(AssertionError):
            GeneralMixingDistribution(vocab_size=vocab_size)
    
    def test_custom_pi_lambda(self, vocab_size, mask_token_id):
        """Test GeneralMixingDistribution with custom pi_lambda."""
        def custom_pi_lambda(log_snr, probs):
            alpha = nn.sigmoid(log_snr)
            mask_vec = nn.one_hot(mask_token_id, vocab_size)
            return alpha[..., None] * probs + (1 - alpha[..., None]) * mask_vec
        
        dist = GeneralMixingDistribution(
            vocab_size=vocab_size,
            mask_token_id=mask_token_id,
            pi_lambda=custom_pi_lambda
        )
        
        log_snr = jnp.array(0.0)
        probs = jnp.ones(vocab_size) / vocab_size
        result = dist.pi_lambda(log_snr, probs)
        
        expected = custom_pi_lambda(log_snr, probs)
        assert jnp.allclose(result, expected)
    
    def test_custom_pi_lambda_from_ids(self, vocab_size):
        """Test GeneralMixingDistribution with custom pi_lambda_from_ids."""
        def custom_pi_lambda(log_snr, probs):
            return probs * 0.5
        
        def custom_pi_lambda_from_ids(log_snr, input_ids):
            return jnp.full((input_ids.shape[0], vocab_size), 0.1)
        
        dist = GeneralMixingDistribution(
            vocab_size=vocab_size,
            pi_lambda=custom_pi_lambda,
            pi_lambda_from_ids=custom_pi_lambda_from_ids
        )
        
        log_snr = jnp.zeros((3,))
        input_ids = jnp.array([1, 5, 10])
        result = dist.pi_lambda_from_ids(log_snr, input_ids)
        
        expected = custom_pi_lambda_from_ids(log_snr, input_ids)
        assert jnp.allclose(result, expected)
    
    def test_custom_pi_lambda_prime(self, vocab_size):
        """Test GeneralMixingDistribution with custom pi_lambda_prime."""
        def custom_pi_lambda(log_snr, probs):
            return probs
        
        def custom_pi_lambda_prime(log_snr, probs):
            return jnp.zeros_like(probs)
        
        dist = GeneralMixingDistribution(
            vocab_size=vocab_size,
            pi_lambda=custom_pi_lambda,
            pi_lambda_prime=custom_pi_lambda_prime
        )
        
        log_snr = jnp.array(0.0)
        probs = jnp.ones(vocab_size) / vocab_size
        result = dist.pi_lambda_prime(log_snr, probs)
        
        assert jnp.allclose(result, jnp.zeros_like(probs))


class TestHybridMixingDistribution:
    """Test cases for the HybridMixingDistribution class."""
    
    def test_initialization(self, vocab_size, mask_token_id):
        """Test HybridMixingDistribution initialization."""
        dist = HybridMixingDistribution(
            vocab_size=vocab_size,
            mask_token_id=mask_token_id,
            scale=2.0,
            shift=1.0
        )
        
        assert dist.vocab_size == vocab_size
        assert dist.mask_token_id == mask_token_id
        assert dist.scale == 2.0
        assert dist.shift == 1.0
        assert dist.prior_distribution == Priors.MASKED
        
        # Check mask vector
        assert jnp.allclose(dist.mask_vec, nn.one_hot(mask_token_id, vocab_size))
        
        # Check uniform vector
        expected_uniform = jnp.full((vocab_size,), 1.0 / (vocab_size - 1))
        expected_uniform = expected_uniform.at[mask_token_id].set(0.0)
        assert jnp.allclose(dist.uniform_vec, expected_uniform)
    
    def test_pi_lambda(self, vocab_size, mask_token_id):
        """Test HybridMixingDistribution pi_lambda computation."""
        dist = HybridMixingDistribution(
            vocab_size=vocab_size,
            mask_token_id=mask_token_id,
            scale=1.0,
            shift=0.0
        )
        
        # Test with log_snr = 0 (alpha = 0.5)
        log_snr = jnp.array(0.0)
        result = dist.pi_lambda(log_snr, None)
        
        alpha = 0.5
        expected = alpha * dist.uniform_vec + (1 - alpha) * dist.mask_vec
        assert jnp.allclose(result, expected)
        
        # Test with extreme log_snr values
        log_snr = jnp.array([-10.0, 10.0])
        result = dist.pi_lambda(log_snr, None)
        
        # For very negative log_snr, should be close to mask vector
        assert jnp.allclose(result[0], dist.mask_vec, atol=1e-4)
        
        # For very positive log_snr, should be close to uniform vector
        assert jnp.allclose(result[1], dist.uniform_vec, atol=1e-4)
    
    def test_pi_lambda_from_ids(self, vocab_size, mask_token_id):
        """Test that pi_lambda_from_ids ignores input_ids."""
        dist = HybridMixingDistribution(
            vocab_size=vocab_size,
            mask_token_id=mask_token_id
        )
        
        log_snr = jnp.zeros((3,))
        input_ids = jnp.array([1, 5, 10])
        
        result_from_ids = dist.pi_lambda_from_ids(log_snr, input_ids)
        result_direct = jnp.stack([dist.pi_lambda(log_snr[i], None) for i in range(3)])
        
        assert jnp.allclose(result_from_ids, result_direct)
    
    def test_pi_lambda_prime(self, vocab_size, mask_token_id):
        """Test HybridMixingDistribution derivative computation."""
        dist = HybridMixingDistribution(
            vocab_size=vocab_size,
            mask_token_id=mask_token_id,
            scale=1.0,
            shift=0.0
        )
        
        log_snr = jnp.array(0.0)
        result = dist.pi_lambda_prime(log_snr, None)
        
        # Compute expected derivative
        alpha = nn.sigmoid(log_snr)
        alpha_prime = dist.scale * alpha * (1 - alpha)
        expected = alpha_prime * (dist.uniform_vec - dist.mask_vec)
        
        assert jnp.allclose(result, expected)
        
        # Verify with finite differences
        epsilon = 1e-6
        pi_plus = dist.pi_lambda(log_snr + epsilon, None)
        pi_minus = dist.pi_lambda(log_snr - epsilon, None)
        finite_diff = (pi_plus - pi_minus) / (2 * epsilon)
        
        assert jnp.allclose(result, finite_diff, rtol=1e-4)
    
    def test_scale_and_shift_effects(self, vocab_size, mask_token_id):
        """Test the effects of scale and shift parameters."""
        # Test with different scale values
        scales = [0.5, 1.0, 2.0]
        log_snr = jnp.array(0.0)
        
        for scale in scales:
            dist = HybridMixingDistribution(
                vocab_size=vocab_size,
                mask_token_id=mask_token_id,
                scale=scale,
                shift=0.0
            )
            result = dist.pi_lambda(log_snr, None)
            alpha = nn.sigmoid(scale * log_snr)
            expected = alpha * dist.uniform_vec + (1 - alpha) * dist.mask_vec
            assert jnp.allclose(result, expected)
        
        # Test with different shift values
        shifts = [-1.0, 0.0, 1.0]
        
        for shift in shifts:
            dist = HybridMixingDistribution(
                vocab_size=vocab_size,
                mask_token_id=mask_token_id,
                scale=1.0,
                shift=shift
            )
            result = dist.pi_lambda(log_snr, None)
            alpha = nn.sigmoid(log_snr + shift)
            expected = alpha * dist.uniform_vec + (1 - alpha) * dist.mask_vec
            assert jnp.allclose(result, expected)


class TestMixingSchedule:
    """Test cases for the MixingSchedule class."""
    
    def test_initialization(self, vocab_size, mask_token_id):
        """Test MixingSchedule initialization."""
        rate = LinearMixingRate()
        dist = HybridMixingDistribution(
            vocab_size=vocab_size,
            mask_token_id=mask_token_id
        )
        
        schedule = MixingSchedule(rate=rate, distribution=dist)
        
        assert schedule._rate == rate
        assert schedule._distribution == dist
    
    def test_rate_methods_delegation(self):
        """Test that rate methods are properly delegated."""
        rate = Mock(spec=MixingRate)
        dist = Mock(spec=MixingDistribution)
        
        schedule = MixingSchedule(rate=rate, distribution=dist)
        
        # Test p_log_snr delegation
        log_snr = jnp.array(0.0)
        schedule.p_log_snr(log_snr)
        rate.p_log_snr.assert_called_once_with(log_snr)
        
        # Test sample_log_snr delegation
        key = jax.random.PRNGKey(0)
        shape = (10,)
        schedule.sample_log_snr(key, shape)
        rate.sample_log_snr.assert_called_once_with(key, shape)
        
        # Test log_snr_from_time delegation
        time = jnp.array(0.5)
        schedule.log_snr_from_time(time)
        rate.log_snr_from_time.assert_called_once_with(time)
        
        # Test time_from_log_snr delegation
        schedule.time_from_log_snr(log_snr)
        rate.time_from_log_snr.assert_called_once_with(log_snr)
    
    def test_distribution_properties_delegation(self, vocab_size, mask_token_id):
        """Test that distribution properties are properly delegated."""
        rate = LinearMixingRate()
        dist = HybridMixingDistribution(
            vocab_size=vocab_size,
            mask_token_id=mask_token_id
        )
        
        schedule = MixingSchedule(rate=rate, distribution=dist)
        
        assert schedule.vocab_size == vocab_size
        assert schedule.prior_distribution == Priors.MASKED
        assert schedule.mask_token_id == mask_token_id
    
    def test_marginal_probs(self, vocab_size, mask_token_id):
        """Test marginal probability computation."""
        rate = LinearMixingRate()
        dist = HybridMixingDistribution(
            vocab_size=vocab_size,
            mask_token_id=mask_token_id
        )
        
        schedule = MixingSchedule(rate=rate, distribution=dist)
        
        log_snr = jnp.array(0.0)
        probs = jnp.ones(vocab_size) / vocab_size
        
        result = schedule.marginal_probs(log_snr, probs)
        
        # Compute expected result
        alpha = rate.alpha_from_log_snr(log_snr)
        pi_lambda = dist.pi_lambda(log_snr, probs)
        expected = alpha * probs + (1 - alpha) * pi_lambda
        
        assert jnp.allclose(result, expected)
    
    def test_marginal_probs_from_ids(self, vocab_size, mask_token_id):
        """Test marginal probability computation from input IDs."""
        rate = LinearMixingRate()
        dist = HybridMixingDistribution(
            vocab_size=vocab_size,
            mask_token_id=mask_token_id
        )
        
        schedule = MixingSchedule(rate=rate, distribution=dist)
        
        log_snr = jnp.array([0.0, 0.0, 0.0])
        input_ids = jnp.array([1, 5, 10])
        
        result = schedule.marginal_probs_from_ids(log_snr, input_ids)
        
        assert result.shape == (3, vocab_size)
        
        # Check that probabilities sum to 1
        assert jnp.allclose(result.sum(axis=-1), 1.0)
        
        # Check that the correct indices have increased probability
        alpha = rate.alpha_from_log_snr(log_snr)
        for i, id in enumerate(input_ids):
            assert result[i, id] >= alpha[i]
        
        # Test with scalar inputs
        log_snr_scalar = jnp.array(0.0)
        input_id_scalar = jnp.array(5)
        result_scalar = schedule.marginal_probs_from_ids(log_snr_scalar, input_id_scalar)
        assert result_scalar.shape == (vocab_size,)
    
    def test_sample_marginals(self, key, vocab_size, mask_token_id):
        """Test sampling from marginal distribution."""
        rate = LinearMixingRate()
        dist = HybridMixingDistribution(
            vocab_size=vocab_size,
            mask_token_id=mask_token_id
        )
        
        schedule = MixingSchedule(rate=rate, distribution=dist)
        
        log_snr = jnp.array([0.0, -2.0, 2.0])
        labels = jnp.array([1, 5, 10])
        
        samples = schedule.sample_marginals(key, log_snr, labels)
        
        assert samples.shape == labels.shape
        assert jnp.all(samples >= 0)
        assert jnp.all(samples < vocab_size)
        
        # With high log_snr, samples should mostly match labels
        high_log_snr = jnp.full((100,), 10.0)
        high_labels = jnp.full((100,), 42)
        high_samples = schedule.sample_marginals(key, high_log_snr, high_labels)
        assert jnp.mean(high_samples == 42) > 0.95
        
        # With low log_snr, samples should be closer to prior
        low_log_snr = jnp.full((100,), -10.0)
        low_labels = jnp.full((100,), 42)
        low_samples = schedule.sample_marginals(key, low_log_snr, low_labels)
        # For masked prior, should mostly be mask token
        assert jnp.mean(low_samples == mask_token_id) > 0.95
    
    def test_sample_prior(self, key, vocab_size, mask_token_id):
        """Test sampling from prior distribution."""
        # Test MASKED prior
        rate = LinearMixingRate()
        dist = HybridMixingDistribution(
            vocab_size=vocab_size,
            mask_token_id=mask_token_id
        )
        schedule = MixingSchedule(rate=rate, distribution=dist)
        
        shape = (10, 5)
        samples = schedule.sample_prior(key, shape)
        assert samples.shape == shape
        assert jnp.all(samples == mask_token_id)
        
        # Test UNIFORM prior
        class UniformDist(MixingDistribution):
            def __init__(self, vocab_size):
                super().__init__(
                    vocab_size=vocab_size,
                    prior_distribution=Priors.UNIFORM
                )
            def pi_lambda(self, log_snr, probs):
                return probs
        
        dist = UniformDist(vocab_size=vocab_size)
        schedule = MixingSchedule(rate=rate, distribution=dist)
        
        samples = schedule.sample_prior(key, shape)
        assert samples.shape == shape
        assert jnp.all(samples >= 0)
        assert jnp.all(samples < vocab_size)
        
        # Test custom prior distribution
        custom_prior = jnp.zeros(vocab_size)
        custom_prior = custom_prior.at[5].set(1.0)  # All probability on token 5
        
        class CustomDist(MixingDistribution):
            def __init__(self, vocab_size, prior):
                super().__init__(
                    vocab_size=vocab_size,
                    prior_distribution=prior
                )
            def pi_lambda(self, log_snr, probs):
                return probs
        
        dist = CustomDist(vocab_size=vocab_size, prior=custom_prior)
        schedule = MixingSchedule(rate=rate, distribution=dist)
        
        samples = schedule.sample_prior(key, shape)
        assert jnp.all(samples == 5)


class TestCreateMixingSchedule:
    """Test cases for the create_mixing_schedule factory function."""
    
    def test_linear_rate_string(self, vocab_size, mask_token_id):
        """Test creating schedule with 'linear' rate string."""
        schedule = create_mixing_schedule(
            rate="linear",
            distribution="hybrid",
            vocab_size=vocab_size,
            mask_token_id=mask_token_id
        )
        
        assert isinstance(schedule, MixingSchedule)
        assert isinstance(schedule._rate, LinearMixingRate)
        assert isinstance(schedule._distribution, HybridMixingDistribution)
    
    def test_invalid_rate_string(self):
        """Test error handling for invalid rate string."""
        with pytest.raises(ValueError, match="Unknown MixingRate type"):
            create_mixing_schedule(rate="invalid", distribution="hybrid", vocab_size=100)
    
    def test_general_distribution_string(self, vocab_size):
        """Test creating schedule with 'general' distribution string."""
        def custom_pi_lambda(log_snr, probs):
            return probs
        
        schedule = create_mixing_schedule(
            rate="linear",
            distribution="general",
            vocab_size=vocab_size,
            pi_lambda=custom_pi_lambda
        )
        
        assert isinstance(schedule._distribution, GeneralMixingDistribution)
    
    def test_general_distribution_requires_pi_lambda(self, vocab_size):
        """Test that general distribution requires pi_lambda."""
        with pytest.raises(ValueError, match="pi_lambda must be provided"):
            create_mixing_schedule(
                rate="linear",
                distribution="general",
                vocab_size=vocab_size
            )
    
    def test_hybrid_distribution_string(self, vocab_size, mask_token_id):
        """Test creating schedule with 'hybrid' distribution string."""
        schedule = create_mixing_schedule(
            rate="linear",
            distribution="hybrid",
            vocab_size=vocab_size,
            mask_token_id=mask_token_id,
            hybrid_scale=2.0,
            hybrid_shift=1.0
        )
        
        assert isinstance(schedule._distribution, HybridMixingDistribution)
        assert schedule._distribution.scale == 2.0
        assert schedule._distribution.shift == 1.0
    
    def test_invalid_distribution_string(self, vocab_size):
        """Test error handling for invalid distribution string."""
        with pytest.raises(ValueError, match="Unknown MixingDistribution type"):
            create_mixing_schedule(
                rate="linear",
                distribution="invalid",
                vocab_size=vocab_size
            )
    
    def test_distribution_requires_vocab_size(self):
        """Test that string distributions require vocab_size."""
        with pytest.raises(ValueError, match="vocab_size must be provided"):
            create_mixing_schedule(rate="linear", distribution="hybrid")
    
    def test_with_custom_objects(self, vocab_size, mask_token_id):
        """Test creating schedule with custom rate and distribution objects."""
        rate = LinearMixingRate(min_log_snr=-8.0, max_log_snr=8.0)
        dist = HybridMixingDistribution(
            vocab_size=vocab_size,
            mask_token_id=mask_token_id,
            scale=3.0
        )
        
        schedule = create_mixing_schedule(rate=rate, distribution=dist)
        
        assert schedule._rate == rate
        assert schedule._distribution == dist
    
    def test_full_integration(self, key, vocab_size, mask_token_id):
        """Test full integration of created schedule."""
        schedule = create_mixing_schedule(
            rate="linear",
            distribution="hybrid",
            vocab_size=vocab_size,
            mask_token_id=mask_token_id,
            prior_distribution=Priors.MASKED
        )
        
        # Test sampling log SNR
        log_snr = schedule.sample_log_snr(key, (10,))
        assert log_snr.shape == (10,)
        
        # Test marginal probabilities
        labels = jnp.array([1, 5, 10])
        log_snr = jnp.array([0.0, 0.0, 0.0])
        marginals = schedule.marginal_probs_from_ids(log_snr, labels)
        assert marginals.shape == (3, vocab_size)
        
        # Test sampling
        samples = schedule.sample_marginals(key, log_snr, labels)
        assert samples.shape == labels.shape


class TestEdgeCasesAndValidation:
    """Test edge cases and validation for all components."""
    
    def test_extreme_log_snr_values(self, vocab_size, mask_token_id):
        """Test behavior with extreme log SNR values."""
        rate = LinearMixingRate()
        dist = HybridMixingDistribution(
            vocab_size=vocab_size,
            mask_token_id=mask_token_id
        )
        schedule = MixingSchedule(rate=rate, distribution=dist)
        
        # Test very large positive log SNR
        log_snr = jnp.array(100.0)
        alpha = schedule.alpha_from_log_snr(log_snr)
        assert jnp.isclose(alpha, 1.0, atol=1e-10)
        
        # Test very large negative log SNR
        log_snr = jnp.array(-100.0)
        alpha = schedule.alpha_from_log_snr(log_snr)
        assert jnp.isclose(alpha, 0.0, atol=1e-10)
    
    def test_numerical_stability(self, vocab_size):
        """Test numerical stability in computations."""
        rate = LinearMixingRate()
        
        # Test log_snr_from_alpha with extreme alpha values
        alpha = jnp.array([1e-10, 1 - 1e-10])
        log_snr = rate.log_snr_from_alpha(alpha)
        assert jnp.isfinite(log_snr).all()
        
        # Test round-trip stability
        recovered_alpha = rate.alpha_from_log_snr(log_snr)
        assert jnp.allclose(alpha, recovered_alpha, rtol=1e-6)
    
    def test_batch_dimensions(self, vocab_size, mask_token_id):
        """Test that operations handle batch dimensions correctly."""
        rate = LinearMixingRate()
        dist = HybridMixingDistribution(
            vocab_size=vocab_size,
            mask_token_id=mask_token_id
        )
        schedule = MixingSchedule(rate=rate, distribution=dist)
        
        # Test with 2D batch
        batch_shape = (4, 3)
        log_snr = jnp.zeros(batch_shape)
        input_ids = jax.random.randint(jax.random.PRNGKey(0), batch_shape, 0, vocab_size)
        
        marginals = schedule.marginal_probs_from_ids(log_snr, input_ids)
        assert marginals.shape == batch_shape + (vocab_size,)
        
        # Test with 3D batch
        batch_shape = (2, 4, 3)
        log_snr = jnp.zeros(batch_shape)
        probs = jnp.ones(batch_shape + (vocab_size,)) / vocab_size
        
        pi_lambda = dist.pi_lambda(log_snr, probs)
        assert pi_lambda.shape == batch_shape + (vocab_size,)
    
    def test_gradient_flow(self, vocab_size):
        """Test that gradients flow through operations correctly."""
        def loss_fn(scale):
            dist = HybridMixingDistribution(
                vocab_size=vocab_size,
                mask_token_id=vocab_size-1,
                scale=scale,
                shift=0.0
            )
            log_snr = jnp.array(1.0)  # Changed from 0.0 to get non-zero gradient
            pi = dist.pi_lambda(log_snr, None)
            return jnp.sum(pi**2)
        
        scale = jnp.array(1.0)
        grad = jax.grad(loss_fn)(scale)
        assert jnp.isfinite(grad)
        # Note: gradient may be zero at certain points, just check it's finite