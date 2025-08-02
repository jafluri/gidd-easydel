import random
import typing as tp

import jax
import jax.numpy as jnp
import flax.nnx as nn
from jax.sharding import NamedSharding, PartitionSpec
from transformers import PreTrainedTokenizerBase

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.utils.helpers import get_logger
from easydel.trainers.trainer import Trainer
from easydel.trainers.trainer_protocol import TrainerConfigureFunctionOutput


from ._utils import create_constant_length_dataset
from ._fn import training_step
from .loss import GiddLoss
from .schedule import MixingSchedule, create_mixing_schedule
from .diffusion_config import DiffusionConfig

if tp.TYPE_CHECKING:
    from datasets import Dataset
else:
    Dataset = tp.Any

logger = get_logger(__name__)


class DiffusionTrainer(Trainer):
    arguments: DiffusionConfig
    tokenizer: PreTrainedTokenizerBase
    mixing_schedule: MixingSchedule
    loss_fn: GiddLoss

    def __init__(
        self,
        arguments: DiffusionConfig,
        tokenizer: PreTrainedTokenizerBase,
        model: EasyDeLBaseModule | EasyDeLState | None = None,
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | dict[str, Dataset] | None = None,
        append_eos_token: bool = True,
        seed: int | None = None,
    ):
        assert isinstance(arguments, DiffusionConfig), "passed argument must be a `DiffusionConfig`."
        assert model is not None, "You must pass a `model` to the DiffusionTrainer."

        if seed is None:
            seed = random.randint(0, 2**31 - 1)
        self.key = jax.random.PRNGKey(seed)

        self.arguments = arguments
        self.tokenizer = tokenizer
        self.mixing_schedule = create_mixing_schedule(
            rate=arguments.mixing_rate,
            min_log_snr=arguments.min_log_snr,
            max_log_snr=arguments.max_log_snr,
            distribution="hybrid",
            vocab_size=len(tokenizer),
            prior_distribution="masked",
            mask_token_id=tokenizer.mask_token_id,
            hybrid_scale=arguments.hybrid_mixing_scale,
            hybrid_shift=arguments.hybrid_mixing_shift,
        )
        self.loss_fn = GiddLoss(
            mixing_schedule=self.mixing_schedule,
            vocab_size=len(tokenizer),
            beta_is_div=arguments.beta_is_div,
            mask_token_id=tokenizer.mask_token_id,
        )

        if not isinstance(model, EasyDeLState):
            model = model.to_state()

        if train_dataset is not None:
            train_dataset = self._prepare_dataset(
                train_dataset,
                dataset_tokens_field=arguments.dataset_tokens_field,
                max_sequence_length=arguments.max_sequence_length,
                num_of_sequences=arguments.total_batch_size,
                append_eos_token=append_eos_token,
            )
        if eval_dataset is not None:
            eval_dataset = self._prepare_dataset(
                eval_dataset,
                dataset_tokens_field=arguments.dataset_tokens_field,
                max_sequence_length=arguments.max_sequence_length,
                num_of_sequences=arguments.eval_batch_size,
                append_eos_token=append_eos_token,
            )

        super().__init__(
            arguments=arguments,
            dataset_train=train_dataset,
            dataset_eval=eval_dataset,
            model_state=model,
            data_collator=self.prepare_batch,
        )
        logger.info("Initialized DiffusionTrainer")

    def prepare_batch(self, batch: dict[str, jax.Array]) -> dict[str, jax.Array]:
        self.key, curr_key = jax.random.split(self.key, 2)
        batch["rng_key"] = curr_key
        return batch
    
    def _sample_log_snr(self, key: jax.Array, shape: tuple[int, ...]) -> jax.Array:
        """
        Sample log-SNR values for the given shape based on the configured noise parameters.
        Args:
            key: Random key for sampling
            shape: Shape of the sequence (batch_size, sequence_length)
        Returns:
            jax.Array: Sampled log-SNR values for each token in the sequence
        """
        if len(shape) != 2:
            raise ValueError(f"Expected shape to be 2D (batch_size, sequence_length), got {len(shape)}D shape: {shape}")

        batch_size, seq_len = shape

        key, snr_key, ind_key, lin_key = jax.random.split(key, 4)
        log_snr = self.mixing_schedule.sample_log_snr(snr_key, shape)

        r = jax.random.uniform(key, (batch_size,))

        if self.arguments.noise_p_independent > 0:
            # Sample independent SNR for each token
            is_independent = r < self.arguments.noise_p_independent
            log_snr = jnp.where(is_independent[:, None], log_snr, log_snr[:, 0, None])
        else:
            is_independent = jnp.zeros(batch_size, dtype=bool)
            log_snr = jnp.broadcast_to(log_snr[:, 0, None], (batch_size, seq_len))

        if self.arguments.noise_p_linear > 0:
            # Sample linear SNR based on token position
            is_linear = ~is_independent & (r < self.arguments.noise_p_linear + self.arguments.noise_p_independent)
            linear_t = jnp.linspace(0, 1, seq_len + 2, device=self.device)[1:-1]
            linear_log_snr = self.mixing_schedule.log_snr_from_time(linear_t)
            log_snr = jnp.where(is_linear[:, None], linear_log_snr[None, :], log_snr)

        # Ensure log_snr is broadcasted correctly
        assert log_snr.shape == (batch_size, seq_len), "log_snr shape mismatch"
        return log_snr

    def _sample_noise_mask(self, key: jax.Array, shape: tuple[int, ...]) -> jax.Array:
        """
        Sample a noise mask for the given shape based on the configured noise parameters.
        
        Args:
            key: Random key for sampling
            shape: Shape of the sequence (batch_size, sequence_length)
            
        Returns:
            Boolean mask where True indicates tokens that should have noise applied
        """
        batch_size, seq_len = shape

        # Start with all tokens receiving noise
        noise_mask = jnp.ones(shape, dtype=bool)

        # Split key for different sampling operations
        key, prompt_key, infill_key = jax.random.split(key, 3)

        # Sample which sequences get prompt conditioning (noise-free prefix)
        if self.arguments.noise_mask_p_prompt > 0:
            has_prompt_mask = jax.random.bernoulli(
                prompt_key, 
                self.arguments.noise_mask_p_prompt, 
                shape=(batch_size,)
            )

            # Sample fraction of prompt tokens for all sequences
            prompt_key, k = jax.random.split(prompt_key, 2)
            r = jax.random.uniform(k, (batch_size,))
            prompt_frac = jnp.arccos(1 - 2*r) / jnp.pi * self.arguments.noise_mask_max_cond_frac

            # Create position masks for prompts
            positions = jnp.arange(seq_len)[None, :]  # (1, seq_len)
            promp_mask = positions <= (prompt_frac[:, None] * (seq_len - 1))  # (batch_size, seq_len)

            # Apply prompt conditioning where has_prompt_mask is True
            prompt_mask = has_prompt_mask[:, None] & promp_mask
            noise_mask = noise_mask & ~prompt_mask

        # Sample which sequences get infilling conditioning (random noise-free tokens)
        if self.arguments.noise_mask_p_infilling > 0:
            has_infill_mask = jax.random.bernoulli(
                infill_key, 
                self.arguments.noise_mask_p_infilling, 
                shape=(batch_size,)
            )

            # Sample fraction of infill tokens for all sequences
            infill_key, k1, k2 = jax.random.split(infill_key, 3)
            r1 = jax.random.uniform(k1, (batch_size,))
            infill_frac = jnp.arccos(1 - 2*r1) / jnp.pi * self.arguments.noise_mask_max_cond_frac

            # Sample positions for infill tokens
            infill_p = jax.random.uniform(k2, (batch_size, seq_len))
            infill_mask = infill_p < infill_frac[:, None]  # (batch_size, seq_len)

            # Apply infill conditioning where has_infill_mask is True
            infill_mask = has_infill_mask[:, None] & infill_mask
            noise_mask = noise_mask & ~infill_mask

        return noise_mask

    def configure_functions(self) -> TrainerConfigureFunctionOutput:
        """
        Configures and JIT-compiles the training and evaluation step functions.

        This method sets up the necessary functions for training and evaluation, including:
            - Initialization of the model state.
            - Sharding of the model parameters and optimizer state.
            - JIT-compilation of the training and evaluation step functions.

        Returns:
            TrainerConfigureFunctionOutput: An object containing the configured functions and other relevant information.
        """
        logger.info("Configuring functions for DiffusionTrainer...")
        mesh = self.model.mesh

        empty_sharding = NamedSharding(spec=PartitionSpec(), mesh=mesh)

        self._train_shared_fn_static_args = (
            self.loss_fn,
            self._sample_log_snr,
            self._sample_noise_mask,
            self.mixing_schedule.sample_marginals,
            self.arguments.loss_config,
            self.scheduler,
            self.arguments.step_partition_spec,
            self.arguments.gradient_accumulation_steps,
            True,  # is_train
        )

        static_argnames = (2, 3, 4, 5, 6, 7, 8, 9, 10)
        sharded_training_step_function = jax.jit(
            training_step,
            in_shardings=(self.state_shardings, empty_sharding),
            out_shardings=(self.state_shardings, empty_sharding),
            donate_argnums=(0,),
            static_argnums=static_argnames,
        )

        self._eval_shared_fn_static_args = self._train_shared_fn_static_args[:-1] + (False,)  # is_train=False

        sharded_evaluation_step_function = jax.jit(
            training_step,
            in_shardings=(self.state_shardings, empty_sharding),
            out_shardings=empty_sharding,
            static_argnums=static_argnames,
        )

        self.arguments.ensure_checkpoint_path()

        logger.info("Functions configured successfully.")
        return TrainerConfigureFunctionOutput(
            sharded_training_step_function=sharded_training_step_function,
            sharded_evaluation_step_function=sharded_evaluation_step_function,
            mesh=mesh,
            checkpoint_manager=self.arguments.get_streaming_checkpointer(),
        )

    def _prepare_dataset(
        self,
        dataset,
        dataset_tokens_field,
        max_sequence_length,
        num_of_sequences,
        append_eos_token=True,
    ):
        """
        Prepares a packed dataloader from the given dataset.

        This method is designed for efficient training of language models by packing multiple
        sequences from the dataset into a single sample. This can be particularly beneficial
        for handling long sequences and optimizing GPU/TPU utilization.

        Args:
            processing_class: The processing_class used for text encoding.
            dataset (Dataset): The dataset to prepare.
            dataset_text_field (str): The name of the text field in the dataset.
            max_sequence_length (int): The maximum length of each packed sequence.
            num_of_sequences (int): The number of sequences to pack into a single sample.
            chars_per_token (float): The average number of characters per token, used for estimating
                the number of tokens in a text sequence.
            formatting_func (tp.Callable, optional): A function to format each sample from the dataset
                before packing. It should take a sample as input and return a dictionary with a "text"
                key containing the processed text. Defaults to None.
            append_eos_token (bool, optional): Whether to append a special concatenation token
                between packed sequences. Defaults to True.

        Returns:
            Dataset: The processed dataset with packed sequences.

        Raises:
            ValueError: If both `dataset_text_field` and `formatting_func` are None, or if there's
                an error during dataset packing.
        """
        if dataset_tokens_field is not None:
            constant_length_iterator = create_constant_length_dataset(
                dataset,
                tokens_field=dataset_tokens_field,
                seq_length=max_sequence_length,
                eos_token_id=self.tokenizer.eos_token_id,
                batch_size=num_of_sequences,
                append_concat_token=append_eos_token,
            )

            def data_generator(inner_constant_length_iterator):
                yield from inner_constant_length_iterator()

            # Import Only and Only when needed, don't dst the runtime.
            try:
                from datasets import IterableDataset
                from datasets.arrow_writer import SchemaInferenceError
                from datasets.builder import DatasetGenerationError
            except ImportError as exc:
                raise ImportError(
                    "Could not import `datasets` from Hugging Face. Make sure to install the "
                    "library using `pip install datasets`."
                ) from exc
            try:
                packed_dataset = IterableDataset.from_generator(
                    data_generator,
                    gen_kwargs={"inner_constant_length_iterator": constant_length_iterator},
                )
            except (DatasetGenerationError, SchemaInferenceError) as exc:
                raise ValueError(
                    "Error occurred while packing the dataset. "
                    "Make sure that your dataset has enough samples to at least yield one packed sequence.\n"
                    f"External Information : {exc}"
                ) from exc
            return packed_dataset
        else:
            raise ValueError(
                "You need to pass a `dataset_text_field` or `formatting_func` argument to the SFTTrainer if you want "
                "to use the `ConstantLengthDataset`."
            )

