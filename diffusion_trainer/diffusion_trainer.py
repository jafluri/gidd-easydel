import typing as tp

import jax
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
from .diffusion_config import DiffusionConfig

if tp.TYPE_CHECKING:
    from datasets import Dataset
else:
    Dataset = tp.Any

logger = get_logger(__name__)


class DiffusionTrainer(Trainer):
    teacher_state: EasyDeLState
    arguments: DiffusionConfig  # type hinting

    def __init__(
        self,
        arguments: DiffusionConfig,
        tokenizer: PreTrainedTokenizerBase,
        model: EasyDeLBaseModule | EasyDeLState | None = None,
        loss_fn: GiddLoss | None = None,
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | dict[str, Dataset] | None = None,
    ):
        self.tokenizer = tokenizer
        assert isinstance(arguments, DiffusionConfig), "passed argument must be a `DiffusionConfig`."

        self.arguments = arguments
        self.loss_fn = loss_fn

        if not isinstance(model, EasyDeLState):
            model = model.to_state()

        if train_dataset is not None:
            train_dataset = self._prepare_dataset(
                train_dataset,
                dataset_tokens_field=arguments.dataset_tokens_field,
                max_sequence_length=arguments.max_sequence_length,
                num_of_sequences=arguments.total_batch_size,
                append_eos_token=arguments.append_eos_token,
            )
        if eval_dataset is not None:
            eval_dataset = self._prepare_dataset(
                eval_dataset,
                dataset_tokens_field=arguments.dataset_tokens_field,
                max_sequence_length=arguments.max_sequence_length,
                num_of_sequences=arguments.eval_batch_size,
                append_eos_token=arguments.append_eos_token,
            )

        super().__init__(
            arguments=arguments,
            dataset_train=train_dataset,
            dataset_eval=eval_dataset,
            model_state=model,
        )

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
        mesh = self.model.mesh

        empty_sharding = NamedSharding(spec=PartitionSpec(), mesh=mesh)

        self._train_shared_fn_static_args = (
            self.loss_fn,
            self.arguments.loss_config,
            self.scheduler,
            self.arguments.step_partition_spec,
            self.arguments.gradient_accumulation_steps,
            True,  # is_train
        )

        static_argnames = (3, 4, 5, 6, 7, 8)
        sharded_training_step_function = jax.jit(
            training_step,
            in_shardings=(self.state_shardings, empty_sharding),
            out_shardings=(self.state_shardings, empty_sharding),
            donate_argnums=(0,),
            static_argnums=static_argnames,
        )

        self._eval_shared_fn_static_args = (
            self.loss_fn,
            self.arguments.loss_config,
            self.scheduler,
            self.arguments.step_partition_spec,
            self.arguments.gradient_accumulation_steps,
            False,  # is_train
        )

        sharded_evaluation_step_function = jax.jit(
            training_step,
            in_shardings=(self.state_shardings, empty_sharding),
            out_shardings=empty_sharding,
            static_argnums=static_argnames,
        )

        flops_per_tkn = self.teacher_state.model.flops_per_token(include_loss=True, include_backward=True)

        self._extra_forward_flops_per_token = flops_per_tkn
        self._extra_backward_flops_per_token = flops_per_tkn

        self.arguments.ensure_checkpoint_path()
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
                append_eos_token=append_eos_token,
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

