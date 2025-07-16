# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
        data_collator: _default_collator | None = None,
    ):
        self.tokenizer = tokenizer
        assert isinstance(arguments, DiffusionConfig), "passed argument must be a `DiffusionConfig`."

        self.arguments = arguments
        self.loss_fn = loss_fn

        if not isinstance(model, EasyDeLState):
            model = model.to_state()

        super().__init__(
            arguments=arguments,
            dataset_train=train_dataset,
            dataset_eval=eval_dataset,
            model_state=model,
            data_collator=data_collator,
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

    @staticmethod
    def _prepare_dataloader(
        dataset,
        dataset_tokens_field,
        max_sequence_length,
        num_of_sequences,
        append_concat_token=True,
        add_special_tokens=True,
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
            append_concat_token (bool, optional): Whether to append a special concatenation token
                between packed sequences. Defaults to True.
            add_special_tokens (bool, optional): Whether to add special tokens (like BOS, EOS)
                during tokenization. Defaults to True.

        Returns:
            Dataset: The processed dataset with packed sequences.

        Raises:
            ValueError: If both `dataset_text_field` and `formatting_func` are None, or if there's
                an error during dataset packing.
        """
        if dataset_tokens_field is not None:
            constant_length_iterator = create_constant_length_dataset(
                processing_class=processing_class,
                dataset=dataset,
                dataset_text_field=dataset_text_field,
                formatting_func=formatting_func,
                seq_length=max_sequence_length,
                infinite=False,
                num_of_sequences=num_of_sequences,
                chars_per_token=chars_per_token,
                eos_token_id=tokenizer.eos_token_id,
                append_concat_token=append_concat_token,
                add_special_tokens=add_special_tokens,
            )

            def data_generator(inner_constant_length_iterator):
                yield from inner_constant_length_iterator()

            # Import Only and Only when needed, don't dst the runtime.
            try:
                from datasets import Dataset
                from datasets.arrow_writer import SchemaInferenceError
                from datasets.builder import DatasetGenerationError
            except ImportError as exc:
                raise ImportError(
                    "Could not import `datasets` from Hugging Face. Make sure to install the "
                    "library using `pip install datasets`."
                ) from exc
            try:
                packed_dataset = Dataset.from_generator(
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

