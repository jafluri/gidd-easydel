
import os
from pprint import pprint

import ray
from eformer.executor.ray import TpuAcceleratorConfig, execute

# Initialize Ray for distributed computing. This must be done once per application.
# ray.init()

SAVE_DIRECTORY = os.environ.get("SAVE_DIRECTORY", "outputs/diffusion_trainer")

# --- Configuration Constants ---
TOKENIZER_ID = "dvruette/nemotron-cc-bpe"

# Your Weights & Biases entity (username or organization) for experiment logging.
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", None)

# --- Environment and TPU Configuration ---
# These environment variables are passed to each Ray worker to ensure they have
# access to necessary tokens and use efficient shared memory for caching.
# TPU_EXECUTION_ENV_VARS = {
#     "EASYDEL_AUTO": "1",  # Enables EasyDeL's automatic sharding configuration.
#     "HF_TOKEN": os.environ.get("HF_TOKEN_FOR_EASYDEL", ""),  # Hugging Face token.
#     "HF_DATASETS_CACHE": "/dev/shm/huggingface-dataset",  # RAM-disk for dataset cache.
#     "HF_HOME": "/dev/shm/huggingface",  # RAM-disk for model cache.
#     "HF_DATASETS_OFFLINE": "0",  # Allow online dataset access.
#     "WANDB_API_KEY": os.environ.get("WANDB_API_KEY_FOR_EASYDEL", ""),  # W&B API key.
# }

# Additional pip packages to install on each Ray worker environment.
TPU_PIP_PACKAGES = []

# Print the environment variables for verification.
# pprint(TPU_EXECUTION_ENV_VARS)

# # Defines the TPU environment for Ray, specifying the accelerator type and worker setup.
# tpu_config = TpuAcceleratorConfig(
#     "v4-64",  # Using a TPU v4 pod slice with 64 chips. Adjust to your hardware.
#     execution_env={
#         "env_vars": TPU_EXECUTION_ENV_VARS,
#         "pip": TPU_PIP_PACKAGES,
#     },
# )





# @execute(tpu_config)
# @ray.remote
def main():
    """
    The main function for the distillation training process, executed as a
    remote task on the TPU cluster via Ray.
    """
    # Imports are inside the function to ensure they are available in the
    # separate Ray worker process.
    import easydel as ed  # noqa
    import jax
    from jax import numpy as jnp
    from transformers import AutoTokenizer
    from datasets import load_dataset

    from diffusion_trainer import DiffusionTrainer, DiffusionConfig
    
    # jax.config.update('jax_disable_jit', True)

    logger = ed.utils.get_logger(__name__)

    # --- Basic Training Parameters ---
    max_length = 1024
    total_batch_size = 8

    hidden_size = 768
    head_dim = 64

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)

    # --- Model Definition ---
    model = ed.GiddForDiffusionLM(
        config=ed.GiddConfig(
            vocab_size=len(tokenizer),
            hidden_size=hidden_size,
            intermediate_size=4*hidden_size,
            num_hidden_layers=12,
            num_attention_heads=hidden_size // head_dim,
            head_dim=head_dim,
            max_position_embeddings=max_length,
            resid_scale=4.0,
            init_scale=0.4,
            emb_init_scale=0.1,
            head_init_scale=0.0,
            sharding_axis_dims=(1, jax.process_count(), 1, -1, 1),
            partition_axis=ed.PartitionAxis(),
            gradient_checkpointing=ed.EasyDeLGradientCheckPointers.NOTHING_SAVEABLE,
            attn_mechanism=ed.AttentionMechanisms.SDPA,
            attn_dtype=jnp.bfloat16,
            # scan_layers=True,
        ),
        dtype=jnp.bfloat16,
        param_dtype=jnp.bfloat16,
        precision=jax.lax.Precision.DEFAULT,
        rngs=ed.Rngs(0),
    ).shard_model()  # Shard the newly created model across devices.

    # --- Configuration ---
    arguments = DiffusionConfig(
        num_train_epochs=1,
        total_batch_size=total_batch_size,
        use_wandb=True,
        wandb_entity=WANDB_ENTITY,
        do_last_save=True,
        max_sequence_length=max_length,
        # This is MANDATORY for streaming datasets. It tells the trainer how many
        # steps constitute one "epoch". Should be ~ (total_dataset_size // total_batch_size).
        per_epoch_training_steps=98_000_000,
        learning_rate=0.75 / hidden_size**0.5,
        learning_rate_end=0.05 / hidden_size**0.5,
        optimizer=ed.EasyDeLOptimizers.ADAMW,
        scheduler=ed.EasyDeLSchedulers.COSINE,
        warmup_steps=2000,
        weight_decay=0.02,
        save_directory=SAVE_DIRECTORY,
        save_steps=1_000,
        save_total_limit=0,
        save_optimizer_state=False,
        clip_grad=1.0,
        report_steps=50,
        log_steps=10,
        progress_bar_type="json",
        track_memory=True,
        use_grain=False,
    )

    # pprint(jax.tree.map(lambda x: x.shape if hasattr(x, "shape") else x, model.parameters))

    # with jax.profiler.trace("./jax-trace", create_perfetto_link=False):

    # --- Streaming Dataset Setup ---
    # informs = [
    #     ed.TextDatasetInform(content_field="tokens", path="dvruette/nemotron-cc-65btok", split="train"),
    #     # ed.TextDatasetInform( # sample of reading from bucket.
    #     #     content_field="text",
    #     #     data_files="gs://your-bucket/raw/dclm/a3b142c/**",
    #     #     split="train",
    #     #     path=ed.DatasetType.JSON,
    #     # ),
    #     # ed.TextDatasetInform(
    #     #     content_field="content",
    #     #     data_files="gs://your-bucket/raw/starcoderdata-720c8c/9fc30b5/**/*.parquet",
    #     #     split="train",
    #     # ),
    # ]
    # mixture = ed.DatasetMixture(batch_size=1, informs=informs)
    # train_dataset = ed.DataManager.create_dataset_from_mixture(mixture)

    train_dataset = load_dataset(
        "dvruette/nemotron-cc-65btok",
        split="train",
        num_proc=32,
    )

    # --- Trainer Setup and Execution ---
    trainer = DiffusionTrainer(
        arguments=arguments,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=None,
    )

    # trainer.memory_monitor.start_monitoring()

    logger.info("Starting training...")
    trainer.train()

if __name__ == "__main__":
    out = main()
