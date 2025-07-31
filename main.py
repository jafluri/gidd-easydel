import argparse
import os
from pprint import pprint

import ray
from eformer.executor.ray import TpuAcceleratorConfig, execute

# Initialize Ray for distributed computing. This must be done once per application.
ray.init()

SAVE_DIRECTORY = os.environ.get("SAVE_DIRECTORY", "outputs/diffusion_trainer")

WANDB_ENTITY = os.environ.get("WANDB_ENTITY", None)

DATA_FILES = os.environ.get("DATA_FILES", "gs://nemotron-cc_europe-west4/Nemotron-CC/**/*.parquet")

# --- Environment and TPU Configuration ---
# These environment variables are passed to each Ray worker to ensure they have
# access to necessary tokens and use efficient shared memory for caching.
EXECUTION_ENV_VARS = {
    "EASYDEL_AUTO": "1",  # Enables EasyDeL's automatic sharding configuration.
    "HF_TOKEN": os.environ.get("HF_TOKEN_FOR_EASYDEL", ""),  # Hugging Face token.
    "HF_DATASETS_CACHE": "/dev/shm/huggingface-dataset",  # RAM-disk for dataset cache.
    "HF_HOME": "/dev/shm/huggingface",  # RAM-disk for model cache.
    "HF_DATASETS_OFFLINE": "0",  # Allow online dataset access.
    "WANDB_API_KEY": os.environ.get("WANDB_API_KEY_FOR_EASYDEL", ""),  # W&B API key.
}

# Additional pip packages to install on each Ray worker environment.
PIP_PACKAGES = []

# Print the environment variables for verification.
pprint(EXECUTION_ENV_VARS)

# Defines the TPU environment for Ray, specifying the accelerator type and worker setup.
acc_config = TpuAcceleratorConfig(
    "v3-8",
    execution_env={
        "env_vars": EXECUTION_ENV_VARS,
        "pip": PIP_PACKAGES,
    },
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the diffusion training process.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument("--max_seq_len", type=int, default=512, help="Maximum sequence length for the model.")
    parser.add_argument("--batch_size", type=int, default=8, help="Total batch size for training.")
    parser.add_argument("--num_layers", type=int, default=8, help="Number of layers in the model.")
    parser.add_argument("--hidden_size", type=int, default=512, help="Hidden size of the model.")
    parser.add_argument("--head_dim", type=int, default=64, help="Dimension of each attention head.")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate for the optimizer.")
    parser.add_argument("--init_scale", type=float, default=0.4, help="Initial scale for model parameters.")
    parser.add_argument("--emb_init_scale", type=float, default=0.1, help="Initial scale for embedding parameters.")
    parser.add_argument("--resid_scale", type=float, default=4.0, help="Scale for residual connections.")
    parser.add_argument("--hybrid_mixing_scale", type=float, default=1.0, help="Scale for hybrid mixing schedule.")
    parser.add_argument("--hybrid_mixing_shift", type=float, default=0.0, help="Shift for hybrid mixing schedule.")
    parser.add_argument("--tokenizer_id", type=str, default="dvruette/nemotron-cc-bpe", help="Tokenizer ID for the model.")
    parser.add_argument("--save_directory", type=str, default=SAVE_DIRECTORY, help="Directory to save model checkpoints.")
    parser.add_argument("--wandb_entity", type=str, default=WANDB_ENTITY, help="Weights & Biases entity for logging.")
    parser.add_argument("--data_files", type=str, default=DATA_FILES, help="Path to training data files.")
    return parser.parse_args()

ARGS = parse_args()


@execute(acc_config)
@ray.remote
def main():
    """
    The main function for the distillation training process, executed as a
    remote task on the TPU cluster via Ray.
    """
    # Imports are inside the function to ensure they are available in the
    # separate Ray worker process.
    from train import train  # noqa

    try:
        pprint(ARGS)
        train(ARGS)
    except Exception as e:
        import traceback
        print("An error occurred during training:")
        traceback.print_exc()
        raise e


if __name__ == "__main__":
    out = main()
