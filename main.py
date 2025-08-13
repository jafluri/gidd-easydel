import os
from pprint import pprint

import ray
from eformer.executor.ray import TpuAcceleratorConfig, execute

# Initialize Ray for distributed computing. This must be done once per application.
ray.init(runtime_env={"py_modules": [os.path.join(os.getcwd(), "gidd_easydel")]})

SAVE_DIRECTORY = os.environ.get("SAVE_DIRECTORY", "outputs/diffusion_trainer")

WANDB_ENTITY = os.environ.get("WANDB_ENTITY", None)

DATA_FILES = os.environ.get("DATA_FILES", "gs://nemotron-cc_europe-west4/Nemotron-CC/**/*.parquet")

TPU_VERSION = os.environ.get("TPU_VERSION", "v3-8")

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
# PIP_PACKAGES = ["dask"]
PIP_PACKAGES = []

# Print the environment variables for verification.
pprint(EXECUTION_ENV_VARS)

# Defines the TPU environment for Ray, specifying the accelerator type and worker setup.
acc_config = TpuAcceleratorConfig(
    TPU_VERSION,
    execution_env={
        "env_vars": EXECUTION_ENV_VARS,
        "pip": PIP_PACKAGES,
    },
)


from args import parse_args
ARGS = parse_args(SAVE_DIRECTORY, WANDB_ENTITY, DATA_FILES)


@execute(acc_config)
@ray.remote
def main():
    """
    The main function for the distillation training process, executed as a
    remote task on the TPU cluster via Ray.
    """
    # Imports are inside the function to ensure they are available in the
    # separate Ray worker process.
    from gidd_easydel.train import train  # noqa

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
