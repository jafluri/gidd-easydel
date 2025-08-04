import os
from pprint import pprint

from args import parse_args
from train import train


SAVE_DIRECTORY = os.environ.get("SAVE_DIRECTORY", "outputs/diffusion_trainer")

WANDB_ENTITY = os.environ.get("WANDB_ENTITY", None)

DATA_FILES = os.environ.get("DATA_FILES", "/local/home/dvruette/nemotron_tokenized/**/*.parquet")


def main():
    try:
        args = parse_args(SAVE_DIRECTORY, WANDB_ENTITY, DATA_FILES)
        pprint(args)
        train(args)
    except Exception as e:
        import traceback
        print("An error occurred during training:")
        traceback.print_exc()
        raise e


if __name__ == "__main__":
    out = main()
