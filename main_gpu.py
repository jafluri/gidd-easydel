import os
import multiprocessing as mp
from pprint import pprint

from args import parse_args

SAVE_DIRECTORY = os.environ.get("SAVE_DIRECTORY", "outputs/diffusion_trainer")

WANDB_ENTITY = os.environ.get("WANDB_ENTITY", None)

DATA_FILES = os.environ.get("DATA_FILES", "/local/home/dvruette/nemotron-cc_tokenized_shuffled/")


def main():
        from gidd_easydel.train import train
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
    mp.freeze_support()
    mp.set_start_method('forkserver', force=True)
    out = main()
