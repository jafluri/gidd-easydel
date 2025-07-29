import argparse
import os
from pprint import pprint


SAVE_DIRECTORY = os.environ.get("SAVE_DIRECTORY", "outputs/diffusion_trainer")

TOKENIZER_ID = "dvruette/nemotron-cc-bpe"

WANDB_ENTITY = os.environ.get("WANDB_ENTITY", None)

DATA_FILES = os.environ.get("DATA_FILES", "/local/home/dvruette/nemotron_tokenized/**/*.parquet")


def parse_args():
    parser = argparse.ArgumentParser(description="Run the diffusion training process.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument("--max_seq_len", type=int, default=512, help="Maximum sequence length for the model.")
    parser.add_argument("--batch_size", type=int, default=16, help="Total batch size for training.")
    parser.add_argument("--num_layers", type=int, default=12, help="Number of layers in the model.")
    parser.add_argument("--hidden_size", type=int, default=768, help="Hidden size of the model.")
    parser.add_argument("--head_dim", type=int, default=64, help="Dimension of each attention head.")
    parser.add_argument("--lr", type=float, default=0.75, help="Learning rate for the optimizer.")
    parser.add_argument("--init_scale", type=float, default=0.4, help="Initial scale for model parameters.")
    parser.add_argument("--emb_init_scale", type=float, default=0.1, help="Initial scale for embedding parameters.")
    parser.add_argument("--resid_scale", type=float, default=4.0, help="Scale for residual connections.")
    parser.add_argument("--tokenizer_id", type=str, default="dvruette/nemotron-cc-bpe", help="Tokenizer ID for the model.")
    parser.add_argument("--save_directory", type=str, default=SAVE_DIRECTORY, help="Directory to save model checkpoints.")
    parser.add_argument("--wandb_entity", type=str, default=WANDB_ENTITY, help="Weights & Biases entity for logging.")
    parser.add_argument("--data_files", type=str, default=DATA_FILES, help="Path to training data files.")
    return parser.parse_args()


def main():
    """
    The main function for the training process.
    """
    # Imports are inside the function to ensure they are available in the
    # separate Ray worker process.
    from train import train  # noqa

    try:
        args = parse_args()
        pprint(args)
        train(args)
    except Exception as e:
        import traceback
        print("An error occurred during training:")
        traceback.print_exc()
        raise e


if __name__ == "__main__":
    out = main()
