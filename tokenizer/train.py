"""
Assumes a dataset of jsonl files in the same format as the neox training set.
"""

import argparse
import random
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob

import tqdm
import zstd
from loguru import logger
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers, Regex, normalizers
from transformers import PreTrainedTokenizerFast

BOS_TOKEN = "<|begin_of_text|>"
EOS_TOKEN = "<|end_of_text|>"
PAD_TOKEN = "<|padding|>"
MASK_TOKEN = "<|mask|>"
WHITESPACE_TOKENS = [
    *["  " * i for i in range(1, 13)],
]



def _decompress_and_parse(j):
    with open(j, "rb") as f:
        byte_data = f.read()
        try:
            data = zstd.decompress(byte_data)
            data = data.decode("utf-8")
            docs = [json.loads(line.rstrip("\n|\r")) for line in data.splitlines()]
            return docs
        except zstd.Error as e:
            logger.error(f"Error decompressing {j}")


def _decompress_and_count_lines(j):
    docs = _decompress_and_parse(j)
    if docs is None:
        return 0, 0
    lines = len(docs)
    chars = sum(len(doc["text"]) for doc in docs if "text" in doc)
    return lines, chars


def get_lines_and_chars(all_jsonls, max_workers=32, show_progress=True):
    """
    Returns the total number of lines in all jsonl files
    :param all_jsonls: list of jsonl files
    :return: int, total number of lines
    """
    total_lines = 0
    total_chars = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_decompress_and_count_lines, j) for j in all_jsonls]
        for future in tqdm.tqdm(as_completed(futures), total=len(futures), disable=not show_progress, desc="Counting lines", dynamic_ncols=True):
            try:
                lines, chars = future.result()
                total_lines += lines
                total_chars += chars
            except zstd.Error as e:
                logger.error(f"Error decompressing: {e}")
    return total_lines, total_chars


def json_iterator(all_jsonls, text_key="text"):
    for j in all_jsonls:
        docs = _decompress_and_parse(j)
        if docs is None:
            continue
        for doc in docs:
            yield doc[text_key]
        


def train_tokenizer(
    input_dir: str, save_path: str, tokenizer_type: str = "BPE", vocab_size: int = 131_072, max_num_files: int = None
):
    """
    Trains a tokenizer on all the json files in `input_dir` and saves it to `save_path`

    :param input_dir: input directory containing jsonl files
    :param save_path: path to save tokenizer to
    :param tokenizer_type: type of tokenizer to train.
    :param vocab_size: int, size of tokenizer's vocab
    :return:
    """

    all_jsonls = sorted(glob(f"{input_dir}/**/*.jsonl.zstd", recursive=True))
    rng = random.Random(0)
    rng.shuffle(all_jsonls)
    if max_num_files is not None:
        all_jsonls = all_jsonls[:max_num_files]

    total_lines, total_chars = get_lines_and_chars(all_jsonls)
    print(f"Total lines: {total_lines}, Total characters: {total_chars} ({total_chars / 1024**3:.2f} GB)")

    print(f"Training tokenizer on {len(all_jsonls)} files")

    if tokenizer_type == "BPE":
        model = models.BPE()
    else:
        raise NotImplementedError(f"Tokenizer type {tokenizer_type} not implemented")
    tokenizer = Tokenizer(model)

    # Customize pre-tokenization and decoding
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        # pre_tokenizers.UnicodeScripts(),
        pre_tokenizers.Split(
            Regex(r"[^\r\n\p{L}\p{N}]?[\p{L}]+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"),
            behavior="isolated",
            invert=False
        ),
        pre_tokenizers.ByteLevel(add_prefix_space=True, trim_offsets=True, use_regex=False),
    ])
    tokenizer.decoder = decoders.ByteLevel(add_prefix_space=True, trim_offsets=True, use_regex=False)
    tokenizer.normalizer = normalizers.NFC()

    # And then train
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, special_tokens=[BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, MASK_TOKEN, *WHITESPACE_TOKENS]
    )

    tokenizer.model._clear_cache()
    tokenizer.model._resize_cache(16000)
    tokenizer.train_from_iterator(json_iterator(all_jsonls), trainer, length=total_lines)

    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"{BOS_TOKEN} $A {EOS_TOKEN}",
        pair=f"{BOS_TOKEN} $A {EOS_TOKEN} $B {EOS_TOKEN}",
        special_tokens=[
            (BOS_TOKEN, tokenizer.token_to_id(BOS_TOKEN)),
            (EOS_TOKEN, tokenizer.token_to_id(EOS_TOKEN)),
        ]
    )

    # And Save it
    if save_path:
        # tokenizer.save(save_path, pretty=True)
        tokenizer_fast = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            bos_token=BOS_TOKEN,
            eos_token=EOS_TOKEN,
            pad_token=PAD_TOKEN,
            mask_token=MASK_TOKEN,
        )
        tokenizer_fast.add_tokens(WHITESPACE_TOKENS, special_tokens=False)
        tokenizer_fast.save_pretrained(save_path)
        print(f"Tokenizer saved at {save_path}")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(
        description="script for training a multilingual "
        "HF tokenizer on CC dumps with upweighting for low resource languages"
    )
    parser.add_argument(
        "--json_input_dir",
        type=str,
        help="Path to folder containing tokenizer training data in jsonl format",
    )
    parser.add_argument(
        "--max_num_files",
        type=int,
        default=None,
        help="Maximum number of files to use for training. If None, all files will be used.",
    )
    parser.add_argument(
        "--tokenizer_output_path",
        type=str,
        help="Path to which your trained tokenizer will be saved (should be a directory)",
    )
    parser.add_argument(
        "--tokenizer_type",
        type=str,
        help="type of tokenizer to train, currently only BPE is supported",
        choices=["BPE"],
        default="BPE",
    )
    parser.add_argument(
        "-v",
        "--vocab_size",
        help="vocabulary size of tokenizer, default=131k",
        type=int,
        default=131_072,
    )
    args_parsed = parser.parse_args(input_args)
    return args_parsed


def main(args):
    train_tokenizer(
        args.json_input_dir,
        save_path=args.tokenizer_output_path,
        tokenizer_type=args.tokenizer_type,
        vocab_size=args.vocab_size,
        max_num_files=args.max_num_files,
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)