import typing as tp
import warnings
import random
import numpy as np
import jax.numpy as jnp


def create_constant_length_dataset(
    dataset,
    tokens_field: str | None = None,
    infinite: bool = False,
    seq_length: int = 1024,
    eos_token_id: int = 0,
    shuffle: bool = True,
    batch_size: int = 512,
    shuffle_buffer_batch_factor: int = 16,
    append_concat_token: bool = True,
) -> tp.Callable[[], tp.Iterator[dict[str, jnp.ndarray]]]:
    """
    Creates a generator function that yields constant length chunks of tokens from a stream of text files.

    Args:
        dataset: Dataset with text files.
        tokens_field: Name of the field in the dataset that contains the text.
        infinite: If True the iterator is reset after dataset reaches end else stops.
        seq_length: Length of token sequences to return.
        eos_token_id: Id of the end of sequence token if the passed processing_class does not have an EOS token.
        shuffle: Shuffle the examples before they are returned.
        batch_size: Batch size for the dataset. Used to compute the shuffle buffer size.
        shuffle_buffer_batch_factor: Factor to compute the shuffle buffer size. The shuffle buffer size is
            `batch_size * shuffle_buffer_batch_factor`.
        append_concat_token: If true, appends eos_token_id at the end of each sample being packed.

    Returns:
        A generator function that yields fixed-length token arrays as jnp.ndarray
    """

    def constant_length_generator() -> tp.Iterator[jnp.ndarray]:
        iterator = iter(dataset)

        buffer: list[int] = []
        shuffle_buffer: list[jnp.ndarray] = []
        shuffle_buffer_size = batch_size * shuffle_buffer_batch_factor

        while True:
            try:
                tokens = next(iterator)[tokens_field]
                if not isinstance(tokens, list):
                    assert isinstance(tokens, (np.ndarray, jnp.ndarray)), (
                        f"Expected tokens to be a list or np.ndarray or jnp.ndarray, got {type(tokens)}"
                    )
                    tokens = tokens.tolist()
                # append EOS token
                buffer.extend(tokens)
                if append_concat_token and len(buffer) % seq_length != 0:
                    tokens.append(eos_token_id)

                while len(buffer) >= seq_length:
                    # Pop the first seq_length tokens to form a complete example
                    example = {"input_ids": jnp.array(buffer[:seq_length], dtype="i4")}
                    buffer = buffer[seq_length:]
                    if shuffle:
                        if len(shuffle_buffer) < shuffle_buffer_size:
                            shuffle_buffer.append(example)
                        else:
                            idx = random.randrange(0, shuffle_buffer_size)
                            yield shuffle_buffer[idx]
                            shuffle_buffer[idx] = example
                    else:
                        yield example
            except StopIteration:
                if infinite:
                    iterator = iter(dataset)
                    warnings.warn(
                        "The dataset reached end and the iterator is reset to the start.",
                        stacklevel=1,
                    )
                else:
                    break

        if len(shuffle_buffer) == 0:
            raise ValueError(
                "The dataset is empty or does not contain enough samples to yield a single packed sequence."
            )

        for idx in random.sample(range(len(shuffle_buffer)), len(shuffle_buffer)):
            yield shuffle_buffer[idx]


    return constant_length_generator