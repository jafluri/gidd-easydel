

def create_constant_length_dataset(
    processing_class,
    dataset,
    dataset_text_field: str | None = None,
    formatting_func: tp.Callable | None = None,
    infinite: bool = False,
    seq_length: int = 1024,
    num_of_sequences: int = 1024,
    chars_per_token: float = 3.6,
    eos_token_id: int = 0,
    shuffle: bool = True,
    append_concat_token: bool = True,
    add_special_tokens: bool = True,
) -> tp.Callable[[], tp.Iterator[dict[str, jnp.ndarray]]]:
    """
    Creates a generator function that yields constant length chunks of tokens from a stream of text files.

    Args:
        processing_class: The processor used for processing the data.
        dataset: Dataset with text files.
        dataset_text_field: Name of the field in the dataset that contains the text.
        formatting_func: Function that formats the text before tokenization.
        infinite: If True the iterator is reset after dataset reaches end else stops.
        seq_length: Length of token sequences to return.
        num_of_sequences: Number of token sequences to keep in buffer.
        chars_per_token: Number of characters per token used to estimate number of tokens in text buffer.
        eos_token_id: Id of the end of sequence token if the passed processing_class does not have an EOS token.
        shuffle: Shuffle the examples before they are returned.
        append_concat_token: If true, appends eos_token_id at the end of each sample being packed.
        add_special_tokens: If true, processing_class adds special tokens to each sample being packed.

    Returns:
        A generator function that yields dictionaries containing input_ids and attention_mask as jnp.arrays
    """
    if processing_class.eos_token_id is None:
        warnings.warn(
            "The passed processing_class does not have an EOS token. We will use the passed eos_token_id instead which "
            f"corresponds to {eos_token_id}. If this is not the correct EOS token, "
            "make sure to pass the correct eos_token_id.",
            stacklevel=1,
        )

    concat_token_id = processing_class.eos_token_id if processing_class.eos_token_id else eos_token_id
    max_buffer_size = seq_length * chars_per_token * num_of_sequences

    # Input validation and formatting function setup
    if dataset_text_field is not None and formatting_func is not None:
        warnings.warn(
            "Only one of `dataset_text_field` and `formatting_func` should be provided. "
            "Ignoring `dataset_text_field` and using `formatting_func`.",
            stacklevel=1,
        )

    if formatting_func is not None:
        if formatting_func.__code__.co_argcount > 1:
            warnings.warn(
                "The passed formatting_func has more than one argument. Usually that "
                "function should have a single argument `example` which corresponds "
                "to the dictionary returned by each element of the dataset. Make sure you know what you are doing.",
                stacklevel=1,
            )
    elif dataset_text_field is not None:
        formatting_func = lambda x: x[dataset_text_field]  # noqa
    else:
        raise ValueError("Either `dataset_text_field` or `formatting_func` should be provided.")

    def constant_length_generator() -> tp.Iterator[dict[str, jnp.ndarray]]:
        iterator = iter(dataset)
        more_examples = True

        while more_examples:
            buffer, buffer_len = [], 0

            # Fill the buffer
            while True:
                if buffer_len >= max_buffer_size:
                    break
                try:
                    prompt = formatting_func(next(iterator))
                    if isinstance(prompt, list):
                        prompt = "".join(p for p in prompt)
                    buffer.append(prompt)
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    if infinite:
                        iterator = iter(dataset)
                        warnings.warn(
                            "The dataset reached end and the iterator is reset to the start.",
                            stacklevel=1,
                        )
                    else:
                        more_examples = False
                        break

            if shuffle:
                random.shuffle(buffer)

            # Tokenize all texts in the buffer
            tokens = processing_class(
                text=buffer,
                add_special_tokens=add_special_tokens,
                truncation=False,
            )
            tokenized_inputs = tokens["input_ids"]
            attention_masks = tokens["attention_mask"]
            # Concatenate all tokens and attention masks
            all_token_ids = []
            all_attention_masks = []
            for tokenized_input, attention_mask in zip(tokenized_inputs, attention_masks, strict=False):
                if append_concat_token:
                    tokenized_input = [*tokenized_input, concat_token_id]
                    attention_mask = [*attention_mask, 1]
                all_token_ids.extend(tokenized_input)
                all_attention_masks.extend(attention_mask)

            # Create fixed-length examples
            examples = []
            examples_attention_masks = []
            for i in range(0, len(all_token_ids), seq_length):
                input_ids = all_token_ids[i : i + seq_length]
                org_attention_masks = all_attention_masks[i : i + seq_length]
                if len(input_ids) == seq_length:
                    examples.append(input_ids)
                    examples_attention_masks.append(org_attention_masks)

            if shuffle:
                # Shuffle examples while keeping pairs together
                combined = list(zip(examples, examples_attention_masks, strict=False))
                random.shuffle(combined)
                examples, examples_attention_masks = zip(*combined, strict=False)

            # Yield examples
            for example, example_attention_mask in zip(examples, examples_attention_masks, strict=False):
                yield {
                    "input_ids": jnp.asarray(example, dtype="i4"),
                    "attention_mask": jnp.asarray(example_attention_mask, dtype="i4"),
                }

    return constant_length_generator