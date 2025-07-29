
import easydel as ed  # noqa

import random

import jax
from jax import numpy as jnp
from transformers import AutoTokenizer
from datasets import load_dataset

from diffusion_trainer import DiffusionTrainer, DiffusionConfig

def train(args):
    # jax.config.update('jax_disable_jit', True)


    logger = ed.utils.get_logger(__name__)

    logger.info("Process count: %d, device count: %d, process index: %d",
                jax.process_count(), jax.local_device_count(), jax.process_index())

    # --- Basic Training Parameters ---
    seed = args.seed
    random.seed(seed)

    max_length = args.max_seq_len
    total_batch_size = args.batch_size

    num_layers = args.num_layers
    hidden_size = args.hidden_size
    head_dim = args.head_dim

    lr = args.lr / hidden_size**0.5
    init_scale = args.init_scale
    emb_init_scale = args.emb_init_scale
    resid_scale = args.resid_scale

    # lr = 5e-4
    # init_scale = 0.02
    # emb_init_scale = 0.02
    # resid_scale = num_layers

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id)

    # --- Model Definition ---
    model = ed.GiddForDiffusionLM(
        config=ed.GiddConfig(
            vocab_size=len(tokenizer),
            hidden_size=hidden_size,
            intermediate_size=4*hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=hidden_size // head_dim,
            head_dim=head_dim,
            max_position_embeddings=max_length,
            resid_scale=resid_scale,
            init_scale=init_scale,
            emb_init_scale=emb_init_scale,
            head_init_scale=0.0,
            use_qk_norm=True,
            sharding_axis_dims=(1, jax.process_count(), 1, -1, 1),  # FSDP
            # sharding_axis_dims=(-1, 1, 1, 1, 1),  # DP
            # sharding_axis_dims=(1, 1, 1, -1, 1),  # TP
            partition_axis=ed.PartitionAxis(),
            gradient_checkpointing=ed.EasyDeLGradientCheckPointers.NOTHING_SAVEABLE,
            attn_mechanism=ed.AttentionMechanisms.SDPA,
            attn_dtype=jnp.bfloat16,
            attention_bias=False,
            mlp_bias=True,
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
        wandb_entity=args.wandb_entity,
        do_last_save=True,
        max_sequence_length=max_length,
        # This is MANDATORY for streaming datasets. It tells the trainer how many
        # steps constitute one "epoch". Should be ~ (total_dataset_size // total_batch_size).
        per_epoch_training_steps=98_000_000,
        learning_rate=lr,
        learning_rate_end=0.1 * lr,
        optimizer=ed.EasyDeLOptimizers.ADAMW,
        scheduler=ed.EasyDeLSchedulers.COSINE,
        warmup_steps=2000,
        weight_decay=0.02,
        save_directory=args.save_directory,
        save_steps=1_000,
        save_total_limit=0,
        save_optimizer_state=False,
        clip_grad=1.0,
        report_steps=50,
        log_steps=10,
        progress_bar_type="json",
        track_memory=60,
        use_grain=False,
    )

    train_dataset = load_dataset(
        "parquet",
        data_files=args.data_files,
        split="train",
        streaming=True,
    )
    train_dataset = train_dataset.shuffle(seed=random.randint(0, 2**32 - 1))

    # --- Trainer Setup and Execution ---
    trainer = DiffusionTrainer(
        arguments=arguments,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=None,
        seed=seed,
    )

    if trainer.memory_monitor is not None:
        trainer.memory_monitor.start_monitoring()

    logger.info("Starting training...")
    trainer.train()
