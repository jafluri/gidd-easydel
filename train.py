
import functools
import random
import typing as tp
from copy import deepcopy

import easydel as ed

import jax
import optax
import chex
from jax import numpy as jnp
from transformers import AutoTokenizer
from datasets import load_dataset
from eformer.optimizers import OptimizerFactory, SchedulerFactory, SchedulerConfig

from diffusion_trainer import DiffusionTrainer, DiffusionConfig
from model import GiddForDiffusionLM, GiddConfig


def wsd_lr_schedule(total_steps: int, base_lr: float, warmup_steps: int = 0, cooldown_steps: int = 0) -> tp.Callable[[chex.Numeric], chex.Numeric]:
    """
    Implements a warmup-stable-decay learning rate schedule.
    
    Args:
        base_lr (float): Base learning rate.
        warmup_steps (int): Number of steps for warmup.
        cooldown_steps (int): Number of steps for decay.
        total_steps (int): Total number of training steps.
        curr_step (int): Current training step.
    
    Returns:
        float: Adjusted learning rate for the current step.
    """

    def sqrt1m_schedule(init_value: float, decay_steps: int) -> tp.Callable[[chex.Numeric], chex.Numeric]:
        def schedule(count: chex.Numeric) -> chex.Numeric:
            count = jnp.minimum(count, decay_steps)
            return init_value * (1 - (count / decay_steps)**0.5)
        return schedule

    return optax.schedules.join_schedules([
        optax.schedules.linear_schedule(0, base_lr, warmup_steps),
        optax.schedules.constant_schedule(base_lr),
        sqrt1m_schedule(base_lr, cooldown_steps),
    ], [warmup_steps, total_steps - cooldown_steps])


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

    # lr = args.lr / hidden_size**0.5
    lr = args.lr
    init_scale = args.init_scale
    emb_init_scale = args.emb_init_scale
    resid_scale = args.resid_scale

    # lr = 5e-4
    # init_scale = 0.02
    # emb_init_scale = 0.02
    # resid_scale = num_layers

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id)

    # --- Model Definition ---
    model = GiddForDiffusionLM(
        config=GiddConfig(
            vocab_size=len(tokenizer),
            hidden_size=hidden_size,
            intermediate_size=4*hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=hidden_size // head_dim,
            head_dim=head_dim,
            is_causal=False,
            max_position_embeddings=max_length,
            resid_scale=resid_scale,
            init_scale=init_scale,
            emb_init_scale=emb_init_scale,
            head_init_scale=0.0,
            weight_scaling="fan_in",
            # weight_scaling=1.0,
            use_qk_norm=True,
            # sharding_axis_dims=(1, jax.process_count(), 1, -1, 1),  # FSDP + TP
            sharding_axis_dims=(1, -1, 1, 1, 1),  # FSDP
            # sharding_axis_dims=(-1, 1, 1, 1, 1),  # DP
            # sharding_axis_dims=(1, 1, 1, -1, 1),  # TP
            partition_axis=ed.PartitionAxis(),
            gradient_checkpointing=ed.EasyDeLGradientCheckPointers.NOTHING_SAVEABLE,
            # attn_mechanism=ed.AttentionMechanisms.SDPA,
            attn_mechanism=ed.AttentionMechanisms.VANILLA,
            attn_dtype=jnp.bfloat16,
            # attn_dtype=jnp.float32,
            attention_bias=False,
            mlp_bias=True,
            # scan_layers=True,
        ),
        dtype=jnp.bfloat16,
        param_dtype=jnp.bfloat16,
        # dtype=jnp.float32,
        # param_dtype=jnp.float32,
        precision=jax.lax.Precision.HIGH,
        rngs=ed.Rngs(0),
    )#.shard_model()  # Shard the newly created model across devices.


    # def print_array_type(p):
    #     jax.debug.inspect_array_sharding(p, callback=print)
    #     jax.debug.visualize_array_sharding(p)

    # jax.tree.map(print_array_type, model.graphstate)

    # model = ed.LlamaForCausalLM(
    #     config=ed.LlamaConfig(
    #         vocab_size=len(tokenizer),
    #         hidden_size=hidden_size,
    #         intermediate_size=4*hidden_size,
    #         num_hidden_layers=num_layers,
    #         num_attention_heads=hidden_size // head_dim,
    #         head_dim=head_dim,
    #         use_cache=False,
    #         max_position_embeddings=max_length,
    #         sharding_axis_dims=(1, jax.process_count(), 1, -1, 1),  # FSDP
    #         # sharding_axis_dims=(-1, 1, 1, 1, 1),  # DP
    #         # sharding_axis_dims=(1, 1, 1, -1, 1),  # TP
    #         partition_axis=ed.PartitionAxis(),
    #         gradient_checkpointing=ed.EasyDeLGradientCheckPointers.NOTHING_SAVEABLE,
    #         attn_mechanism=ed.AttentionMechanisms.VANILLA,
    #         attn_dtype=jnp.bfloat16,
    #         # attn_dtype=jnp.float32,
    #         # scan_layers=True,
    #     ),
    #     dtype=jnp.bfloat16,
    #     param_dtype=jnp.bfloat16,
    #     # dtype=jnp.float32,
    #     # param_dtype=jnp.float32,
    #     precision=jax.lax.Precision.HIGH,
    #     rngs=ed.Rngs(0),
    # )

    # class CustomDiffusionConfig(DiffusionConfig):
    #     # override the `get_optimizer_and_scheduler` method to implement per-layer learning rates
    #     def get_optimizer_and_scheduler(self, steps):
    #         optimizer_kwargs = deepcopy(self.optimizer_kwargs)
    #         steps = optimizer_kwargs.pop("steps")
    #         warmup_steps = optimizer_kwargs.pop("warmup_steps", 0)
    #         cooldown_steps = optimizer_kwargs.pop("cooldown_steps", 0)
    #         clip_grad = optimizer_kwargs.pop("clip_grad", None)
    #         weight_decay = optimizer_kwargs.pop("weight_decay", 0.0)


    #         adam_kwargs = dict(b1=0.9, b2=0.99, eps=1e-12)

    #         bulk_schedule = wsd_lr_schedule(
    #             total_steps=steps,
    #             base_lr=lr / hidden_size**0.5,
    #             warmup_steps=warmup_steps,
    #             cooldown_steps=cooldown_steps,
    #         )
    #         qk_scale_schedule = wsd_lr_schedule(
    #             total_steps=steps,
    #             base_lr=lr,
    #             warmup_steps=warmup_steps,
    #             cooldown_steps=cooldown_steps,
    #         )
    #         other_schedule = wsd_lr_schedule(
    #             total_steps=steps,
    #             base_lr=lr / hidden_size**0.5,
    #             warmup_steps=warmup_steps,
    #             cooldown_steps=cooldown_steps,
    #         )

    #         def param_label_fn(params: tp.Any) -> str:
    #             def label_leaf(path: str, param: chex.Array) -> str:
    #                 path = ''.join(str(k) for k in path)

    #                 if "qk_scale" in path:
    #                     return "qk_scale"
    #                 elif "norm" in path:
    #                     return "other_wd_params"
    #                 elif "embed_tokens" in path or param.ndim < 2:
    #                     return "other_nowd_params"
    #                 else:
    #                     return "bulk_params"

    #             labels = jax.tree.map_with_path(label_leaf, params)
    #             return labels

    #         optimizer = optax.multi_transform({
    #             "bulk_params": optax.adamw(learning_rate=bulk_schedule, weight_decay=weight_decay, **adam_kwargs),
    #             "qk_scale": optax.adamw(learning_rate=qk_scale_schedule, weight_decay=0.0, **adam_kwargs),
    #             "other_wd_params": optax.adamw(learning_rate=other_schedule, weight_decay=weight_decay, **adam_kwargs),
    #             "other_nowd_params": optax.adamw(learning_rate=other_schedule, weight_decay=0.0, **adam_kwargs),
    #         }, param_label_fn)

    #         if clip_grad:
    #             tx = optax.chain(
    #                 optax.clip_by_global_norm(clip_grad),
    #                 optimizer,
    #             )
    #         else:
    #             tx = optimizer

    #         if optimizer_kwargs.get("gradient_accumulation_steps", 0) > 1:
    #             tx = optax.MultiSteps(tx, optimizer_kwargs["gradient_accumulation_steps"])
            
    #         # the LR schedule returned here is only used for logging purposes
    #         return optimizer, bulk_schedule

    # --- Configuration ---
    # arguments = CustomDiffusionConfig(
    arguments = DiffusionConfig(
        model_name="gidd",  # for wandb run name
        num_train_epochs=1,
        total_batch_size=total_batch_size,
        use_wandb=True,
        wandb_entity=args.wandb_entity,
        do_last_save=True,
        max_sequence_length=max_length,
        # This is MANDATORY for streaming datasets. It tells the trainer how many
        # steps constitute one "epoch". Should be ~ (total_dataset_size // total_batch_size).
        per_epoch_training_steps=98_000_000,
        max_training_steps=100_000,
        hybrid_mixing_scale=args.hybrid_mixing_scale,
        hybrid_mixing_shift=args.hybrid_mixing_shift,
        learning_rate=lr,
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
        track_memory=20.0,
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
    with jax.profiler.trace("/tmp/jax-trace"):
        trainer.train()
