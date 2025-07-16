from dataclasses import field

from eformer.pytree import auto_pytree

from easydel.utils.compiling_utils import hash_fn
from easydel.trainers.training_configurations import TrainingArguments


@auto_pytree
class DiffusionConfig(TrainingArguments):
    trainer_prefix: str | None = field(
        default="diffusiontrainer", metadata={"help": "default prefix name for trainer."}
    )
    # loss arguments
    beta_is_div: float = field(default=1.0, metadata={"help": "Weight of the IS-divergence loss"})
    # mixing schedule
    min_log_snr: float = field(default=-10.0, metadata={"help": "Min. log-SNR for the mixing schedule"})
    max_log_snr: float = field(default=-10.0, metadata={"help": "Max. log-SNR for the mixing schedule"})
    mixing_sigmoid_scale: float = field(default=1.0, metadata={
        "help": (
            "Sigmoid scale of the mixing distribution. "
            "Determines how quickly mixing transitions from uniform to masking."
        )
    })
    mixing_sigmoid_bias: float = field(default=0.0, metadata={
        "help": (
            "Sigmoid bias of the mixing distribution. "
            "Determines at which SNR mixing transitions from uniform to masking."
        )
    })
    # noise sampling
    noise_p_independent: float = field(default=1.0, metadata={"help": "Probability that SNR is sampled is independently for each token."})
    noise_p_linear: float = field(default=0.0, metadata={"help": "Probability that SNR is sampled linearly increasing in token position."})
    noise_mask_p_prompt: float = field(default=0.1, metadata={"help": "Probability that a sample has a noise-free prompt."})
    noise_mask_p_infilling: float = field(default=0.1, metadata={"help": "Probability that a sample has noise-free tokens (chosen uniformly at random)."})
    noise_mask_max_cond_frac: float = field(default=1.0, metadata={"help": "Max. fraction of noise-free tokens for each conditioning type (prompt, infilling)."})

    __hash__ = hash_fn
