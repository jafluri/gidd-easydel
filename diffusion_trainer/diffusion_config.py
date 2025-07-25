from dataclasses import field

from eformer.pytree import auto_pytree

from easydel.utils.compiling_utils import hash_fn
from easydel.trainers.training_configurations import TrainingArguments

from .schedule import MixingRate


@auto_pytree
class DiffusionConfig(TrainingArguments):
    trainer_prefix: str | None = field(
        default="diffusiontrainer", metadata={"help": "default prefix name for trainer."}
    )
    dataset_tokens_field: str = field(default="tokens", metadata={
        "help": "Name of the field in the dataset that contains the tokenized text."
    })
    # loss arguments
    beta_is_div: float = field(default=1.0, metadata={"help": "Weight of the IS-divergence loss"})
    # noise sampling
    noise_p_independent: float = field(default=0.5, metadata={"help": "Probability that SNR is sampled independently for each token."})
    noise_p_linear: float = field(default=0.0, metadata={"help": "Probability that SNR is sampled linearly increasing in token position."})
    noise_mask_p_prompt: float = field(default=0.1, metadata={"help": "Probability that a sample has a noise-free prompt."})
    noise_mask_p_infilling: float = field(default=0.1, metadata={"help": "Probability that a sample has noise-free tokens (chosen uniformly at random)."})
    noise_mask_max_cond_frac: float = field(default=1.0, metadata={"help": "Max. fraction of noise-free tokens for each conditioning type (prompt, infilling)."})
    # mixing schedule
    mixing_rate: str | MixingRate = field(default="linear", metadata={"help": "Mixing schedule rate, can be 'linear' or any instance of `MixingRate`."})
    min_log_snr: float = field(default=-9.0, metadata={"help": "Min. log-SNR for the mixing schedule"})
    max_log_snr: float = field(default=9.0, metadata={"help": "Max. log-SNR for the mixing schedule"})
    hybrid_mixing_scale: float = field(default=1.0, metadata={"help": "Sigmoid scale for the hybrid mixing schedule."})
    hybrid_mixing_shift: float = field(default=0.0, metadata={"help": "Sigmoid shift for the hybrid mixing schedule."})

    __hash__ = hash_fn
