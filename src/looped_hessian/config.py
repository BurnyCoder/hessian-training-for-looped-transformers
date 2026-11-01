"""Experiment configuration: dataclasses with the preregistered defaults, plus .env paths.

Local: every number here is either taken from the cited papers or fixed in
reports/preregistration.md; the resolved config of each run is written to its run folder
(`config.json`) so results are reproducible from the folder alone.

Global: `main.py` builds a `RunConfig` from these defaults and CLI flags, and passes the
sub-configs to training (`train.py`), evaluation (`evaluate.py`) and curvature measurement
(`measure.py`).
"""

import os
from dataclasses import asdict, dataclass, field

from dotenv import load_dotenv

from looped_hessian.model import ModelConfig

# Paths and device come from .env (see .env.example); code defaults apply when unset.
load_dotenv()


@dataclass
class TrainConfig:
    arm: str = "baseline"  # "baseline" or "hutchinson" (Stage 1 arms)
    steps: int = 30_000  # set by the preregistered throughput rule (addendum 1)
    batch_size: int = 64  # Fan et al. 2024 / Kuo et al. 2026
    lr: float = 1e-4  # Fan et al. / Kuo et al.: AdamW 1e-4, cosine decay to 0
    weight_decay: float = 0.01  # torch.optim.AdamW default
    beta1: float = 0.9
    beta2: float = 0.999
    grad_clip: float = 1.0
    seed: int = 0
    lam: float = 5e-5  # λ = σ²/2 with σ = 0.01, NSO's central σ (preregistration Sec. 4)
    sub_batch: int = 4  # problems used for the penalty (addendum 1: 16 needs 10 GB > 8 GB VRAM)
    penalty_every: int = 8  # apply the penalty every k steps with weight k·λ (addendum 1)
    max_train_len: int = 19  # train on n ∈ [1, 19] (Fan et al.: lengths [1, 20))
    loop_jitter: int = 5  # Kuo et al.: K ~ U{n-5, ..., n+5}
    max_train_loops: int = 20  # Kuo et al.: T_max = 20
    log_every: int = 100
    curvature_every: int = 1000  # in-training 10-probe tr(H) on a fixed 128-problem batch
    curvature_probes: int = 10
    curvature_batch: int = 128
    eval_every: int = 2000  # quick learning-curve evaluation (not the final evaluation)
    eval_count: int = 200


@dataclass
class EvalConfig:
    id_lengths: tuple = tuple(range(1, 20))
    ood_lengths: tuple = tuple(range(20, 61, 5))  # Kuo et al.: {20, 25, ..., 60}
    n_per_length: int = 1000
    extra_loops: int = 5  # evaluate K = 1 .. n + 5
    seed_base: int = 10_000  # test set for length n uses seed seed_base + n
    chunk_size: int = 250


@dataclass
class MeasureConfig:
    batch: int = 512  # fixed measurement batch from the training distribution
    seed: int = 20_000
    probes: int = 100
    power_iters: int = 50
    power_tol: float = 1e-3
    fisher_samples: int = 10
    sharpness_rhos: tuple = (0.01, 0.02, 0.05)
    sharpness_draws: int = 10
    loop_ks: tuple = (1, 2, 4, 8, 12, 16, 20)  # loop-count curvature study
    loop_probes: int = 20
    loop_power_iters: int = 20
    chunk_size: int = 32


@dataclass
class RunConfig:
    name: str
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    measure: MeasureConfig = field(default_factory=MeasureConfig)
    output_dir: str = field(default_factory=lambda: os.getenv("LH_OUTPUT_DIR", "results"))
    checkpoint_dir: str = field(default_factory=lambda: os.getenv("LH_CHECKPOINT_DIR", "checkpoints"))
    device: str = field(default_factory=lambda: os.getenv("LH_DEVICE", "cuda"))

    @property
    def run_dir(self) -> str:
        return os.path.join(self.output_dir, self.name)

    @property
    def ckpt_dir(self) -> str:
        return os.path.join(self.checkpoint_dir, self.name)

    def to_dict(self) -> dict:
        return asdict(self)
