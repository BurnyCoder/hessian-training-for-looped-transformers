"""Experiment configuration: dataclasses with the preregistered defaults, plus .env paths.

Local: protocol choices and machine settings are serialized into the run's config.json.
Reproduction also uses the matching code, environment and recorded data seeds.

Global: `main.py` builds a `RunConfig` from these defaults and CLI flags, and passes the
sub-configs to training (`train.py`), evaluation (`evaluate.py`) and curvature measurement
(`measure.py`).
"""

import json
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
    lr: float = 1e-4  # AdamW 1e-4 (Fan et al. / Kuo et al.), cosine decay to 0 over training (Kuo et al.)
    weight_decay: float = 0.01  # torch.optim.AdamW default
    beta1: float = 0.9
    beta2: float = 0.999
    grad_clip: float = 1.0
    seed: int = 0
    lam: float = 5e-5  # λ = σ²/2 with σ = 0.01, NSO's central σ (preregistration Sec. 4)
    sub_batch: int = 4  # first four problems, chosen after the timings recorded in Addendum 1
    penalty_every: int = 8  # apply the penalty every k steps with weight k·λ (addendum 1)
    max_train_len: int = 19  # train on n ∈ [1, 19] (Fan et al.: lengths [1, 20))
    loop_jitter: int = 5  # Kuo et al.: K ~ U{n-5, ..., n+5}
    max_train_loops: int = 20  # Kuo et al.: T_max = 20
    log_every: int = 100
    curvature_every: int = 2000  # scheduled interim probe interval in optimizer-loop indices
    curvature_probes: int = 10
    curvature_batch: int = 128
    eval_every: int = 2000  # quick learning-curve evaluation (not the final evaluation)
    eval_count: int = 200
    quick_seed_base: int = 50_000
    curvature_data_seed: int = 30_000
    curvature_probe_seed: int = 40_000


@dataclass
class EvalConfig:
    id_lengths: tuple = tuple(range(1, 20))
    ood_lengths: tuple = tuple(range(20, 61, 5))  # Kuo et al.: {20, 25, ..., 60}
    n_per_length: int | dict = 1000
    extra_loops: int = 5  # evaluate K = 1 .. n + 5
    seed_base: int = 10_000  # test set for length n uses seed seed_base + n
    example_seed_base: int = 60_000
    chunk_size: int = 250


@dataclass
class MeasureConfig:
    batch: int = 128  # fixed measurement batch (addendum 2: 512 -> 128 for time)
    seed: int = 20_000
    probes: int = 100
    power_iters: int = 50
    power_tol: float = 1e-3
    fisher_samples: int = 10
    sharpness_rhos: tuple = (0.01, 0.02, 0.05)
    sharpness_draws: int = 10
    loop_ks: tuple = (1, 2, 4, 8, 12, 16, 20)  # loop-count curvature study
    loop_probes: int = 10  # addendum 2: 20 -> 10
    loop_power_iters: int = 10  # addendum 2: 20 -> 10
    chunk_size: int = 16  # machine-specific measurement choice in preregistration Addendum 2


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

    def write_new_config(self):
        """Start a new evidence record without associating old outcomes with new weights."""
        os.makedirs(self.run_dir, exist_ok=True)
        if os.path.isdir(self.ckpt_dir) and os.listdir(self.ckpt_dir):
            raise FileExistsError(f"Existing checkpoints in {self.ckpt_dir}. Choose a fresh checkpoint directory or run name.")
        if os.listdir(self.run_dir):
            raise FileExistsError(f"Existing evidence in {self.run_dir}. Choose a fresh results directory or run name.")
        # Exclusive creation prevents simultaneous starts in the same results directory.
        with open(os.path.join(self.run_dir, "config.json"), "x", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)

    @classmethod
    def from_record(cls, paths):
        """Recover the run's actual numeric settings when measuring an existing checkpoint."""
        with open(os.path.join(paths.run_dir, "config.json"), encoding="utf-8") as handle:
            record = json.load(handle)
        if record["name"] != paths.name:
            raise ValueError("the run directory and recorded experiment name differ")
        with open(os.path.join(paths.run_dir, "train_summary.json"), encoding="utf-8") as handle:
            completed = json.load(handle)
        if (completed.get("name") != paths.name or completed.get("status", "completed") != "completed"
                or completed.get("steps_done") != record["train"]["steps"]):
            raise ValueError("final measurement requires completed training at the recorded budget")
        return cls(name=paths.name, model=ModelConfig(**record["model"]), train=TrainConfig(**record["train"]),
                   eval=EvalConfig(**record["eval"]), measure=MeasureConfig(**record["measure"]),
                   output_dir=paths.output_dir, checkpoint_dir=paths.checkpoint_dir, device=paths.device)


def smoke_config(arm):
    """Reserve separate paired plumbing streams for initialization, data and all diagnostics."""
    return RunConfig(name=f"smoke_{arm}", output_dir=os.path.join(os.getenv("LH_OUTPUT_DIR", "results"), "smoke"),
                     checkpoint_dir=os.path.join(os.getenv("LH_CHECKPOINT_DIR", "checkpoints"), "smoke"),
                     train=TrainConfig(arm=arm, steps=60, seed=71_000, log_every=20, curvature_every=30,
                                       eval_every=30, eval_count=20, curvature_probes=2, curvature_batch=16,
                                       curvature_data_seed=72_000, curvature_probe_seed=73_000, quick_seed_base=74_000),
                     eval=EvalConfig(id_lengths=(1, 5, 19), ood_lengths=(20, 30), n_per_length=20,
                                     seed_base=70_000, example_seed_base=75_000),
                     measure=MeasureConfig(batch=32, probes=3, power_iters=3, fisher_samples=2, sharpness_draws=2,
                                           loop_ks=(1, 4), loop_probes=2, loop_power_iters=2, seed=76_000))
