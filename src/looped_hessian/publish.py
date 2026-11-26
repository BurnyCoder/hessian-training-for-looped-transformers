"""Upload Stage 1 checkpoints, configs and results to Hugging Face and add them to the collection.

Local: uses the huggingface_hub Python API (`create_repo`, `upload_folder`, `create_collection`,
`add_collection_item`; https://huggingface.co/docs/huggingface_hub/guides/upload and
https://huggingface.co/docs/huggingface_hub/guides/collections). The Hub client uses its
configured credentials. The model card is generated from `results/stage1_summary.json`,
so its numbers match the README and the paper.

Global: called by `main.py publish` after `main.py report`.
"""

import json
import os
import shutil
import tempfile

from huggingface_hub import HfApi

REPO_ID = os.getenv("LH_HF_REPO_ID", "BurnyCoder/hessian-looped-transformers")
COLLECTION_TITLE = "Hessian training for looped transformers"
GITHUB_URL = "https://github.com/BurnyCoder/hessian-training-for-looped-transformers"


def model_card(summary: dict) -> str:
    tables = summary["tables"]
    return f"""---
license: mit
tags:
- looped-transformer
- hessian
- flat-minima
- length-generalization
- binary-addition
---

# Hessian-trace regularization for looped transformers

The [registered binary addition study]({GITHUB_URL}) compares ordinary cross-entropy with an intermittent,
single-probe Hutchinson penalty on a token-normalized sub-batch. The per-length oracle uses the evaluation
labels. Fixed-depth accuracy measures a prespecified stopping rule. Each uploaded arm includes its resolved configuration.

{tables['headline']}

## Results

{tables['results_table']}

## Files

- `stage1_<arm>/final.pt`, `stage1_<arm>/init.pt`: `torch.save` dicts with `model` (state dict) and `config`.
  Load with `looped_hessian.train.load_checkpoint` from the GitHub repository.
- `stage1_<arm>/*.json`: evaluation, curvature and training-summary files used for every reported number.

Single seed per arm: these checkpoints support descriptive, not confirmatory, conclusions (see the
preregistration and paper in the GitHub repository).
"""


def publish(results_dir: str = "results", checkpoint_dir: str = "checkpoints", prefix: str = "stage1_") -> None:
    api = HfApi()
    with open(os.path.join(results_dir, f"{prefix}summary.json"), encoding="utf-8") as f:
        summary = json.load(f)
    if not summary.get("arms"):
        raise ValueError("no completed, evaluated arm is available for a checkpoint release")
    with tempfile.TemporaryDirectory() as tmp:
        for arm in summary["arms"]:
            run = prefix + arm
            dst = os.path.join(tmp, run)
            os.makedirs(dst)
            for name in ("final.pt", "init.pt"):
                shutil.copy(os.path.join(checkpoint_dir, run, name), dst)
            for name in os.listdir(os.path.join(results_dir, run)):
                if name.endswith((".json", ".jsonl", ".log")):
                    shutil.copy(os.path.join(results_dir, run, name), dst)
        shutil.copy(os.path.join(results_dir, f"{prefix}summary.json"), tmp)
        with open(os.path.join(tmp, "README.md"), "w", encoding="utf-8") as f:
            f.write(model_card(summary))
        api.create_repo(REPO_ID, repo_type="model", exist_ok=True)
        api.upload_folder(repo_id=REPO_ID, folder_path=tmp, commit_message="Upload Stage 1 checkpoints and results")
    collection = api.create_collection(title=COLLECTION_TITLE, exists_ok=True,
                                       description="Preregistered experiments on Hessian-trace regularization "
                                                   "of looped transformers.")
    api.add_collection_item(collection.slug, item_id=REPO_ID, item_type="model", exists_ok=True)
    print(f"published https://huggingface.co/{REPO_ID} in collection https://huggingface.co/collections/{collection.slug}")
