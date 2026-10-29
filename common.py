import os

import numpy as np
from datasets import load_from_disk
from transformers import AutoFeatureExtractor

ROUTER_BASE = "openai/whisper-tiny"
LABELS = ["moonshine", "whisper", "conformer"]  # class index -> STT model to route to
DATA = "data/mix"  # domain-balanced ESB mix (prepare_esb.py); "data" = the original Loquacious set
WEIGHT_CAP = 10.0  # stop a single hallucination loop from dominating the loss

feature_extractor = AutoFeatureExtractor.from_pretrained(ROUTER_BASE)


def recording_group(id_, spk_id):
    """Speaker (or, when unknown, source recording) so validation never shares a voice with training."""
    if spk_id:
        return spk_id
    if "-PLENARY-" in id_:
        return id_.split("_")[0]  # parliament session
    return id_.rsplit("-", 3)[0]  # YouTube video id


def router_val_mask(ds, val_frac=0.15):
    """Boolean mask over `ds` rows that fall in the speaker-disjoint validation split (deterministic)."""
    groups = [recording_group(i, s) for i, s in zip(ds["ID"], ds["spk_id"], strict=True)]
    rng = np.random.default_rng(0)
    uniq = sorted(set(groups))
    val_groups = set(rng.choice(uniq, size=int(len(uniq) * val_frac), replace=False))
    return np.array([g in val_groups for g in groups])


def load_router_splits(data=DATA):
    """Speaker-disjoint train / val / test splits.

    data/mix ships explicit splits. For the Loquacious layout (train_scored + official test_scored) the
    validation split is carved out of train by speaker: a random split leaks voices across the boundary and
    lets the router memorize "this voice -> whisper", inflating validation relative to test.
    """
    if os.path.exists(f"{data}/val_scored"):
        return {s: load_from_disk(f"{data}/{s}_scored") for s in ("train", "val", "test")}
    ds = load_from_disk(f"{data}/train_scored")
    is_val = router_val_mask(ds)
    return {
        "train": ds.select(np.where(~is_val)[0]),
        "val": ds.select(np.where(is_val)[0]),
        "test": load_from_disk(f"{data}/test_scored"),
    }


def error_matrix(ds, labels=LABELS):
    """(N, K) word errors of each candidate model on each clip."""
    return np.stack([np.array(ds[f"err_{m}"]) for m in labels], 1)


def add_targets(ds):
    """regret[k] = extra word errors from routing to model k instead of the clip's best model (capped)."""
    err = error_matrix(ds)
    regret = np.minimum(err - err.min(1, keepdims=True), WEIGHT_CAP).astype(np.float32)
    return ds.add_column("regret", regret.tolist())


def to_features(batch):
    feats = feature_extractor(
        [a["array"] for a in batch["audio"]], sampling_rate=16_000, return_tensors="pt", return_attention_mask=True
    )
    out = {"input_features": feats.input_features, "attention_mask": feats.attention_mask}
    if "regret" in batch:
        out["regret"] = batch["regret"]
    return out


def routed_errors(choice, err):
    """Word errors on each clip when routed to model index `choice` (err: (N, K) from error_matrix)."""
    return err[np.arange(len(err)), np.asarray(choice, dtype=int)]
