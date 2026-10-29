import numpy as np
from datasets import load_from_disk
from transformers import AutoFeatureExtractor

MOONSHINE_BASE = "UsefulSensors/moonshine-tiny"  # the router's encoder
LABELS = ["moonshine", "whisper", "conformer"]  # class index -> STT model to route to
DATA = "data/speakers"  # speaker-diverse chunks (prepare_speakers.py)

feature_extractor = AutoFeatureExtractor.from_pretrained(MOONSHINE_BASE)
# Pad Moonshine batches to a multiple of 2s: MPS compiles a graph per input shape and never frees them, so
# arbitrary lengths leak memory until OOM. The padding is masked out, so outputs don't change.
MOONSHINE_PAD = 2 * 16_000


def load_router_splits(data=DATA):
    """Train / val / test splits, disjoint by speaker or recording (assigned in prepare_speakers.py)."""
    return {s: load_from_disk(f"{data}/{s}_scored") for s in ("train", "val", "test")}


def error_matrix(ds, labels=LABELS):
    """(N, K) word errors of each candidate model on each clip."""
    return np.stack([np.array(ds[f"err_{m}"]) for m in labels], 1)


TARGETS = ("cost",)  # per-clip training targets, passed through the feature transform


def add_targets(ds):
    """cost[k] = model k's word errors minus the clip's mean over models, uncapped (the diff loss target).

    Centering cancels the difficulty every model shares, which is most of the variance; the argmin is unchanged.
    """
    err = error_matrix(ds)
    cost = (err - err.mean(1, keepdims=True)).astype(np.float32)
    return ds.add_column("cost", cost.tolist())


def to_moonshine_features(batch):
    """Raw waveform padded to the longest clip in the batch (Moonshine takes variable length; no 30s window)."""
    feats = feature_extractor(
        [a["array"] for a in batch["audio"]],
        sampling_rate=16_000,
        return_tensors="pt",
        return_attention_mask=True,
        padding=True,
        pad_to_multiple_of=MOONSHINE_PAD,
    )
    out = {"input_values": feats.input_values, "attention_mask": feats.attention_mask}
    return out | {t: batch[t] for t in TARGETS if t in batch}


def routed_errors(choice, err):
    """Word errors on each clip when routed to model index `choice` (err: (N, K) from error_matrix)."""
    return err[np.arange(len(err)), np.asarray(choice, dtype=int)]
