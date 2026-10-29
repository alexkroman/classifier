"""Transcribe every clip with every STT model and record per-clip word-error counts."""

import math
import shutil

import jiwer
import numpy as np
import torch
from datasets import load_from_disk
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    MoonshineForConditionalGeneration,
    WhisperForConditionalGeneration,
)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MODELS = {
    "moonshine": ("UsefulSensors/moonshine-tiny", MoonshineForConditionalGeneration),
    "whisper": ("openai/whisper-tiny.en", WhisperForConditionalGeneration),
    # 13M Conformer-CTC (NeMo), trained on NeMo's multi-domain English set: a different architecture, decoder
    # and training data from the tiny pair, and non-autoregressive, so it cannot fall into repetition loops.
    "conformer": ("nvidia/stt_en_conformer_ctc_small", "nemo"),
}
# Whisper's English normalizer (lowercase, strip punctuation, standardize spellings/numbers) is the
# standard way to score ASR fairly across models with different formatting habits.
normalize = AutoProcessor.from_pretrained("openai/whisper-tiny.en").tokenizer.normalize


def make_nemo_transcriber(name, repo):
    from nemo.collections.asr.models import ASRModel  # heavy import; only needed for NeMo models
    from nemo.utils import logging

    logging.setLevel(logging.ERROR)
    model = ASRModel.from_pretrained(repo).eval()  # CPU: small CTC models are fast, and NeMo's MPS support is patchy

    def transcribe(batch):
        arrays = [np.asarray(a["array"], dtype=np.float32) for a in batch["audio"]]
        return {f"hyp_{name}": [h.text for h in model.transcribe(arrays, batch_size=len(arrays), verbose=False)]}

    return transcribe


def make_transcriber(name):
    repo, cls = MODELS[name]
    if cls == "nemo":
        return make_nemo_transcriber(name, repo)
    processor = AutoProcessor.from_pretrained(repo)
    model = cls.from_pretrained(repo, dtype=torch.float32).to(DEVICE).eval()  # v5 keeps checkpoint dtype

    @torch.inference_mode()
    def transcribe(batch):
        arrays = [a["array"] for a in batch["audio"]]
        # Whisper always pads to a fixed 30s window; Moonshine takes variable length with a mask.
        pad = {} if name == "whisper" else {"padding": True, "return_attention_mask": True}
        inputs = processor(arrays, sampling_rate=16_000, return_tensors="pt", **pad).to(DEVICE)
        # Both autoregressive models can fall into repetition loops on hard audio; cap output length by audio
        # duration (~6.5 tok/s is Moonshine's guidance) and apply the same cap to both so neither gets an edge.
        max_new = math.ceil(max(batch["duration"]) * 6.5) + 10
        ids = model.generate(**inputs, max_new_tokens=max_new)
        return {f"hyp_{name}": processor.batch_decode(ids, skip_special_tokens=True)}

    return transcribe


def word_errors(ref, hyp):
    ref, hyp = normalize(ref), normalize(hyp)
    if not ref.strip():
        return len(hyp.split())
    out = jiwer.process_words(ref, hyp)
    return out.substitutions + out.deletions + out.insertions


def score(ds, desc="", models=MODELS):
    """Add hyp_<model>, err_<model> for every model, plus n_words. Sorts by duration for efficient batching."""
    # sort by length so batches have similar durations (less padding, tighter token cap)
    ds = ds.sort("duration")
    text = ds["text"]
    for name in models:
        transcribe = make_transcriber(name)
        # collect hypotheses and attach with add_column, so the audio isn't rewritten per model (disk)
        hyps = []
        for i in tqdm(range(0, len(ds), 16), desc=f"{desc}/{name}"):
            hyps += transcribe(ds[i : i + 16])[f"hyp_{name}"]
        ds = ds.add_column(f"hyp_{name}", hyps)
        ds = ds.add_column(f"err_{name}", [word_errors(r, h) for r, h in zip(text, hyps, strict=True)])
    return ds.add_column("n_words", [len(normalize(t).split()) for t in text])


if __name__ == "__main__":
    for split in ("train", "test"):
        ds = score(load_from_disk(f"data/{split}"), desc=split)
        ds.save_to_disk(f"data/{split}_scored")
        shutil.rmtree(f"data/{split}")  # scored copy has the audio too; save disk
        for m in MODELS:
            print(f"{split} {m} corpus WER: {sum(ds[f'err_{m}']) / sum(ds['n_words']):.4f}")
