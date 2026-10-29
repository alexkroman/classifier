"""Score one more STT model on the already-transcribed splits (adds hyp_<name> / err_<name> columns in place)."""

import argparse
import shutil

from datasets import load_from_disk
from tqdm import tqdm

from transcribe import MODELS, make_transcriber, word_errors

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("name", choices=list(MODELS))
    ap.add_argument("--splits", nargs="+", default=["test", "train"])
    cli = ap.parse_args()

    transcribe = make_transcriber(cli.name)
    for split in cli.splits:
        path = f"data/{split}_scored"
        ds = load_from_disk(path)
        if f"err_{cli.name}" in ds.column_names:
            print(f"{split}: already has {cli.name}, skipping")
            continue
        hyps = []
        for i in tqdm(range(0, len(ds), 16), desc=f"{split}/{cli.name}"):
            hyps += transcribe(ds[i : i + 16])[f"hyp_{cli.name}"]
        ds = ds.add_column(f"hyp_{cli.name}", hyps)
        ds = ds.add_column(f"err_{cli.name}", [word_errors(r, h) for r, h in zip(ds["text"], hyps, strict=True)])
        # can't overwrite a dataset that is memory-mapped from the same directory: write aside, then swap
        ds.save_to_disk(path + ".tmp")
        del ds
        shutil.rmtree(path)
        shutil.move(path + ".tmp", path)
        ds = load_from_disk(path)
        print(f"{split} {cli.name} corpus WER: {sum(ds[f'err_{cli.name}']) / sum(ds['n_words']):.4f}")
