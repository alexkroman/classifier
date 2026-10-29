"""Compare training targets for the 2-model (Moonshine vs Whisper) Loquacious router, fast, on cached encoder features.

With the encoder frozen, the router head is linear up to the classifier (layer-weighted sum -> projector ->
masked mean), and linear maps commute with the mean. So pooling each encoder layer once and training the
head on the cached (N, layers, d) array is *exactly* the same model as train_router.py, in seconds instead
of 20 minutes, which lets every variant run over several seeds.

Variants:
  baseline   cross-entropy weighted by min(|err_m - err_w|, 10); ties get weight 0 (current train_router.py)
  min_diff2  same, but clips where the models differ by a single word also get weight 0
  regress    predict log1p(errors) of each model separately (all clips, ties included); route on the gap
Each is scored at the default decision rule and with a "prefer Moonshine" margin tuned on validation.
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from common import ROUTER_BASE, WEIGHT_CAP, routed_errors, router_val_mask, to_features
from router_model import WhisperRouter

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
COLS = ["err_moonshine", "err_whisper", "n_words", "source"]


@torch.inference_mode()
def pooled_layers(ds, encoder, batch_size=32):
    """(N, L+1, D) mean of every encoder hidden state over the real (non-padding) frames of each clip."""
    out = []
    for i in tqdm(range(0, len(ds), batch_size), desc="encode"):
        feats = to_features(ds[i : i + batch_size])
        hs = encoder(feats["input_features"].to(DEVICE), output_hidden_states=True).hidden_states
        mask = feats["attention_mask"][:, ::2].unsqueeze(-1).to(DEVICE)  # 3000 mel frames -> 1500 encoder frames
        h = torch.stack(hs, 1)  # (B, L+1, T, D)
        out.append(((h * mask.unsqueeze(1)).sum(2) / mask.sum(1).unsqueeze(1)).cpu().numpy())
    return np.concatenate(out)


def load_cached(split, encoder):
    path = f"data/emb_{ROUTER_BASE.split('/')[-1]}_{split}.npy"
    ds = load_from_disk(f"data/{split}_scored")
    if not os.path.exists(path):
        np.save(path, pooled_layers(ds, encoder))
    meta = ds.remove_columns([c for c in ds.column_names if c not in COLS + ["ID", "spk_id"]])
    return np.load(path), meta


class Head(torch.nn.Module):
    """Same parameters as WhisperRouter's trainable part: layer_weights, projector, classifier."""

    def __init__(self, n_layers, d, n_out, proj=256):
        super().__init__()
        self.layer_weights = torch.nn.Parameter(torch.ones(n_layers) / n_layers)
        self.projector = torch.nn.Linear(d, proj)
        self.classifier = torch.nn.Linear(proj, n_out)

    def forward(self, x):  # x: (B, L+1, D)
        x = (x * torch.softmax(self.layer_weights, 0).view(1, -1, 1)).sum(1)
        return self.classifier(self.projector(x))


def whisper_score(variant, out):
    """Higher = more in favour of Whisper; 0 (regress) / 0.5 (CE) is the default decision boundary."""
    if variant == "regress":
        return (out[:, 0] - out[:, 1]).numpy()  # predicted log1p(err_m) - log1p(err_w)
    return out.softmax(-1)[:, 1].numpy()


def train_head(variant, x_tr, y_tr, x_va, y_va, seed, epochs=8, lr=1e-3, bs=16):
    torch.manual_seed(seed)
    head = Head(x_tr.shape[1], x_tr.shape[2], 2)
    opt = torch.optim.AdamW(head.parameters(), lr=lr)
    steps = epochs * int(np.ceil(len(x_tr) / bs))
    warm = int(0.1 * steps)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, (steps - s) / (steps - warm)))
    em, ew = y_tr["err_moonshine"], y_tr["err_whisper"]
    diff = em - ew
    labels = torch.tensor((diff > 0).astype(int))
    weight = np.minimum(np.abs(diff), WEIGHT_CAP).astype(np.float32)
    if variant == "min_diff2":
        weight[np.abs(diff) < 2] = 0
    weight = torch.tensor(weight)
    log_err = torch.tensor(np.log1p(np.stack([em, ew], 1)), dtype=torch.float32)
    x_tr, x_va = torch.tensor(x_tr), torch.tensor(x_va)

    best, best_state = np.inf, None
    for _ in range(epochs):
        head.train()
        perm = torch.randperm(len(x_tr), generator=torch.Generator().manual_seed(seed + _))
        for i in range(0, len(perm), bs):
            b = perm[i : i + bs]
            out = head(x_tr[b])
            if variant == "regress":
                loss = F.mse_loss(out, log_err[b])
            else:
                w = weight[b]
                loss = (F.cross_entropy(out, labels[b], reduction="none") * w).sum() / w.sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        head.eval()  # select the epoch on validation routed WER, like load_best_model_at_end
        with torch.no_grad():
            s = whisper_score(variant, head(x_va))
        wer = routed_wer(s >= default_cut(variant), y_va)
        if wer < best:
            best, best_state = wer, {k: v.clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_state)
    return head


def default_cut(variant):
    return 0.0 if variant == "regress" else 0.5


def routed_wer(choose_whisper, y):
    err = np.stack([y["err_moonshine"], y["err_whisper"]], 1)
    return routed_errors(choose_whisper, err).sum() / y["n_words"].sum()


def tune_cut(variant, s, y):
    """Best cutoff on validation, searched only on the Moonshine-favouring side of the default."""
    lo = default_cut(variant)
    grid = np.linspace(lo, 1.0 if variant == "regress" else 0.8, 31)
    return grid[np.argmin([routed_wer(s >= c, y) for c in grid])]


def discrimination(s, y):
    """AUC over contested clips, and mean AUC within each source (only within-source signal is per-clip)."""
    c = y["err_moonshine"] != y["err_whisper"]
    win = y["err_whisper"] < y["err_moonshine"]
    within = [
        roc_auc_score(win[k], s[k]) for src in np.unique(y["source"]) if (k := c & (y["source"] == src)).sum() > 30
    ]
    return roc_auc_score(win[c], s[c]), float(np.mean(within))


def as_arrays(meta):
    return {c: np.array(meta[c]) for c in COLS}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--save", default="", help="variant to export as a WhisperRouter to models/router_<variant>")
    cli = ap.parse_args()

    router = WhisperRouter.from_pretrained(ROUTER_BASE, num_labels=2, use_weighted_layer_sum=True)
    encoder = router.encoder.to(DEVICE).eval()
    x_all, meta_all = load_cached("train", encoder)
    x_te, meta_te = load_cached("test", encoder)
    is_val = router_val_mask(meta_all)
    y_all, y_te = as_arrays(meta_all), as_arrays(meta_te)
    y_tr = {k: v[~is_val] for k, v in y_all.items()}
    y_va = {k: v[is_val] for k, v in y_all.items()}
    x_tr, x_va = x_all[~is_val], x_all[is_val]

    print(f"\nval:  moonshine {routed_wer(np.zeros(len(x_va), bool), y_va):.4f}")
    print(
        f"test: moonshine {routed_wer(np.zeros(len(x_te), bool), y_te):.4f}   "
        f"per-clip oracle {np.minimum(y_te['err_moonshine'], y_te['err_whisper']).sum() / y_te['n_words'].sum():.4f}\n"
    )
    header = (
        f"{'variant':10s} {'val WER':>8s} {'test WER':>9s} {'->whisper':>9s} | "
        f"{'tuned cut':>9s} {'test WER':>9s} {'->whisper':>9s} | {'test AUC':>8s} {'within':>7s}"
    )
    print(header + "\n" + "-" * len(header))
    for variant in ("baseline", "min_diff2", "regress"):
        rows = []
        for seed in range(cli.seeds):
            head = train_head(variant, x_tr, y_tr, x_va, y_va, seed)
            with torch.no_grad():
                s_va, s_te = (whisper_score(variant, head(torch.tensor(x))) for x in (x_va, x_te))
            d, cut = default_cut(variant), tune_cut(variant, s_va, y_va)
            rows.append(
                [
                    routed_wer(s_va >= d, y_va),
                    routed_wer(s_te >= d, y_te),
                    (s_te >= d).mean(),
                    cut,
                    routed_wer(s_te >= cut, y_te),
                    (s_te >= cut).mean(),
                    *discrimination(s_te, y_te),
                ]
            )
            if variant == cli.save and seed == 0:
                # regress: logits = -predicted log errors, so softmax[:, 1] >= 0.5 iff Whisper is predicted better
                sign = -1.0 if variant == "regress" else 1.0
                router.layer_weights.data = head.layer_weights.data.clone()
                router.projector.load_state_dict(head.projector.state_dict())
                router.classifier.weight.data = sign * head.classifier.weight.data
                router.classifier.bias.data = sign * head.classifier.bias.data
                router.config.id2label, router.config.label2id = (
                    {0: "moonshine", 1: "whisper"},
                    {"moonshine": 0, "whisper": 1},
                )
                router.save_pretrained(f"models/router_{variant}")
        m, sd = np.mean(rows, 0), np.std(rows, 0)
        print(
            f"{variant:10s} {m[0]:8.4f} {m[1]:9.4f} {m[2]:9.0%} | "
            f"{m[3]:9.2f} {m[4]:9.4f} {m[5]:9.0%} | {m[6]:8.3f} {m[7]:7.3f}"
        )
        print(
            f"{'  ± sd':10s} {sd[0]:8.4f} {sd[1]:9.4f} {'':9s} | "
            f"{sd[3]:9.2f} {sd[4]:9.4f} {'':9s} | {sd[6]:8.3f} {sd[7]:7.3f}"
        )
