"""Train a tiny Whisper-encoder audio classifier to route each clip to the STT model that will do best."""

import argparse

import numpy as np
from transformers import Trainer, TrainingArguments

from common import (
    DATA,
    LABELS,
    ROUTER_BASE,
    add_targets,
    error_matrix,
    feature_extractor,
    load_router_splits,
    routed_errors,
    to_features,
)
from router_model import WhisperRouter


class RegretTrainer(Trainer):
    """Minimize expected regret, sum_k P(k) * regret_k: routed word errors in expectation.

    Ties cost nothing (every tied model has regret 0), and with two models this is the same objective as
    cross-entropy weighted by the errors at stake.
    """

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        regret = inputs.pop("regret")
        outputs = model(**inputs)
        loss = (outputs.logits.softmax(-1) * regret).sum(-1).mean()
        return (loss, outputs) if return_outputs else loss


def make_compute_metrics(val):
    err, n_words = error_matrix(val), np.array(val["n_words"])

    def compute_metrics(pred):
        choice = pred.predictions.argmax(-1)
        metrics = {"routed_wer": routed_errors(choice, err).sum() / n_words.sum()}
        return metrics | {f"frac_{m}": (choice == k).mean() for k, m in enumerate(LABELS)}

    return compute_metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument(
        "--finetune-encoder", action="store_true", help="default: freeze encoder, train layer weights + head"
    )
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--out", default="models/router")
    args_cli = ap.parse_args()

    splits = load_router_splits(args_cli.data)
    train, val = add_targets(splits["train"]), add_targets(splits["val"])
    wins = np.bincount(np.array(train["regret"]).argmin(1), minlength=len(LABELS)) / len(train)
    print(
        f"train {len(train)} clips, val {len(val)}; best model (ties -> first): "
        + ", ".join(f"{m} {w:.0%}" for m, w in zip(LABELS, wins, strict=True))
    )

    model = WhisperRouter.from_pretrained(
        ROUTER_BASE,
        num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)),
        label2id={label: i for i, label in enumerate(LABELS)},
        use_weighted_layer_sum=True,  # learn which encoder layers carry the routing signal
    )
    if not args_cli.finetune_encoder:
        model.freeze_encoder()  # ~8M frozen params; only layer weights + projector + classifier train

    args = TrainingArguments(
        output_dir=f"checkpoints/{args_cli.out.split('/')[-1]}",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=args_cli.lr,
        warmup_steps=0.1,  # fraction of total steps (transformers 5 replaced warmup_ratio)
        num_train_epochs=args_cli.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="routed_wer",
        greater_is_better=False,
        logging_steps=20,
        remove_unused_columns=False,  # keep `regret` for the custom loss; features come from the transform
        label_names=["regret"],
        # Forked DataLoader workers segfault on macOS once MPS is initialized (the eval loader forks after
        # training has touched the GPU); log-mel extraction is cheap enough to run in the main process.
        dataloader_num_workers=0,
        report_to="none",
        seed=0,
    )
    trainer = RegretTrainer(
        model=model,
        args=args,
        train_dataset=train.with_transform(to_features),
        eval_dataset=val.with_transform(to_features),
        compute_metrics=make_compute_metrics(val),
    )
    trainer.train()
    trainer.save_model(args_cli.out)
    feature_extractor.save_pretrained(args_cli.out)
    print("best val metrics:", trainer.evaluate())
