"""Train a Moonshine-encoder audio classifier to route each clip to the STT model that will do best."""

import argparse

import numpy as np
import torch
from transformers import AutoConfig, Trainer, TrainingArguments

from common import (
    DATA,
    LABELS,
    MOONSHINE_BASE,
    add_targets,
    error_matrix,
    feature_extractor,
    load_router_splits,
    routed_errors,
    to_moonshine_features,
)
from router_model import MoonshineRouter, load_router


class RouterTrainer(Trainer):
    """Train the router so that argmax(logits) approximates the Bayes rule argmin_k E[word errors of model k | audio].

    The logits are the *negated* predicted cost of each model (its word errors minus the clip's mean over models),
    fit with Huber loss. Regression recovers conditional means, so the argmin is the Bayes rule (consistent);
    centering cancels shared clip difficulty; no cap, since corpus WER pays in full for a hallucination loop. Huber
    (delta in words) keeps those rare loops from dominating the gradient. Also tried: soft cross-entropy to
    softmax(-regret / tau) (only consistent as tau -> inf; drifts toward the most frequent winner) and expected
    regret (consistent, but its gradient vanishes once the softmax saturates: it collapsed within one epoch to
    always routing to the best single model).
    """

    def __init__(self, *args, huber_delta=5.0, encoder_lr=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.huber_delta, self.encoder_lr = huber_delta, encoder_lr

    def create_optimizer(self):
        # A fine-tuned encoder needs a far smaller LR than the freshly initialized head: at the head's 1e-3 the
        # pretrained features are wrecked within a few hundred steps. The schedule scales both groups alike.
        if self.optimizer is None and self.encoder_lr is not None:
            params = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
            groups = [
                {"params": [p for n, p in params if not n.startswith("encoder.")], "lr": self.args.learning_rate},
                {"params": [p for n, p in params if n.startswith("encoder.")], "lr": self.encoder_lr},
            ]
            self.optimizer = torch.optim.AdamW(groups, weight_decay=self.args.weight_decay)
        return super().create_optimizer()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        cost = inputs.pop("cost")
        outputs = model(**inputs)
        loss = torch.nn.functional.huber_loss(-outputs.logits, cost, delta=self.huber_delta)
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
        "--init-from",
        help="start from a saved router (head included). LP-FT: train the head on the frozen encoder first, then "
        "--finetune-encoder --init-from that model, so a random head's early gradients don't wreck the pretrained "
        "features",
    )
    ap.add_argument(
        "--finetune-encoder", action="store_true", help="default: freeze encoder, train layer weights + head"
    )
    ap.add_argument("--lr", type=float, default=1e-3, help="head LR")
    ap.add_argument("--encoder-lr", type=float, default=1e-5, help="encoder LR with --finetune-encoder")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--head-hidden", type=int, default=256, help="GELU layer before the classifier; 0 = linear")
    ap.add_argument("--huber-delta", type=float, default=5.0, help="Huber delta, in word errors")
    ap.add_argument("--specaug-time", type=float, default=0.0, help="fraction of encoder frames masked")
    ap.add_argument("--specaug-feat", type=float, default=0.0, help="fraction of encoder channels masked")
    ap.add_argument("--out", default="models/router")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="continue from the latest checkpoint in checkpoints/<out name> (weights, optimizer, LR schedule, RNG); "
        "pass the same flags as the interrupted run",
    )
    args_cli = ap.parse_args()

    splits = load_router_splits(args_cli.data)
    train, val = add_targets(splits["train"]), add_targets(splits["val"])
    wins = np.bincount(error_matrix(train).argmin(1), minlength=len(LABELS)) / len(train)
    print(
        f"train {len(train)} clips, val {len(val)}; best model (ties -> first): "
        + ", ".join(f"{m} {w:.0%}" for m, w in zip(LABELS, wins, strict=True))
    )

    if args_cli.init_from:
        model = load_router(args_cli.init_from)  # --head-hidden comes from the saved config
    else:
        config = AutoConfig.from_pretrained(
            MOONSHINE_BASE,
            num_labels=len(LABELS),
            id2label=dict(enumerate(LABELS)),
            label2id={label: i for i, label in enumerate(LABELS)},
        )
        # custom fields: transformers 5 drops unknown config kwargs, so set them after
        config.classifier_proj_size, config.head_hidden = 256, args_cli.head_hidden
        model = MoonshineRouter.from_moonshine(MOONSHINE_BASE, config)
    model.config.specaug_time, model.config.specaug_feat = args_cli.specaug_time, args_cli.specaug_feat
    if not args_cli.finetune_encoder:
        model.freeze_encoder()  # ~8M frozen params; only layer weights + projector + head train

    args = TrainingArguments(
        output_dir=f"checkpoints/{args_cli.out.split('/')[-1]}",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=args_cli.lr,
        warmup_steps=0.1,  # fraction of total steps (transformers 5 replaced warmup_ratio)
        num_train_epochs=args_cli.epochs,
        # Validation is logged every epoch for monitoring only; the saved model is the last epoch. Picking the
        # best epoch on val bakes val noise (epoch-to-epoch swings are ~0.05 WER pts) into the model choice.
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        logging_steps=20,
        remove_unused_columns=False,  # keep `cost` for the custom loss; features come from the transform
        label_names=["cost"],
        # Forked DataLoader workers segfault on macOS once MPS is initialized (the eval loader forks after
        # training has touched the GPU); feature extraction is cheap enough to run in the main process.
        dataloader_num_workers=0,
        report_to="none",
        seed=0,
    )
    trainer = RouterTrainer(
        model=model,
        args=args,
        train_dataset=train.with_transform(to_moonshine_features),
        eval_dataset=val.with_transform(to_moonshine_features),
        compute_metrics=make_compute_metrics(val),
        huber_delta=args_cli.huber_delta,
        encoder_lr=args_cli.encoder_lr if args_cli.finetune_encoder else None,
    )
    trainer.train(resume_from_checkpoint=True if args_cli.resume else None)
    trainer.save_model(args_cli.out)
    feature_extractor.save_pretrained(args_cli.out)
    print("best val metrics:", trainer.evaluate())
