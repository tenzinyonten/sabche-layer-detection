#!/usr/bin/env python3
"""
Train mmBERT-base as a BIO token classifier for one layer (tsawa, sabche or chapter). The
layer is the label name. The 5-label scheme (`multi`) adds a quotation class and is only used
for tsawa experiments; the final models use plain `bio`.

Usage:
    python src/train.py --label-name TSAWA --scheme bio --weight-scheme inv \
        --epochs 15 --evals-per-epoch 4 --patience 3 --no-grad-checkpointing --skip-test \
        --output-dir runs/tsawa
    python src/train.py --label-name SABCHE --scheme bio --weight-scheme inv --epochs 8 \
        --evals-per-epoch 4 --patience 3 --no-grad-checkpointing --skip-test --output-dir runs/sabche
    python src/train.py --label-name CHAPTER --scheme bio --weight-scheme sqrt_inv --epochs 8 \
        --evals-per-epoch 4 --patience 3 --skip-test --output-dir runs/chapter
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoConfig, AutoModelForTokenClassification, EarlyStoppingCallback,
    Trainer, TrainingArguments, set_seed,
)

NEG = -1.0e9

DEFAULT_DATASETS = {"TSAWA": "Yontenn/formatting-tsawa-v6",
                    "SABCHE": "Yontenn/formatting-sabche-v1",
                    "CHAPTER": "Yontenn/formatting-chapter-v1"}
PRIMARY = "TSAWA"
SCHEMES: dict = {}
ENTITIES: dict = {}    # label ids that open / continue each entity type


def configure_label(name: str) -> None:
    """Fill SCHEMES and ENTITIES for the entity called `name`."""
    global PRIMARY
    PRIMARY = name = name.upper()
    SCHEMES.clear()
    SCHEMES.update({
        "bio":   ["O", f"B-{name}", f"I-{name}"],
        "io":    ["O", name],
        "bioe":  ["O", f"B-{name}", f"I-{name}", f"E-{name}"],
        "multi": ["O", f"B-{name}", f"I-{name}", "B-QUOTE", "I-QUOTE"],
    })
    ENTITIES.clear()
    ENTITIES.update({
        "bio":   {name: (1, 2, None)},
        "io":    {name: (1, 1, None)},
        "bioe":  {name: (1, 2, 3)},
        "multi": {name: (1, 2, None), "QUOTE": (3, 4, None)},
    })


configure_label("TSAWA")


# label conversion

def convert_row(labels, scheme: str, source_is_multi: bool):
    """Stored labels -> the scheme's label ids."""
    if scheme == "multi":
        if not source_is_multi:
            raise SystemExit("--scheme multi needs the 5-label v4 dataset")
        return labels
    # collapse a 5-label source down to tsawa-only
    if source_is_multi:
        labels = [0 if x in (3, 4) else x for x in labels]
    if scheme == "bio":
        return labels
    if scheme == "io":
        return [1 if x == 2 else x for x in labels]
    out = list(labels)  # bioe
    for i in range(len(out)):
        if out[i] in (1, 2):
            nxt = out[i + 1] if i + 1 < len(out) else -100
            if nxt != 2:
                out[i] = 3
    return out


# Viterbi

def transition_matrix(scheme: str, break_penalty: float) -> np.ndarray:
    labels = SCHEMES[scheme]
    n = len(labels)
    m = np.zeros((n, n))
    if scheme == "io":
        return m
    ent = ENTITIES[scheme]
    # a continuation label may only follow its own begin or continuation
    for j, nxt in enumerate(labels):
        for tag, (b, i, e) in ent.items():
            if j == i and i != b:
                for p in range(n):
                    if p not in (b, i):
                        m[p, j] = NEG
            if e is not None and j == e:
                for p in range(n):
                    if p not in (b, i):
                        m[p, j] = NEG
    # leaving a span costs the penalty
    for tag, (b, i, e) in ent.items():
        closers = {i} if e is None else {i, e}
        for p in (b, i):
            for q in range(n):
                if q not in closers:
                    m[p, q] -= break_penalty
    return m


def viterbi(logits: np.ndarray, scheme: str, break_penalty: float) -> np.ndarray:
    if scheme == "io":
        return logits.argmax(-1)
    T, C = logits.shape
    trans = transition_matrix(scheme, break_penalty)
    dp = np.full((T, C), NEG)
    bp = np.zeros((T, C), dtype=np.int64)
    dp[0] = logits[0]
    for tag, (b, i, e) in ENTITIES[scheme].items():
        if i != b:
            dp[0, i] = NEG          # cannot open mid-span
        if e is not None:
            dp[0, e] = NEG
    for t in range(1, T):
        s = dp[t - 1][:, None] + trans
        bp[t] = s.argmax(0)
        dp[t] = s.max(0) + logits[t]
    path = np.zeros(T, dtype=np.int64)
    path[-1] = int(dp[-1].argmax())
    for t in range(T - 1, 0, -1):
        path[t - 1] = bp[t, path[t]]
    return path


def spans_of(seq, scheme: str, tag: str):
    """Inclusive (start, end) spans of one entity type."""
    b, i, e = ENTITIES[scheme][tag]
    out, start = [], None
    for k, v in enumerate(seq):
        if v == b and b != i:
            if start is not None:
                out.append((start, k - 1))
            start = k
        elif v == b and b == i:          # io
            if start is None:
                start = k
        elif v == i:
            if start is None:
                start = k
        elif e is not None and v == e:
            if start is None:
                start = k
            out.append((start, k)); start = None
        else:
            if start is not None:
                out.append((start, k - 1)); start = None
    if start is not None:
        out.append((start, len(seq) - 1))
    return out


# IoU@0.5 - mirrors src/layer_detection/v2_metrics.py

def iou(a, b):
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    if hi < lo:
        return 0.0
    inter = hi - lo + 1
    return inter / ((a[1] - a[0] + 1) + (b[1] - b[0] + 1) - inter)


def count_matches(gold, pred, thr=0.5):
    cands = sorted(((iou(g, p), gi, pi)
                    for pi, p in enumerate(pred)
                    for gi, g in enumerate(gold)
                    if iou(g, p) >= thr), reverse=True)
    ug, up, n = set(), set(), 0
    for _, gi, pi in cands:
        if gi in ug or pi in up:
            continue
        ug.add(gi); up.add(pi); n += 1
    return n


def build_metrics(scheme: str, break_penalty: float):
    tags = list(ENTITIES[scheme])

    def compute_metrics(eval_pred):
        logits = np.asarray(eval_pred[0])
        labels = np.asarray(eval_pred[1])
        res = {}
        acc = {t: [0, 0, 0] for t in tags}      # tp, fp, fn
        acc_argmax = {t: [0, 0, 0] for t in tags}

        for lg, lab in zip(logits, labels):
            m = lab != -100
            dec = viterbi(lg[m], scheme, break_penalty)
            arg = lg[m].argmax(-1)
            for t in tags:
                g = spans_of(lab[m], scheme, t)
                for seq, store in ((dec, acc), (arg, acc_argmax)):
                    p = spans_of(seq, scheme, t)
                    tp = count_matches(g, p)
                    store[t][0] += tp
                    store[t][1] += len(p) - tp
                    store[t][2] += len(g) - tp

        def prf(tp, fp, fn):
            pr = tp / (tp + fp) if tp + fp else 0.0
            rc = tp / (tp + fn) if tp + fn else 0.0
            return pr, rc, (2 * pr * rc / (pr + rc) if pr + rc else 0.0)

        for t in tags:
            tp, fp, fn = acc[t]
            pr, rc, f1 = prf(tp, fp, fn)
            key = "iou50" if t == PRIMARY else f"{t.lower()}_iou50"
            res[f"{key}_f1"] = f1
            res[f"{key}_precision"] = pr
            res[f"{key}_recall"] = rc
            res[f"{key}_n_gold"] = tp + fn
            res[f"{key}_n_pred"] = tp + fp
            res[f"{key}_overproposal"] = (tp + fp) / (tp + fn) if tp + fn else 0.0
            a_tp, a_fp, a_fn = acc_argmax[t]
            res[f"{key}_argmax_f1"] = prf(a_tp, a_fp, a_fn)[2]
        return res
    return compute_metrics



class WeightedTrainer(Trainer):
    def __init__(self, class_weights, **kw):
        super().__init__(**kw)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.CrossEntropyLoss(
            weight=self.class_weights.to(device=logits.device, dtype=logits.dtype),
            ignore_index=-100)
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        inputs["labels"] = labels
        return (loss, outputs) if return_outputs else loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label-name", default="TSAWA",
                    help="the layer's entity name, e.g. TSAWA, SABCHE or CHAPTER; sets the "
                         "label names and the default --dataset / --output-dir")
    ap.add_argument("--dataset", default=None,
                    help="local dir or Hugging Face repo; default: the layer's dataset on the Hub "
                         "(see DEFAULT_DATASETS; needs HF_TOKEN if private)")
    ap.add_argument("--base-model", default="jhu-clsp/mmBERT-base")
    ap.add_argument("--output-dir", default=None, help="default: runs/<layer>")
    ap.add_argument("--scheme", default="bio", choices=list(SCHEMES))
    ap.add_argument("--weight-scheme", default="inv",
                    choices=["none", "inv", "sqrt_inv", "manual"])
    ap.add_argument("--manual-weights", default=None)
    ap.add_argument("--break-penalty", type=float, default=5.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=10)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--grad-clip", type=float, default=0.3)
    ap.add_argument("--warmup-ratio", type=float, default=0.06)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument(
        "--evals-per-epoch",
        type=int,
        default=1,
        help="When > 1, evaluate/save every steps_per_epoch // evals_per_epoch "
        "steps and multiply --patience by this so patience stays in epochs.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--no-bf16", action="store_true")
    ap.add_argument("--no-grad-checkpointing", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--skip-test", action="store_true")
    ap.add_argument("--smoke-test", action="store_true")
    args = ap.parse_args()

    configure_label(args.label_name)
    layer = PRIMARY.lower()
    args.dataset = args.dataset or DEFAULT_DATASETS.get(PRIMARY, f"{layer}/data/{layer}_dataset")
    args.output_dir = args.output_dir or f"runs/{layer}"
    scheme = args.scheme
    names = SCHEMES[scheme]
    n_labels = len(names)
    set_seed(args.seed)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    print(f"scheme: {scheme.upper()} ({n_labels} labels: {', '.join(names)})")

    ds = (load_from_disk(args.dataset) if Path(args.dataset).exists()
          else load_dataset(args.dataset, token=os.environ.get("HF_TOKEN")))
    print({k: len(v) for k, v in ds.items()})
    keep = ["input_ids", "attention_mask", "labels"]
    ds = ds.remove_columns([c for c in ds["train"].column_names if c not in keep])

    # detect whether the stored labels already carry quotation
    probe = np.array(ds["train"][0]["labels"])
    src_multi = bool((probe > 2).any())
    for i in range(1, min(50, len(ds["train"]))):
        if (np.array(ds["train"][i]["labels"]) > 2).any():
            src_multi = True
            break
    print(f"source dataset: {f'5-label ({layer} + quotation)' if src_multi else f'3-label ({layer} only)'}")

    if scheme != "bio" or src_multi:
        ds = ds.map(lambda b: {"labels": [convert_row(r, scheme, src_multi)
                                          for r in b["labels"]]},
                    batched=True, batch_size=64, desc=f"-> {scheme.upper()}")
    ds.set_format("torch")

    train_ds, eval_ds = ds["train"], ds["validation"]
    if args.smoke_test:
        train_ds = train_ds.select(range(min(32, len(train_ds))))
        eval_ds = eval_ds.select(range(min(8, len(eval_ds))))
        args.epochs = 1

    counts = np.zeros(n_labels, dtype=np.int64)
    for b in ds["train"].with_format("numpy").iter(batch_size=64):
        lab = b["labels"].reshape(-1)
        counts += np.bincount(lab[lab != -100], minlength=n_labels)
    total = counts.sum()
    print("\ntrain label counts:")
    for c, nm in enumerate(names):
        print(f"  {nm:>9}: {counts[c]:>12,}  ({100*counts[c]/total:6.3f}%)")

    n_o = float(counts[0])
    if args.weight_scheme == "none":
        w = np.ones(n_labels)
    elif args.weight_scheme == "manual":
        w = np.array([float(x) for x in args.manual_weights.split(",")])
    else:
        r = np.array([1.0] + [n_o / max(counts[c], 1) for c in range(1, n_labels)])
        w = np.sqrt(r) if args.weight_scheme == "sqrt_inv" else r
        w[0] = 1.0
    print(f"\nweights ({args.weight_scheme}): " +
          "  ".join(f"{nm}={v:.1f}" for nm, v in zip(names, w)))
    print("  NOTE: with the quotation class present these differ from the")
    print("  3-label run; QUOTE is the larger positive class in train.")
    class_weights = torch.tensor(w, dtype=torch.float32)

    cfg = AutoConfig.from_pretrained(
        args.base_model, num_labels=n_labels,
        id2label={i: n for i, n in enumerate(names)},
        label2id={n: i for i, n in enumerate(names)})
    try:
        model = AutoModelForTokenClassification.from_pretrained(
            args.base_model, config=cfg, attn_implementation="sdpa")
    except Exception:
        model = AutoModelForTokenClassification.from_pretrained(
            args.base_model, config=cfg)

    sig = inspect.signature(TrainingArguments.__init__).parameters
    ta = dict(
        output_dir=str(out), learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        weight_decay=args.weight_decay, max_grad_norm=args.grad_clip,
        warmup_ratio=args.warmup_ratio,
        bf16=not args.no_bf16 and torch.cuda.is_available(),
        gradient_checkpointing=not args.no_grad_checkpointing,
        logging_steps=25, save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="iou50_f1",   # primary layer only; quotation is auxiliary
        greater_is_better=True,
        report_to=os.environ.get("REPORT_TO", "none"),
        run_name=args.run_name or f"{layer}-{scheme}-{args.weight_scheme}",
        seed=args.seed, dataloader_num_workers=2,
    )
    eval_key = "eval_strategy" if "eval_strategy" in sig else "evaluation_strategy"
    evals_per_epoch = max(1, int(args.evals_per_epoch))
    steps_per_epoch = max(1, len(train_ds) // (args.batch_size * args.grad_accum))
    early_patience = args.patience
    if evals_per_epoch > 1:
        eval_every = max(1, steps_per_epoch // evals_per_epoch)
        ta[eval_key] = "steps"
        ta["save_strategy"] = "steps"
        ta["eval_steps"] = eval_every
        ta["save_steps"] = eval_every
        early_patience = args.patience * evals_per_epoch
        print(
            f"\n  evals_per_epoch={evals_per_epoch}: {eval_key}=steps "
            f"eval_steps=save_steps={eval_every} "
            f"(steps_per_epoch={steps_per_epoch}); "
            f"early_stopping_patience={early_patience} "
            f"({args.patience} epochs × {evals_per_epoch})"
        )
    else:
        ta[eval_key] = "epoch"
        ta["save_strategy"] = "epoch"
    if "warmup_ratio" not in sig:
        spe = max(1, len(train_ds) // (args.batch_size * args.grad_accum))
        steps = int(spe * args.epochs * args.warmup_ratio)
        ta.pop("warmup_ratio")
        if "warmup_steps" in sig:
            ta["warmup_steps"] = steps
            print(f"\n  warmup_ratio unsupported; warmup_steps={steps}")
    dropped = [k for k in ta if k not in sig]
    if dropped:
        print(f"  dropping unsupported args: {dropped}")
        ta = {k: v for k, v in ta.items() if k in sig}

    trainer_kw = dict(
        model=model,
        args=TrainingArguments(**ta),
        train_dataset=train_ds, eval_dataset=eval_ds,
        compute_metrics=build_metrics(scheme, args.break_penalty),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_patience)],
    )
    trainer = WeightedTrainer(class_weights=class_weights, **trainer_kw)
    trainer.train(resume_from_checkpoint=args.resume or None)

    val = trainer.evaluate(eval_ds, metric_key_prefix="val")
    print("\nVALIDATION:", json.dumps(val, indent=2, default=float))
    test = None
    if not args.skip_test:
        test = trainer.evaluate(ds["test"], metric_key_prefix="test")
        print("\nTEST:", json.dumps(test, indent=2, default=float))
    else:
        print("\nTEST: skipped. Frozen split, final run only.")

    print("\nreference (validation, Viterbi): BIO inv 0.323 | BIOE 0.320 | "
          "IO 0.309 | joint 0.395 | quotation model 0.853")
    if scheme == "multi":
        print("Quotation metrics are diagnostic only. Test quotation density is")
        print("3.2% against train's 7.0%, so quote F1 on test is not comparable.")

    trainer.save_model(str(out / "best"))
    (out / "results.json").write_text(json.dumps({
        "args": vars(args), "scheme": scheme,
        "label_counts": counts.tolist(), "weights": w.tolist(),
        "validation": {k: float(v) for k, v in val.items()
                       if isinstance(v, (int, float))},
        "test": ({k: float(v) for k, v in test.items()
                  if isinstance(v, (int, float))} if test else None),
    }, indent=2))
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()