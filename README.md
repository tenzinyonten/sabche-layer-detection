# sabche-layer-detection

Finding **sabche** (ས་བཅད, outline headings) in Tibetan commentaries with a fine-tuned
`jhu-clsp/mmBERT-base` token classifier. This repo has the whole pipeline: the audit of the
raw annotations, how the spans were cleaned, how the split was built, how the model was
trained and how it is scored.

- Dataset: [Yontenn/formatting-sabche-v1](https://huggingface.co/datasets/Yontenn/formatting-sabche-v1)
- Model: [Yontenn/mmbert-sabche-v1](https://huggingface.co/Yontenn/mmbert-sabche-v1)

## 1. What sabche is

A commentary gives each section it is about to explain an outline heading, for example
`གཉིས་པ་༼ཚུལ་ཁྲིམས་ཕར་ཕྱིན་གྱི་རབ་དབྱེ་༽ནི` ("second, the divisions of the perfection of
ethics"). These headings usually open with an ordinal (དང་པོ, གཉིས་པ, གསུམ་པ) and often
close with ནི. The task is to mark each heading, by character offsets, in a whole book. The
annotations come from OpenPecha `.opf` books (`Sabche.yml`).

## 2. Data preparation

Source: 539 OpenPecha books (266 old batch, 273 new batch). 333 have a Sabche layer (138 old,
195 new) with 43,223 spans. The raw annotations are never edited. Every fix is written to a
sidecar CSV in `data/`. More detail is in `docs/PIPELINE.md`.

**What the audit found**

| | old batch | new batch |
|---|---|---|
| Spans | 18,845 | 24,378 |
| Next span separated only by punctuation | 0.4% | 25.6% |
| Self-overlap or nesting | 0 | 0 |
| Cut in the middle of a syllable (start or end) | 21.3% | 1.4% |
| Open with an ordinal | 60.6% | 56.8% |
| Long spans (40+ characters) found word for word in another book | 12.9% | 14.0% |

**Decisions, each with the number it moved**

| Step | What was done | Effect |
|---|---|---|
| No merging of adjacent headings | In the new batch a quarter of headings sit next to another one. I read samples from 99 books: they are separate one-per-line outline headings, not one split heading, so they stay separate. | 0 spans merged. |
| Re-align drifted offsets | Some old-batch books have offsets shifted by an amount that changes part way through a book (for example P000063: +116, then +229, then +291). A Viterbi search over ordinal-heading anchors picks one offset per stretch. A book is accepted when the share of heading-like spans rises by 20 points or more and ends reach 60% or more. | 30 old-batch books re-aligned, 4,173 spans shifted. 227 spans still unverified after that were dropped. |
| Snap off-by-one edges | Walk at most 3 characters to a syllable boundary (sabche spans end before the shad). Larger errors are left alone. | 1,359 old and 352 new spans snapped, 29 left dirty. Mid-syllable spans in the old batch: 20.5% to 0.1%. Heading-like: 77.7% to 97.7%. |
| Cross-layer overlap check | Sabche against Chapter, Tsawa, Quotation, Citation and Commentary. | 0.1% (old) and 0.6% (new) overlap another layer, 3.3% of new-batch spans touch Commentary. Too small to matter for a one-layer model, so no precedence rule was applied. |
| Book exclusions | 3 books whose offsets cannot be recovered, 2 with no heading-like spans, 5 where only the outline is annotated and the headings in the body are not. | 10 books out. 42,292 spans stay (18,456 old, 23,836 new). |
| Split | Books already in the tsawa split keep their split. Books that share headings are grouped and never separated. | See section 3. |

Two caveats are in the data and were left as is. About 29% of bare `Nth-པ་ནི།` headings with no
title are annotated, so the convention is inconsistent. And 21 kept books have titled-heading
recall between 0.2 and 0.5, so they are only partly annotated.

**Leakage.** Two checks on the test split:
- Long spans (40+ characters) found word for word in a train book: val 1.6% (26 of 1,630),
  test 2.49% (43 of 1,728).
- All 4,079 test spans, any length: 85.8% appear in no train book, 6.8% (279) in one or two
  train books, 2.9% in three to nine, 4.4% in ten or more (stock phrases like `གཉིས་པ་ནི`).
  Only the 6.8% is real document-level overlap, and it is concentrated in one lamrim group.
  The split was left as is.

## 3. Dataset

Hugging Face: [Yontenn/formatting-sabche-v1](https://huggingface.co/datasets/Yontenn/formatting-sabche-v1)
(card in `data/dataset_card.md`). Windows of 8,192 tokens with stride 5,120 (mmBERT
tokenizer), BIO labels (`O`, `B-SABCHE`, `I-SABCHE`). The frozen split is
`data/split_frozen.csv`, 83/8.5/8.5 by window count, test frozen.

| Split | Books (old / new) | Windows | Spans |
|---|---|---|---|
| train | 261 (113 / 148) | 8,886 | 35,122 |
| validation | 33 (10 / 23) | 911 | 3,091 |
| test | 29 (11 / 18) | 963 | 4,079 |

## 4. Training

`src/train.py`, giving [Yontenn/mmbert-sabche-v1](https://huggingface.co/Yontenn/mmbert-sabche-v1).

| | |
|---|---|
| Base model | `jhu-clsp/mmBERT-base`, token classification, 3 labels |
| Learning rate / batch size | 1e-5 / 8 |
| Epochs | 8, early stopping with patience 3, evaluated 4 times per epoch |
| Class weights | inverse frequency: `O` 1.0, `B` 1298.0, `I` 25.0 |
| Other | weight decay 0.01, gradient clip 0.3, warmup 6%, seed 42 |

## 5. Evaluation

Predictions are decoded with Viterbi into character spans (break penalty 5.0 for the validation score logged during training, 4.0 for test). A predicted
span is correct when its IoU with a gold span is at least 0.5, matched greedily one to one.

| | F1 | Precision | Recall |
|---|---|---|---|
| Validation (per window) | 0.867 | 0.882 | 0.852 |
| Test (per window) | 0.9647 | 0.9629 | 0.9666 |
| Test, whole books (29 books) | 0.962 | 0.954 | 0.970 |
| Test, old batch (11 books) | 0.958 | 0.940 | 0.977 |
| Test, new batch (18 books) | 0.964 | 0.964 | 0.965 |
| Test, median book | 0.979 | | |

Per-window scores count a span twice when it falls in two overlapping windows. The
whole-book numbers decode each book once and are the ones to quote. Validation was only scored
per window, and not split by batch. The book-level test numbers can be checked without a
GPU: `python src/score_spans.py --split test --per-book --model mmbert-sabche-v1=results/mmbert-sabche-v1/test/spans`.

## 6. Summary

| Model | Score | Notes |
|---|---|---|
| Joint multi-label baseline (sabche score only, before per-layer split) | 0.394 F1 | Viterbi, IoU 0.5. Plain argmax gave 0.199. Validation, 2,084 gold sabche spans. |
| Sabche model, validation, plain argmax | 0.753 F1 | per window |
| Sabche model, validation, Viterbi | 0.867 F1 | per window |
| Sabche model, test, plain argmax | 0.912 F1 | per window |
| Sabche model, test, Viterbi | **0.9647 F1** | per window |
| Sabche model, test, whole books | **0.962 F1** | 29 books, P 0.954 / R 0.970 |

The joint baseline is from the joint model's own repo (`layer_detection_model_train`,
checkpoint v1.3) and used a different validation set, so it shows the size of the gain and
not a strict comparison.

## Layout

```
data/     dataset card, frozen split, cleaned spans, book verdicts, excluded books, gold spans,
          and the tsawa audit and split that the sabche split builds on
src/      audit, clean, split, build dataset, train, evaluate
docs/     PIPELINE.md (why each decision was made), RESULTS.md (all numbers)
results/  mmBERT test predictions as character offsets
```

Book texts are not included. They come from OpenPecha.
