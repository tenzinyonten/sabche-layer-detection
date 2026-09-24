# Pipeline and decisions

Run from the repo root. Every step reads the previous step's CSV and writes a new one, so the
cloned `.opf` files are never changed. Book texts are not in this repo, so the audit, clean
and build steps need them cloned into `data/raw_opf/`.

## 1. Audit

`src/audit_sabche.py` reads the raw `Sabche.yml` of every book and reports numbers only. Of
539 books, 333 have a Sabche layer (138 old batch, 195 new batch) with 43,223 spans.

| | old | new |
|---|---|---|
| Median span length (p10 to p90) | 43 (20 to 204) | 36 (16 to 160) |
| Next span separated only by punctuation | 0.4% | 25.6% |
| Self-overlap or nesting | 0 | 0 |
| Overlaps Chapter, Tsawa, Quotation or Citation | 0.1% | 0.6% |
| Overlaps Commentary | 0.0% | 3.3% |
| Opens with an ordinal | 60.6% | 56.8% |
| Closes with ནི | 66.0% | 63.0% |
| Cut mid-syllable (start or end) | 21.3% | 1.4% |
| Long spans (40+ chars) found verbatim in another book | 12.9% | 14.0% |

Boundary check note: `check_boundary_snapping.check_span` reports about 92% bad ends. That
is an artifact, because it expects the shad inside the span and sabche spans stop before it.
The mid-syllable test used here only counts a cut as bad when the characters on both sides of
it are non-boundary.

## 2. Cleaning (`src/clean_sabche_spans.py`)

Outputs `data/sabche_spans_clean.csv` (one row per span, with a `dropped` flag) and
`data/sabche_book_verdicts.csv` (one row per book).

**No merging of adjacent headings.** A quarter of new-batch headings sit right next to
another one (only punctuation between them). Reading samples from 99 books showed they are
separate one-per-line outline headings such as `(2) ...`, `{1} ...`, or `ལ་གསུམ།` followed by
`དང་པོ་...ནི`. So nothing is merged. This is the opposite of the tsawa decision.

**Re-aligning drifted offsets.** Some old-batch books have offsets shifted by a constant that
changes part way through (P000063: +116, then +229, then +291; P000250: +665 throughout;
P000185: +360, then +205). For each book the script builds anchors from ordinal headings whose
length matches the span, runs a Viterbi search for a piecewise-constant offset, and accepts the
result when the heading-like rate rises by 20 points or more and ends reach 60% or more.
The verdict file has 30 old-batch books re-aligned (298 keep, 30 realigned, 5 exclude in
total). That shifted 4,173 spans. After the shift, spans that had to be snapped and are still
not heading-like were dropped: 227 spans.

**Snapping.** Only off-by-one errors are fixed: walk at most 3 characters to a syllable
boundary, with the end stopping before the shad. 1,359 old and 352 new spans were snapped, 29
old-batch spans stay dirty (more than 3 characters off), and 1 span was trimmed where snapping
created an overlap (P000128).

Effect on the kept spans:

| | mid-syllable raw to clean | heading-like raw to clean |
|---|---|---|
| old | 20.5% to 0.1% | 77.7% to 97.7% |
| new | 1.4% to 0.0% | 99.0% to 99.4% |

**Cross-layer overlap.** Sabche overlaps other layers on 0.1% (old) and 0.6% (new) of spans,
and 3.3% of new-batch spans touch Commentary. That does not affect a one-layer model, so no
precedence rule is applied (unlike tsawa and quotation).

**Book exclusions** (10 books, `data/sabche_excluded_books.csv`):

| Reason | Books |
|---|---|
| offsets unrecoverable (spans on body text, re-alignment fails) | P000016, P000017, P000226 |
| no heading-like spans | I82AB29CD (8 spans), IC3A7006F (1 span) |
| under-annotated: only the outline is annotated, not the headings in the body | I100E7DAD, I7A79CC95, I9779A606, IB8F4E5BE, P000031 |

The first heuristic flagged 16 books. Reading them kept 6 that really have one-per-line
headings (IFF541F44, I55D7A55C, IE166DA22, I78C84F85, I8698A1DF, IAAADCAA0). A low span count
on its own is not a reason to exclude: 31 books with under 10 spans are kept. After cleaning,
42,292 spans are active (18,456 old, 23,836 new).

Known gaps, left as they are: bare `Nth-པ་ནི།` headings with no title are annotated only 29%
of the time (1,698 of 5,924), 21 kept books have titled-heading recall between 0.2 and 0.5,
and the 227 dropped spans are labelled `O` even though most are probably real headings.

## 3. Split (`src/prepare_sabche_split.py`)

- Books already in the tsawa split (`data/tsawa_split_frozen.csv`) keep their split. 116 of
  them remain after exclusions, 3 were moved by conflict resolution.
- Books are linked into must-link groups from two sources: the tsawa groups, and Sabche text
  sharing (10% or more of a book's long-span characters found verbatim in the other book over
  at least 2 spans, or 10 or more shared spans at any share). 34 links that came from one
  shared boilerplate title are ignored. The absolute 10-span rule was added after leakage
  stayed high: P000263 (train) contained 63 and 55 headings from test books P000015 and P000177.
- When a group has members in different splits it goes to test, then val, then train, so no
  frozen val or test book leaves evaluation. I36A7A668 and I7D476A8B moved from train to test
  (a lamrim outline shared with I3F4A91F5) and IAE1E11E5 from train to validation.
- The remaining groups are placed greedily, largest first, toward 83/8.5/8.5 by window count.
  The books fixed by the tsawa split already fill 8.9% of test windows, so test got no new
  groups.
- 240 groups, 34 with more than one book, the largest with 24 books.

An earlier version of this split used 76/12/12. The 83/8.5/8.5 version keeps the same spans,
exclusions and groups and only changes the targets. Only the final split is in this repo.

Leakage in the final split:

| | long spans (40+ chars) verbatim in a train book |
|---|---|
| val | 1.6% (26 of 1,630) |
| test | 2.49% (43 of 1,728) |

For all 4,079 test spans of any length: 3,501 (85.8%) are in no train book, 279 (6.8%) in one
or two, 119 (2.9%) in three to nine, and 180 (4.4%) in ten or more (stock phrases like
`གཉིས་པ་ནི`, 144 of them under 15 characters). Only the 6.8% is real document-level overlap,
and it is concentrated in one lamrim group: I7D476A8B shares 14 headings with train book
P000218, and I36A7A668 shares 10 each with I54375004 and P000100. The link rule missed these
because it only counts spans of 40+ characters. The split was left as is.

Gold spans are varied: of all 42,292, 35,541 (84.0%) are unique strings after normalising
whitespace, numbering and trailing shad, and the top 100 strings cover 5.3%.

## 4. Dataset build (`src/build_sabche_dataset.py`)

Uses the tsawa builder (`src/build_tsawa_dataset.py`, functions `label_tokens`,
`sliding_windows`, `pack_window`) with the label names `O`, `B-SABCHE`, `I-SABCHE`. Windows are
8,192 tokens with stride 5,120 (mmBERT tokenizer), labels use the token-start rule, and
CLS, SEP and padding are -100. Every one of the 42,292 spans has at least one B token. Counts
are in `data/sabche_dataset_stats.json`.

| Label | train | val | test |
|---|---|---|---|
| O | 69,878,137 | 7,190,531 | 7,602,369 |
| B-SABCHE | 53,837 | 4,734 | 5,920 |
| I-SABCHE | 2,797,816 | 258,495 | 274,716 |

Counts are per window, so tokens in overlapping windows count twice.

## 5. Training

```
python src/train.py --dataset Yontenn/formatting-sabche-v1 --scheme bio \
    --weight-scheme inv --epochs 8 --evals-per-epoch 4 --patience 3 \
    --no-grad-checkpointing --skip-test --output-dir runs/sabche
```

The reported run used the tsawa training script with the label names in the dataset renamed
to `B-SABCHE` and `I-SABCHE`, because that script had `TSAWA` hard-coded. `src/train.py` is
the same script with the label name as a parameter. The class weights are inverse frequency
from the train counts (`O` 1.0, `B` 1298.0, `I` 25.0). Tsawa's `B` weight was around 1700 and
was not reused. Test is skipped in training and scored once at the end.

## 6. Evaluation

- `src/eval_viterbi_iou.py` scores a model on a tokenized split at window level (Viterbi
  decoding, IoU 0.5, greedy one-to-one, inclusive offsets) and can dump the predicted spans.
- `src/mmbert_dump_to_chars.py` converts that dump to character offsets per book.
- `src/score_spans.py` scores offset files against `data/sabche_gold.csv` per book, split by
  batch. This is where the whole-book test numbers come from, and it needs no GPU.
- `src/pred_ordinal_check.py` checks whether predicted spans start on an ordinal as often as
  gold spans do.

Window-level scoring counts a span twice when it falls in two overlapping windows, so the
whole-book scores are the ones to quote.
