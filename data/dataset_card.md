---
pretty_name: Formatting Sabche
language:
- bo
task_categories:
- token-classification
tags:
- tibetan
- sabche
- bio
- mmbert
- openpecha
size_categories:
- 10K<n<100K
license: other
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: validation
    path: data/validation-*
  - split: test
    path: data/test-*
---

# Sabche (ས་བཅད་, outline heading) dataset - mmBERT BIO

Binary token classification for the Sabche layer, built with the tsawa
pipeline's labeling/windowing. Generated 2026-09-22.

Split: 83/8.5/8.5 by window count (`sabche_split_frozen.csv`).

| item | value |
|---|---|
| tokenizer | `jhu-clsp/mmBERT-base` (fast, offsets) |
| window / stride | 8192 (8190 content + CLS/SEP) / 5120 - same as tsawa |
| labels | `O`=0, `B-SABCHE`=1, `I-SABCHE`=2; CLS/SEP/pad = -100 |
| label rule | token-start rule (`build_tsawa_dataset.label_tokens`) |
| columns | input_ids, attention_mask, labels, token_start, token_end, char_start, char_end, pecha_id, source_batch, window_index, n_tokens_doc, coverage_pct (no `features` column) |
| span source | `data/sabche_spans_clean.csv` (dropped=False) |
| split | `data/split_frozen.csv` (**test frozen**) |
| metric (for training) | IoU ≥ 0.5, greedy one-to-one, inclusive offsets - same as tsawa/quotation |

## Pipeline

```bash
python src/audit_sabche.py                  # A: audit (read-only)
python src/clean_sabche_spans.py            # B: sidecar + book verdicts
python src/prepare_sabche_split.py --val-frac 0.085 --test-frac 0.085
python src/build_sabche_dataset.py --split-file data/split_frozen.csv \
    --out-dir sabche_dataset --stats-json data/sabche_dataset_stats.json
```

## A. Audit (raw Sabche.yml offsets)

| | old (P) | new (I) |
|---|---|---|
| books / with Sabche | 266 / 138 | 273 / 195 |
| spans | 18,845 | 24,378 |
| length median (p10-p90) | 43 (20-204) | 36 (16-160) |
| next span separated only by punctuation | 0.4% | 25.6% |
| self-overlap / nested | 0 / 0 | 0 / 0 |
| overlap Chapter, Tsawa, Quotation or Citation | 0.1% | 0.6% |
| overlap Commentary | 0.0% | 3.3% |
| opens with ordinal (དང་པོ…) | 60.6% | 56.8% |
| closes …ནི | 66.0% | 63.0% |
| mid-syllable start/end (either) | 21.3% | 1.4% |
| long spans (≥40 chars) found verbatim in another book | 12.9% | 14.0% |

Notes:
- `check_boundary_snapping.check_span` reports about 92% bad ends. That's an
  artifact: Sabche spans stop *before* the shad, and that check expects the
  shad inside. The mid-syllable test used here counts a cut as bad only when
  the characters on both sides of it are non-boundary.
- Cross-layer overlap doesn't affect this binary model, so no precedence
  rule is applied.
- About 29% (1,698 of 5,924) of bare `Nth-པ་ནི།` headings with no title are
  annotated, so the convention is inconsistent. Left as is.

## B. Cleaning decisions

| decision | rule | evidence |
|---|---|---|
| **no merge** of adjacent spans | new-batch neighbours are separate one-per-line outline headings (`(2) …`, `{1} …`, `ལ་གསུམ།` then `དང་པོ་…ནི`) | read samples from 99 books |
| **re-align drifted offsets** | Viterbi over length-matched ordinal-heading anchors, piecewise-constant offset; accept if heading-like rate rises ≥ 20 points and ends ≥ 60% | P000250 +665 throughout; P000063 +116, then +229, then +291; P000185 +360, then +2052 |
| **snap off-by-one only** | walk ≤ 3 chars to a syllable boundary (end stops before the shad); leave the rest | P000128 `ག\|ཉིས་པ` |
| drop residual spans in re-aligned books | snapped **and** not heading-like after the shift | 227 spans |
| trim overlap created by snapping | earlier span's end = next span's start | 1 span (P000128) |
| exclude books | see table below | read samples before excluding |

Cleaning results (all spans):

| | old | new |
|---|---|---|
| spans | 18,845 | 24,378 |
| shifted (re-aligned) | 4,173 (30 books) | 0 |
| snapped | 1,359 | 352 |
| left dirty (> 3 chars) | 29 | 0 |
| dropped: re-aligned but unverified | 227 | 0 |
| dropped: book excluded | 147 (+ under-annotated) | 9 (+ under-annotated) |
| **active in dataset** | **18,456** | **23,836** |

On the kept spans:

| | mid-syllable raw → clean | heading-like raw → clean |
|---|---|---|
| old | 20.5% → 0.1% | 77.7% → 97.7% |
| new | 1.4% → 0.0% | 99.0% → 99.4% |

Excluded books (10, `sabche_excluded_books.csv`):

| reason | books |
|---|---|
| offsets unrecoverable (spans on body text, re-alignment fails) | P000016, P000017, P000226 |
| no heading-like spans | I82AB29CD (8 spans), IC3A7006F (1 span) |
| under-annotated: outline/TOC annotated, body headings not (titled-heading recall < 0.2) | I100E7DAD, I7A79CC95, I9779A606, IB8F4E5BE, P000031 |

The first heuristic flagged 16 books. Reading them kept 6: IFF541F44,
I55D7A55C, IE166DA22, I78C84F85, I8698A1DF and IAAADCAA0 have real
one-per-line headings. Low span count on its own isn't used as an exclusion
reason: 31 books with fewer than 10 spans are kept.

## B. Split

- All 116 tsawa split books that remain keep their split, except the 3 moved by
  conflict resolution.
- Must-link groups combine two sources:
  - tsawa split `group_id`
  - Sabche verbatim sharing: ≥ 10% of a book's long-span characters over
    ≥ 2 spans, or ≥ 10 shared spans at any share.
- 34 single-span links are ignored. They come from a shared boilerplate
  title; for example, `དཔེ་དེབ་བལྟ་བར་བསྐུལ་བའི་གཏམ…` links 9 books through
  I8698A1DF.
- The ≥ 10 absolute rule was added after leakage stayed high: P000263
  (train) contains 63 and 55 headings from test books P000015 and P000177.
- A group with conflicting fixed members goes to test, then val, then train,
  so no frozen val/test book leaves eval. Moves:
  - I36A7A668 and I7D476A8B: train → test (lamrim outline shared with
    I3F4A91F5)
  - IAE1E11E5: train → val
- New groups are placed greedily, largest first, toward 83/8.5/8.5 by window
  count. The tsawa split-fixed groups alone already fill 8.9% of test windows,
  so test received no new groups: every test book comes from tsawa split's test
  set or the moved lamrim group.
- 240 groups: 34 with more than one book; the largest has 24 books
  (I048844DB).

| split | books | old / new | from tsawa split | windows | spans |
|---|---|---|---|---|---|
| train | 261 | 113 / 148 | 75 | 8,886 (82.6%) | 35,122 |
| validation | 33 | 10 / 23 | 18 | 911 (8.5%) | 3,091 |
| test | 29 | 11 / 18 | 23 | 963 (8.9%) | 4,079 |

Leakage (long spans verbatim in any train book):

| split | tsawa split as-is, Sabche books only | this split (8.5/8.5) |
|---|---|---|
| val | 0.5% | **1.6% (26 / 1,630)** |
| test | 1.4% | **2.49% (43 / 1,728)** |

**Test leakage, all span lengths:** 14.2% raw text overlap in test, but 84%
of gold spans are unique text and most overlap is stock phrasing (e.g.
numbered heading templates), not document leakage. Real document-level
leakage is ~6.8%, concentrated in one book group (lamrim). The split is left
as is.

| test spans (4,079) found verbatim in… | spans | of which < 15 chars |
|---|---|---|
| no train book | 3,501 (85.8%) | 140 |
| 1-2 train books (document-level) | 279 (6.8%) | 55 |
| 3-9 train books | 119 (2.9%) | 43 |
| 10+ train books (stock phrases) | 180 (4.4%) | 144 |

- The 1-2 book overlap is concentrated in the lamrim group: I7D476A8B
  shares 14 headings with train book P000218, and I36A7A668 shares 10 each
  with I54375004 and P000100. The link rule missed these because it only
  counts spans of 40+ characters.
- Gold span diversity (all 42,292 spans, whitespace / `༈` / numbering /
  trailing shad normalized): 35,541 unique strings (84.0%), and 32,304 of
  them occur only once. The top 10 strings cover 2.3% of spans and the top
  100 cover 5.3%. The most common are `གཉིས་པ་ནི` (316), `གསུམ་པ་ནི` (182)
  and `དང་པོ་ནི` (89). Linked
pairs straddling splits: 4 → 0.

## C. Label distribution and class weights (train)

Counts are per window, so tokens in overlapping windows are counted twice.

| label | train | val | test |
|---|---|---|---|
| O | 69,878,137 | 7,190,531 | 7,602,369 |
| B-SABCHE | 53,837 | 4,734 | 5,920 |
| I-SABCHE | 2,797,816 | 258,495 | 274,716 |

Inverse-frequency weights from **this** train set (`N / (K · n_c)`):

| scheme | O | B-SABCHE | I-SABCHE |
|---|---|---|---|
| BIO (K=3) | 0.347 | 450.31 | 8.67 |
| IO (K=2, B folded into I) | 0.520 | - | 12.75 |

Tsawa's B weight was about 1700×. Don't reuse it here.

Every one of the 42,292 active spans has at least one B token (0 lost to
tokenization).

## Caveats

- Metric: `src/eval_viterbi_iou.py` (`inclusive_iou`, `match_iou`, `score`): IoU >= 0.5,
  inclusive offsets, greedy best-first one-to-one matching, on this dataset's 3 BIO
  labels. Scoring is per window, so a span that falls in two overlapping windows is
  counted twice, the same as for tsawa. `src/score_spans.py` scores whole books.
- Titled-heading recall is between 0.2 and 0.5 in 21 kept books, so they're
  partly annotated and some real headings are labeled O.
- Bare `Nth-པ་ནི།` headings are inconsistently annotated (29%).
- The re-aligned books rely on an anchor heuristic. 227 spans were dropped
  as unverified, and their true headings are labeled O.
