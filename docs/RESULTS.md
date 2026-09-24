# Results

## How it is scored

A predicted span is correct when its character IoU with a gold span is at least 0.5, matched
greedily one to one. Test scores are whole-book: each book is decoded once (overlapping windows
de-duplicated) and the scores are micro-averaged over books. The validation score is the one
the trainer logged, per window: the windows overlap, so a heading in the overlap counts twice.

## Test split (29 books, 4,079 gold spans)

Whole books:

| | F1 | Precision | Recall |
|---|---|---|---|
| All 29 books | 0.962 | 0.954 | 0.970 |
| Old batch (11 books) | 0.958 | 0.940 | 0.977 |
| New batch (18 books) | 0.964 | 0.964 | 0.965 |
| Median book | 0.979 | | |

The model predicts 4,144 spans for 4,079 gold. Reproduce without a GPU:

```
python src/score_spans.py --split test --per-book \
  --model mmbert-sabche-v1=results/mmbert-sabche-v1/test/spans
```

### Per book (whole books)

| Book | Batch | Gold | F1 | False positives |
|---|---|---|---|---|
| P000083 | old | 225 | 0.998 | 1 |
| P000067 | old | 382 | 0.992 | 5 |
| I3F4A91F5 | new | 367 | 0.991 | 5 |
| P000027 | old | 52 | 0.990 | 1 |
| P000164 | old | 313 | 0.989 | 7 |
| IE5895799 | new | 227 | 0.989 | 5 |
| I52248444 | new | 236 | 0.989 | 4 |
| P000118 | old | 42 | 0.988 | 1 |
| I07240379 | new | 118 | 0.987 | 2 |
| P000144 | old | 107 | 0.986 | 3 |
| IFA88A536 | new | 254 | 0.982 | 4 |
| I9B6A4525 | new | 196 | 0.979 | 3 |
| I9AEEF96A | new | 22 | 0.978 | 1 |
| I9D9C7AC9 | new | 42 | 0.977 | 2 |
| ICDC84458 | new | 181 | 0.962 | 11 |
| IC6F06BCD | new | 122 | 0.955 | 7 |
| IDB6093E9 | new | 154 | 0.945 | 10 |
| P000037 | old | 161 | 0.943 | 16 |
| P000013 | old | 44 | 0.926 | 7 |
| I0FCFA88F | new | 248 | 0.925 | 10 |
| I36A7A668 | new | 88 | 0.918 | 11 |
| I7D476A8B | new | 55 | 0.915 | 9 |
| P000242 | old | 261 | 0.907 | 39 |
| IC05A6BE0 | new | 8 | 0.824 | 2 |
| I575514A8 | new | 77 | 0.775 | 2 |
| P000269 | old | 29 | 0.276 | 21 |
| I319DAFF7 | new | 15 | 1.000 | 0 |
| IF3ACC3E1 | new | 45 | 1.000 | 0 |
| P000247 | old | 8 | 1.000 | 0 |

Twenty-six of the 29 books score above 0.9. The weak ones are P000269 (0.276, 21 false
positives on 29 gold spans), I575514A8 and IC05A6BE0 (only 8 gold spans). Some false positives
are probably real headings that were never annotated (see the known gaps in
`docs/PIPELINE.md`), and I did not check them one by one.

## Validation

At the last epoch (8), scored per window during training with break penalty 5.0:

| Decoding | F1 | Precision | Recall |
|---|---|---|---|
| Plain argmax | 0.753 | | |
| Viterbi | 0.867 | 0.882 | 0.852 |

That is 4,600 predicted spans for 4,761 gold windows-spans. Validation was not scored per
whole book and not split by batch.

