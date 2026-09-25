"""
Viterbi decoding for 3-label BIO token classification (O / B-x / I-x), for any layer.
"""

from __future__ import annotations

import numpy as np

NEG = -1.0e9
O, B, I = 0, 1, 2


# decoding

def transition_matrix(break_penalty: float) -> np.ndarray:
    """
    3-label BIO transitions. I may only follow B or I; every span exit costs
    `break_penalty`, so fragmenting a span is penalised.
    """
    labels = ["O", "B", "I"]
    n = len(labels)
    m = np.zeros((n, n), dtype=np.float64)
    for i, prev in enumerate(labels):
        for j, nxt in enumerate(labels):
            if nxt == "I" and prev not in ("B", "I"):
                m[i, j] = NEG
                continue
            if prev == "O":
                continue
            # leaving a span (B/I -> anything that is not I) costs the penalty
            if nxt != "I":
                m[i, j] -= break_penalty
    return m


def viterbi(logits: np.ndarray, break_penalty: float) -> np.ndarray:
    """logits [T, C] -> best legal label sequence [T]."""
    T, C = logits.shape
    trans = transition_matrix(break_penalty)
    dp = np.full((T, C), NEG)
    bp = np.zeros((T, C), dtype=np.int64)
    dp[0] = logits[0]
    dp[0, I] = NEG  # a window cannot open mid-span
    for t in range(1, T):
        scores = dp[t - 1][:, None] + trans  # [C_prev, C_next]
        bp[t] = scores.argmax(axis=0)
        dp[t] = scores.max(axis=0) + logits[t]
    path = np.zeros(T, dtype=np.int64)
    path[-1] = int(dp[-1].argmax())
    for t in range(T - 1, 0, -1):
        path[t - 1] = bp[t, path[t]]
    return path


def spans_from_bio(seq: np.ndarray) -> list[tuple[int, int]]:
    """BIO -> inclusive (start, end) spans, matching the team's convention."""
    out, start = [], None
    for i, v in enumerate(seq):
        if v == B:
            if start is not None:
                out.append((start, i - 1))
            start = i
        elif v == I:
            if start is None:
                start = i
        else:
            if start is not None:
                out.append((start, i - 1))
                start = None
    if start is not None:
        out.append((start, len(seq) - 1))
    return out
