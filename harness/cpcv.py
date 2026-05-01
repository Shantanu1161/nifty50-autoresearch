"""Combinatorial Purged Cross-Validation (Lopez de Prado, 2018).

Splits the timeline into N sequential blocks. For every C(N, k) combination
of k blocks selected as the test set, the remaining N-k form the train set.
Train samples whose label horizon (exit_date) overlaps with any test block
are *purged*. After each test block we apply an *embargo* of E days to drop
training samples that follow.

Yields (train_idx, test_idx) pairs of integer positions into `event_times`.
"""
from __future__ import annotations
from itertools import combinations
import numpy as np
import pandas as pd


def cpcv_splits(
    event_times: pd.Series,  # index = event start (t), values = exit_date
    n_splits: int = 10,
    k_test: int = 2,
    embargo_days: int = 5,
):
    n = len(event_times)
    if n == 0:
        return
    starts = pd.to_datetime(event_times.index)
    exits = pd.to_datetime(event_times.values)

    # block boundaries by event-position (equal-count blocks)
    edges = np.linspace(0, n, n_splits + 1, dtype=int)
    blocks = [(edges[i], edges[i + 1]) for i in range(n_splits)]

    all_pos = np.arange(n)

    for combo in combinations(range(n_splits), k_test):
        test_pos = np.concatenate([all_pos[blocks[b][0] : blocks[b][1]] for b in combo])
        # determine the time spans of each test block (start..exit)
        test_spans = []
        for b in combo:
            lo, hi = blocks[b]
            if lo >= hi:
                continue
            span_start = starts[lo]
            span_end = exits[lo:hi].max()
            test_spans.append((span_start, span_end))

        train_pos = np.setdiff1d(all_pos, test_pos, assume_unique=True)

        # purge: drop training events whose label horizon overlaps any test span
        keep_mask = np.ones(len(train_pos), dtype=bool)
        for ts, te in test_spans:
            ev_start = starts[train_pos]
            ev_end = exits[train_pos]
            overlap = (ev_start <= te) & (ev_end >= ts)
            keep_mask &= ~overlap

        # embargo: drop training events that *start* within E days after a test span ends
        if embargo_days > 0:
            emb = pd.Timedelta(days=embargo_days)
            for ts, te in test_spans:
                ev_start = starts[train_pos]
                in_embargo = (ev_start > te) & (ev_start <= te + emb)
                keep_mask &= ~in_embargo

        train_pos = train_pos[keep_mask]
        if len(train_pos) == 0 or len(test_pos) == 0:
            continue
        yield train_pos, test_pos
