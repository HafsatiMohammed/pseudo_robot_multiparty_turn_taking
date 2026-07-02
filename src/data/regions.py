"""
Speech-region construction and interval math.

A "region" is a merged (onset, offset) interval of continuous speech for one
speaker, built from word tokens. All activity queries here are used by the
timing-feature extractor; causality (only words with start < t) is enforced by
the caller before merging.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from .ami import Word

Region = Tuple[float, float]


def merge_words_to_regions(
    words: Sequence[Word],
    gap_threshold: float = 0.3,
    drop_punc: bool = True,
) -> List[Region]:
    """
    Merge consecutive lexical words into speech regions.

    Two words join the same region if the silence between them is <= gap_threshold.
    Zero-width punctuation tokens are dropped (they carry no acoustic duration).
    """
    toks = [w for w in words if not (drop_punc and w.is_punc)]
    # keep only tokens with positive or zero duration and valid ordering
    toks = sorted(toks, key=lambda w: (w.start, w.end))
    regions: List[Region] = []
    for w in toks:
        if w.end < w.start:
            continue
        if not regions:
            regions.append((w.start, w.end))
            continue
        last_start, last_end = regions[-1]
        if w.start - last_end <= gap_threshold:
            regions[-1] = (last_start, max(last_end, w.end))
        else:
            regions.append((w.start, w.end))
    return regions


def filter_regions_before(regions: Sequence[Region], t: float) -> List[Region]:
    """Keep only regions that STARTED before t (onset < t). A region may extend
    past t (speech ongoing at the decision point) -- that is causal and kept."""
    return [(s, e) for (s, e) in regions if s < t]


def covers(regions: Sequence[Region], tau: float) -> bool:
    """True if any region covers instant tau (onset <= tau <= offset)."""
    for s, e in regions:
        if s <= tau <= e:
            return True
    return False


def active_letters_at(regions_by_spk: Dict[str, List[Region]], tau: float) -> List[str]:
    """Letters whose speech covers instant tau."""
    return [spk for spk, regs in regions_by_spk.items() if covers(regs, tau)]


def ongoing_run_onset(regions: Sequence[Region], t: float) -> float | None:
    """If a region covers t, return its onset (start of the ongoing run); else None.
    If several regions cover t (shouldn't happen for one speaker after merge),
    return the earliest onset."""
    onsets = [s for s, e in regions if s <= t <= e]
    return min(onsets) if onsets else None


def last_offset_before(regions: Sequence[Region], t: float) -> float | None:
    """Latest offset that ends at or before t (most recent finished speech)."""
    ends = [e for s, e in regions if e <= t]
    return max(ends) if ends else None


def time_with_min_speakers(
    regions_by_spk: Dict[str, List[Region]],
    win_start: float,
    win_end: float,
    k: int,
) -> float:
    """
    Seconds within [win_start, win_end] during which >= k speakers are
    simultaneously active. Intervals are clipped to the window. Per-speaker
    regions are assumed non-overlapping (true after merge), so a sweep with
    +1/-1 events gives the simultaneous-speaker count.
    """
    if win_end <= win_start:
        return 0.0
    events: List[Tuple[float, int]] = []
    for regs in regions_by_spk.values():
        for s, e in regs:
            cs, ce = max(s, win_start), min(e, win_end)
            if ce > cs:
                events.append((cs, +1))
                events.append((ce, -1))
    if not events:
        return 0.0
    # sort by time; process exits before entries at equal time so touching
    # intervals of the SAME count don't spuriously bump the level
    events.sort(key=lambda x: (x[0], x[1]))
    total = 0.0
    count = 0
    prev_t = events[0][0]
    for tt, delta in events:
        if tt > prev_t and count >= k:
            total += tt - prev_t
        count += delta
        prev_t = tt
    return total
