"""
RoBERTa text feature caching (frozen encoder, offline).

Gold-transcript word embeddings (oracle upper bound; ASR is the follow-up).
Frozen eval()/no_grad, never trained.

Per sample:
  Take the human words available at t (end <= t, i.e. fully uttered before the
  decision point -- no future leakage), ordered by time. Tokenize, take last
  hidden states, mean-pool subwords back to words via word_ids, then place each
  word vector on the 120 x 50 ms grid by its [start, end] span. Frames with no
  active word are zero. -> [120, 768] (sequence mode, preserves recency).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch

from .align import place_words_on_grid
from ..data.ami import Word


@dataclass
class TimedWord:
    text: str
    start: float
    end: float


def events_to_words(events: List[dict]) -> List["TimedWord"]:
    """Convert manifest text_context events (THE text source) to TimedWords.

    Drops punctuation-only tokens; keeps each word/utterance event with its span.
    Caller must already have cut events at end <= t (see ManifestRecord.human_text_events)."""
    out: List[TimedWord] = []
    for e in events:
        if e.get("is_punc"):
            continue
        txt = (e.get("text") or "").strip()
        if not txt or e.get("start") is None or e.get("end") is None:
            continue
        out.append(TimedWord(text=txt, start=float(e["start"]), end=float(e["end"])))
    out.sort(key=lambda w: (w.start, w.end))
    return out


def collect_human_words(
    corpus,
    meeting_id: str,
    human_letters: List[str],
    context_start: float,
    t: float,
) -> List[TimedWord]:
    """Human lexical words with start >= context_start and end <= t, time-ordered.

    end <= t is the causal cut for text: only words fully uttered before the
    decision point (a word still in progress at t is not yet 'available')."""
    words: List[TimedWord] = []
    for letter in human_letters:
        for w in corpus.read_words(meeting_id, letter):
            if w.is_punc:
                continue
            if w.end <= t and w.end >= context_start and w.text:
                words.append(TimedWord(text=w.text, start=w.start, end=w.end))
    words.sort(key=lambda w: (w.start, w.end))
    return words


class RoBERTaCacher:
    def __init__(self, model_name: str = "roberta-base", device: str = "cpu", max_length: int = 512):
        from transformers import AutoModel, AutoTokenizer

        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.model.requires_grad_(False)
        self.model.to(device)
        self.hidden_dim = self.model.config.hidden_size

    def _pool_subwords(self, hidden: np.ndarray, word_ids: List, W: int) -> np.ndarray:
        """Mean-pool subword hidden states [num_tokens, D] back to [W, D] by word_ids.
        Tokens with word_id None (specials/padding) or >= W (truncated) are skipped;
        words with no surviving subword stay zero."""
        vecs = np.zeros((W, self.hidden_dim), dtype=np.float32)
        cnt = np.zeros(W, dtype=np.int64)
        for ti, wid in enumerate(word_ids):
            if wid is None or wid >= W:
                continue
            vecs[wid] += hidden[ti]
            cnt[wid] += 1
        cnt[cnt == 0] = 1
        vecs /= cnt[:, None]
        return vecs

    @torch.no_grad()
    def encode_words_batch(self, word_lists: List[List[str]]) -> List[np.ndarray]:
        """Batched form of ``encode_words``: one RoBERTa forward for B word-lists.

        Empty lists get a (0, D) array and are excluded from the forward. Padding is
        attention-masked, so each non-empty sample's per-word vectors match encoding it
        alone (real-token outputs are padding-invariant), modulo tiny batched-matmul
        float noise. Returns a list of [W_i, D] arrays aligned to ``word_lists``."""
        results: List[np.ndarray] = [np.zeros((0, self.hidden_dim), dtype=np.float32)
                                     for _ in word_lists]
        idxs = [i for i, wl in enumerate(word_lists) if wl]
        if not idxs:
            return results
        batch = [word_lists[i] for i in idxs]
        enc = self.tokenizer(
            batch,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
        )
        out = self.model(**{k: v.to(self.device) for k, v in enc.items()})
        hidden = out.last_hidden_state.float().cpu().numpy()  # [Bn, Tpad, D]
        for row, i in enumerate(idxs):
            results[i] = self._pool_subwords(hidden[row], enc.word_ids(row), len(word_lists[i]))
        return results

    def encode_words(self, words: List[str]) -> np.ndarray:
        """[W, D] one mean-pooled vector per input word (batch of 1)."""
        return self.encode_words_batch([words])[0]

    def encode_sample_batch(
        self,
        timed_words_list: List[List[TimedWord]],
        context_starts: List[float],
        num_bins: int = 120,
        bin_dur: float = 0.05,
    ) -> List[np.ndarray]:
        """Batched form of ``encode_sample``: one RoBERTa forward for B samples.
        Returns a list of [num_bins, D] float16 grids aligned to the inputs."""
        vecs_list = self.encode_words_batch([[w.text for w in tw] for tw in timed_words_list])
        results: List[np.ndarray] = []
        for tw, vecs, cs in zip(timed_words_list, vecs_list, context_starts):
            if not tw:
                grid = np.zeros((num_bins, self.hidden_dim), dtype=np.float32)
            else:
                spans: List[Tuple[float, float]] = [(w.start, w.end) for w in tw]
                grid = place_words_on_grid(vecs, spans, clip_start=cs, num_bins=num_bins, bin_dur=bin_dur)
            # Frozen features feed a GRU; float16 on disk is plenty and halves cache size.
            results.append(grid.astype(np.float16))
        return results

    def encode_sample(
        self,
        timed_words: List[TimedWord],
        context_start: float,
        num_bins: int = 120,
        bin_dur: float = 0.05,
    ) -> np.ndarray:
        """-> [num_bins, D] grid-aligned word embeddings, float16 (batch of 1)."""
        return self.encode_sample_batch([timed_words], [context_start],
                                        num_bins=num_bins, bin_dur=bin_dur)[0]
