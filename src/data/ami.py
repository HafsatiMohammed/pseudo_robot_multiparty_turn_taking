"""
Local AMI corpus access.

Reads the NXT-style annotation XML and resolves the per-meeting
speaker-letter <-> headset-channel mapping. This is the ONLY module that knows
the on-disk AMI layout; everything else goes through it.

Layout (verified, see docs/DATA_SCHEMA.md):
    <ami_root>/ami_manual_1.6.1/corpusResources/meetings.xml   (speaker<->channel map)
    <ami_root>/ami_manual_1.6.1/words/<meeting>.<spk>.words.xml (word tokens, ISO-8859-1)
    <ami_root>/headset/<meeting>/audio/<meeting>.Headset-<ch>.wav

Notes:
  - Group size varies (3/4/5 speakers); never assume exactly A-D.
  - The letter<->channel map MUST be read per meeting (do not assume A==Headset-0).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(frozen=True)
class Word:
    """A single transcription token from a words XML file."""
    start: float
    end: float
    text: str
    is_punc: bool


@dataclass(frozen=True)
class Speaker:
    letter: str          # nxt_agent, e.g. "A"
    channel: int         # headset channel index, e.g. 0 -> Headset-0.wav


@dataclass
class Meeting:
    meeting_id: str
    speakers: Dict[str, Speaker]   # letter -> Speaker
    duration: Optional[float]

    @property
    def letters(self) -> List[str]:
        return sorted(self.speakers.keys())

    def channel(self, letter: str) -> int:
        return self.speakers[letter].channel


class AMICorpus:
    """Indexes meetings.xml and reads per-speaker word tokens, with caching."""

    def __init__(self, ami_root: str):
        self.ami_root = Path(ami_root)
        self.annot_root = self.ami_root / "ami_manual_1.6.1"
        self.words_dir = self.annot_root / "words"
        self.meetings_xml = self.annot_root / "corpusResources" / "meetings.xml"
        if not self.meetings_xml.exists():
            raise FileNotFoundError(
                f"meetings.xml not found at {self.meetings_xml}. "
                f"Check the AMI root path ({self.ami_root})."
            )
        self._meetings: Optional[Dict[str, Meeting]] = None
        self._words_cache: Dict[str, List[Word]] = {}

    # -- meeting / speaker mapping --------------------------------------
    @property
    def meetings(self) -> Dict[str, Meeting]:
        if self._meetings is None:
            self._meetings = self._parse_meetings()
        return self._meetings

    def _parse_meetings(self) -> Dict[str, Meeting]:
        tree = ET.parse(self.meetings_xml)
        root = tree.getroot()
        out: Dict[str, Meeting] = {}
        for m in root.iter("meeting"):
            obs = m.get("observation")
            if obs is None:
                continue
            dur = m.get("duration")
            speakers: Dict[str, Speaker] = {}
            for sp in m.findall("speaker"):
                letter = sp.get("nxt_agent")
                ch = sp.get("channel")
                if letter is None or ch is None:
                    continue
                speakers[letter] = Speaker(letter=letter, channel=int(ch))
            out[obs] = Meeting(
                meeting_id=obs,
                speakers=speakers,
                duration=float(dur) if dur is not None else None,
            )
        return out

    def get_meeting(self, meeting_id: str) -> Meeting:
        try:
            return self.meetings[meeting_id]
        except KeyError:
            raise KeyError(
                f"Meeting '{meeting_id}' not found in meetings.xml "
                f"({len(self.meetings)} meetings indexed)."
            )

    # -- audio paths ----------------------------------------------------
    def headset_wav(self, meeting_id: str, letter: str) -> Path:
        ch = self.get_meeting(meeting_id).channel(letter)
        return (
            self.ami_root
            / "headset"
            / meeting_id
            / "audio"
            / f"{meeting_id}.Headset-{ch}.wav"
        )

    # -- words ----------------------------------------------------------
    def read_words(
        self,
        meeting_id: str,
        letter: str,
        include_vocalsound: bool = False,
    ) -> List[Word]:
        """
        Parse <meeting>.<letter>.words.xml -> list of Word, sorted by start.

        Lexical <w> tokens only by default. <vocalsound> can be optionally
        included as (non-lexical) activity; <disfmarker>/<gap> are always
        skipped (zero-width markers, not speech). Tokens missing timestamps are
        skipped. Punctuation (<w punc="true">) is kept (is_punc=True) -- callers
        that build speech regions should drop zero-width punc tokens; the text
        branch keeps "?" for question detection.
        """
        key = f"{meeting_id}.{letter}|{int(include_vocalsound)}"
        if key in self._words_cache:
            return self._words_cache[key]

        path = self.words_dir / f"{meeting_id}.{letter}.words.xml"
        words: List[Word] = []
        if not path.exists():
            self._words_cache[key] = words
            return words

        tree = ET.parse(path)
        root = tree.getroot()
        for el in root.iter():
            tag = el.tag.split("}")[-1]  # strip any namespace
            if tag == "w":
                kind = "w"
            elif tag == "vocalsound" and include_vocalsound:
                kind = "vocalsound"
            else:
                continue
            st, et = el.get("starttime"), el.get("endtime")
            if st is None or et is None:
                continue
            try:
                start, end = float(st), float(et)
            except ValueError:
                continue
            is_punc = (kind == "w") and (el.get("punc") == "true")
            text = (el.text or "").strip() if kind == "w" else "<vocalsound>"
            words.append(Word(start=start, end=end, text=text, is_punc=is_punc))

        words.sort(key=lambda w: (w.start, w.end))
        self._words_cache[key] = words
        return words
