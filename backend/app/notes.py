# Splits a continuous pitch track into discrete notes. A new note starts
# whenever singing stops (an unvoiced gap) or whenever the pitch settles on
# a clearly different value (a sustained change of more than ~0.7 semitone,
# which is about a third of the way to the next note). We require the new
# pitch to hold for a few frames so a fast vibrato or slide does not get
# chopped into dozens of tiny fake notes.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.pitch import PitchTrack

PITCH_CHANGE_THRESHOLD = 0.7  # semitones
MIN_HOLD_FRAMES = 3  # frames the new pitch must hold before we split
MIN_NOTE_FRAMES = 2  # shorter runs are dropped as noise


@dataclass
class Note:
    start: float  # seconds
    end: float  # seconds
    midi: float  # median pitch over the note


def segment_notes(track: PitchTrack) -> list[Note]:
    voiced = track.voiced
    midi = track.midi
    times = track.times
    n = len(midi)

    notes: list[Note] = []
    run_start = None
    run_pitch_ref = None

    def close_run(end_idx: int) -> None:
        nonlocal run_start
        if run_start is None:
            return
        if end_idx - run_start >= MIN_NOTE_FRAMES:
            segment = midi[run_start:end_idx]
            notes.append(
                Note(
                    start=float(times[run_start]),
                    end=float(times[end_idx - 1]),
                    midi=float(np.median(segment)),
                )
            )
        run_start = None

    i = 0
    while i < n:
        if not voiced[i]:
            close_run(i)
            i += 1
            continue

        if run_start is None:
            run_start = i
            run_pitch_ref = midi[i]
            i += 1
            continue

        if abs(midi[i] - run_pitch_ref) > PITCH_CHANGE_THRESHOLD:
            # candidate pitch change: only split if it holds for a bit
            hold_end = min(i + MIN_HOLD_FRAMES, n)
            window = midi[i:hold_end]
            still_voiced = voiced[i:hold_end].all()
            holds = still_voiced and len(window) > 0 and (
                np.abs(window - midi[i]).max() <= PITCH_CHANGE_THRESHOLD
            )
            if holds:
                close_run(i)
                run_start = i
                run_pitch_ref = midi[i]
            # else: treat as a passing blip, keep the current run going
        i += 1

    close_run(n)
    return notes


def merge_short_notes(
    notes: list[Note],
    min_duration: float = 0.14,
    max_gap: float = 0.06,
    same_pitch_tol: float = 0.5,
) -> list[Note]:
    """Stop a sustained note from being chopped into pieces.

    1. Fragments of one note - neighbours closer than max_gap in time and
       within same_pitch_tol semitones - are joined.
    2. Any note shorter than min_duration is absorbed into the neighbour
       (within max_gap) whose pitch is closest. A short note with no such
       neighbour is a real short note and stays.

    Merged pitch is the duration-weighted mean of the parts.
    """

    def join(a: Note, b: Note) -> Note:
        da, db = max(a.end - a.start, 1e-6), max(b.end - b.start, 1e-6)
        return Note(start=a.start, end=b.end, midi=(a.midi * da + b.midi * db) / (da + db))

    def touching(a: Note, b: Note) -> bool:
        return b.start - a.end <= max_gap

    merged = list(notes)

    changed = True
    while changed:
        changed = False
        for i in range(len(merged) - 1):
            a, b = merged[i], merged[i + 1]
            if touching(a, b) and abs(a.midi - b.midi) <= same_pitch_tol:
                merged[i : i + 2] = [join(a, b)]
                changed = True
                break

    while True:
        short = [i for i, n in enumerate(merged) if n.end - n.start < min_duration]
        target = None
        for i in sorted(short, key=lambda k: merged[k].end - merged[k].start):
            options = []
            if i > 0 and touching(merged[i - 1], merged[i]):
                options.append(i - 1)
            if i + 1 < len(merged) and touching(merged[i], merged[i + 1]):
                options.append(i + 1)
            if options:
                j = min(options, key=lambda k: abs(merged[k].midi - merged[i].midi))
                target = (i, j)
                break
        if target is None:
            return merged
        i, j = target
        lo, hi = min(i, j), max(i, j)
        merged[lo : hi + 1] = [join(merged[lo], merged[hi])]
