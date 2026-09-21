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
