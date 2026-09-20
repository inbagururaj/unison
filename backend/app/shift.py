# Rebuilds the singer's take note by note: each take note is time-stretched
# to match how long the corresponding reference note should take (from the
# DTW alignment), then pitch-shifted toward the reference note's pitch by
# an amount controlled by "snap strength" (0% = untouched, 100% = exactly
# on pitch). The reshaped notes are stitched back together with short
# crossfades so the joins do not click.

from __future__ import annotations

import numpy as np

from app.align import AlignmentResult
from app.notes import Note

try:
    import pyrubberband as pyrb

    _HAS_RUBBERBAND = True
except Exception:  # pyrubberband raises OSError if the rubberband CLI is missing
    _HAS_RUBBERBAND = False

import librosa

FADE_SECONDS = 0.015
MIN_STRETCH_RATIO = 0.7
MAX_STRETCH_RATIO = 1.4


def _pitch_shift(samples: np.ndarray, sample_rate: int, semitones: float) -> np.ndarray:
    if abs(semitones) < 1e-3 or len(samples) == 0:
        return samples
    if _HAS_RUBBERBAND:
        try:
            return pyrb.pitch_shift(samples, sample_rate, semitones).astype(np.float32)
        except Exception:
            pass  # fall through to librosa if the rubberband CLI misbehaves at runtime
    return librosa.effects.pitch_shift(
        samples, sr=sample_rate, n_steps=semitones
    ).astype(np.float32)


def _time_stretch(samples: np.ndarray, ratio: float) -> np.ndarray:
    """ratio = desired_output_len / input_len."""
    if abs(ratio - 1.0) < 1e-3 or len(samples) == 0:
        return samples
    rate = 1.0 / ratio  # librosa's rate is input_len / output_len
    return librosa.effects.time_stretch(samples, rate=rate).astype(np.float32)


def _closest_ref_note(take_note: Note, ref_notes: list[Note], alignment: AlignmentResult) -> Note | None:
    if not ref_notes:
        return None
    mid_take = (take_note.start + take_note.end) / 2
    mapped = alignment.take_to_ref(mid_take)

    for note in ref_notes:
        if note.start <= mapped <= note.end:
            return note

    return min(ref_notes, key=lambda n: abs((n.start + n.end) / 2 - mapped))


def _crossfade_append(output: list[np.ndarray], chunk: np.ndarray, sample_rate: int) -> None:
    fade_len = min(int(FADE_SECONDS * sample_rate), len(chunk) // 2)
    if not output or fade_len <= 0:
        output.append(chunk)
        return

    prev = output[-1]
    if len(prev) < fade_len:
        output.append(chunk)
        return

    fade_out = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
    fade_in = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)

    overlap = prev[-fade_len:] * fade_out + chunk[:fade_len] * fade_in
    output[-1] = np.concatenate([prev[:-fade_len], overlap])
    output.append(chunk[fade_len:])


def correct_take(
    take_samples: np.ndarray,
    take_sample_rate: int,
    take_notes: list[Note],
    ref_notes: list[Note],
    alignment: AlignmentResult,
    snap_strength: float,
) -> np.ndarray:
    """snap_strength is 0-100."""
    strength = max(0.0, min(100.0, snap_strength)) / 100.0
    output: list[np.ndarray] = []
    cursor = 0.0

    for note in take_notes:
        start_sample = int(note.start * take_sample_rate)
        end_sample = int(note.end * take_sample_rate)
        chunk = take_samples[start_sample:end_sample]
        if len(chunk) == 0:
            continue

        # fill any silent gap before this note with the original audio
        gap_start_sample = int(cursor * take_sample_rate)
        if gap_start_sample < start_sample:
            output.append(take_samples[gap_start_sample:start_sample])

        ref_note = _closest_ref_note(note, ref_notes, alignment)

        ref_start = alignment.take_to_ref(note.start)
        ref_end = alignment.take_to_ref(note.end)
        target_duration = max(ref_end - ref_start, 1e-3)
        original_duration = note.end - note.start
        ratio = target_duration / original_duration if original_duration > 0 else 1.0
        ratio = max(MIN_STRETCH_RATIO, min(MAX_STRETCH_RATIO, ratio))

        stretched = _time_stretch(chunk, ratio)

        semitones = 0.0
        if ref_note is not None:
            semitones = (ref_note.midi - note.midi) * strength

        shifted = _pitch_shift(stretched, take_sample_rate, semitones)

        _crossfade_append(output, shifted, take_sample_rate)
        cursor = note.end

    tail_start_sample = int(cursor * take_sample_rate)
    if tail_start_sample < len(take_samples):
        output.append(take_samples[tail_start_sample:])

    if not output:
        return take_samples.copy()

    return np.concatenate(output).astype(np.float32)
