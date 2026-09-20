# Tracks the melody of a mono audio signal over time. We use librosa's pyin
# (probabilistic YIN), which returns a pitch estimate per frame plus a
# confidence flag for whether that frame is actually voiced (singing) or
# silence/noise. We lightly smooth the voiced pitch so single-frame jitter
# does not get mistaken for a real note change later on.

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

FMIN = librosa.note_to_hz("C2")  # ~65 Hz, low end of a singing voice
FMAX = librosa.note_to_hz("C6")  # ~1047 Hz, high end of a singing voice
FRAME_LENGTH = 2048
HOP_LENGTH = 512
SMOOTH_WINDOW = 5  # frames, must be odd


@dataclass
class PitchTrack:
    times: np.ndarray  # seconds, shape (n,)
    midi: np.ndarray  # MIDI note number (float), NaN where unvoiced
    voiced: np.ndarray  # bool, shape (n,)


def _smooth_voiced_runs(midi: np.ndarray, voiced: np.ndarray, window: int) -> np.ndarray:
    """Median-smooth midi pitch within each contiguous voiced run.

    We never smooth across an unvoiced gap, since that would blend two
    unrelated notes together.
    """
    smoothed = midi.copy()
    half = window // 2
    n = len(midi)

    run_start = None
    for i in range(n + 1):
        in_run = i < n and voiced[i]
        if in_run and run_start is None:
            run_start = i
        elif not in_run and run_start is not None:
            run_end = i  # exclusive
            for j in range(run_start, run_end):
                lo = max(run_start, j - half)
                hi = min(run_end, j + half + 1)
                smoothed[j] = float(np.median(midi[lo:hi]))
            run_start = None

    return smoothed


def track_pitch(samples: np.ndarray, sample_rate: int) -> PitchTrack:
    f0, voiced_flag, _voiced_prob = librosa.pyin(
        samples,
        fmin=FMIN,
        fmax=FMAX,
        sr=sample_rate,
        frame_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH,
    )

    times = librosa.times_like(f0, sr=sample_rate, hop_length=HOP_LENGTH)
    voiced = np.asarray(voiced_flag, dtype=bool) & ~np.isnan(f0)

    midi = np.full_like(f0, np.nan, dtype=np.float64)
    midi[voiced] = librosa.hz_to_midi(f0[voiced])

    midi = _smooth_voiced_runs(midi, voiced, SMOOTH_WINDOW)

    return PitchTrack(times=times, midi=midi, voiced=voiced)
