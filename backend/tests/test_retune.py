# Tests for the continuous retuner, using a harmonic tone through a vowel
# formant filter so WORLD has something voice-like to analyze.

from __future__ import annotations

import numpy as np

from app.align import align
from app.pitch import track_pitch
from app.retune import retune

SR = 22050


def _voice(midi: float, seconds: float, vibrato_cents: float = 0.0) -> np.ndarray:
    n = int(seconds * SR)
    t = np.arange(n) / SR
    f0 = 440.0 * 2 ** ((midi - 69) / 12) * 2 ** (vibrato_cents / 1200 * np.sin(2 * np.pi * 5.5 * t))
    phase = np.cumsum(2 * np.pi * f0 / SR)
    sig = sum(np.sin(h * phase) / h for h in range(1, 20))
    fade = np.minimum(1, np.minimum(t, seconds - t) / 0.03)
    return (0.4 * sig / np.abs(sig).max() * fade).astype(np.float32)


def _median_midi(samples: np.ndarray) -> float:
    track = track_pitch(samples, SR)
    return float(np.nanmedian(track.midi[track.voiced]))


def test_flat_take_is_pulled_to_reference_pitch():
    ref = _voice(64.0, 1.5)
    take = _voice(62.7, 1.5)  # 1.3 semitones flat
    out = retune(take, ref, SR, align(ref, SR, take, SR), 100)
    assert abs(_median_midi(out) - 64.0) < 0.3


def test_snap_zero_leaves_pitch_alone():
    ref = _voice(64.0, 1.5)
    take = _voice(62.7, 1.5)
    out = retune(take, ref, SR, align(ref, SR, take, SR), 0)
    assert abs(_median_midi(out) - 62.7) < 0.3


def test_half_snap_lands_halfway():
    ref = _voice(64.0, 1.5)
    take = _voice(62.0, 1.5)
    out = retune(take, ref, SR, align(ref, SR, take, SR), 50)
    assert abs(_median_midi(out) - 63.0) < 0.3


def test_singer_an_octave_low_snaps_to_the_octave_not_the_reference_register():
    ref = _voice(64.0, 1.5)
    take = _voice(52.6, 1.5)  # an octave below, 0.6 semitone flat of E3
    out = retune(take, ref, SR, align(ref, SR, take, SR), 100)
    assert abs(_median_midi(out) - 52.0) < 0.3


def test_output_follows_reference_length():
    ref = _voice(64.0, 1.5)
    take = _voice(64.0, 2.0)
    out = retune(take, ref, SR, align(ref, SR, take, SR), 100)
    assert abs(len(out) / SR - 1.5) < 0.1
