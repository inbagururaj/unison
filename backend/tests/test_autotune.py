# Tests for the key/scale autotune engine: key detection, snapping to the
# scale, timing preservation, retune speed, and formant preservation. Uses a
# harmonic source through vowel formant filters so WORLD has something
# voice-like to analyze.

from __future__ import annotations

import librosa
import numpy as np
import pyworld as pw
from scipy.signal import lfilter

from app.autotune import autotune, detect_key, scale_grid
from app.pitch import track_pitch

SR = 22050
FORMANTS = [(700, 90), (1200, 110), (2600, 160)]


def _voice(midis, seconds_each=0.6, vibrato_cents=0.0, rest=0.0) -> np.ndarray:
    parts = []
    for m in midis:
        n = int(seconds_each * SR)
        t = np.arange(n) / SR
        f0 = 440.0 * 2 ** ((m - 69) / 12) * 2 ** (vibrato_cents / 1200 * np.sin(2 * np.pi * 5.5 * t))
        phase = np.cumsum(2 * np.pi * f0 / SR)
        sig = sum(np.sin(h * phase) / h for h in range(1, 30) if h * f0.max() < SR / 2)
        fade = np.minimum(1, np.minimum(t, seconds_each - t) / 0.03)
        parts.append(sig * fade)
        parts.append(np.zeros(int(rest * SR)))
    x = np.concatenate(parts)
    for freq, bw in FORMANTS:
        r = np.exp(-np.pi * bw / SR)
        x = lfilter([1 - r], [1, -2 * r * np.cos(2 * np.pi * freq / SR), r * r], x)
    return (0.5 * x / np.abs(x).max()).astype(np.float32)


def _voiced_midi(samples: np.ndarray) -> np.ndarray:
    track = track_pitch(samples, SR)
    return track.midi[track.voiced]


def _pitch_classes(tonic: int, scale: str) -> set[int]:
    return {int(round(m)) % 12 for m in scale_grid(tonic, scale)}


def test_detects_key_of_a_g_major_melody():
    # G A B C D E F# G, weighted toward G and D
    melody = [67, 69, 71, 72, 74, 76, 78, 79, 74, 67, 71, 67]
    key = detect_key(track_pitch(_voice(melody, 0.4), SR))
    # relative minor has the same notes, which is all snapping cares about
    assert _pitch_classes(key.tonic, key.scale) == _pitch_classes(7, "major")
    assert abs(key.tuning_cents) < 15


def test_detects_tuning_offset():
    melody = np.array([60, 62, 64, 65, 67, 64, 60]) + 0.3  # 30 cents sharp
    key = detect_key(track_pitch(_voice(melody, 0.4), SR))
    assert abs(key.tuning_cents - 30) < 10


def test_off_key_note_snaps_to_nearest_scale_note():
    take = _voice([63.7])  # between D#/E, nearest C-major note is E (64)
    out = autotune(take, SR, scale_grid(0, "major"), 100, retune_speed_ms=60)
    assert abs(np.median(_voiced_midi(out)) - 64.0) < 0.15


def test_strength_scales_the_correction():
    take = _voice([63.4])
    out = autotune(take, SR, scale_grid(0, "major"), 50, retune_speed_ms=60)
    assert abs(np.median(_voiced_midi(out)) - 63.7) < 0.15


def test_strength_zero_returns_the_take_unchanged():
    take = _voice([63.4])
    out = autotune(take, SR, scale_grid(0, "major"), 0)
    assert np.array_equal(out, take)


def test_timing_is_preserved_exactly():
    take = _voice([60.4, 62.6, 64.3], seconds_each=0.5, rest=0.25)
    out = autotune(take, SR, scale_grid(0, "major"), 100)
    assert len(out) == len(take)

    # sound starts and stops in the same places (10 ms frames, active = within 30 dB of peak)
    def active(x):
        rms = librosa.feature.rms(y=x, frame_length=440, hop_length=220)[0]
        return 20 * np.log10(rms / rms.max() + 1e-9) > -30

    before, after = active(take), active(out)
    assert np.mean(before == after) > 0.97
    edges_before = np.flatnonzero(np.diff(before.astype(int)))
    edges_after = np.flatnonzero(np.diff(after.astype(int)))
    assert len(edges_before) == len(edges_after)
    assert np.abs(edges_before - edges_after).max() <= 2  # within 20 ms


def test_slow_retune_keeps_vibrato_fast_retune_flattens_it():
    take = _voice([64.4], seconds_each=1.5, vibrato_cents=50)
    grid = scale_grid(0, "major")
    slow = _voiced_midi(autotune(take, SR, grid, 100, retune_speed_ms=200))
    fast = _voiced_midi(autotune(take, SR, grid, 100, retune_speed_ms=0))
    original_spread = np.std(_voiced_midi(take))
    assert np.std(slow) > 0.7 * original_spread
    assert np.std(fast) < 0.3 * original_spread
    # either way the note's centre lands on E
    assert abs(np.median(slow) - 64.0) < 0.15
    assert abs(np.median(fast) - 64.0) < 0.15


def test_formants_stay_put_where_a_plain_pitch_shift_moves_them():
    take = _voice([48.0], seconds_each=1.2)  # low voice, so harmonics sample the envelope densely

    def envelope(x):
        x = x.astype(np.float64)
        f0, t = pw.harvest(x, SR, f0_floor=65, f0_ceil=1050)
        sp = pw.cheaptrick(x, f0, t, SR)
        return np.log(sp[f0 > 0].mean(axis=0) + 1e-12)

    def distance(x):
        # mean log-spectral difference below 4 kHz, ignoring overall level
        band = np.linspace(0, SR / 2, len(ref_env)) < 4000
        d = envelope(x) - ref_env
        d = d - d[band].mean()
        return float(np.abs(d[band]).mean())

    ref_env = envelope(take)
    # a one-note grid forces a full one-semitone shift
    tuned = autotune(take, SR, np.array([49.0]), 100, retune_speed_ms=0)
    assert abs(np.median(_voiced_midi(tuned)) - 49.0) < 0.15
    naive = librosa.effects.pitch_shift(take, sr=SR, n_steps=1.0)

    assert distance(tuned) < 0.6 * distance(naive)
