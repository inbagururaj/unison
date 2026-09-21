# Shared synthetic-audio helpers for tests. We use plain sine waves so
# tests do not depend on any real recording and have a known-correct pitch
# to check against.

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 22050


def make_tone(freq_hz: float, duration_seconds: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    n = int(duration_seconds * sample_rate)
    t = np.arange(n) / sample_rate
    # small fade in/out avoids click artifacts that can confuse pyin at the edges
    fade = min(int(0.02 * sample_rate), n // 4)
    signal = 0.5 * np.sin(2 * np.pi * freq_hz * t).astype(np.float32)
    if fade > 0:
        ramp = np.linspace(0, 1, fade, dtype=np.float32)
        signal[:fade] *= ramp
        signal[-fade:] *= ramp[::-1]
    return signal


def make_melody(freqs_hz: list[float], note_duration_seconds: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    return np.concatenate([make_tone(f, note_duration_seconds, sample_rate) for f in freqs_hz])
