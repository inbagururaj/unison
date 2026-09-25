# Generates a more voice-like synthetic pair than make_samples.py: a
# harmonic source (glottal-ish, 1/h rolloff) filtered through vowel formants,
# with vibrato, glides between notes, breath noise and rests. The reference
# sings a melody in tune; the take sings the same melody with per-note
# detuning, slow pitch drift, different vibrato and looser timing.
# No real recordings, so nothing to license.

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import lfilter

SAMPLE_RATE = 22050
OUT_DIR = Path(__file__).resolve().parent.parent / "samples"

# (midi, seconds); None midi = rest
MELODY = [(60, 0.6), (62, 0.6), (64, 0.9), (None, 0.3), (65, 0.6), (67, 0.6), (64, 0.6), (60, 1.0)]
FORMANTS = [(700, 90), (1200, 110), (2600, 160)]  # an "ah" vowel: (freq, bandwidth)


def midi_to_hz(m):
    return 440.0 * 2 ** ((np.asarray(m, dtype=float) - 69) / 12)


def _pitch_curve(notes, glide=0.06, vib_hz=5.5, vib_cents=35, seed=0, detune=None, drift=0.0):
    rng = np.random.default_rng(seed)
    parts = []
    last = notes[0][0]
    for i, (m, dur) in enumerate(notes):
        n = int(dur * SAMPLE_RATE)
        if m is None:
            parts.append(np.full(n, np.nan))
            continue
        target = m + (detune[i] if detune is not None else 0.0)
        curve = np.full(n, float(target))
        g = min(int(glide * SAMPLE_RATE), n // 2)
        prev = last if last is not None else target
        curve[:g] = np.linspace(prev, target, g)
        parts.append(curve)
        last = target
    midi = np.concatenate(parts)
    t = np.arange(len(midi)) / SAMPLE_RATE
    phase = rng.uniform(0, 2 * np.pi)
    vibrato = (vib_cents / 100.0) * np.sin(2 * np.pi * vib_hz * t + phase) * np.clip(t % 1.0 * 3, 0, 1)
    walk = np.cumsum(rng.normal(0, 1, len(t))) / np.sqrt(len(t))
    return midi + vibrato + drift * walk


def _render(midi_curve, seed=0):
    rng = np.random.default_rng(seed)
    voiced = ~np.isnan(midi_curve)
    f0 = np.where(voiced, midi_to_hz(np.nan_to_num(midi_curve, nan=60.0)), 0.0)
    phase = np.cumsum(2 * np.pi * f0 / SAMPLE_RATE)
    source = np.zeros(len(f0))
    for h in range(1, 40):
        freq_ok = (f0 * h) < SAMPLE_RATE / 2 * 0.95
        source += np.where(freq_ok, np.sin(h * phase) / h, 0.0)
    source *= voiced

    out = source
    for freq, bw in FORMANTS:
        r = np.exp(-np.pi * bw / SAMPLE_RATE)
        theta = 2 * np.pi * freq / SAMPLE_RATE
        out = lfilter([1 - r], [1, -2 * r * np.cos(theta), r * r], out)

    env = np.convolve(voiced.astype(float), np.hanning(int(0.03 * SAMPLE_RATE)), "same")
    env /= max(env.max(), 1e-9)
    breath = rng.normal(0, 1, len(out)) * 0.03 * env
    sig = out / max(np.abs(out).max(), 1e-9) * 0.6 * env + breath
    return sig.astype(np.float32)


def make_reference():
    return _render(_pitch_curve(MELODY, seed=1), seed=1)


def make_take():
    rng = np.random.default_rng(11)
    detune = rng.uniform(-1.3, 1.0, len(MELODY))  # some notes badly off, some okay
    stretched = [(m, d * rng.uniform(0.9, 1.25)) for m, d in MELODY]
    return _render(
        _pitch_curve(stretched, seed=3, detune=detune, vib_hz=6.3, vib_cents=60, drift=0.5),
        seed=3,
    )


def main():
    OUT_DIR.mkdir(exist_ok=True)
    sf.write(str(OUT_DIR / "voice_reference.wav"), make_reference(), SAMPLE_RATE)
    sf.write(str(OUT_DIR / "voice_take.wav"), make_take(), SAMPLE_RATE)
    print("wrote samples/voice_reference.wav and samples/voice_take.wav")


if __name__ == "__main__":
    main()
