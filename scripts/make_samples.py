# Generates the synthetic demo samples in samples/. We never ship real
# copyrighted audio - the reference is a sine-wave melody and the take is
# an intentionally off-pitch, slightly mistimed version of the same
# melody, so the app has something legal to demo end to end.

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 22050
OUT_DIR = Path(__file__).resolve().parent.parent / "samples"

# a short melody: C4 D4 E4 F4 G4, each held for one second
MELODY_MIDI = [60, 62, 64, 65, 67]
NOTE_SECONDS = 1.0


def midi_to_hz(midi: float) -> float:
    return 440.0 * 2 ** ((midi - 69) / 12)


def tone(freq_hz: float, duration: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    n = int(duration * sample_rate)
    t = np.arange(n) / sample_rate
    fade = min(int(0.02 * sample_rate), n // 4)
    signal = 0.4 * np.sin(2 * np.pi * freq_hz * t).astype(np.float32)
    if fade > 0:
        ramp = np.linspace(0, 1, fade, dtype=np.float32)
        signal[:fade] *= ramp
        signal[-fade:] *= ramp[::-1]
    return signal


def make_reference() -> np.ndarray:
    return np.concatenate([tone(midi_to_hz(m), NOTE_SECONDS) for m in MELODY_MIDI])


def make_take() -> np.ndarray:
    # off-pitch (flat by a semitone to a semitone and a half) and each
    # note held slightly longer, so alignment and pitch shift both have
    # something real to correct
    rng = np.random.default_rng(7)
    chunks = []
    for m in MELODY_MIDI:
        detune = -1.0 - rng.uniform(0, 0.5)
        chunks.append(tone(midi_to_hz(m + detune), NOTE_SECONDS * 1.15))
    return np.concatenate(chunks)


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    sf.write(str(OUT_DIR / "reference.wav"), make_reference(), SAMPLE_RATE)
    sf.write(str(OUT_DIR / "take.wav"), make_take(), SAMPLE_RATE)
    print(f"wrote {OUT_DIR / 'reference.wav'}")
    print(f"wrote {OUT_DIR / 'take.wav'}")


if __name__ == "__main__":
    main()
