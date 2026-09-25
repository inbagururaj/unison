# Runs the reference-following engines on one reference/take pair and prints
# comparable numbers, and writes each result to samples/engines_<name>.wav
# (plus samples/engines_take.wav) so they can be A/B'd by ear. The numbers
# are proxies; they cannot say which one sounds best.
#
#   cd backend
#   uv run python ../scripts/compare_engines.py [--ref r.wav --take t.wav] [--strength 100]
#
#   pitch error   median cents from the output's pitch to the reference's at the same time
#   timbre drift  mean log-spectral-envelope difference (dB, below 4 kHz, level removed)
#                 between the output and your take, averaged over voiced frames:
#                 how much the voice's formants/timbre moved (0 = untouched)
#   silence       loudness of the output inside the take's own silent gaps,
#                 relative to the take's voiced loudness (dB; more negative = quieter)
#   length        output length vs the reference's

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pyworld as pw
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app import shift  # noqa: E402
from app.align import align  # noqa: E402
from app.notes import segment_notes  # noqa: E402
from app.notes_world import notes_world  # noqa: E402
from app.pitch import track_pitch  # noqa: E402
from app.retune import retune  # noqa: E402


def envelope(y, sr):
    x = y.astype(np.float64)
    f0, t = pw.harvest(x, sr, f0_floor=65, f0_ceil=1050)
    return np.log(pw.cheaptrick(x, f0, t, sr)[f0 > 0].mean(axis=0) + 1e-12)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ref", default=ROOT / "samples/voice_reference.wav")
    p.add_argument("--take", default=ROOT / "samples/voice_take.wav")
    p.add_argument("--strength", type=float, default=100.0)
    a = p.parse_args()
    ref, sr = sf.read(str(a.ref), dtype="float32")
    take, _ = sf.read(str(a.take), dtype="float32")

    ref_track, take_track = track_pitch(ref, sr), track_pitch(take, sr)
    alignment = align(ref, sr, take, sr)
    ref_notes, take_notes = segment_notes(ref_track), segment_notes(take_track)

    runs = {
        "notes_legacy": lambda: shift.correct_take(take, sr, take_notes, ref_notes, alignment, a.strength, shift.LEGACY_CONFIG),
        "notes": lambda: shift.correct_take(take, sr, take_notes, ref_notes, alignment, a.strength, shift.DEFAULT_CONFIG),
        "notes_world": lambda: notes_world(take, sr, len(ref) / sr, take_notes, ref_notes, alignment, a.strength),
        "retune": lambda: retune(take, ref, sr, alignment, a.strength),
    }

    take_env = envelope(take, sr)
    band = np.linspace(0, sr / 2, len(take_env)) < 4000
    # the take's own silent gaps, in take time: where it is quiet for at least 100 ms
    hop = int(0.01 * sr)
    rms = np.sqrt(np.convolve(take**2, np.ones(hop) / hop, "same"))[::hop]
    quiet = rms < 0.03 * rms.max()
    voiced_level = float(np.sqrt(np.mean(take[np.repeat(~quiet, hop)[: len(take)]] ** 2))) if (~quiet).any() else 1.0

    sf.write(str(ROOT / "samples/engines_take.wav"), take, sr)
    print(f"take {len(take) / sr:.2f}s, reference {len(ref) / sr:.2f}s, strength {a.strength:.0f}")
    print(f"{'engine':14s} {'time':>6s} {'pitch err':>10s} {'within 25c':>10s} {'timbre drift':>13s} {'silence':>9s} {'length diff':>12s}")
    for name, run in runs.items():
        t0 = time.perf_counter()
        out = run()
        el = time.perf_counter() - t0
        sf.write(str(ROOT / f"samples/engines_{name}.wav"), out, sr)

        tr = track_pitch(out, sr)
        idx = np.round(np.interp(tr.times, ref_track.times, np.arange(len(ref_track.times)))).astype(int)
        both = tr.voiced & ref_track.voiced[idx]
        cents = np.abs(tr.midi[both] - ref_track.midi[idx[both]]) * 100
        d = envelope(out, sr) - take_env
        drift = float(np.abs((d - d[band].mean())[band]).mean() * 20 / np.log(10))

        # output inside the take's silent gaps, mapped to output time by the alignment
        gap_times = np.arange(len(quiet))[quiet] * 0.01
        # (approximate for the notes engines: their timeline only roughly follows the alignment)
        out_times = np.array([alignment.take_to_ref(t) for t in gap_times])
        idxs = np.clip((out_times * sr).astype(int), 0, len(out) - 1)
        leak = float(np.sqrt(np.mean(out[idxs] ** 2))) if len(idxs) else 0.0
        sil = 20 * np.log10(leak / voiced_level + 1e-9)

        print(
            f"{name:14s} {el:5.1f}s {np.median(cents):8.0f} c {100 * (cents < 25).mean():9.0f}% "
            f"{drift:11.2f} dB {sil:7.0f} dB {abs(len(out) - len(ref)) / sr * 1000:9.0f} ms"
        )
    print("wrote samples/engines_*.wav (listen to these)")


if __name__ == "__main__":
    main()
