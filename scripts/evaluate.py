# Runs the correction pipeline on a reference/take pair and prints how close
# the result is to the reference, so tuning changes can be compared with
# numbers instead of by ear.
#
#   cd backend && uv run python ../scripts/evaluate.py [reference.wav take.wav] [snap 0-100]
#
# Metric: pitch error in cents against the reference, on frames where both
# are voiced, measured with an independent tracker (pyin, not the WORLD
# tracker used inside the corrector). "before" maps the take onto the
# reference timeline with the DTW alignment first.

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app.align import align  # noqa: E402
from app.pitch import track_pitch  # noqa: E402
from app.retune import retune  # noqa: E402


def _errors_cents(ref_track, other_times, other_midi, other_voiced) -> np.ndarray:
    ref_idx = np.round(np.interp(other_times, ref_track.times, np.arange(len(ref_track.times)))).astype(int)
    both = other_voiced & ref_track.voiced[ref_idx]
    return np.abs(other_midi[both] - ref_track.midi[ref_idx[both]]) * 100.0


def _report(label: str, cents: np.ndarray) -> None:
    if len(cents) == 0:
        print(f"{label:8s} no overlapping voiced frames")
        return
    print(
        f"{label:8s} median {np.median(cents):6.1f}c  mean {cents.mean():6.1f}c  "
        f"within 25c {100 * (cents < 25).mean():5.1f}%  within 50c {100 * (cents < 50).mean():5.1f}%  "
        f"({len(cents)} frames)"
    )


def main() -> None:
    ref_path = sys.argv[1] if len(sys.argv) > 2 else ROOT / "samples" / "voice_reference.wav"
    take_path = sys.argv[2] if len(sys.argv) > 2 else ROOT / "samples" / "voice_take.wav"
    snap = float(sys.argv[3]) if len(sys.argv) > 3 else 100.0

    ref, sr = sf.read(str(ref_path), dtype="float32")
    take, sr_t = sf.read(str(take_path), dtype="float32")
    assert sr == sr_t, "reference and take must share a sample rate"

    t0 = time.perf_counter()
    alignment = align(ref, sr, take, sr)
    out = retune(take, ref, sr, alignment, snap)
    elapsed = time.perf_counter() - t0

    ref_track = track_pitch(ref, sr)
    take_track = track_pitch(take, sr)
    out_track = track_pitch(out, sr)

    # before: take frames, placed on the reference timeline by the alignment
    take_on_ref_times = np.array([alignment.take_to_ref(t) for t in take_track.times])
    before = _errors_cents(ref_track, take_on_ref_times, take_track.midi, take_track.voiced)
    after = _errors_cents(ref_track, out_track.times, out_track.midi, out_track.voiced)

    print(f"snap {snap:.0f}  process {elapsed:.1f}s  ref {len(ref)/sr:.2f}s  take {len(take)/sr:.2f}s  out {len(out)/sr:.2f}s")
    _report("before", before)
    _report("after", after)

    out_path = ROOT / "samples" / "eval_output.wav"
    sf.write(str(out_path), out, sr)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
