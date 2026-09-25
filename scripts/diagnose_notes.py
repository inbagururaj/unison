# Diagnoses why the "notes" engine can sound choppy, and re-checks the
# numbers after each fix stage (shift.STAGES). For a reference/take pair it
# prints, per stage: segment counts, stretch-ratio behaviour, clamp hits,
# pitch-shift jumps between neighbouring notes, crossfade length, and - from
# the actual output audio - loudness dips/bumps and click size at the joins,
# pitch error against the reference, and length/timing.
#
# These are proxies for "choppy"; none of them is a listening test.
#
#   cd backend
#   uv run python ../scripts/diagnose_notes.py [--ref r.wav --take t.wav] [--stage 0,3]

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app import shift  # noqa: E402
from app.align import align  # noqa: E402
from app.notes import segment_notes  # noqa: E402
from app.pitch import track_pitch  # noqa: E402


def _rms(x):
    return float(np.sqrt(np.mean(x**2))) + 1e-9


def join_metrics(out: np.ndarray, joins: list[int], sr: int):
    """Loudness change (dB) and click size at each join between touching notes."""
    ms = lambda t: int(t * sr / 1000)  # noqa: E731
    dips, clicks = [], []
    for j in joins:
        if j - ms(60) < 0 or j + ms(60) >= len(out):
            continue
        centre = _rms(out[j - ms(8) : j + ms(8)])
        around = 0.5 * (_rms(out[j - ms(55) : j - ms(35)]) + _rms(out[j + ms(35) : j + ms(55)]))
        dips.append(20 * np.log10(centre / around))
        d = np.abs(np.diff(out[j - ms(30) : j + ms(30)]))
        near = d[ms(30) - ms(1) : ms(30) + ms(1)].max()
        clicks.append(near / (np.median(d) + 1e-9))
    return np.array(dips), np.array(clicks)


def pitch_step_at_joins(y, joins, sr):
    """Largest pitch change (semitones) within +-15 ms of each join, from a 5 ms pitch track."""
    import pyworld as pw

    x = y.astype(np.float64)
    f0, t = pw.harvest(x, sr, f0_floor=65, f0_ceil=1050, frame_period=5.0)
    f0 = pw.stonemask(x, f0, t, sr)
    midi = np.where(f0 > 0, 69 + 12 * np.log2(np.maximum(f0, 1e-9) / 440), np.nan)
    steps = []
    for j in joins:
        w = midi[max(0, int((j / sr - 0.015) / 0.005)) : int((j / sr + 0.015) / 0.005) + 1]
        w = w[~np.isnan(w)]
        steps.append(np.ptp(w) if len(w) > 1 else np.nan)
    return np.array(steps)


def run_stage(name, cfg, ref, take, sr, ref_track, take_track, ref_notes, take_notes, alignment):
    out, info = shift.correct_take_debug(take, sr, take_notes, ref_notes, alignment, 100, cfg)
    notes = info["notes"]
    dur = np.array([n.end - n.start for n in notes])
    used = np.array(info["ratios_used"])
    semis = np.array(info["semitones"])
    adj = np.array(info["adjacent"], dtype=bool)

    adj_ratio = np.abs(np.diff(used))
    adj_ratio_max = adj_ratio.max() if len(adj_ratio) else 0.0
    step = np.abs(np.diff(semis)) if len(semis) > 1 else np.array([0.0])
    step = step[adj] if adj.any() and len(step) == len(adj) else step
    if cfg.glide_ms > 0:
        step = step * min(1.0, 20.0 / cfg.glide_ms)  # change inside any 20 ms window
    jump = step.max() if len(step) else 0.0

    # the same measurements at the same musical spots in the original take, so
    # a dip that is already in the source (a breath, a fade) is not blamed on the processing
    take_joins = [
        int((notes[i].end + notes[i + 1].start) / 2 * sr) for i in range(len(notes) - 1) if adj[i]
    ]
    dips, clicks = join_metrics(out, info["joins"], sr)
    dips0, clicks0 = join_metrics(take, take_joins, sr)
    if len(dips) == len(dips0) and len(dips):
        extra = dips - dips0
    else:
        extra = np.array([])
    steps, steps0 = pitch_step_at_joins(out, info["joins"], sr), pitch_step_at_joins(take, take_joins, sr)
    out_track = track_pitch(out, sr)
    idx = np.round(np.interp(out_track.times, ref_track.times, np.arange(len(ref_track.times)))).astype(int)
    both = out_track.voiced & ref_track.voiced[idx]
    cents = np.abs(out_track.midi[both] - ref_track.midi[idx[both]]) * 100 if both.any() else np.array([np.nan])

    print(f"-- {name}")
    print(f"   segments {len(notes)} (raw {len(take_notes)}), <150 ms: {(dur < 0.15).sum()}, min {dur.min() * 1000:.0f} ms")
    print(
        f"   stretch ratio used: {used.min():.2f} / {np.median(used):.2f} / {used.max():.2f} (min/med/max), "
        f"clamp hits {info['clamp_hits']}, max change between neighbours {adj_ratio_max:.2f}"
    )
    print(f"   pitch-shift step between touching notes (max, per 20 ms): {jump:.2f} st")
    print(f"   crossfade {info['crossfade_ms']:.0f} ms, {len(info['joins'])} touching joins")
    if len(extra):
        print(
            f"   at joins, beyond what the take itself has: loudness {extra.min():+.1f}..{extra.max():+.1f} dB "
            f"(worst {np.abs(extra).max():.1f}), click ratio worst {clicks.max():.1f}x vs {clicks0.max():.1f}x in the take"
        )
    if len(steps) == len(steps0) and len(steps):
        extra_step = steps - steps0
        print(
            f"   pitch step at joins (within 30 ms): output max {np.nanmax(steps):.2f} st vs take {np.nanmax(steps0):.2f} st; "
            f"extra added by processing: max {np.nanmax(extra_step):+.2f}, mean {np.nanmean(extra_step):+.2f} st"
        )
    print(
        f"   pitch error vs reference: median {np.nanmedian(cents):.0f} c;  "
        f"length {len(out) / sr:.2f}s vs reference {len(ref) / sr:.2f}s (diff {abs(len(out) - len(ref)) / sr * 1000:.0f} ms)"
    )
    return out


def diagnose(ref, take, sr, label, stages, save_prefix=None):
    ref_track, take_track = track_pitch(ref, sr), track_pitch(take, sr)
    alignment = align(ref, sr, take, sr)
    ref_notes, take_notes = segment_notes(ref_track), segment_notes(take_track)
    print(f"== {label}  (take {len(take) / sr:.2f}s, reference {len(ref) / sr:.2f}s)")
    for name, cfg in shift.STAGES.items():
        if stages and name.split()[0] not in stages:
            continue
        out = run_stage(name, cfg, ref, take, sr, ref_track, take_track, ref_notes, take_notes, alignment)
        if save_prefix:
            sf.write(f"{save_prefix}_{name.split()[0]}.wav", out, sr)
    print()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ref")
    p.add_argument("--take")
    p.add_argument("--stage", help="comma-separated stage numbers, e.g. 0,5 (default: all)")
    p.add_argument("--save", action="store_true", help="write samples/notes_stage_N.wav for listening")
    a = p.parse_args()
    stages = set(a.stage.split(",")) if a.stage else None
    pairs = [(a.ref, a.take, "given pair")] if a.ref and a.take else [
        (ROOT / "samples/voice_reference.wav", ROOT / "samples/voice_take.wav", "voice-like pair"),
        (ROOT / "samples/reference.wav", ROOT / "samples/take.wav", "sine pair"),
    ]
    for r, t, label in pairs:
        ref, sr = sf.read(str(r), dtype="float32")
        take, _ = sf.read(str(t), dtype="float32")
        diagnose(ref, take, sr, label, stages, str(ROOT / "samples/notes_stage") if a.save else None)


if __name__ == "__main__":
    main()
