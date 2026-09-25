# Continuous pitch correction, the engine behind the app. Instead of
# chopping the take into notes, we decompose it with the WORLD vocoder into
# three things that can be edited independently: pitch (f0), spectral
# envelope (the voice's timbre and vowel shape) and aperiodicity (breath).
# We then:
#
#   1. warp the take's frames onto the reference's timeline (smoothed DTW),
#   2. compute, for every 5 ms frame, how far the take's pitch is from the
#      reference's, smooth that error so we fix drift and wrong notes but
#      keep the singer's own vibrato and slides, and add it to the take's
#      pitch, and
#   3. resynthesize with the original envelope, so formants are untouched
#      (no chipmunk effect) and there are no note-boundary clicks.

from __future__ import annotations

import librosa
import numpy as np
import pyworld as pw
from scipy.ndimage import median_filter, uniform_filter1d

from app.align import AlignmentResult

FRAME_PERIOD_MS = 5.0
F0_FLOOR = 65.0
F0_CEIL = 1050.0

RETUNE_SMOOTH_MS = 120.0  # window for smoothing the pitch error; lower = tighter, more robotic
TIMING_SMOOTH_SECONDS = 0.2  # smoothing of the DTW path, hides jitter in the warp
REF_MEDIAN_FRAMES = 5  # kills single-frame octave glitches in the reference pitch


def analyze(samples: np.ndarray, sample_rate: int) -> dict:
    x = samples.astype(np.float64)
    f0, t = pw.harvest(
        x, sample_rate, f0_floor=F0_FLOOR, f0_ceil=F0_CEIL, frame_period=FRAME_PERIOD_MS
    )
    f0 = pw.stonemask(x, f0, t, sample_rate)
    sp = pw.cheaptrick(x, f0, t, sample_rate)
    ap = pw.d4c(x, f0, t, sample_rate)
    return {"f0": f0, "t": t, "sp": sp, "ap": ap}


def _f0_to_midi(f0: np.ndarray) -> np.ndarray:
    midi = np.full(len(f0), np.nan)
    voiced = f0 > 0
    midi[voiced] = librosa.hz_to_midi(f0[voiced])
    return midi


def _fill_nan_nearest(values: np.ndarray) -> np.ndarray:
    ok = ~np.isnan(values)
    if not ok.any():
        return np.zeros_like(values)
    idx = np.arange(len(values))
    return np.interp(idx, idx[ok], values[ok])


def _runs(mask: np.ndarray):
    """Yield (start, end) of each contiguous True run in mask (end exclusive)."""
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    for start, end in zip(edges[::2], edges[1::2]):
        yield int(start), int(end)


def _ref_to_take_times(alignment: AlignmentResult, ref_times: np.ndarray, strength: float) -> np.ndarray:
    """For each reference frame time, the take time that should play there."""
    # DTW plateaus repeat ref times; keep the mean take time for each one
    uniq_ref, inverse = np.unique(alignment.ref_times, return_inverse=True)
    take_for_ref = np.bincount(inverse, weights=alignment.take_times) / np.bincount(inverse)

    mapped = np.interp(ref_times, uniq_ref, take_for_ref)
    win = max(1, int(TIMING_SMOOTH_SECONDS * 1000 / FRAME_PERIOD_MS))
    mapped = uniform_filter1d(mapped, win, mode="nearest")
    mapped = np.maximum.accumulate(mapped)

    return (1.0 - strength) * ref_times + strength * mapped


def _interp_frames(matrix: np.ndarray, pos: np.ndarray, log: bool) -> np.ndarray:
    n = len(matrix)
    pos = np.clip(pos, 0, n - 1)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, n - 1)
    w = (pos - lo)[:, None]
    if log:
        m = np.log(np.maximum(matrix, 1e-16))
        return np.exp((1 - w) * m[lo] + w * m[hi])
    return (1 - w) * matrix[lo] + w * matrix[hi]


def retune(
    take_samples: np.ndarray,
    ref_samples: np.ndarray,
    sample_rate: int,
    alignment: AlignmentResult,
    snap_strength: float,
    timing_strength: float = 1.0,
    take_analysis: dict | None = None,
    ref_analysis: dict | None = None,
) -> np.ndarray:
    """Return the take re-pitched (and re-timed) onto the reference.

    snap_strength is 0-100: how much of the pitch error is corrected.
    timing_strength is 0-1: 0 keeps the take's own timing, 1 follows the reference.
    """
    strength = max(0.0, min(100.0, snap_strength)) / 100.0
    take = take_analysis or analyze(take_samples, sample_rate)
    ref = ref_analysis or analyze(ref_samples, sample_rate)

    # output frames live on the reference timeline
    n_out = len(ref["f0"])
    out_times = ref["t"]
    take_times = _ref_to_take_times(alignment, out_times, timing_strength)
    pos = take_times / (FRAME_PERIOD_MS / 1000.0)

    take_midi = _f0_to_midi(take["f0"])
    take_voiced = ~np.isnan(take_midi)
    take_midi_filled = _fill_nan_nearest(take_midi)

    pos_clipped = np.clip(pos, 0, len(take_midi) - 1)
    midi_out = np.interp(pos_clipped, np.arange(len(take_midi)), take_midi_filled)
    voiced_out = take_voiced[np.round(pos_clipped).astype(int)]

    ref_midi = _f0_to_midi(ref["f0"])
    ref_voiced = ~np.isnan(ref_midi)
    ref_midi = np.where(ref_voiced, ref_midi, 0.0)
    ref_midi = median_filter(ref_midi, size=REF_MEDIAN_FRAMES, mode="nearest")
    ref_midi = np.where(ref_voiced, ref_midi, np.nan)

    # pitch error, folded to the nearest octave so a singer in a different
    # octave than the reference is pulled to the nearest matching note
    err = ref_midi - midi_out
    err = (err + 6.0) % 12.0 - 6.0
    err[~voiced_out] = np.nan

    smooth = max(1, int(RETUNE_SMOOTH_MS / FRAME_PERIOD_MS))
    correction = np.zeros(n_out)
    for start, end in _runs(voiced_out):
        seg = err[start:end]
        ok = ~np.isnan(seg)
        if not ok.any():
            continue  # nothing to follow here, leave it alone
        idx = np.arange(len(seg))
        seg = np.interp(idx, idx[ok], seg[ok])
        correction[start:end] = uniform_filter1d(seg, smooth, mode="nearest")

    f0_out = np.zeros(n_out)
    f0_out[voiced_out] = librosa.midi_to_hz(midi_out[voiced_out] + strength * correction[voiced_out])

    sp = _interp_frames(take["sp"], pos, log=True)
    ap = _interp_frames(take["ap"], pos, log=False)
    ap = np.clip(ap, 0.0, 1.0)

    out = pw.synthesize(
        np.ascontiguousarray(f0_out),
        np.ascontiguousarray(sp),
        np.ascontiguousarray(ap),
        sample_rate,
        FRAME_PERIOD_MS,
    )

    peak = float(np.max(np.abs(out))) if len(out) else 0.0
    if peak > 0.99:
        out = out * (0.99 / peak)
    return out.astype(np.float32)
