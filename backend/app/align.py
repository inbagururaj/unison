# Time-aligns the singer's take to the reference using dynamic time warping
# (DTW).
#
# In plain words: the singer never sings at exactly the reference's tempo -
# they rush some words and drag others. DTW finds, for every moment in the
# take, the best-matching moment in the reference, while requiring the
# matches to stay in order (you cannot match take-second-5 to a reference
# moment earlier than the one you matched take-second-4 to). It does this
# by building a cost matrix of "how different do these two moments sound"
# for every (take frame, reference frame) pair, then finding the cheapest
# path through that matrix from the start to the end. The result is a
# warping path: a list of (reference_time, take_time) pairs describing how
# the take's timeline bends to line up with the reference's timeline.
#
# We compare MFCCs (a standard timbre/phoneme-shape feature) rather than
# raw pitch, since DTW works best on features that change smoothly and
# distinctly over time, and MFCCs capture syllable-level structure even
# when the take is off pitch.

from __future__ import annotations

from dataclasses import dataclass, replace

import librosa
import numpy as np
from scipy.spatial.distance import cdist

HOP_LENGTH = 512
N_MFCC = 20


@dataclass(frozen=True)
class AlignConfig:
    slope_limit: bool = False  # bound the local warp ratio to [1/2, 2]
    open_end: bool = False  # the path may end before the take's last frame
    # an early end/late start must match this much better (per step) than using
    # the whole take. Measured on synthetic pairs: matched endings score within
    # 0.7% of the full end, real leftovers (silence, an extra-long hold) 8-19%
    # better, a uniformly slower last phrase (nothing left over) 2.7% better.
    open_end_margin: float = 0.05
    open_end_min_fraction: float = 0.5  # never start after / end before half of the take
    open_end_quiet_db: float = 35.0  # only take material this far below the take's peak may be left out
    # fall back to plain DTW steps if the slope-limited path matches the audio
    # more than this much worse than plain DTW does: that happens when a note
    # is held > 2x the reference's length, which the [1/2, 2] limit cannot
    # follow (measured: a 2.5x fermata 4.1% worse; every other case tied or
    # better, the closest +0.2%). None disables the check.
    fallback_margin: float | None = 0.02


LEGACY_ALIGN_CONFIG = AlignConfig()
DEFAULT_ALIGN_CONFIG = AlignConfig(slope_limit=True, open_end=True)
ALIGN_CONFIGS = {"default": DEFAULT_ALIGN_CONFIG, "legacy": LEGACY_ALIGN_CONFIG}

_SLOPE_STEPS = np.array([[1, 1], [1, 2], [2, 1]])
_SLOPE_WEIGHTS = np.array([1.0, 1.5, 1.5])  # each (1,2)/(2,1) step covers 3 frames vs the diagonal's 2


@dataclass
class AlignmentResult:
    # warping path, oldest to newest, one entry per matched frame pair
    ref_times: np.ndarray
    take_times: np.ndarray

    def take_to_ref(self, take_time: float) -> float:
        """Map a moment in the take's timeline to the matching reference moment."""
        return float(np.interp(take_time, self.take_times, self.ref_times))


def _mfcc(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    mfcc = librosa.feature.mfcc(
        y=samples, sr=sample_rate, n_mfcc=N_MFCC, hop_length=HOP_LENGTH
    )
    # normalize each coefficient so no single one dominates the distance
    mean = mfcc.mean(axis=1, keepdims=True)
    std = mfcc.std(axis=1, keepdims=True) + 1e-8
    return (mfcc - mean) / std


def _cosine_cost(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """What librosa.sequence.dtw(X, Y, metric="cosine") computes internally."""
    return cdist(X.T, Y.T, metric="cosine")


def _loud_frames(samples: np.ndarray, quiet_db: float, n_frames: int) -> np.ndarray:
    """Per MFCC frame: is the take within quiet_db of its loudest frame?"""
    rms = librosa.feature.rms(y=samples, hop_length=HOP_LENGTH)[0][:n_frames]
    rms = np.pad(rms, (0, n_frames - len(rms)))
    peak = float(rms.max()) if len(rms) else 0.0
    if peak <= 0:
        return np.zeros(n_frames, dtype=bool)
    return 20 * np.log10(np.maximum(rms, 1e-12) / peak) > -quiet_db


def _pick_end(D: np.ndarray, config: AlignConfig, last_loud: int) -> int:
    """Take frame to end on (on the reference's last frame): the last one unless an earlier end is clearly cheaper.

    Only frames after last_loud (quiet: silence, noise floor) may be left out.
    """
    n, m = D.shape
    js = np.arange(max(int(m * config.open_end_min_fraction), last_loud), m)
    per_step = D[n - 1, js] / (n + js + 1)
    if not np.isfinite(per_step).any():
        return m - 1
    best = int(np.nanargmin(np.where(np.isfinite(per_step), per_step, np.nan)))
    full = per_step[-1]
    if not np.isfinite(full) or per_step[best] < (1 - config.open_end_margin) * full:
        return int(js[best])
    return m - 1


def _dtw_path(cost: np.ndarray, config: AlignConfig, loud: np.ndarray) -> np.ndarray | None:
    """Slope-limited and/or open-ended path, oldest first, or None if no path fits."""
    steps = {"step_sizes_sigma": _SLOPE_STEPS, "weights_mul": _SLOPE_WEIGHTS} if config.slope_limit else {}
    m = cost.shape[1]
    start, end = 0, m - 1
    if config.open_end:
        # the same rule at both boundaries: leading material is an "end" of the reversed signals
        loud_idx = np.flatnonzero(loud)
        first_loud = int(loud_idx[0]) if len(loud_idx) else 0
        last_loud = int(loud_idx[-1]) if len(loud_idx) else m - 1
        D_rev = librosa.sequence.dtw(C=cost[::-1, ::-1], backtrack=False, **steps)
        start = m - 1 - _pick_end(D_rev, config, m - 1 - first_loud)
        D = librosa.sequence.dtw(C=cost[:, start:], backtrack=False, **steps)
        end = start + _pick_end(D, config, last_loud - start)
    sub = cost[:, start : end + 1]
    D = librosa.sequence.dtw(C=sub, backtrack=False, **steps)
    if not np.isfinite(D[-1, -1]):
        return None  # lengths too different for a [1/2, 2] warp
    _D, wp = librosa.sequence.dtw(C=sub, **steps)
    wp = wp[::-1].copy()
    wp[:, 1] += start
    return wp


def _mean_cost(cost: np.ndarray, wp: np.ndarray) -> float:
    return float(cost[wp[:, 0], wp[:, 1]].mean())


def _warp_path(cost: np.ndarray, config: AlignConfig, loud: np.ndarray | None = None) -> np.ndarray:
    """(ref_frame, take_frame) pairs, oldest to newest, for a (ref x take) cost matrix.

    loud marks the take frames that must not be left out by the open end/start
    (default: all of them).
    """
    _D, legacy = librosa.sequence.dtw(C=cost)  # the original call, unchanged
    legacy = legacy[::-1]
    if not config.slope_limit and not config.open_end:
        return legacy
    if loud is None:
        loud = np.ones(cost.shape[1], dtype=bool)
    wp = _dtw_path(cost, config, loud)
    if wp is None:
        return legacy
    if config.slope_limit and config.fallback_margin is not None:
        # a held note more than 2x the reference's length cannot be followed
        # inside the slope limit, so the path is forced off the matching
        # frames; if it matches clearly worse than plain DTW, use plain DTW
        lo = int(wp[0, 1]); hi = int(wp[-1, 1])
        plain = legacy[(legacy[:, 1] >= lo) & (legacy[:, 1] <= hi)]
        if len(plain) and _mean_cost(cost, wp) > (1 + config.fallback_margin) * _mean_cost(cost, plain):
            plain_open = _dtw_path(cost, replace(config, slope_limit=False), loud)
            return legacy if plain_open is None else plain_open
    return wp


def align(
    ref_samples: np.ndarray,
    ref_sample_rate: int,
    take_samples: np.ndarray,
    take_sample_rate: int,
    config: AlignConfig = DEFAULT_ALIGN_CONFIG,
) -> AlignmentResult:
    ref_features = _mfcc(ref_samples, ref_sample_rate)
    take_features = _mfcc(take_samples, take_sample_rate)

    # frame-to-frame distance for every (ref frame, take frame) pair; the
    # warping path is the cheapest route through it from start to end
    cost = _cosine_cost(ref_features, take_features)
    warp_path = _warp_path(cost, config, _loud_frames(take_samples, config.open_end_quiet_db, take_features.shape[1]))
    ref_frames = warp_path[:, 0]
    take_frames = warp_path[:, 1]

    ref_times = librosa.frames_to_time(ref_frames, sr=ref_sample_rate, hop_length=HOP_LENGTH)
    take_times = librosa.frames_to_time(take_frames, sr=take_sample_rate, hop_length=HOP_LENGTH)

    # the path can repeat a frame on plateaus; np.interp needs take_times
    # strictly increasing, so keep only the first occurrence of each step
    _, unique_idx = np.unique(take_times, return_index=True)
    unique_idx = np.sort(unique_idx)

    return AlignmentResult(ref_times=ref_times[unique_idx], take_times=take_times[unique_idx])
