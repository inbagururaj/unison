# Tests for the DTW alignment's boundary handling: trailing silence in the
# take is left out instead of being squeezed into the reference's last note,
# sung material is never left out, a note held more than 2x the reference's
# length falls back to plain DTW, and the legacy config is the original call.

from __future__ import annotations

import librosa
import numpy as np

from app.align import (
    DEFAULT_ALIGN_CONFIG,
    HOP_LENGTH,
    LEGACY_ALIGN_CONFIG,
    _cosine_cost,
    _loud_frames,
    _mfcc,
    _warp_path,
    align,
)
from app.retune import FRAME_PERIOD_MS, _ref_to_take_times
from tests.test_autotune import _voice

SR = 22050
MELODY = [60, 62, 64, 65, 67, 64, 60]
FRAME_S = HOP_LENGTH / SR


def _sing(midis, durations, gap=0.08) -> np.ndarray:
    parts = []
    for m, d in zip(midis, durations):
        parts += [_voice([m], seconds_each=d), np.zeros(int(gap * SR), np.float32)]
    return np.concatenate(parts[:-1])


def _pair(take_durations, trailing_silence=0.0):
    ref = _sing(MELODY, [0.6] * len(MELODY))
    take = _sing(np.array(MELODY) + 0.4, take_durations)
    singing_end = len(take) / SR
    take = np.concatenate([take, np.zeros(int(trailing_silence * SR), np.float32)])
    return ref, take, singing_end


def _max_rate(ref, al, lo, hi) -> float:
    """Largest take-seconds per output-second over any 100 ms between take times lo and hi."""
    grid = np.arange(0, len(ref) / SR, FRAME_PERIOD_MS / 1000)
    mapped = _ref_to_take_times(al, grid, 1.0)
    rate = np.gradient(mapped, grid)
    inside = (mapped >= lo) & (mapped <= hi)
    w = int(0.1 / (FRAME_PERIOD_MS / 1000))
    return float(np.max(np.convolve(np.where(inside, rate, 0.0), np.ones(w) / w, "same")))


def test_trailing_silence_is_left_out_not_squeezed_into_the_last_note():
    # case E from the boundary diagnosis: the take stops recording 0.8 s late
    ref, take, singing_end = _pair([0.62] * len(MELODY), trailing_silence=0.8)
    ref_end = len(ref) / SR

    new = align(ref, SR, take, SR, DEFAULT_ALIGN_CONFIG)
    # the path finishes where the singing does; the silence after it is unused
    assert new.take_times[-1] < singing_end + 0.1
    # so the reference's end lines up with the end of the singing
    assert abs(np.interp(ref_end, new.ref_times, new.take_times) - singing_end) < 0.1
    # and the last note plays at the reference's pace, no fast burst
    last_note = singing_end - 0.62
    assert _max_rate(ref, new, last_note, singing_end) < 2.2

    # legacy (for A/B) must consume the whole take, silence included
    old = align(ref, SR, take, SR, LEGACY_ALIGN_CONFIG)
    assert old.take_times[-1] > len(take) / SR - 2 * FRAME_S


def test_sung_material_is_never_left_out():
    # the same pair as above, where the open end does leave the silence out;
    # with every frame marked as loud (sung), nothing may be left out
    ref, take, _ = _pair([0.62] * len(MELODY), trailing_silence=0.8)
    cost = _cosine_cost(_mfcc(ref, SR), _mfcc(take, SR))
    m = cost.shape[1]
    measured = _loud_frames(take, DEFAULT_ALIGN_CONFIG.open_end_quiet_db, m)
    assert _warp_path(cost, DEFAULT_ALIGN_CONFIG, measured)[-1, 1] < m - 20  # silence really is quiet: left out
    path = _warp_path(cost, DEFAULT_ALIGN_CONFIG, np.ones(m, dtype=bool))
    assert path[0, 1] == 0 and path[-1, 1] == m - 1

    # and a real held note (1.8x) is kept whole
    ref, take, singing_end = _pair([0.62] * (len(MELODY) - 1) + [1.08])
    al = align(ref, SR, take, SR, DEFAULT_ALIGN_CONFIG)
    assert al.take_times[0] < 2 * FRAME_S
    assert al.take_times[-1] > singing_end - 2 * FRAME_S


def test_note_held_past_the_slope_limit_falls_back_to_plain_dtw():
    # a mid-take fermata, 2.5x the reference's note: [1/2, 2] cannot follow it
    ref, take, _ = _pair([0.62, 0.62, 1.5, 0.62, 0.62, 0.62, 0.62])
    cost = _cosine_cost(_mfcc(ref, SR), _mfcc(take, SR))
    assert np.array_equal(_warp_path(cost, DEFAULT_ALIGN_CONFIG), _warp_path(cost, LEGACY_ALIGN_CONFIG))


def test_legacy_config_is_the_original_dtw_call():
    ref, take, _ = _pair([0.62, 0.55, 0.7, 0.6, 0.66, 0.58, 0.64], trailing_silence=0.3)
    X, Y = _mfcc(ref, SR), _mfcc(take, SR)
    _D, original = librosa.sequence.dtw(X=X, Y=Y, metric="cosine")
    assert np.array_equal(_warp_path(_cosine_cost(X, Y), LEGACY_ALIGN_CONFIG), original[::-1])
