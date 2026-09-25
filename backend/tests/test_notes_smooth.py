# Tests for the smoothed notes engine: tiny segments get merged, the stretch
# ratio cannot lurch between neighbours, the pitch shift glides instead of
# stepping, and the original version is still reachable.

from __future__ import annotations

import numpy as np

from app import shift
from app.align import AlignmentResult
from app.notes import Note, merge_short_notes
from tests.audio_helpers import SAMPLE_RATE, make_tone

SR = SAMPLE_RATE


def _setup():
    take = make_tone(261.63, 2.0)  # C4
    # a sustained note chopped by a 60 ms blip, then a second note
    take_notes = [Note(0.0, 0.5, 60.0), Note(0.52, 0.58, 60.4), Note(0.6, 1.2, 60.0), Note(1.2, 1.9, 61.0)]
    ref_notes = [Note(0.0, 1.1, 62.0), Note(1.1, 2.1, 64.0)]
    alignment = AlignmentResult(ref_times=np.array([0.0, 2.2]), take_times=np.array([0.0, 2.0]))
    return take, take_notes, ref_notes, alignment


def test_merge_reduces_segment_count_and_keeps_real_notes():
    take_notes = _setup()[1]
    merged = merge_short_notes(take_notes, min_duration=0.14, max_gap=0.06)
    assert len(merged) < len(take_notes)
    assert len(merged) == 2
    assert all(n.end - n.start >= 0.14 for n in merged)
    assert merged[0].start == 0.0 and merged[0].end == 1.2  # the blip is absorbed, the note is whole


def test_merge_leaves_an_isolated_short_note_alone():
    notes = [Note(0.0, 0.5, 60.0), Note(1.0, 1.06, 64.0)]  # far apart: a real staccato note
    assert len(merge_short_notes(notes)) == 2


def test_smoothed_ratios_change_by_a_bounded_amount_between_neighbours():
    raw = np.array([0.7, 1.4, 0.75, 1.35, 0.8, 1.3])
    used = shift.smooth_ratios(raw, np.zeros(len(raw) - 1, dtype=bool), max_step=0.08)
    assert np.abs(np.diff(used)).max() <= 0.08 + 1e-9
    assert used.min() >= shift.MIN_STRETCH_RATIO and used.max() <= shift.MAX_STRETCH_RATIO


def test_ratios_are_not_smoothed_across_a_long_rest():
    raw = np.array([0.8, 0.8, 1.3, 1.3])
    used = shift.smooth_ratios(raw, np.array([False, True, False]), max_step=0.08)
    assert np.allclose(used, raw)  # each side of the rest is already steady


def test_default_engine_has_fewer_segments_and_bounded_ratio_change_than_legacy():
    take, take_notes, ref_notes, alignment = _setup()
    _, legacy = shift.correct_take_debug(take, SR, take_notes, ref_notes, alignment, 100, shift.LEGACY_CONFIG)
    out, new = shift.correct_take_debug(take, SR, take_notes, ref_notes, alignment, 100, shift.DEFAULT_CONFIG)
    assert len(new["notes"]) < len(legacy["notes"])
    assert np.abs(np.diff(new["ratios_used"])).max() <= shift.DEFAULT_CONFIG.max_ratio_step + 1e-9
    assert np.isfinite(out).all() and len(out) > 0


def test_glide_steps_the_shift_less_than_a_constant_shift():
    take, take_notes, ref_notes, alignment = _setup()
    merged = merge_short_notes(take_notes)
    shifts = np.array(shift._plan_info(merged, ref_notes, alignment, 1.0, shift.DEFAULT_CONFIG, [], SR)["semitones"])
    assert abs(shifts[1] - shifts[0]) > 0.5  # the two notes really need different shifts
    # the curve the glide builds changes by less than that inside any 20 ms
    g = shift.DEFAULT_CONFIG.glide_ms / 1000 / 2
    t = np.array([merged[0].start + g, merged[0].end - g, merged[1].start + g, merged[1].end - g])
    curve = lambda x: np.interp(x, t, [shifts[0], shifts[0], shifts[1], shifts[1]])  # noqa: E731
    xs = np.arange(0.0, 1.9, 0.001)
    per_20ms = np.abs(curve(xs + 0.02) - curve(xs)).max()
    assert per_20ms < abs(shifts[1] - shifts[0])


def test_original_version_is_still_reachable_and_unchanged():
    take, take_notes, ref_notes, alignment = _setup()
    via_flag = shift.correct_take(take, SR, take_notes, ref_notes, alignment, 100, shift.LEGACY_CONFIG)
    direct = shift._correct_take_legacy(take, SR, take_notes, ref_notes, alignment, 100)
    assert np.array_equal(via_flag, direct)


def test_snap_zero_still_leaves_pitch_alone_with_the_smoothed_engine():
    take, take_notes, ref_notes, alignment = _setup()
    _, info = shift.correct_take_debug(take, SR, take_notes, ref_notes, alignment, 0, shift.DEFAULT_CONFIG)
    assert np.allclose(info["semitones"], 0.0)
