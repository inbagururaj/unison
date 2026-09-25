# Tests for the vocoder-based notes engine: pitch lands on the reference notes,
# timing follows the reference, formants stay put on a big upward shift,
# silences stay silent, and strength 0 leaves the pitch alone.

from __future__ import annotations

import librosa
import numpy as np
import pyworld as pw

from app.align import AlignmentResult
from app.notes import Note, segment_notes
from app.notes_world import notes_world
from app.pitch import track_pitch
from tests.test_autotune import _voice

SR = 22050


def _pair(take_midis=(62.6, 60.4), ref_midis=(64.0, 62.0), each=0.7, rest=0.3, ref_scale=1.0):
    take = _voice(list(take_midis), seconds_each=each, rest=rest)
    take_dur = len(take) / SR
    ref_dur = take_dur * ref_scale
    take_notes = segment_notes(track_pitch(take, SR))
    starts = [i * (each + rest) for i in range(len(ref_midis))]
    ref_notes = [Note(s * ref_scale + 0.05, (s + each) * ref_scale - 0.05, m) for s, m in zip(starts, ref_midis)]
    alignment = AlignmentResult(ref_times=np.array([0.0, ref_dur]), take_times=np.array([0.0, take_dur]))
    return take, take_notes, ref_notes, alignment, ref_dur


def _note_medians(out, ref_notes):
    tr = track_pitch(out, SR)
    res = []
    for n in ref_notes:
        sel = tr.voiced & (tr.times > n.start + 0.1) & (tr.times < n.end - 0.1)
        res.append(float(np.median(tr.midi[sel])))
    return res


def test_each_note_lands_on_its_reference_note():
    take, take_notes, ref_notes, alignment, ref_dur = _pair()
    out = notes_world(take, SR, ref_dur, take_notes, ref_notes, alignment, 100)
    for got, note in zip(_note_medians(out, ref_notes), ref_notes):
        assert abs(got - note.midi) < 0.3


def test_output_follows_the_reference_length_and_timing():
    take, take_notes, ref_notes, alignment, ref_dur = _pair(ref_scale=0.85)
    out = notes_world(take, SR, ref_dur, take_notes, ref_notes, alignment, 100)
    assert abs(len(out) / SR - ref_dur) < 0.05
    # the notes still land inside their (compressed) reference windows
    for got, note in zip(_note_medians(out, ref_notes), ref_notes):
        assert abs(got - note.midi) < 0.3


def test_strength_zero_leaves_the_pitch_alone():
    take, take_notes, ref_notes, alignment, ref_dur = _pair()
    out = notes_world(take, SR, ref_dur, take_notes, ref_notes, alignment, 0)
    for got, orig in zip(_note_medians(out, ref_notes), (62.6, 60.4)):
        assert abs(got - orig) < 0.3


def test_silences_stay_silent():
    take, take_notes, ref_notes, alignment, ref_dur = _pair()
    out = notes_world(take, SR, ref_dur, take_notes, ref_notes, alignment, 100)
    rest = out[int(0.78 * SR) : int(0.92 * SR)]  # inside the gap between the two notes
    voiced = out[int(0.2 * SR) : int(0.6 * SR)]
    assert np.sqrt(np.mean(rest**2)) < 0.05 * np.sqrt(np.mean(voiced**2))


def test_big_upward_shift_keeps_formants_where_a_plain_pitch_shift_moves_them():
    take = _voice([48.0], seconds_each=1.2)
    take_notes = [Note(0.05, 1.15, 48.0)]
    ref_notes = [Note(0.05, 1.15, 55.0)]  # a fifth up, +7 semitones
    alignment = AlignmentResult(ref_times=np.array([0.0, 1.2]), take_times=np.array([0.0, 1.2]))
    out = notes_world(take, SR, 1.2, take_notes, ref_notes, alignment, 100)

    def envelope(x):
        x = x.astype(np.float64)
        f0, t = pw.harvest(x, SR, f0_floor=65, f0_ceil=1050)
        return np.log(pw.cheaptrick(x, f0, t, SR)[f0 > 0].mean(axis=0) + 1e-12)

    ref_env = envelope(take)
    band = np.linspace(0, SR / 2, len(ref_env)) < 4000

    def distance(x):
        d = envelope(x) - ref_env
        return float(np.abs((d - d[band].mean())[band]).mean())

    naive = librosa.effects.pitch_shift(take, sr=SR, n_steps=7.0)
    assert abs(np.median(track_pitch(out, SR).midi[track_pitch(out, SR).voiced]) - 55.0) < 0.3
    assert distance(out) < 0.6 * distance(naive)
