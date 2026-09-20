import numpy as np

from app.align import AlignmentResult
from app.notes import Note
from app.pitch import track_pitch
from app.shift import correct_take
from tests.audio_helpers import SAMPLE_RATE, make_tone


def _identity_alignment(duration: float) -> AlignmentResult:
    times = np.array([0.0, duration])
    return AlignmentResult(ref_times=times, take_times=times)


def test_shifting_a_note_by_n_semitones_measures_n_semitones_after():
    duration = 1.5
    take_signal = make_tone(440.0, duration_seconds=duration)  # A4, MIDI 69

    take_note = Note(start=0.0, end=duration, midi=69.0)
    ref_note = Note(start=0.0, end=duration, midi=72.0)  # 3 semitones up (C5)

    corrected = correct_take(
        take_signal,
        SAMPLE_RATE,
        take_notes=[take_note],
        ref_notes=[ref_note],
        alignment=_identity_alignment(duration),
        snap_strength=100.0,
    )

    track = track_pitch(corrected, SAMPLE_RATE)
    voiced_midi = track.midi[track.voiced]

    assert abs(np.median(voiced_midi) - 72.0) < 0.5


def test_snap_strength_zero_leaves_pitch_unchanged():
    duration = 1.5
    take_signal = make_tone(440.0, duration_seconds=duration)

    take_note = Note(start=0.0, end=duration, midi=69.0)
    ref_note = Note(start=0.0, end=duration, midi=72.0)

    corrected = correct_take(
        take_signal,
        SAMPLE_RATE,
        take_notes=[take_note],
        ref_notes=[ref_note],
        alignment=_identity_alignment(duration),
        snap_strength=0.0,
    )

    track = track_pitch(corrected, SAMPLE_RATE)
    voiced_midi = track.midi[track.voiced]

    assert abs(np.median(voiced_midi) - 69.0) < 0.5
