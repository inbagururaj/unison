from app.notes import segment_notes
from app.pitch import track_pitch
from tests.audio_helpers import SAMPLE_RATE, make_melody


def test_segments_a_two_note_melody():
    # A4 (440 Hz, MIDI 69) then C5 (523.25 Hz, MIDI 72): a 3-semitone jump
    signal = make_melody([440.0, 523.25], note_duration_seconds=1.0)
    track = track_pitch(signal, SAMPLE_RATE)

    notes = segment_notes(track)

    assert len(notes) == 2
    assert abs(notes[0].midi - 69.0) < 0.75
    assert abs(notes[1].midi - 72.0) < 0.75
    assert notes[0].end <= notes[1].start + 0.05


def test_single_steady_note_is_one_note():
    signal = make_melody([440.0], note_duration_seconds=1.5)
    track = track_pitch(signal, SAMPLE_RATE)

    notes = segment_notes(track)

    assert len(notes) == 1
