import numpy as np

from app.pitch import track_pitch
from tests.audio_helpers import SAMPLE_RATE, make_tone


def test_tracks_a_steady_tone_at_the_right_pitch():
    # A4 = 440 Hz = MIDI 69
    signal = make_tone(440.0, duration_seconds=1.5)
    track = track_pitch(signal, SAMPLE_RATE)

    assert track.voiced.mean() > 0.7

    voiced_midi = track.midi[track.voiced]
    assert abs(np.median(voiced_midi) - 69.0) < 0.5


def test_silence_is_mostly_unvoiced():
    signal = np.zeros(int(1.0 * SAMPLE_RATE), dtype=np.float32)
    track = track_pitch(signal, SAMPLE_RATE)

    assert track.voiced.mean() < 0.1
