# The notes engine's decisions, carried out the way the retune engine works.
#
# The classic notes engine cuts the take into notes and time-stretches and
# pitch-shifts each chunk of audio on its own, which is where the choppy,
# smeared sound comes from. This one keeps the same decisions but applies
# them to WORLD parameters instead of to audio:
#
#   1. Analyze the whole take once: f0 (pitch), spectral envelope (the voice's
#      timbre and vowel shape) and aperiodicity (breath), every 5 ms.
#   2. Pitch: each take note gets the shift the notes engine would give it,
#      (matched reference note - take note) * strength, with merged tiny
#      segments and a glide between neighbouring notes instead of a step.
#      The shift is added to f0 only. The envelope is never touched, so
#      formants stay put however far the note moves (no chipmunk effect), and
#      the singer's own vibrato and slides inside a note survive.
#   3. Timing: the smoothed DTW mapping from retune warps the *frames* onto
#      the reference's timeline. Nothing is stretched as audio, and there are
#      no per-note stretch ratios to clamp or lurch.
#   4. Frames keep their voiced/unvoiced status from the take. Unvoiced and
#      silent frames stay f0 = 0 with their own envelope and aperiodicity, so
#      breaths, consonants and silence carry over untouched.
#   5. Resynthesize once.

from __future__ import annotations

import librosa
import numpy as np
import pyworld as pw

from app.align import AlignmentResult
from app.notes import Note, merge_short_notes
from app.retune import (
    FRAME_PERIOD_MS,
    _f0_to_midi,
    _fill_nan_nearest,
    _interp_frames,
    _ref_to_take_times,
    analyze,
)
from app.shift import DEFAULT_CONFIG, _closest_ref_note, shift_curve


def notes_world(
    take_samples: np.ndarray,
    sample_rate: int,
    ref_duration: float,
    take_notes: list[Note],
    ref_notes: list[Note],
    alignment: AlignmentResult,
    snap_strength: float,
    timing_strength: float = 1.0,
    analysis: dict | None = None,
) -> np.ndarray:
    """Return the take re-pitched note by note and re-timed onto the reference.

    snap_strength is 0-100. timing_strength is 0-1 (0 keeps the take's own
    timing). The output runs for ref_duration seconds, the reference's length.
    """
    strength = max(0.0, min(100.0, snap_strength)) / 100.0
    cfg = DEFAULT_CONFIG
    a = analysis or analyze(take_samples, sample_rate)

    # the same note clean-up the smoothed notes engine does
    take_notes = merge_short_notes(take_notes, cfg.merge_ms / 1000, cfg.max_gap_ms / 1000)
    ref_notes = merge_short_notes(ref_notes, cfg.merge_ms / 1000, cfg.max_gap_ms / 1000)
    semis = np.array(
        [
            (r.midi - n.midi) * strength if (r := _closest_ref_note(n, ref_notes, alignment)) else 0.0
            for n in take_notes
        ]
    )
    shift_at = shift_curve(take_notes, semis, max(cfg.glide_ms, 1.0))

    # output frames live on the reference timeline; each pulls a take frame
    n_out = max(1, int(ref_duration * 1000 / FRAME_PERIOD_MS) + 1)
    out_times = np.arange(n_out) * FRAME_PERIOD_MS / 1000.0
    take_times = _ref_to_take_times(alignment, out_times, timing_strength)
    pos = take_times / (FRAME_PERIOD_MS / 1000.0)

    take_midi = _f0_to_midi(a["f0"])
    take_voiced = ~np.isnan(take_midi)
    filled = _fill_nan_nearest(take_midi)
    pos_c = np.clip(pos, 0, len(take_midi) - 1)
    midi_out = np.interp(pos_c, np.arange(len(take_midi)), filled)
    voiced_out = take_voiced[np.round(pos_c).astype(int)]

    # inside a note the shift is that note's; between notes it glides; outside
    # all notes (a dropped blip, the run-up to the first note) np.interp holds
    # the nearest note's shift
    shift = shift_at(take_times) if shift_at is not None else np.zeros(n_out)
    f0_out = np.zeros(n_out)
    f0_out[voiced_out] = librosa.midi_to_hz(midi_out[voiced_out] + shift[voiced_out])

    sp = _interp_frames(a["sp"], pos, log=True)
    ap = np.clip(_interp_frames(a["ap"], pos, log=False), 0.0, 1.0)

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
