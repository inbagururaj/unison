# Key/scale pitch correction in the style of Auto-Tune / BandLab AutoPitch.
# Unlike the other two engines this never looks at the reference's timing:
# the reference is only used to detect which key and scale to snap to.
#
#   1. Detect the key: build a duration-weighted pitch-class histogram of the
#      reference's pitch track and correlate it with the Krumhansl-Kessler
#      major/minor key profiles (24 candidate keys). We also measure the
#      reference's tuning offset from A440 so the snap grid matches it.
#   2. Decompose the take with the WORLD vocoder into pitch (f0), spectral
#      envelope (formants / timbre) and aperiodicity (breathiness), every
#      5 ms, keeping every frame exactly where it is.
#   3. For each frame pick the nearest scale note (chosen from a median-
#      filtered pitch with hysteresis, so vibrato does not flicker between
#      notes but real note changes are followed immediately),
#      compute the correction = target - pitch. Each note's average error is
#      always corrected; the movement within the note (vibrato, scoops,
#      slides) is smoothed by the "retune speed": 0 flattens it (the robotic
#      effect), larger values let it through. We are offline, so the
#      smoothing is zero-phase (no lag at note starts, unlike a real-time
#      plug-in).
#   4. Resynthesize with only f0 changed. Same frame count, same envelope,
#      same aperiodicity: timing, formants and breath are untouched.
#   5. Vocoding is not transparent even with the pitch unchanged, so the
#      resynthesized audio is only used where it is needed: unvoiced audio
#      (breaths, consonants, silence) and voiced phrases that are already in
#      tune keep the take's original samples, with short crossfades at the
#      edges of the phrases that were corrected.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyworld as pw
from scipy.ndimage import gaussian_filter1d, median_filter, uniform_filter1d

from app.pitch import PitchTrack

FRAME_PERIOD_MS = 5.0
F0_FLOOR = 65.0
F0_CEIL = 1050.0

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# semitone offsets from the tonic
SCALES: dict[str, list[int]] = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "harmonic minor": [0, 2, 3, 5, 7, 8, 11],
    "major pentatonic": [0, 2, 4, 7, 9],
    "minor pentatonic": [0, 3, 5, 7, 10],
    "chromatic": list(range(12)),
}

# Krumhansl-Kessler key profiles, index 0 = tonic
_MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

DEFAULT_RETUNE_SPEED_MS = 120.0
TARGET_WINDOW_MS = 160.0  # median window for choosing the target note: cancels vibrato, keeps note steps
NOTE_HYSTERESIS = 0.25  # semitones a new note must win by before we switch to it
NOTE_STEP_MS = 15.0  # smoothing of the step between two notes' offsets
ONSET_MS = 200.0  # first target of a phrase comes from this much pitch (> one vibrato cycle), not one frame
OCTAVE_FIX_FRAMES = 15  # window for repairing single-frame octave errors in f0
PASSTHROUGH_CENTS = 3.0  # voiced phrases whose largest correction is below this keep the original audio
CROSSFADE_MS = 20.0  # blend between original and resynthesized audio at phrase edges


@dataclass
class KeyEstimate:
    tonic: int  # 0 = C ... 11 = B
    scale: str  # "major" or "minor"
    tuning_cents: float  # reference tuning relative to A440
    confidence: float  # correlation of the best key profile, -1..1

    @property
    def name(self) -> str:
        return f"{NOTE_NAMES[self.tonic]} {self.scale}"


def detect_key(track: PitchTrack) -> KeyEstimate:
    midi = track.midi[track.voiced]
    if len(midi) < 10:
        return KeyEstimate(tonic=0, scale="major", tuning_cents=0.0, confidence=0.0)

    # circular mean of the fractional part = how far the singer sits from A440
    angles = 2 * np.pi * (midi - np.round(midi))
    tuning = float(np.angle(np.mean(np.exp(1j * angles))) / (2 * np.pi))

    pitch_classes = np.round(midi - tuning).astype(int) % 12
    hist = np.bincount(pitch_classes, minlength=12).astype(float)

    best = (-2.0, 0, "major")
    for tonic in range(12):
        for scale, profile in (("major", _MAJOR_PROFILE), ("minor", _MINOR_PROFILE)):
            r = float(np.corrcoef(hist, np.roll(profile, tonic))[0, 1])
            if np.isfinite(r) and r > best[0]:
                best = (r, tonic, scale)

    return KeyEstimate(tonic=best[1], scale=best[2], tuning_cents=100 * tuning, confidence=best[0])


def scale_grid(tonic: int, scale: str, tuning_cents: float = 0.0) -> np.ndarray:
    """Every allowed MIDI pitch (float, tuning applied) from MIDI 0 to 127."""
    offsets = SCALES[scale]
    notes = [octave * 12 + tonic + o for octave in range(-1, 12) for o in offsets]
    grid = np.array(sorted(n for n in notes if 0 <= n <= 127), dtype=float)
    return grid + tuning_cents / 100.0


def analyze(samples: np.ndarray, sample_rate: int) -> dict:
    x = samples.astype(np.float64)
    f0, t = pw.harvest(x, sample_rate, f0_floor=F0_FLOOR, f0_ceil=F0_CEIL, frame_period=FRAME_PERIOD_MS)
    f0 = pw.stonemask(x, f0, t, sample_rate)
    sp = pw.cheaptrick(x, f0, t, sample_rate)
    ap = pw.d4c(x, f0, t, sample_rate)
    return {"f0": f0, "t": t, "sp": sp, "ap": ap}


def _runs(mask: np.ndarray):
    """Yield (start, end) of each contiguous True run in mask (end exclusive)."""
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    for start, end in zip(edges[::2], edges[1::2]):
        yield int(start), int(end)


def _fix_octave_errors(midi: np.ndarray) -> np.ndarray:
    """Fold frames that sit ~an octave away from their neighbours back into place."""
    fixed = midi.copy()
    for start, end in _runs(~np.isnan(midi)):
        seg = midi[start:end]
        local = median_filter(seg, size=min(OCTAVE_FIX_FRAMES, len(seg)), mode="nearest")
        octaves = np.round((seg - local) / 12.0)
        fixed[start:end] = seg - 12.0 * octaves
    return fixed


def _choose_targets(midi: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Per voiced frame, the scale note to pull toward (NaN where unvoiced)."""
    targets = np.full(len(midi), np.nan)
    size = int(TARGET_WINDOW_MS / FRAME_PERIOD_MS) | 1
    for start, end in _runs(~np.isnan(midi)):
        seg = midi[start:end]
        centre = median_filter(seg, size=min(size, len(seg)), mode="nearest")
        onset = np.median(centre[: max(1, int(ONSET_MS / FRAME_PERIOD_MS))])
        current = grid[np.argmin(np.abs(grid - onset))]
        half = size // 2
        for i, p in enumerate(centre):
            # within half a window of the phrase edges the median only sees
            # part of a vibrato cycle, so do not switch notes there
            nearest = grid[np.argmin(np.abs(grid - p))]
            if half <= i < len(seg) - half and abs(p - nearest) + NOTE_HYSTERESIS < abs(p - current):
                current = nearest
            targets[start + i] = current
    return targets


def correction_curve(midi: np.ndarray, grid: np.ndarray, retune_speed_ms: float) -> np.ndarray:
    """Semitones to add to each frame (0 where unvoiced), before strength scaling.

    The raw correction (target - pitch) is split in two:
      - a per-note offset (the median error over each target note), which is
        always applied in full, with only a short step between notes, and
      - the movement within the note (vibrato, scoops, drift, glides between
        notes), which is smoothed by the retune speed: at 0 it is removed
        entirely (hard snap), at slow speeds most of it is left alone.
    So a slow speed keeps the performance's expression but still centres
    every note, instead of only half-correcting short notes.
    """
    targets = _choose_targets(midi, grid)
    raw = targets - midi
    out = np.zeros(len(midi))
    sigma = max(0.0, retune_speed_ms) / 2.0 / FRAME_PERIOD_MS
    step_sigma = min(sigma, NOTE_STEP_MS / FRAME_PERIOD_MS)
    for start, end in _runs(~np.isnan(midi)):
        seg, seg_targets = raw[start:end], targets[start:end]
        offset = np.empty_like(seg)
        changes = np.flatnonzero(np.diff(seg_targets)) + 1
        for a, b in zip(np.r_[0, changes], np.r_[changes, len(seg)]):
            offset[a:b] = np.median(seg[a:b])
        movement = seg - offset
        if sigma > 0:
            movement = gaussian_filter1d(movement, sigma, mode="nearest")
            offset = gaussian_filter1d(offset, step_sigma, mode="nearest")
        out[start:end] = offset + movement
    return out


def _resynthesis_weight(needs: np.ndarray, n_samples: int, sample_rate: int) -> np.ndarray:
    """Per-sample mix of resynthesized (1) vs original (0) audio from a per-frame mask."""
    frame_times = np.arange(len(needs)) * FRAME_PERIOD_MS / 1000.0
    sample_times = np.arange(n_samples) / sample_rate
    w = np.interp(sample_times, frame_times, needs.astype(float))
    return uniform_filter1d(w, max(1, int(CROSSFADE_MS / 1000.0 * sample_rate)), mode="nearest")


def autotune(
    take_samples: np.ndarray,
    sample_rate: int,
    grid: np.ndarray,
    strength: float,
    retune_speed_ms: float = DEFAULT_RETUNE_SPEED_MS,
    analysis: dict | None = None,
) -> np.ndarray:
    """Pitch-correct the take toward grid without touching its timing.

    strength is 0-100. retune_speed_ms is 0 (instant, robotic) and up (slower, more natural).
    Output has exactly as many samples as the input.
    """
    s = max(0.0, min(100.0, strength)) / 100.0
    if s == 0.0 or len(take_samples) == 0:
        return take_samples.astype(np.float32).copy()

    a = analysis or analyze(take_samples, sample_rate)
    f0 = a["f0"]
    voiced = f0 > 0
    midi = np.full(len(f0), np.nan)
    midi[voiced] = 69.0 + 12.0 * np.log2(f0[voiced] / 440.0)
    midi = _fix_octave_errors(midi)

    correction = s * correction_curve(midi, grid, retune_speed_ms)
    f0_out = np.zeros(len(f0))
    f0_out[voiced] = 440.0 * 2.0 ** ((midi[voiced] + correction[voiced] - 69.0) / 12.0)

    needs = np.zeros(len(f0), dtype=bool)
    for start, end in _runs(voiced):
        needs[start:end] = np.max(np.abs(correction[start:end])) * 100.0 >= PASSTHROUGH_CENTS
    if not needs.any():
        return take_samples.astype(np.float32).copy()

    out = pw.synthesize(
        np.ascontiguousarray(f0_out),
        np.ascontiguousarray(a["sp"]),
        np.ascontiguousarray(a["ap"]),
        sample_rate,
        FRAME_PERIOD_MS,
    )

    # WORLD pads to a whole number of frames; trim/pad back to the input length
    n = len(take_samples)
    out = out[:n] if len(out) >= n else np.pad(out, (0, n - len(out)))

    w = _resynthesis_weight(needs, n, sample_rate)
    out = w * out + (1.0 - w) * take_samples.astype(np.float64)

    peak = float(np.max(np.abs(out)))
    if peak > 0.99:
        out = out * (0.99 / peak)
    return out.astype(np.float32)
