# Gives a rough opinion on whether the reference audio looks like a solo
# singing voice, so we can warn the user instead of silently producing a
# bad result. This is a heuristic score, not a hard gate: three signals are
# combined - how much of the clip is voiced, how much energy looks
# percussive/instrumental (via harmonic-percussive separation), and how
# often the pitch jumps around implausibly (a sign of multiple overlapping
# voices or instruments rather than one melodic line).

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

from app.pitch import PitchTrack

IMPLAUSIBLE_JUMP_SEMITONES = 12  # bigger than an octave, frame to frame
WARN_THRESHOLD = 0.5


@dataclass
class SoloCheckResult:
    score: float  # 0 (not solo vocal) to 1 (clearly solo vocal)
    warn: bool
    reason: str


def check_solo_vocal(samples: np.ndarray, sample_rate: int, track: PitchTrack) -> SoloCheckResult:
    voiced_ratio = float(np.mean(track.voiced)) if len(track.voiced) else 0.0

    harmonic, percussive = librosa.effects.hpss(samples)
    harmonic_energy = float(np.sum(harmonic**2))
    percussive_energy = float(np.sum(percussive**2))
    total_energy = harmonic_energy + percussive_energy
    percussive_ratio = percussive_energy / total_energy if total_energy > 0 else 0.0

    voiced_midi = track.midi[track.voiced]
    if len(voiced_midi) > 1:
        jumps = np.abs(np.diff(voiced_midi))
        jump_rate = float(np.mean(jumps > IMPLAUSIBLE_JUMP_SEMITONES))
    else:
        jump_rate = 0.0

    score = (
        0.5 * voiced_ratio
        + 0.3 * (1.0 - min(percussive_ratio * 2, 1.0))
        + 0.2 * (1.0 - min(jump_rate * 5, 1.0))
    )
    score = max(0.0, min(1.0, score))

    reasons = []
    if voiced_ratio < 0.4:
        reasons.append("much of the clip has no clear singing pitch")
    if percussive_ratio > 0.35:
        reasons.append("there is a lot of percussive/instrumental energy")
    if jump_rate > 0.05:
        reasons.append("the pitch jumps around in ways a single voice usually does not")

    warn = score < WARN_THRESHOLD
    if warn:
        reason = "this reference may not be a clean solo vocal: " + "; ".join(reasons or ["low confidence overall"]) + "."
    else:
        reason = "this reference looks like a solo vocal."

    return SoloCheckResult(score=score, warn=warn, reason=reason)
