# Rebuilds the singer's take note by note: each take note is time-stretched
# to match how long the corresponding reference note should take (from the
# DTW alignment), then pitch-shifted toward the reference note's pitch by
# an amount controlled by "snap strength" (0% = untouched, 100% = exactly
# on pitch). The reshaped notes are stitched back together with short
# crossfades so the joins do not click.
#
# Two versions live here, selectable by a NotesConfig (see STAGES at the
# bottom for the presets, and DEFAULT_CONFIG for what the app uses):
#   LEGACY_CONFIG   the original: note by note, constant shift per note,
#                   15 ms linear crossfade, no context. Left untouched.
#   the smoothed    merges tiny segments, limits how fast the stretch ratio
#   version         changes between neighbouring notes, gives every chunk
#                   real audio context on both sides, uses a longer overlap
#                   at touching notes, and glides the pitch shift between
#                   note centres instead of stepping it.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline

from app.align import AlignmentResult
from app.notes import Note, merge_short_notes

try:
    import pyrubberband as pyrb

    _HAS_RUBBERBAND = True
except Exception:  # pyrubberband raises OSError if the rubberband CLI is missing
    _HAS_RUBBERBAND = False

import librosa

FADE_SECONDS = 0.015
MIN_STRETCH_RATIO = 0.7
MAX_STRETCH_RATIO = 1.4


def _pitch_shift(samples: np.ndarray, sample_rate: int, semitones: float) -> np.ndarray:
    if abs(semitones) < 1e-3 or len(samples) == 0:
        return samples
    if _HAS_RUBBERBAND:
        try:
            return pyrb.pitch_shift(samples, sample_rate, semitones).astype(np.float32)
        except Exception:
            pass  # fall through to librosa if the rubberband CLI misbehaves at runtime
    return librosa.effects.pitch_shift(
        samples, sr=sample_rate, n_steps=semitones
    ).astype(np.float32)


def _time_stretch(samples: np.ndarray, ratio: float) -> np.ndarray:
    """ratio = desired_output_len / input_len."""
    if abs(ratio - 1.0) < 1e-3 or len(samples) == 0:
        return samples
    rate = 1.0 / ratio  # librosa's rate is input_len / output_len
    return librosa.effects.time_stretch(samples, rate=rate).astype(np.float32)


def _closest_ref_note(take_note: Note, ref_notes: list[Note], alignment: AlignmentResult) -> Note | None:
    if not ref_notes:
        return None
    mid_take = (take_note.start + take_note.end) / 2
    mapped = alignment.take_to_ref(mid_take)

    for note in ref_notes:
        if note.start <= mapped <= note.end:
            return note

    return min(ref_notes, key=lambda n: abs((n.start + n.end) / 2 - mapped))


def _crossfade_append(output: list[np.ndarray], chunk: np.ndarray, sample_rate: int) -> None:
    fade_len = min(int(FADE_SECONDS * sample_rate), len(chunk) // 2)
    if not output or fade_len <= 0:
        output.append(chunk)
        return

    prev = output[-1]
    if len(prev) < fade_len:
        output.append(chunk)
        return

    fade_out = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
    fade_in = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)

    overlap = prev[-fade_len:] * fade_out + chunk[:fade_len] * fade_in
    output[-1] = np.concatenate([prev[:-fade_len], overlap])
    output.append(chunk[fade_len:])


def _correct_take_legacy(
    take_samples: np.ndarray,
    take_sample_rate: int,
    take_notes: list[Note],
    ref_notes: list[Note],
    alignment: AlignmentResult,
    snap_strength: float,
    trace: list | None = None,
) -> np.ndarray:
    """The original implementation, unchanged apart from the optional trace
    (a list that receives, per note, the output position where it starts)."""
    strength = max(0.0, min(100.0, snap_strength)) / 100.0
    output: list[np.ndarray] = []
    cursor = 0.0

    for note in take_notes:
        start_sample = int(note.start * take_sample_rate)
        end_sample = int(note.end * take_sample_rate)
        chunk = take_samples[start_sample:end_sample]
        if len(chunk) == 0:
            continue

        # fill any silent gap before this note with the original audio
        gap_start_sample = int(cursor * take_sample_rate)
        if gap_start_sample < start_sample:
            output.append(take_samples[gap_start_sample:start_sample])

        ref_note = _closest_ref_note(note, ref_notes, alignment)

        ref_start = alignment.take_to_ref(note.start)
        ref_end = alignment.take_to_ref(note.end)
        target_duration = max(ref_end - ref_start, 1e-3)
        original_duration = note.end - note.start
        ratio = target_duration / original_duration if original_duration > 0 else 1.0
        ratio = max(MIN_STRETCH_RATIO, min(MAX_STRETCH_RATIO, ratio))

        stretched = _time_stretch(chunk, ratio)

        semitones = 0.0
        if ref_note is not None:
            semitones = (ref_note.midi - note.midi) * strength

        shifted = _pitch_shift(stretched, take_sample_rate, semitones)

        if trace is not None:
            fade = min(int(FADE_SECONDS * take_sample_rate), len(shifted) // 2)
            trace.append(sum(len(a) for a in output) - fade // 2)
        _crossfade_append(output, shifted, take_sample_rate)
        cursor = note.end

    tail_start_sample = int(cursor * take_sample_rate)
    if tail_start_sample < len(take_samples):
        output.append(take_samples[tail_start_sample:])

    if not output:
        return take_samples.copy()

    return np.concatenate(output).astype(np.float32)


# ---------------------------------------------------------------------------
# configuration


@dataclass(frozen=True)
class NotesConfig:
    legacy: bool = False
    merge_ms: float = 0.0  # 0 = off; segments shorter than this are absorbed into a neighbour
    max_gap_ms: float = 60.0  # neighbours closer than this count as touching for merging
    smooth_ratios: bool = False  # smooth + limit the stretch ratio across neighbouring notes
    max_ratio_step: float = 0.08  # largest allowed ratio change between adjacent notes
    pad_ms: float = 0.0  # real audio context processed on each side of a chunk, then trimmed
    crossfade_ms: float = 15.0  # overlap at touching notes
    crossfade_shape: str = "linear"  # or "equal_power"
    glide_ms: float = 0.0  # 0 = constant shift per note, else ramp the shift over this long at joins
    adjacent_ms: float = 50.0  # notes closer than this are joined with a crossfade


LEGACY_CONFIG = NotesConfig(legacy=True)

# Each stage adds one change on top of the previous one, so the effect of
# each can be measured on its own (scripts/diagnose_notes.py runs all of them).
STAGES: dict[str, NotesConfig] = {
    "0 legacy": LEGACY_CONFIG,
    "1 merge short segments": NotesConfig(merge_ms=140.0),
    "2 + smooth stretch ratios": NotesConfig(merge_ms=140.0, smooth_ratios=True),
    "3 + context padding": NotesConfig(merge_ms=140.0, smooth_ratios=True, pad_ms=50.0),
    "4 + longer crossfade": NotesConfig(
        merge_ms=140.0, smooth_ratios=True, pad_ms=50.0, crossfade_ms=40.0, crossfade_shape="equal_power"
    ),
    "5 + pitch glide": NotesConfig(
        merge_ms=140.0, smooth_ratios=True, pad_ms=50.0, crossfade_ms=40.0, crossfade_shape="equal_power", glide_ms=60.0
    ),
}
DEFAULT_CONFIG = STAGES["5 + pitch glide"]

RUN_BREAK_SECONDS = 0.15  # a longer gap starts a new run when smoothing ratios


# ---------------------------------------------------------------------------
# the smoothed version


def smooth_ratios(raw: np.ndarray, breaks_after: np.ndarray, max_step: float) -> np.ndarray:
    """Clamp, average over neighbours, then limit the change between neighbours.

    breaks_after[i] is True where a long gap separates note i from note i+1;
    the rhythm cannot lurch across a rest, so runs are smoothed separately.
    """
    clamped = np.clip(raw, MIN_STRETCH_RATIO, MAX_STRETCH_RATIO)
    out = clamped.copy()
    start = 0
    for end in [i + 1 for i in np.flatnonzero(breaks_after)] + [len(raw)]:
        run = clamped[start:end]
        if len(run) > 1:
            padded = np.concatenate([[run[0]], run, [run[-1]]])
            run = 0.25 * padded[:-2] + 0.5 * padded[1:-1] + 0.25 * padded[2:]
            for i in range(1, len(run)):  # forward pass
                run[i] = np.clip(run[i], run[i - 1] - max_step, run[i - 1] + max_step)
            for i in range(len(run) - 2, -1, -1):  # backward pass
                run[i] = np.clip(run[i], run[i + 1] - max_step, run[i + 1] + max_step)
            out[start:end] = run
        start = end
    return np.clip(out, MIN_STRETCH_RATIO, MAX_STRETCH_RATIO)


def shift_curve(notes: list[Note], semis: np.ndarray, glide_ms: float):
    """Pitch shift (semitones) as a function of take time: flat inside each
    note, ramping linearly between neighbours over glide_ms centred on the
    join. Returns None when glide_ms is 0 (constant shift per note)."""
    if glide_ms <= 0 or not notes:
        return None
    g = glide_ms / 1000 / 2
    knots_t, knots_s = [], []
    for note, s in zip(notes, semis):
        if note.end - note.start > 2 * g:
            knots_t += [note.start + g, note.end - g]
            knots_s += [s, s]
        else:
            knots_t.append((note.start + note.end) / 2)
            knots_s.append(s)
    knots_t, knots_s = np.array(knots_t), np.array(knots_s)
    return lambda t: np.interp(t, knots_t, knots_s)


def _resample_variable_rate(x: np.ndarray, semitones: np.ndarray) -> np.ndarray:
    """Read x faster where semitones > 0: raises pitch by a curve, not a constant.

    Changes the length (shorter where shifted up); the caller time-stretches it back.
    """
    rate = 2.0 ** (semitones / 12.0)
    out_time = np.concatenate([[0.0], np.cumsum(1.0 / rate)])[: len(x)]
    n_out = int(out_time[-1]) + 1
    in_pos = np.interp(np.arange(n_out), out_time, np.arange(len(x)))
    return CubicSpline(np.arange(len(x)), x)(in_pos).astype(np.float32)


def _fades(n: int, shape: str) -> tuple[np.ndarray, np.ndarray]:
    u = np.linspace(0.0, 1.0, n, dtype=np.float32)
    if shape == "equal_power":
        return np.cos(0.5 * np.pi * u), np.sin(0.5 * np.pi * u)
    return 1.0 - u, u


def _slice_padded(samples: np.ndarray, start: int, end: int) -> np.ndarray:
    """samples[start:end] with zeros for anything outside the array."""
    left, right = max(0, -start), max(0, end - len(samples))
    piece = samples[max(0, start) : min(len(samples), end)]
    return np.pad(piece, (left, right)) if left or right else piece


def correct_take_debug(
    take_samples: np.ndarray,
    take_sample_rate: int,
    take_notes: list[Note],
    ref_notes: list[Note],
    alignment: AlignmentResult,
    snap_strength: float,
    config: NotesConfig = DEFAULT_CONFIG,
) -> tuple[np.ndarray, dict]:
    """Same as correct_take, but also returns what was decided, for diagnostics."""
    sr = take_sample_rate
    strength = max(0.0, min(100.0, snap_strength)) / 100.0

    if config.legacy:
        trace: list[int] = []
        out = _correct_take_legacy(take_samples, sr, take_notes, ref_notes, alignment, snap_strength, trace)
        return out, _plan_info(take_notes, ref_notes, alignment, strength, config, trace, sr)

    if config.merge_ms > 0:
        take_notes = merge_short_notes(take_notes, config.merge_ms / 1000, config.max_gap_ms / 1000)
        ref_notes = merge_short_notes(ref_notes, config.merge_ms / 1000, config.max_gap_ms / 1000)
    if not take_notes:
        return take_samples.copy(), _plan_info([], ref_notes, alignment, strength, config, [], sr)

    info = _plan_info(take_notes, ref_notes, alignment, strength, config, [], sr)
    ratios, semis = np.array(info["ratios_used"]), np.array(info["semitones"])
    n = len(take_notes)
    adjacent = [take_notes[i + 1].start - take_notes[i].end < config.adjacent_ms / 1000 for i in range(n - 1)]

    # cores tile the timeline where notes touch: the boundary is the middle of the gap
    core_start = [take_notes[0].start]
    core_end = []
    for i in range(n - 1):
        cut = (take_notes[i].end + take_notes[i + 1].start) / 2 if adjacent[i] else None
        core_end.append(cut if cut is not None else take_notes[i].end)
        core_start.append(cut if cut is not None else take_notes[i + 1].start)
    core_end.append(take_notes[-1].end)
    core_len = [max(1, int(round((core_end[i] - core_start[i]) * sr * ratios[i]))) for i in range(n)]

    # overlap (in output samples) at each touching join, bounded by the shorter neighbour
    xfade = max(0, int(config.crossfade_ms * sr / 1000))
    half = [min(xfade // 2, core_len[i] // 2, core_len[i + 1] // 2) if adjacent[i] else 0 for i in range(n - 1)]

    # the shift curve over the take's own timeline
    shift_at = shift_curve(take_notes, semis, config.glide_ms)

    pad = int(config.pad_ms * sr / 1000)
    pieces, join_positions = [], []
    for i, note in enumerate(take_notes):
        h_left = half[i - 1] if i > 0 else 0
        h_right = half[i] if i < n - 1 else 0
        ratio = float(ratios[i])
        ext_left = pad + int(np.ceil(h_left / ratio))
        ext_right = pad + int(np.ceil(h_right / ratio))
        s0 = int(round(core_start[i] * sr))
        s1 = int(round(core_end[i] * sr))
        src = _slice_padded(take_samples, s0 - ext_left, s1 + ext_right)
        target = max(1, int(round(len(src) * ratio)))

        if shift_at is not None:
            t = (np.arange(len(src)) + s0 - ext_left) / sr
            y = _resample_variable_rate(src, shift_at(t))
            processed = _time_stretch(y, target / max(len(y), 1))
        else:
            processed = _pitch_shift(_time_stretch(src, ratio), sr, float(semis[i]))

        lead = int(round(pad * ratio))
        want = h_left + core_len[i] + h_right
        piece = processed[lead : lead + want]
        if len(piece) < want:
            piece = np.pad(piece, (0, want - len(piece)))
        pieces.append(piece.astype(np.float32))

    out: list[np.ndarray] = []
    total = 0
    cursor = 0  # in take samples
    for i, piece in enumerate(pieces):
        if i > 0 and adjacent[i - 1]:
            h = half[i - 1]
            prev = out.pop()
            if h > 0:
                fade_out, fade_in = _fades(2 * h, config.crossfade_shape)
                overlap = prev[-2 * h :] * fade_out + piece[: 2 * h] * fade_in
                out += [prev[: -2 * h], overlap, piece[2 * h :]]
            else:
                out += [prev, piece]
            join_positions.append(total - 2 * h + h)
            total += len(piece) - 2 * h
        else:
            gap = take_samples[cursor : int(round(core_start[i] * sr))]
            out.append(gap)
            out.append(piece)
            total += len(gap) + len(piece)
        cursor = int(round(core_end[i] * sr))

    out.append(take_samples[cursor:])
    info["joins"] = join_positions
    info["crossfade_ms"] = half and 2 * max(half) * 1000 / sr or config.crossfade_ms
    return np.concatenate(out).astype(np.float32), info


def _plan_info(take_notes, ref_notes, alignment, strength, config, trace, sr) -> dict:
    """The per-note decisions: raw and used stretch ratios, and pitch shifts."""
    raw, semis = [], []
    for note in take_notes:
        ref_note = _closest_ref_note(note, ref_notes, alignment)
        target = max(alignment.take_to_ref(note.end) - alignment.take_to_ref(note.start), 1e-3)
        dur = note.end - note.start
        raw.append(target / dur if dur > 0 else 1.0)
        semis.append((ref_note.midi - note.midi) * strength if ref_note else 0.0)
    raw = np.array(raw)
    clamped = np.clip(raw, MIN_STRETCH_RATIO, MAX_STRETCH_RATIO)
    if config.smooth_ratios and len(raw):
        gaps = np.array([take_notes[i + 1].start - take_notes[i].end for i in range(len(take_notes) - 1)])
        used = smooth_ratios(raw, gaps > RUN_BREAK_SECONDS, config.max_ratio_step)
    else:
        used = clamped
    joins: list[int] = []
    if trace:  # legacy: keep only joins between touching notes
        joins = [
            trace[i]
            for i in range(1, len(take_notes))
            if take_notes[i].start - take_notes[i - 1].end < config.adjacent_ms / 1000
        ]
    return {
        "notes": take_notes,
        "ratios_raw": raw.tolist(),
        "ratios_used": used.tolist(),
        "clamp_hits": int(np.sum((raw < MIN_STRETCH_RATIO) | (raw > MAX_STRETCH_RATIO))),
        "semitones": semis,
        "adjacent": [
            take_notes[i + 1].start - take_notes[i].end < config.adjacent_ms / 1000 for i in range(len(take_notes) - 1)
        ],
        "joins": joins,
        "crossfade_ms": FADE_SECONDS * 1000 if config.legacy else config.crossfade_ms,
        "glide_ms": config.glide_ms,
    }


def correct_take(
    take_samples: np.ndarray,
    take_sample_rate: int,
    take_notes: list[Note],
    ref_notes: list[Note],
    alignment: AlignmentResult,
    snap_strength: float,
    config: NotesConfig = DEFAULT_CONFIG,
) -> np.ndarray:
    """snap_strength is 0-100. config selects the legacy or smoothed version."""
    if config.legacy:
        return _correct_take_legacy(take_samples, take_sample_rate, take_notes, ref_notes, alignment, snap_strength)
    return correct_take_debug(take_samples, take_sample_rate, take_notes, ref_notes, alignment, snap_strength, config)[0]
