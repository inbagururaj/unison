# Runs a correction engine on a reference/take pair and prints numbers, so
# tuning changes can be compared with measurements instead of by ear. The
# numbers say nothing about how natural the result sounds; listen too.
#
#   cd backend
#   uv run python ../scripts/evaluate.py                          # autotune on the synthetic voice pair
#   uv run python ../scripts/evaluate.py --engine retune
#   uv run python ../scripts/evaluate.py --ref r.wav --take t.wav --strength 80 --speed 120 --key A --scale minor
#
# Pitch is measured independently of WORLD's tracker: voicing from pyin, pitch
# values from yin (continuous; pyin's own values sit on a 10-cent grid, too
# coarse to measure a few cents of error).
#
# autotune metrics:
#   scale error   cents from each voiced frame to the nearest note of the key, before/after
#   notes         per note, cents from its mean pitch to its scale note (vibrato averages out)
#   timing        correlation of the loudness envelope before/after (1.0 = identical timing)
#   vibrato kept  ratio of fast (> ~4 Hz) pitch movement inside steady notes, after vs before
#   vocoder only  how much WORLD analysis+resynthesis with the pitch unchanged alters the
#                 spectrum (log-spectral distance in dB) - the cost of the method itself.
#                 For scale: an mp3 round trip at 128 kbps measures ~0.5 dB, librosa's
#                 phase-vocoder shifter at ~0 semitones ~4.5 dB, WORLD ~5 dB (real speech/trumpet)
#   original      share of the output that is the take's untouched samples (unvoiced audio
#                 and already-in-tune phrases are passed through instead of resynthesized)
# retune metrics: pitch error in cents against the time-aligned reference, before/after.

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import pyworld as pw
import soundfile as sf
from scipy.ndimage import median_filter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app.align import align  # noqa: E402
from app.autotune import NOTE_NAMES, analyze, autotune, detect_key, scale_grid  # noqa: E402
from app.pitch import FMAX, FMIN, FRAME_LENGTH, HOP_LENGTH, PitchTrack, track_pitch  # noqa: E402
from app.retune import retune  # noqa: E402

TRACK_SR = 22050


def _load(path, sr):
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    return y.astype(np.float32)


def _measure(y) -> tuple[PitchTrack, PitchTrack]:
    """(precise, coarse): yin values where they agree with pyin, and pyin's own track."""
    track = track_pitch(y, TRACK_SR)
    f0 = librosa.yin(y, fmin=FMIN, fmax=FMAX, sr=TRACK_SR, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
    yin = librosa.hz_to_midi(f0[: len(track.midi)])
    agree = track.voiced & (np.abs(yin - track.midi) < 0.5)  # drop yin's octave/transition spikes
    precise = PitchTrack(times=track.times, midi=np.where(agree, yin, np.nan), voiced=agree)
    return precise, track


def _report(label: str, cents: np.ndarray) -> None:
    if len(cents) == 0:
        print(f"  {label:8s} no voiced frames")
        return
    print(
        f"  {label:8s} median {np.median(cents):6.1f}c  mean {cents.mean():6.1f}c  "
        f"within 10c {100 * (cents < 10).mean():5.1f}%  within 25c {100 * (cents < 25).mean():5.1f}%  "
        f"({len(cents)} frames)"
    )


def _scale_error_cents(track, grid) -> np.ndarray:
    m = track.midi[track.voiced]
    return np.min(np.abs(m[:, None] - grid[None, :]), axis=1) * 100.0


def _note_centre_error_cents(track, grid) -> np.ndarray:
    """Per note: cents from the note's mean pitch (edges trimmed) to its scale note.

    The mean over a whole note averages vibrato out, so this is what a slow
    retune speed should still fix even though it leaves vibrato alone.
    """
    errs = []
    padded = np.concatenate([[False], track.voiced, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    for s, e in zip(edges[::2], edges[1::2]):
        seg = track.midi[s:e]
        if len(seg) < 21:
            continue
        centre = median_filter(seg, size=9, mode="nearest")
        nearest = grid[np.argmin(np.abs(centre[:, None] - grid[None, :]), axis=1)]
        changes = np.flatnonzero(np.diff(nearest)) + 1
        for a, b in zip(np.r_[0, changes], np.r_[changes, len(seg)]):
            if b - a >= 12:  # notes of at least ~280 ms
                errs.append(abs(seg[a + 3 : b - 3].mean() - nearest[a]) * 100.0)
    return np.array(errs)


def _vibrato_depth(track) -> float:
    """RMS of pitch movement faster than ~4 Hz inside steady notes, in cents.

    Frames near note changes are skipped, so snapping a transition is not
    mistaken for vibrato.
    """
    out = []
    padded = np.concatenate([[False], track.voiced, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    for s, e in zip(edges[::2], edges[1::2]):
        seg = track.midi[s:e]
        if len(seg) < 21:
            continue
        slow = np.convolve(seg, np.ones(9) / 9, mode="same")  # ~200 ms moving average
        slope = np.abs(np.roll(slow, -3) - np.roll(slow, 3))
        steady = slope < 0.3
        steady[:6] = steady[-6:] = False
        out.append((seg - slow)[steady])
    both = np.concatenate(out) if out else np.array([])
    return float(np.sqrt(np.mean(both**2)) * 100) if len(both) else 0.0


def _envelope(y, sr):
    return librosa.feature.rms(y=y, frame_length=int(0.02 * sr), hop_length=int(0.01 * sr))[0]


def _log_spectral_distance_db(a, b, sr):
    n = min(len(a), len(b))
    A = np.abs(librosa.stft(a[:n], n_fft=2048, hop_length=512)) + 1e-6
    B = np.abs(librosa.stft(b[:n], n_fft=2048, hop_length=512)) + 1e-6
    loud = 20 * np.log10(A) > 20 * np.log10(A.max()) - 60  # ignore near-silent bins
    return float(np.mean(np.abs(20 * np.log10(A / B))[loud]))


def run_autotune(args) -> None:
    ref = _load(args.ref, TRACK_SR)
    take_hq = _load(args.take, args.sr)
    take = librosa.resample(take_hq, orig_sr=args.sr, target_sr=TRACK_SR)

    key = detect_key(track_pitch(ref, TRACK_SR))
    tonic = key.tonic if args.key == "auto" else NOTE_NAMES.index(args.key)
    scale = key.scale if args.scale == "auto" else args.scale
    grid = scale_grid(tonic, scale, key.tuning_cents)
    print(
        f"detected {key.name} (tuning {key.tuning_cents:+.1f}c, confidence {key.confidence:.2f}); "
        f"using {NOTE_NAMES[tonic]} {scale}"
    )

    t0 = time.perf_counter()
    analysis = analyze(take_hq, args.sr)
    out = autotune(take_hq, args.sr, grid, args.strength, args.speed, analysis)
    elapsed = time.perf_counter() - t0

    # the vocoder's own cost: resynthesize with the pitch left exactly as it was
    passthrough = pw.synthesize(analysis["f0"], analysis["sp"], analysis["ap"], args.sr, 5.0)[: len(take_hq)]

    out_lo = librosa.resample(out, orig_sr=args.sr, target_sr=TRACK_SR)
    before, after = _measure(take)[0], _measure(out_lo)[0]

    env_in, env_out = _envelope(take_hq, args.sr), _envelope(out, args.sr)
    n = min(len(env_in), len(env_out))

    print(
        f"strength {args.strength:.0f}  speed {args.speed:.0f}ms  take {len(take_hq) / args.sr:.2f}s @ {args.sr} Hz  "
        f"process {elapsed:.1f}s"
    )
    print(f"length   in {len(take_hq)} samples, out {len(out)} samples ({'same' if len(out) == len(take_hq) else 'DIFFERENT'})")
    print(f"timing   loudness-envelope correlation {np.corrcoef(env_in[:n], env_out[:n])[0, 1]:.4f}")
    print("scale error (every frame, vibrato counts as error):")
    _report("before", _scale_error_cents(before, grid))
    _report("after", _scale_error_cents(after, grid))
    nb, na = _note_centre_error_cents(before, grid), _note_centre_error_cents(after, grid)
    if len(nb) and len(na):
        print(
            f"notes    mean pitch of each note vs its scale note: median {np.median(nb):.1f}c -> {np.median(na):.1f}c, "
            f"worst {nb.max():.1f}c -> {na.max():.1f}c ({len(na)} notes)"
        )
    vb, va = _vibrato_depth(before), _vibrato_depth(after)
    print(f"vibrato  depth inside steady notes {vb:.1f}c before, {va:.1f}c after ({100 * va / vb if vb else 0:.0f}% kept)")
    print(
        f"vocoder  WORLD resynthesis with unchanged pitch differs from the original by "
        f"{_log_spectral_distance_db(take_hq, passthrough, args.sr):.2f} dB (log-spectral distance)"
    )
    untouched = np.isclose(out, take_hq, atol=1e-6)
    loud = np.abs(take_hq) > 0.01 * np.abs(take_hq).max()
    print(
        f"original {100 * untouched.mean():.0f}% of output samples are the take's original audio "
        f"({100 * untouched[loud].mean():.0f}% of the non-silent ones)"
    )

    sf.write(str(ROOT / "samples" / "eval_output.wav"), out, args.sr)
    sf.write(str(ROOT / "samples" / "eval_take.wav"), take_hq, args.sr)
    print("wrote samples/eval_take.wav and samples/eval_output.wav (listen to both)")


def run_retune(args) -> None:
    ref = _load(args.ref, TRACK_SR)
    take = _load(args.take, TRACK_SR)
    t0 = time.perf_counter()
    alignment = align(ref, TRACK_SR, take, TRACK_SR)
    out = retune(take, ref, TRACK_SR, alignment, args.strength)
    elapsed = time.perf_counter() - t0

    ref_track, take_track, out_track = (_measure(x)[0] for x in (ref, take, out))

    def errors(times, midi, voiced):
        idx = np.round(np.interp(times, ref_track.times, np.arange(len(ref_track.times)))).astype(int)
        both = voiced & ref_track.voiced[idx]
        return np.abs(midi[both] - ref_track.midi[idx[both]]) * 100.0

    take_on_ref = np.array([alignment.take_to_ref(t) for t in take_track.times])
    print(f"strength {args.strength:.0f}  process {elapsed:.1f}s")
    _report("before", errors(take_on_ref, take_track.midi, take_track.voiced))
    _report("after", errors(out_track.times, out_track.midi, out_track.voiced))
    sf.write(str(ROOT / "samples" / "eval_output.wav"), out, TRACK_SR)
    print("wrote samples/eval_output.wav")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--engine", choices=["autotune", "retune"], default="autotune")
    p.add_argument("--ref", default=ROOT / "samples" / "voice_reference.wav")
    p.add_argument("--take", default=ROOT / "samples" / "voice_take.wav")
    p.add_argument("--strength", type=float, default=100.0)
    p.add_argument("--speed", type=float, default=120.0, help="autotune retune speed in ms")
    p.add_argument("--key", default="auto")
    p.add_argument("--scale", default="auto")
    p.add_argument("--sr", type=int, default=44100, help="autotune render sample rate")
    args = p.parse_args()
    (run_autotune if args.engine == "autotune" else run_retune)(args)


if __name__ == "__main__":
    main()
