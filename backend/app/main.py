# The HTTP surface of the app. POST /api/process runs the full pipeline on
# an uploaded reference and take with one of three engines and returns the
# corrected audio plus the data needed to draw the pitch chart:
#
#   autotune  snap the take to the reference's detected key/scale, timing untouched
#   retune    time-warp the take onto the reference and follow its pitch contour
#   notes     the original note-by-note shift and stretch
#
# POST /api/detect-key returns just the detected key of a reference, so the
# UI can show it before processing. In production this same app also serves the built frontend, so a judge can
# run one command and get the whole app on one port.

from __future__ import annotations

import time
import uuid
from pathlib import Path

import librosa
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.align import align
from app.audio_io import SAMPLE_RATE, AudioLoadError, load_upload, write_wav
from app.autotune import DEFAULT_RETUNE_SPEED_MS, NOTE_NAMES, SCALES, KeyEstimate, autotune, detect_key, scale_grid
from app.notes import segment_notes
from app.pitch import PitchTrack, track_pitch
from app.retune import retune
from app.shift import correct_take
from app.solo_check import check_solo_vocal

BACKEND_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BACKEND_DIR / "output"
ENGINES = ("autotune", "retune", "notes")
HQ_SAMPLE_RATE = 44100  # autotune decodes and renders the take at full bandwidth
FRONTEND_DIST = BACKEND_DIR.parent / "frontend" / "dist"

OUTPUT_DIR.mkdir(exist_ok=True)

app = FastAPI(title="unison")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")


def _pitch_track_json(track: PitchTrack) -> dict:
    midi = [None if not v else round(float(m), 3) for m, v in zip(track.midi, track.voiced)]
    times = [round(float(t), 4) for t in track.times]
    return {"times": times, "midi": midi}


def _suffix_for(filename: str | None) -> str:
    if not filename or "." not in filename:
        return ".bin"
    return "." + filename.rsplit(".", 1)[-1].lower()


def _key_json(key: KeyEstimate) -> dict:
    return {
        "tonic": NOTE_NAMES[key.tonic],
        "scale": key.scale,
        "name": key.name,
        "tuning_cents": round(key.tuning_cents, 1),
        "confidence": round(key.confidence, 3),
    }


def _decode(raw: bytes, filename: str | None, sample_rate: int = SAMPLE_RATE):
    try:
        return load_upload(raw, _suffix_for(filename), sample_rate)
    except AudioLoadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/detect-key")
async def detect_key_endpoint(reference: UploadFile = File(...)) -> dict:
    ref_audio = _decode(await reference.read(), reference.filename)
    return _key_json(detect_key(track_pitch(ref_audio.samples, ref_audio.sample_rate)))


@app.post("/api/process")
async def process(
    reference: UploadFile = File(...),
    take: UploadFile = File(...),
    snap_strength: float = Form(50.0),
    engine: str = Form("autotune"),
    key: str = Form("auto"),
    scale: str = Form("auto"),
    retune_speed_ms: float = Form(DEFAULT_RETUNE_SPEED_MS),
) -> dict:
    if engine not in ENGINES:
        raise HTTPException(status_code=400, detail=f"unknown engine {engine!r}. use one of {', '.join(ENGINES)}.")
    if key != "auto" and key not in NOTE_NAMES:
        raise HTTPException(status_code=400, detail=f"unknown key {key!r}.")
    if scale != "auto" and scale not in SCALES:
        raise HTTPException(status_code=400, detail=f"unknown scale {scale!r}.")

    timings: dict[str, float] = {}
    t_start = time.perf_counter()

    def mark(label: str, t_prev: float) -> float:
        now = time.perf_counter()
        timings[label] = round(now - t_prev, 3)
        return now

    ref_audio = _decode(await reference.read(), reference.filename)
    take_bytes = await take.read()
    take_audio = _decode(take_bytes, take.filename)
    # autotune renders the take at full bandwidth (and plays the original back at it)
    take_hq = _decode(take_bytes, take.filename, HQ_SAMPLE_RATE) if engine == "autotune" else take_audio
    t = mark("decode", t_start)

    ref_track = track_pitch(ref_audio.samples, ref_audio.sample_rate)
    take_track = track_pitch(take_audio.samples, take_audio.sample_rate)
    t = mark("pitch_tracking", t)

    solo_result = check_solo_vocal(ref_audio.samples, ref_audio.sample_rate, ref_track)
    t = mark("solo_check", t)

    detected = detect_key(ref_track)
    tonic = detected.tonic if key == "auto" else NOTE_NAMES.index(key)
    used_scale = detected.scale if scale == "auto" else scale
    grid = scale_grid(tonic, used_scale, detected.tuning_cents)
    t = mark("key_detection", t)

    if engine == "autotune":
        corrected = autotune(take_hq.samples, take_hq.sample_rate, grid, snap_strength, retune_speed_ms)
        t = mark("autotune", t)
    else:
        alignment = align(ref_audio.samples, ref_audio.sample_rate, take_audio.samples, take_audio.sample_rate)
        t = mark("alignment", t)
        if engine == "retune":
            corrected = retune(take_audio.samples, ref_audio.samples, take_audio.sample_rate, alignment, snap_strength)
        else:
            corrected = correct_take(
                take_audio.samples,
                take_audio.sample_rate,
                segment_notes(take_track),
                segment_notes(ref_track),
                alignment,
                snap_strength,
            )
        t = mark(engine, t)

    out_rate = take_hq.sample_rate
    corrected_for_tracking = (
        librosa.resample(corrected, orig_sr=out_rate, target_sr=SAMPLE_RATE) if out_rate != SAMPLE_RATE else corrected
    )
    corrected_track = track_pitch(corrected_for_tracking, SAMPLE_RATE)
    t = mark("verify_pitch", t)

    # play back the decoded copies of the inputs too: the browser cannot
    # always decode what was uploaded (video containers, MediaRecorder webm)
    run_id = uuid.uuid4().hex
    output_name = f"{run_id}.wav"
    write_wav(corrected, out_rate, OUTPUT_DIR / output_name)
    write_wav(take_hq.samples, out_rate, OUTPUT_DIR / f"{run_id}_take.wav")
    write_wav(ref_audio.samples, ref_audio.sample_rate, OUTPUT_DIR / f"{run_id}_reference.wav")
    mark("write_output", t)

    timings["total"] = round(time.perf_counter() - t_start, 3)

    lo = np.nanmin(np.concatenate([take_track.midi, corrected_track.midi, [np.inf]]))
    hi = np.nanmax(np.concatenate([take_track.midi, corrected_track.midi, [-np.inf]]))
    visible_grid = [round(float(m), 3) for m in grid if lo - 2 <= m <= hi + 2] if np.isfinite(lo) else []

    return {
        "engine": engine,
        "corrected_audio_url": f"/output/{output_name}",
        "take_audio_url": f"/output/{run_id}_take.wav",
        "reference_audio_url": f"/output/{run_id}_reference.wav",
        "pitch": {
            # autotune never aligns the reference to the take, so drawing it on
            # the take's time axis would be misleading; show the scale instead
            "reference": None if engine == "autotune" else _pitch_track_json(ref_track),
            "before": _pitch_track_json(take_track),
            "after": _pitch_track_json(corrected_track),
            "scale_notes": visible_grid if engine == "autotune" else [],
        },
        "key": {
            "detected": _key_json(detected),
            "used": {"tonic": NOTE_NAMES[tonic], "scale": used_scale, "name": f"{NOTE_NAMES[tonic]} {used_scale}"},
        },
        "solo_warning": {"warn": solo_result.warn, "reason": solo_result.reason, "score": round(solo_result.score, 3)},
        "timings": timings,
    }


if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
