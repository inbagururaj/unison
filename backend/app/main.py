# The HTTP surface of the app. One real endpoint, POST /api/process, which
# runs the full pipeline (decode -> pitch track -> solo check -> align ->
# retune) on an uploaded reference and take, and returns
# the corrected audio plus the data needed to draw the pitch chart. In
# production this same app also serves the built frontend, so a judge can
# run one command and get the whole app on one port.

from __future__ import annotations

import time
import uuid
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.align import align
from app.audio_io import AudioLoadError, load_upload, write_wav
from app.pitch import PitchTrack, track_pitch
from app.retune import retune
from app.solo_check import check_solo_vocal

BACKEND_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BACKEND_DIR / "output"
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


@app.post("/api/process")
async def process(
    reference: UploadFile = File(...),
    take: UploadFile = File(...),
    snap_strength: float = Form(50.0),
) -> dict:
    timings: dict[str, float] = {}
    t_start = time.perf_counter()

    def mark(label: str, t_prev: float) -> float:
        now = time.perf_counter()
        timings[label] = round(now - t_prev, 3)
        return now

    try:
        ref_bytes = await reference.read()
        take_bytes = await take.read()
        ref_audio = load_upload(ref_bytes, _suffix_for(reference.filename))
        take_audio = load_upload(take_bytes, _suffix_for(take.filename))
    except AudioLoadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    t = mark("decode", t_start)

    ref_track = track_pitch(ref_audio.samples, ref_audio.sample_rate)
    take_track = track_pitch(take_audio.samples, take_audio.sample_rate)
    t = mark("pitch_tracking", t)

    solo_result = check_solo_vocal(ref_audio.samples, ref_audio.sample_rate, ref_track)
    t = mark("solo_check", t)

    alignment = align(
        ref_audio.samples, ref_audio.sample_rate, take_audio.samples, take_audio.sample_rate
    )
    t = mark("alignment", t)

    corrected = retune(
        take_audio.samples,
        ref_audio.samples,
        take_audio.sample_rate,
        alignment,
        snap_strength,
    )
    t = mark("retune", t)

    corrected_track = track_pitch(corrected, take_audio.sample_rate)
    t = mark("verify_pitch", t)

    # play back the decoded copies of the inputs too: the browser cannot
    # always decode what was uploaded (video containers, MediaRecorder webm)
    run_id = uuid.uuid4().hex
    output_name = f"{run_id}.wav"
    write_wav(corrected, take_audio.sample_rate, OUTPUT_DIR / output_name)
    write_wav(take_audio.samples, take_audio.sample_rate, OUTPUT_DIR / f"{run_id}_take.wav")
    write_wav(ref_audio.samples, ref_audio.sample_rate, OUTPUT_DIR / f"{run_id}_reference.wav")
    mark("write_output", t)

    timings["total"] = round(time.perf_counter() - t_start, 3)

    return {
        "corrected_audio_url": f"/output/{output_name}",
        "take_audio_url": f"/output/{run_id}_take.wav",
        "reference_audio_url": f"/output/{run_id}_reference.wav",
        "pitch": {
            "reference": _pitch_track_json(ref_track),
            "before": _pitch_track_json(take_track),
            "after": _pitch_track_json(corrected_track),
        },
        "solo_warning": {"warn": solo_result.warn, "reason": solo_result.reason, "score": round(solo_result.score, 3)},
        "timings": timings,
    }


if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
