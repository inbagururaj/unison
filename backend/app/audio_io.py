# Loads any uploaded audio or video file and turns it into a mono wav array.
# We always go through ffmpeg first, even for wav input, so every format
# (webm, mp4, mp3, wav...) ends up decoded the same way at a fixed sample
# rate. That keeps every other module free of format-specific code.

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 22050
MAX_DURATION_SECONDS = 60


class AudioLoadError(ValueError):
    """Raised when an upload cannot be decoded or is out of bounds."""


@dataclass
class LoadedAudio:
    samples: np.ndarray  # mono float32, shape (n,)
    sample_rate: int
    duration_seconds: float


def _run_ffmpeg(src_path: Path, dst_path: Path) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src_path),
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-vn",
        str(dst_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise AudioLoadError(
            "could not decode the uploaded file as audio or video. "
            "check that the file is a valid audio/video file."
        )


def load_upload(raw_bytes: bytes, suffix: str) -> LoadedAudio:
    if not raw_bytes:
        raise AudioLoadError("uploaded file is empty.")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        src_path = tmp_dir / f"input{suffix or '.bin'}"
        dst_path = tmp_dir / "decoded.wav"
        src_path.write_bytes(raw_bytes)

        _run_ffmpeg(src_path, dst_path)

        samples, sr = sf.read(str(dst_path), dtype="float32", always_2d=False)

    if samples.ndim > 1:
        samples = samples.mean(axis=1)

    duration = len(samples) / sr
    if duration < 0.5:
        raise AudioLoadError("audio is too short (must be at least 0.5 seconds).")
    if duration > MAX_DURATION_SECONDS:
        raise AudioLoadError(
            f"audio is too long ({duration:.1f}s). limit is {MAX_DURATION_SECONDS}s."
        )

    return LoadedAudio(samples=samples, sample_rate=sr, duration_seconds=duration)


def write_wav(samples: np.ndarray, sample_rate: int, path: Path) -> None:
    sf.write(str(path), samples, sample_rate)
