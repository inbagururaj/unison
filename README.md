# unison

Upload a vocals-only reference, sing your own take, and get your take
corrected toward the reference: aligned in time, split into notes, and
pitch-shifted note by note. Shows a pitch chart comparing the reference,
your original take, and the corrected result.

Built for a beginner student hackathon. Everything in scope is listed
below; there is no live/real-time mode, no song downloading, no vocal
separation from full mixes, no accounts, and no database.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (manages the Python version and backend dependencies)
- [Node.js](https://nodejs.org/) with npm (for the frontend)
- [ffmpeg](https://ffmpeg.org/) on your PATH (decodes uploaded audio/video)
- rubberband CLI (optional, higher-quality pitch shifting; the app falls
  back to librosa's pitch shifter automatically if it is not installed)

Versions used during development:

| tool | version |
|---|---|
| Python | 3.13.15 (installed and managed by uv) |
| uv | 0.12.17 |
| Node.js | v24.15.0 |
| ffmpeg | 9.0.1 |
| React | 19.3.0 |
| Vite | 8.3.0 |
| TypeScript | 7.0.2 |
| FastAPI | 0.141.1 |
| librosa | 1.0.0 |

## Run it (production, one command per side)

```sh
# build the frontend once
cd frontend
npm install
npm run build

# run the backend, which also serves the built frontend
cd ../backend
uv sync
uv run uvicorn app.main:app --port 8000
```

Open http://127.0.0.1:8000

## Run it (development, two servers)

```sh
# terminal 1: backend
cd backend
uv sync
uv run uvicorn app.main:app --reload --port 8000

# terminal 2: frontend (proxies /api and /output to the backend)
cd frontend
npm install
npm run dev
```

Open http://localhost:5173

## Generate the synthetic demo samples

We never ship real recorded audio. `samples/` holds only synthetic
sine-wave audio: a reference melody and a deliberately off-pitch,
mistimed version of it as the take.

```sh
cd backend
uv run python ../scripts/make_samples.py
```

## Measure quality

`make_voice_samples.py` builds a more voice-like pair (vowel formants,
vibrato, glides, breath noise, an off-key drifting take). `evaluate.py`
runs the corrector on it and prints pitch error in cents against the
reference, before and after, so tuning changes can be compared by number:

```sh
cd backend
uv run python ../scripts/make_voice_samples.py
uv run python ../scripts/evaluate.py                      # default sample pair, snap 100
uv run python ../scripts/evaluate.py ref.wav take.wav 70  # your own files, snap 70
```

Tunable constants live at the top of `backend/app/retune.py`
(`RETUNE_SMOOTH_MS`, `TIMING_SMOOTH_SECONDS`, `REF_MEDIAN_FRAMES`).

## Run the tests

```sh
cd backend
uv run pytest
```

Tests use synthetic sine signals to check pitch tracking accuracy, note
segmentation, and that shifting a note by N semitones measures N
semitones afterward.

## Dependencies and why each one is here

Backend (`backend/pyproject.toml`):

- `fastapi` - the web framework behind `/api/process`
- `uvicorn[standard]` - runs the FastAPI app
- `librosa` - pitch tracking (`pyin`), MFCC features for alignment,
  harmonic-percussive separation for the solo-vocal check, and the
  fallback pitch shifter / time stretcher
- `numpy` - array math underlying every audio module
- `soundfile` - reads/writes wav files
- `pyrubberband` - higher-quality pitch shifting via the rubberband CLI;
  falls back to librosa automatically if the CLI is missing
- `python-multipart` - required by FastAPI to parse the multipart file
  upload
- `pytest` (dev only) - test runner

Frontend (`frontend/package.json`):

- `react`, `react-dom` - UI framework
- `vite` - dev server and production bundler
- `@vitejs/plugin-react` - React fast refresh for Vite
- `typescript` - type checking (strict mode)

No CSS framework, component library, icon library, or chart library -
the UI is hand-written CSS and the pitch chart is hand-coded SVG.

## AI assistance disclosure

Claude Code was used for scaffolding and implementation of this project,
including the dynamic time warping alignment in `backend/app/align.py`.

## Known limitations

- The corrected audio can sound robotic, especially at large pitch
  shifts or stretch ratios near the 0.7-1.4 clamp - this is a known
  tradeoff of phase-vocoder-based pitch/time processing, more so
  without the rubberband CLI installed.
- Works best with a clean, single solo vocal as the reference.
  `solo_check.py` gives a plain-language warning (not a rejection) when
  the reference does not look like one.
- Uploads are capped at 60 seconds.
- Time alignment (`align.py`) compares MFCC features between the
  reference and the take with dynamic time warping; it is not tuned for
  references with heavy instrumental accompaniment.
- Browser recording requires microphone permission and a modern browser
  (`MediaRecorder` support).

## Project structure

```
backend/app/       fastapi app and the audio pipeline
backend/tests/      pytest tests on synthetic signals
frontend/src/        react + typescript ui
samples/              synthetic demo audio (generated, not committed audio you can't regenerate)
scripts/              make_samples.py / make_voice_samples.py generate samples/, evaluate.py scores the corrector
docs/DEVLOG.md        what broke and how it was fixed, per milestone
docs/HOW_IT_WORKS.md  plain-language explanation of each module
```
