# unison

Upload a vocals-only reference, sing your own take, and get your take
pitch-corrected. Three engines, selectable in the UI so you can A/B them
on the same recording:

- **autotune** (default) - detects the key/scale of the reference and
  snaps your take to the nearest note in it, with a retune speed and a
  strength control. Your timing is never changed.
- **retune** - time-warps your take onto the reference and follows the
  reference's pitch line.
- **notes** - the original engine: splits your take into notes, then
  stretches and pitch-shifts each one.

A pitch chart compares your original take and the corrected result (plus
the reference line, or the scale's notes for autotune).

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
runs an engine on a pair and prints numbers, so tuning changes can be
compared by measurement. Numbers do not tell you whether it sounds
natural - listen to `samples/eval_take.wav` vs `samples/eval_output.wav`
as well.

```sh
cd backend
uv run python ../scripts/make_voice_samples.py
uv run python ../scripts/evaluate.py                                    # autotune, synthetic pair
uv run python ../scripts/evaluate.py --ref r.wav --take t.wav --speed 60 --strength 80
uv run python ../scripts/evaluate.py --key A --scale minor              # override the detected key
uv run python ../scripts/evaluate.py --engine retune                    # the retune engine
```

For autotune it prints: pitch error to the scale per frame and per note
(the per-note number averages vibrato out), how much vibrato survived,
whether length and timing are unchanged, how much the WORLD vocoder
alone alters the sound, and how much of the output is untouched original
audio. Tunable constants are at the top of `backend/app/autotune.py`
(and `retune.py` for the retune engine).

## Run the tests

```sh
cd backend
uv run pytest
```

Tests use synthetic signals to check pitch tracking accuracy, note
segmentation, note shifting, the retune engine, and for autotune: key and
tuning detection, snapping to the scale, strength, exact timing/length
preservation, retune speed vs vibrato, and that formants stay put where a
plain pitch shift moves them.

## Dependencies and why each one is here

Backend (`backend/pyproject.toml`):

- `fastapi` - the web framework behind `/api/process`
- `uvicorn[standard]` - runs the FastAPI app
- `librosa` - pitch tracking (`pyin`), MFCC features for alignment,
  harmonic-percussive separation for the solo-vocal check, and the
  fallback pitch shifter / time stretcher
- `numpy` - array math underlying every audio module
- `soundfile` - reads/writes wav files
- `pyrubberband` - higher-quality pitch shifting via the rubberband CLI
  (notes engine only); falls back to librosa automatically if the CLI is
  missing
- `pyworld-prebuilt` - the WORLD vocoder (split a voice into pitch,
  spectral envelope and aperiodicity, then resynthesize), used by the
  autotune and retune engines. Prebuilt wheels of `pyworld`, which does
  not build from source on Python 3.13 here
- `scipy` - filters used by the autotune/retune engines
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

- Not yet checked by ear on a real singing recording. The autotune engine
  was measured on synthetic voices plus real speech and a real trumpet
  (see the devlog), not on real singing.
- Every voiced phrase that gets corrected goes through the WORLD vocoder,
  which changes the sound even when the pitch is left alone (about as
  much as librosa's phase vocoder, by the log-spectral measure in
  `evaluate.py`). Unvoiced audio and already-in-tune phrases are passed
  through untouched to limit this.
- Autotune snaps to a key/scale. If the melody uses notes outside the
  detected scale (chromatic passing notes, borrowed chords), pick
  `chromatic` or the right scale manually.
- The notes engine can sound robotic, especially at large pitch
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
