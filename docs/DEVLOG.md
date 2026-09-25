# Devlog

## m1/m2 - pitch tracking, notes, dtw alignment, note-level shift

Built the core pipeline (`pitch.py`, `notes.py`, `align.py`, `shift.py`)
and verified it against synthetic sine signals with pytest before
touching real audio end to end.

What broke:

- `uv init` scaffolded the backend as an installable package
  (`backend/src/unison_backend/`), which doesn't match how the app is
  actually laid out (`backend/app/`). Fixed by setting
  `[tool.uv] package = false` in `pyproject.toml` and deleting the
  unused `src/` scaffold - this is an app, not a library, it doesn't
  need to be pip-installable.
- The rubberband CLI isn't available via winget on this machine, so
  `pyrubberband` can import fine but fails at runtime when it tries to
  shell out to the `rubberband` binary. `shift.py` catches that and
  falls back to `librosa.effects.pitch_shift`. The shift-accuracy test
  passed using that fallback path, confirming it works well enough on
  its own.
- First cut of `notes.py` chopped fast vibrato into many tiny fake
  notes. Fixed by requiring a pitch change to *hold* for a few frames
  before accepting it as a new note, instead of splitting on any single
  frame that crosses the threshold.

Verified: `uv run pytest` - 6 passed (pitch tracking accuracy, silence
detection, two-note segmentation, single-note stability, shift-by-N-
semitones accuracy, snap-strength-zero leaves pitch unchanged). Ran the
full pipeline on the synthetic samples from the command line: reference
notes measured at exactly 60/62/64/65/67 MIDI, take measured off-pitch
at 58.7/60.6/62.6/63.9/65.9, corrected output measured back at
60/62/64/65/67.

## m3 - api and frontend

Wired `main.py`'s `/api/process` endpoint to the pipeline, then built
the React frontend by hand (no `create-vite` scaffold, since the CLI's
interactive overwrite prompt doesn't work non-interactively against a
pre-created directory - wrote `package.json`, `vite.config.ts`,
`tsconfig.json`, and the source files directly instead).

What broke:

- A leftover uvicorn process from an earlier test run was still holding
  port 8000, so the first "does the built frontend get served"
  check returned 404 - it was hitting the stale process (started
  before `frontend/dist` existed), not the new one. Found it with
  `netstat -ano | grep :8000` and killed it with `taskkill`, then
  confirmed a fresh process on the same port serves the built frontend
  fine.
- `curl -o /tmp/foo.json` from Git Bash writes to the MSYS `/tmp`, which
  isn't the same path a Windows-native `uv run python` sees. Used
  `cygpath -w` to get the real Windows path when reading the file back.

Verified: `npm run build` produces no TypeScript errors. `uv run
uvicorn app.main:app --port 8000` (frontend already built) serves the
app's `index.html`, its JS/CSS assets, and `/api/process` all from one
port. Called `/api/process` against that single production server with
the synthetic samples - got a 200 and a valid corrected-audio URL.

## m4 - polish and docs

Solo-vocal warning, README, and this devlog. No functional changes to
the pipeline; snap strength 100 vs 0 was spot-checked in the m1/m2 tests
already, so the UI just needed to expose the slider correctly.

## m5 - autotune engine (key/scale snapping, timing untouched)

The retune engine still time-warps the take onto the reference, and the
corrected vocal sounded worse than the original take. Added a third
engine, `autotune.py`, modelled on Auto-Tune / AutoPitch: detect the
reference's key, snap the take's pitch to that scale with a retune speed
and strength, and never change timing. It is now the default, and the UI
has an engine dropdown to A/B all three on the same recording.

What broke:

- `pyworld` does not build from source on Python 3.13 on this machine
  (linker failure); `pyworld-prebuilt` ships wheels and exposes the same
  `pyworld` module.
- With vibrato, the target note sometimes locked onto the wrong
  neighbour at the start of a phrase: the first target came from a
  single frame, and a Gaussian-smoothed pitch still carried ~20% of the
  vibrato. Switched to a median filter over ~one vibrato period, picking
  the first target from the first 200 ms, and no note switching within
  half a window of the phrase edges.
- At slow retune speeds note centres were only partly corrected, because
  one smoothing pass mixed each note's error with its neighbours'. Split
  the correction into a per-note offset (always applied) and within-note
  movement (smoothed by retune speed).
- The evaluation itself was misleading at first: pyin reports pitch on a
  10-cent grid (asking for a finer grid broke its voicing detection
  entirely), its 5-frame smoothing hides 6 Hz vibrato, and a short median
  over one vibrato cycle can sit ~25 cents off-centre. `evaluate.py` now
  takes voicing from pyin and pitch values from yin, measures vibrato
  only inside steady notes, and measures note error as the mean over
  each note.
- Timing test: pyin marks exact digital silence as voiced, so comparing
  voicing masks failed on a synthetic take with rests. Compared loudness
  envelopes instead.

Verified (numbers from `scripts/evaluate.py`, default settings: strength
100, speed 120 ms):

- Synthetic voice pair: detected C major correctly. Per-note error 24.9c
  median / 42.9c worst -> 6.2c / 7.9c. 91% of vibrato depth kept. Output
  exactly the same length as the input.
- Retune speed behaves as intended on the synthetic pair: 0 ms leaves
  18% of vibrato (hard snap, notes within 1c), 40 ms 31%, 80 ms 71%,
  120 ms 91%, 200 ms 97%.
- Real recordings (librosa's example files, not committed): a solo
  trumpet (a real player, already mostly in tune: median 4.4c) and a
  read-aloud speech clip. On both, output length was identical and the
  loudness envelope showed 0 ms lag. On the trumpet, per-frame error
  went 4.7c -> 4.3c median at 120 ms and -> 1.3c at 0 ms.
- Cost of the method: WORLD analysis + resynthesis with the pitch left
  unchanged differs from the original by ~4.8 dB (speech) / ~5.5 dB
  (trumpet) log-spectral distance. For scale, a 128 kbps mp3 round trip
  is ~0.5 dB and librosa's phase-vocoder shifter at ~0 semitones is
  ~4.2-4.5 dB. So WORLD is not more transparent than the phase vocoder
  by this measure; its advantage is that it keeps formants when it does
  shift (tested: envelope change under half of librosa's for a
  1-semitone shift).
- Not verified: how any of it sounds on a real singing recording. None
  was available, and nothing here was listened to.
