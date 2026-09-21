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
