# How it works

Plain-language walkthrough of each backend module, in the order the
pipeline runs them. Everything below lives in `backend/app/`.

## 1. audio_io.py - getting audio in a usable shape

Anything the user uploads (webm from browser recording, mp4 video, mp3,
wav, whatever) gets piped through `ffmpeg` as a subprocess, which
converts it to mono wav at a fixed sample rate (22050 Hz). Every other
module only ever deals with that one consistent format, so none of them
need to know or care what the original file was. We also reject files
that are empty, under half a second, or over the 60 second limit.

## 2. pitch.py - tracking the melody

We use librosa's `pyin` algorithm (probabilistic YIN) to estimate the
sung pitch at every moment in the clip. For each small window of audio
(about 23ms) it returns two things: a pitch estimate in Hz, and whether
that window is confidently "voiced" (someone singing a clear pitch) or
not (silence, breath, consonants). We convert Hz to MIDI note numbers
(A4 = 440 Hz = MIDI 69, and each semitone up or down is +1 or -1) because
semitones are what the rest of the app reasons about. We then
median-smooth the pitch a little within each unbroken stretch of singing,
so a single noisy frame doesn't get treated as a real pitch change.

## 3. solo_check.py - is this reference actually a solo voice?

This is a warning, not a gate. It combines three signals into a 0-1
score:

- **voiced ratio**: what fraction of the clip has a confident singing
  pitch. A clip that's mostly silence or noise scores low.
- **percussive energy ratio**: librosa can split audio into a
  "harmonic" part (sustained tones - singing, held notes) and a
  "percussive" part (transients - drums, plosives, noise) using
  harmonic-percussive source separation (HPSS). A lot of percussive
  energy suggests this isn't a clean solo vocal.
- **implausible pitch jump rate**: how often the pitch jumps by more
  than an octave between consecutive frames. A single voice doesn't
  usually leap around like that; overlapping voices or instruments do.

If the combined score is low, the API returns a plain-language reason
(e.g. "there is a lot of percussive/instrumental energy") so the user
understands why, without blocking them from trying anyway.

## 4. align.py - lining up the take's timing with the reference

The singer never sings at exactly the reference's tempo - they rush some
words and drag others. This module uses **dynamic time warping (DTW)**
to figure out the correspondence between "this moment in my take" and
"this moment in the reference."

In plain words: imagine writing out, frame by frame, how different every
moment of the take sounds from every moment of the reference. That's a
big grid (cost matrix). DTW finds the cheapest path through that grid
from the very start to the very end, where "cheapest" means the matched
moments sound as similar as possible, and the path is only allowed to
move forward in time on both sides (you can never match a later moment
in the take to an earlier moment in the reference than you already
matched). Walking that path gives a list of (reference time, take time)
pairs - the *warping path* - describing how the take's timeline bends to
fit the reference's timeline.

We compare **MFCCs** (mel-frequency cepstral coefficients, a standard
feature that captures the rough shape of a sound - closer to "syllable
shape" than to pitch) rather than raw pitch, because MFCCs still carry
useful structure even when the singer is off-key, which is exactly the
case we need alignment to work for. We use librosa's `dtw` function to
compute the cost matrix and the optimal path.

The result is wrapped in `AlignmentResult.take_to_ref(t)`, which
interpolates the warping path to answer "what reference-timeline moment
does take-timeline moment `t` correspond to?" for any `t`, not just the
exact matched frames.

## 5. notes.py - splitting a pitch contour into notes

A pitch track is just a wobbly line over time; a note is a sustained,
mostly-steady pitch. We walk through the track and start a new note
whenever:

- singing stops (an unvoiced gap), or
- the pitch settles on a value more than about 0.7 semitone away from
  the current note's pitch, *and holds there* for a few frames (so a
  quick vibrato wobble or a slide doesn't get chopped into a dozen fake
  notes - it has to actually commit to the new pitch).

Each note is recorded as `(start_time, end_time, median_pitch)`, using
the median so a brief blip inside the note doesn't skew its pitch.

## 6. shift.py - rebuilding the take note by note

Two versions live in this file. The engine dropdown offers both:
"Notes (smoothed)" is the default `DEFAULT_CONFIG`; "Notes (before
smoothing)" is `LEGACY_CONFIG`, the original code left untouched. The
description below is the original; the smoothed version adds, in order
(each stage is a preset in `shift.STAGES`):

1. **Merge short segments** (`notes.merge_short_notes`): fragments of one
   note (neighbours within 60 ms and 0.5 semitone) are joined, and any
   segment under 140 ms is absorbed into the neighbour with the closest
   pitch, so a sustained note is not chopped up.
2. **Smooth stretch ratios**: the per-note ratios are averaged with
   their neighbours, and the change between adjacent notes is limited to
   0.08 (runs are split at rests longer than 150 ms).
3. **Context padding**: each note is stretched and shifted together with
   50 ms of real audio on each side, and the padding is trimmed
   afterwards, so the phase vocoder does not start cold at the edge.
4. **Longer crossfade**: touching notes are tiled at the middle of the
   gap and overlapped by 40 ms with an equal-power fade (was 15 ms
   linear); gaps with silence in them are left alone. The overlap keeps
   the total length, unlike the original.
5. **Pitch glide**: instead of one constant shift per note, the shift is
   interpolated between note centres (a 60 ms ramp centred on each
   join) by reading the audio at a variable rate and then time-stretching
   back to the target length.

For every note in the take:

1. **Find the matching reference note.** We map the take note's
   midpoint through the DTW alignment to get the corresponding moment in
   the reference, then find which reference note contains that moment
   (or is closest to it).
2. **Time-stretch** the note's audio so its duration matches what the
   alignment says the reference note's duration should be, clamped to a
   0.7x-1.4x ratio so we never stretch so much it falls apart.
3. **Pitch-shift** the note toward the reference note's pitch. The full
   semitone difference is scaled by "snap strength" (0% = leave it
   alone, 100% = shift the full difference), using the rubberband CLI
   if it's installed (better quality) or librosa's pitch shifter as a
   fallback.
4. **Crossfade** the reshaped note into the growing output with a short
   (15ms) linear fade, so the join between notes doesn't click.

Any silent gaps between notes are passed through unchanged from the
original take, so timing outside of sung notes isn't altered.

### notes_world.py - the notes decisions, done through the vocoder

The notes engine's stretch-and-shift of separate audio chunks is what
makes it sound choppy. `notes_world.py` keeps its decisions and drops the
audio processing: the whole take is analyzed once with WORLD (pitch,
spectral envelope, aperiodicity, every 5 ms); each take note gets the
same shift as before ((reference note - take note) x strength, tiny
segments merged, a 60 ms glide between notes), added to the pitch frames
only; the frames are warped onto the reference's timeline with the same
smoothed DTW mapping retune uses; and the result is synthesized once.
Voiced/unvoiced status comes from the take, so silence, breaths and
consonants stay as they were, and because the envelope is never touched
the formants do not move however far a note is shifted.

## 7. retune.py - the "retune" engine

An alternative to the note-by-note `shift.py` (the "notes" engine). It
works on the whole take continuously instead of on chopped-up notes:

1. **Decompose** both files with the WORLD vocoder into pitch (f0),
   spectral envelope (timbre/vowel) and aperiodicity (breathiness), every
   5 ms.
2. **Re-time** the take onto the reference's timeline using a smoothed
   version of the DTW path.
3. **Correct pitch**: per frame, compute the gap to the reference pitch
   (folded to the nearest octave, so a singer in another register lands
   on the matching note), smooth that gap over ~120 ms, scale it by snap
   strength and add it to the take's own pitch. Because only the smoothed
   gap is applied, the singer's vibrato and slides survive.
4. **Resynthesize** with the take's original envelope, so the voice keeps
   its character (no chipmunk formant shift) and there are no per-note
   joins to click.

## 8. autotune.py - the "autotune" engine (default)

Works like Auto-Tune or BandLab AutoPitch: snap to a key, never touch
timing. The reference is only used to find the key.

1. **Detect the key.** Count how long the reference spends on each of
   the 12 pitch classes (C, C#, D ...) and compare that histogram with
   the Krumhansl-Kessler profiles, which describe how often each scale
   degree is used in major and minor keys. The best of the 24 matches
   wins. Mixing up a key with its relative minor (C major / A minor) does
   not matter here, because both have the same notes. We also measure
   how far the reference sits from A440 (e.g. +20 cents) and move the
   snap grid to match. The UI shows the result and lets you override key
   and scale (major, minor, harmonic minor, pentatonics, chromatic).
2. **Decompose the take** with WORLD into pitch, spectral envelope and
   aperiodicity every 5 ms, rendered at 44.1 kHz (the other engines use
   22.05 kHz, which drops everything above 11 kHz).
3. **Choose a target note per frame.** The nearest scale note to a
   median-filtered pitch (the median over ~160 ms cancels vibrato but
   keeps note changes sharp), with hysteresis so the target does not
   flicker between two notes.
4. **Correct.** The correction (target - pitch) is split into a per-note
   offset (the note's median error, always applied) and the movement
   within the note (vibrato, scoops, drift, glides). The retune speed
   smooths only the movement: 0 ms flattens it (the robotic effect), and
   slower settings leave it alone while still centring each note.
   Strength scales the whole correction.
5. **Resynthesize** with only the pitch changed, so the frame count,
   formants and breathiness all stay where they were. The output has
   exactly as many samples as the input.
6. **Pass through what does not need changing.** Unvoiced audio
   (consonants, breaths, silence) and phrases already within 3 cents keep
   the original samples, with 20 ms crossfades, because vocoding is never
   perfectly transparent.

## API (main.py)

`POST /api/process` takes the two uploads plus `engine` (autotune,
retune, notes), `snap_strength`, and for autotune `key`, `scale` (or
"auto") and `retune_speed_ms`. It decodes both uploads, tracks pitch,
runs the solo-vocal check, detects the key, runs the chosen engine, and
re-tracks pitch on the corrected result (so the "after" line on the chart
reflects what was actually produced, not just what was intended). It
returns URLs for the corrected audio and decoded copies of both inputs,
the pitch curves for the chart (the reference line is left out for
autotune, since the reference is never aligned to the take, and the
scale's notes are sent instead), the detected and used key, the solo
warning, and per-stage timings.

`POST /api/detect-key` takes just the reference and returns its detected
key, so the UI can show it as soon as the reference is chosen.
