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

## API (main.py)

`POST /api/process` ties all of the above together: decode both
uploads, track pitch on both, run the solo-vocal check on the
reference, align the take to the reference, segment both into notes,
shift the take, and re-track pitch on the corrected result (so the
"after" line on the chart reflects what was actually produced, not just
what was intended). It returns the corrected audio's URL, the three
pitch curves for the chart, the solo warning, and per-stage timings.
