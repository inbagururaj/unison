// Hand-coded SVG line chart: x is seconds, y is note name. Each track is
// drawn as one or more path segments, breaking the line wherever the pitch
// is unvoiced (null) so gaps show as gaps rather than a straight line
// across silence. The three tracks arrive already on one shared time axis
// (the backend maps them), and share one pitch scale here. For the autotune
// engine the allowed scale notes are also drawn as dashed guides.
//
// Click-to-inspect: dots every 0.5s along the reference/corrected lines (and
// a full-area click target) let you snap to a time and see the note at that
// point on all three tracks. Clicking also plays the reference and corrected
// audio together from there (see useDualPlayback in App.tsx); a moving
// playhead line reuses the same marker as the static selection line.

import { useMemo } from "react";
import type { PitchTrackData } from "./api";
import type { CheckedTracks } from "./useDualPlayback";
import { midiToNoteName, midiToNoteNameWithCents } from "./pitchNotes";

const MARK_STEP = 0.5; // seconds between clickable/dot marks

interface Track {
  label: string;
  data: PitchTrackData;
  className: string;
}

interface Props {
  reference: PitchTrackData;
  before: PitchTrackData;
  after: PitchTrackData;
  scaleNotes?: number[];
  selectedTime: number | null;
  onSelectTime: (time: number) => void;
  playheadTime: number | null;
  isPlaying: boolean;
  isPaused: boolean;
  checked: CheckedTracks;
  onStop: () => void;
}

const WIDTH = 680;
const HEIGHT = 260;
const PAD_LEFT = 44;
const PAD_RIGHT = 12;
const PAD_TOP = 12;
const PAD_BOTTOM = 28;
const DOT_HIT_RADIUS = 7; // svg units; chart is ~1 unit per css px at this container width, keeping hit targets >= 12px across

function buildPaths(
  data: PitchTrackData,
  xScale: (t: number) => number,
  yScale: (m: number) => number,
): string[] {
  const paths: string[] = [];
  let current: string | null = null;

  for (let i = 0; i < data.times.length; i++) {
    const midi = data.midi[i];
    if (midi === null) {
      current = null;
      continue;
    }
    const x = xScale(data.times[i]);
    const y = yScale(midi);
    if (current === null) {
      current = `M ${x.toFixed(2)} ${y.toFixed(2)}`;
      paths.push(current);
    } else {
      const segment = `L ${x.toFixed(2)} ${y.toFixed(2)}`;
      paths[paths.length - 1] += ` ${segment}`;
    }
  }

  return paths;
}

// Linear interpolation between the two samples bracketing `t`, matching the
// gaps buildPaths draws: null on either side of the bracket means unvoiced.
function valueAtTime(data: PitchTrackData, t: number): number | null {
  const { times, midi } = data;
  if (times.length === 0) return null;
  if (t <= times[0]) return midi[0];
  const last = times.length - 1;
  if (t >= times[last]) return midi[last];

  for (let i = 0; i < last; i++) {
    if (times[i] <= t && t <= times[i + 1]) {
      const a = midi[i];
      const b = midi[i + 1];
      if (a === null || b === null) return null;
      const span = times[i + 1] - times[i];
      if (span <= 0) return a;
      return a + (b - a) * ((t - times[i]) / span);
    }
  }
  return null;
}

function marksAlong(data: PitchTrackData, maxTime: number): { time: number; value: number }[] {
  const marks: { time: number; value: number }[] = [];
  for (let t = 0; t <= maxTime + 1e-6; t += MARK_STEP) {
    const value = valueAtTime(data, t);
    if (value !== null) marks.push({ time: Math.round(t / MARK_STEP) * MARK_STEP, value });
  }
  return marks;
}

function formatReadoutValue(data: PitchTrackData, t: number): string {
  const value = valueAtTime(data, t);
  return value === null ? "-" : midiToNoteNameWithCents(value);
}

export default function PitchChart({
  reference,
  before,
  after,
  scaleNotes = [],
  selectedTime,
  onSelectTime,
  playheadTime,
  isPlaying,
  isPaused,
  checked,
  onStop,
}: Props) {
  const tracks: Track[] = [
    { label: "Reference", data: reference, className: "pitch-line-reference" },
    { label: "Your take", data: before, className: "pitch-line-before" },
    { label: "Corrected", data: after, className: "pitch-line-after" },
  ];

  const allMidi: number[] = [];
  let maxTime = 0.1;
  for (const t of tracks) {
    for (const m of t.data.midi) if (m !== null) allMidi.push(m);
    for (const s of t.data.times) if (s > maxTime) maxTime = s;
  }

  const midiMin = allMidi.length ? Math.floor(Math.min(...allMidi)) - 2 : 0;
  const midiMax = allMidi.length ? Math.ceil(Math.max(...allMidi)) + 2 : 1;

  const xScale = (t: number) =>
    PAD_LEFT + (t / maxTime) * (WIDTH - PAD_LEFT - PAD_RIGHT);
  const yScale = (m: number) =>
    HEIGHT -
    PAD_BOTTOM -
    ((m - midiMin) / (midiMax - midiMin)) * (HEIGHT - PAD_TOP - PAD_BOTTOM);
  const xToTime = (x: number) =>
    ((x - PAD_LEFT) / (WIDTH - PAD_LEFT - PAD_RIGHT)) * maxTime;

  function snapTime(t: number): number {
    const snapped = Math.round(t / MARK_STEP) * MARK_STEP;
    return Math.min(maxTime, Math.max(0, snapped));
  }

  function handleAreaClick(e: React.MouseEvent<SVGRectElement>) {
    const svg = e.currentTarget.ownerSVGElement;
    if (!svg) return;
    const ctm = svg.getScreenCTM();
    if (!ctm) return;
    const point = svg.createSVGPoint();
    point.x = e.clientX;
    point.y = e.clientY;
    const local = point.matrixTransform(ctm.inverse());
    onSelectTime(snapTime(xToTime(local.x)));
  }

  // dot positions only depend on the reference/corrected data and the time
  // range, not on selection or playhead, so they're recomputed only when a
  // new result arrives (not every animation frame during playback).
  const referenceDots = useMemo(() => marksAlong(reference, maxTime), [reference, maxTime]);
  const afterDots = useMemo(() => marksAlong(after, maxTime), [after, maxTime]);

  if (allMidi.length === 0) {
    return <p className="chart-empty">No pitch data to show.</p>;
  }

  const noteStep = midiMax - midiMin > 24 ? 4 : midiMax - midiMin > 14 ? 2 : 1;
  const yTicks: number[] = [];
  for (let m = Math.ceil(midiMin / noteStep) * noteStep; m <= midiMax; m += noteStep) {
    yTicks.push(m);
  }

  const secondsStep = maxTime > 20 ? 5 : maxTime > 8 ? 2 : 1;
  const xTicks: number[] = [];
  for (let s = 0; s <= maxTime; s += secondsStep) xTicks.push(s);

  const markerTime = playheadTime ?? selectedTime;

  return (
    <>
      <svg
        className="pitch-chart"
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        role="img"
        aria-label="Pitch chart comparing the reference, your take, and the corrected take. Click a point to inspect and play from there."
      >
        {yTicks.map((m) => (
          <g key={`y-${m}`}>
            <line
              x1={PAD_LEFT}
              x2={WIDTH - PAD_RIGHT}
              y1={yScale(m)}
              y2={yScale(m)}
              className="chart-gridline"
            />
            <text x={PAD_LEFT - 8} y={yScale(m) + 3} className="chart-axis-label" textAnchor="end">
              {midiToNoteName(m)}
            </text>
          </g>
        ))}

        {scaleNotes
          .filter((m) => m >= midiMin && m <= midiMax)
          .map((m) => (
            <line
              key={`scale-${m}`}
              x1={PAD_LEFT}
              x2={WIDTH - PAD_RIGHT}
              y1={yScale(m)}
              y2={yScale(m)}
              className="chart-scale-line"
            />
          ))}

        {xTicks.map((s) => (
          <text
            key={`x-${s}`}
            x={xScale(s)}
            y={HEIGHT - 8}
            className="chart-axis-label"
            textAnchor="middle"
          >
            {s}s
          </text>
        ))}

        {tracks.map((track) => (
          <g key={track.label}>
            {buildPaths(track.data, xScale, yScale).map((d, i) => (
              <path key={i} d={d} className={track.className} fill="none" />
            ))}
          </g>
        ))}

        {/* full-area hit target first, in the same z-order as everything above:
            individual dots below are drawn after it, so they win precedence at
            their own position, while any other click falls through to this rect */}
        <rect
          x={PAD_LEFT}
          y={PAD_TOP}
          width={WIDTH - PAD_LEFT - PAD_RIGHT}
          height={HEIGHT - PAD_TOP - PAD_BOTTOM}
          className="chart-hit-area"
          onClick={handleAreaClick}
        />

        {markerTime !== null && (
          <line
            x1={xScale(markerTime)}
            x2={xScale(markerTime)}
            y1={PAD_TOP}
            y2={HEIGHT - PAD_BOTTOM}
            className={`chart-marker-line${isPlaying ? " is-playing" : ""}`}
          />
        )}

        {referenceDots.map((d) => (
          <circle
            key={`ref-dot-${d.time}`}
            cx={xScale(d.time)}
            cy={yScale(d.value)}
            r={2.5}
            className="chart-dot chart-dot-reference"
          />
        ))}
        {afterDots.map((d) => (
          <circle
            key={`after-dot-${d.time}`}
            cx={xScale(d.time)}
            cy={yScale(d.value)}
            r={2.5}
            className="chart-dot chart-dot-after"
          />
        ))}
        {[...referenceDots, ...afterDots].map((d, i) => (
          <circle
            key={`hit-${i}-${d.time}`}
            cx={xScale(d.time)}
            cy={yScale(d.value)}
            r={DOT_HIT_RADIUS}
            className="chart-dot-hit-area"
            onClick={() => onSelectTime(d.time)}
          />
        ))}
      </svg>

      {selectedTime !== null && (
        <div className="chart-readout">
          <span className="chart-readout-time">{selectedTime.toFixed(2)}s</span>
          <span className={`chart-readout-item${checked.reference ? "" : " is-dim"}`}>
            <i className="legend-swatch legend-swatch-reference" /> {formatReadoutValue(reference, selectedTime)}
          </span>
          <span className={`chart-readout-item${checked.before ? "" : " is-dim"}`}>
            <i className="legend-swatch legend-swatch-before" /> {formatReadoutValue(before, selectedTime)}
          </span>
          <span className={`chart-readout-item${checked.after ? "" : " is-dim"}`}>
            <i className="legend-swatch legend-swatch-after" /> {formatReadoutValue(after, selectedTime)}
          </span>
          {isPlaying && isPaused && <span className="chart-readout-state">paused</span>}
          {isPlaying && (
            <button type="button" className="button-secondary chart-stop-button" onClick={onStop}>
              Stop
            </button>
          )}
        </div>
      )}
    </>
  );
}
