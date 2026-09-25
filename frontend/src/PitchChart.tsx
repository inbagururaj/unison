// Hand-coded SVG line chart: x is seconds, y is note name. Each track is
// drawn as one or more path segments, breaking the line wherever the pitch
// is unvoiced (null) so gaps show as gaps rather than a straight line
// across silence. The three tracks arrive already on one shared time axis
// (the backend maps them), and share one pitch scale here. For the autotune
// engine the allowed scale notes are also drawn as dashed guides.

import type { PitchTrackData } from "./api";

const NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];

function midiToNoteName(midi: number): string {
  const rounded = Math.round(midi);
  const name = NOTE_NAMES[((rounded % 12) + 12) % 12];
  const octave = Math.floor(rounded / 12) - 1;
  return `${name}${octave}`;
}

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
}

const WIDTH = 680;
const HEIGHT = 260;
const PAD_LEFT = 44;
const PAD_RIGHT = 12;
const PAD_TOP = 12;
const PAD_BOTTOM = 28;

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

export default function PitchChart({ reference, before, after, scaleNotes = [] }: Props) {
  const tracks: Track[] = [
    { label: "reference", data: reference, className: "pitch-line-reference" },
    { label: "your take", data: before, className: "pitch-line-before" },
    { label: "corrected", data: after, className: "pitch-line-after" },
  ];

  const allMidi: number[] = [];
  let maxTime = 0.1;
  for (const t of tracks) {
    for (const m of t.data.midi) if (m !== null) allMidi.push(m);
    for (const s of t.data.times) if (s > maxTime) maxTime = s;
  }

  if (allMidi.length === 0) {
    return <p className="chart-empty">no pitch data to show.</p>;
  }

  const midiMin = Math.floor(Math.min(...allMidi)) - 2;
  const midiMax = Math.ceil(Math.max(...allMidi)) + 2;

  const xScale = (t: number) =>
    PAD_LEFT + (t / maxTime) * (WIDTH - PAD_LEFT - PAD_RIGHT);
  const yScale = (m: number) =>
    HEIGHT -
    PAD_BOTTOM -
    ((m - midiMin) / (midiMax - midiMin)) * (HEIGHT - PAD_TOP - PAD_BOTTOM);

  const noteStep = midiMax - midiMin > 24 ? 4 : midiMax - midiMin > 14 ? 2 : 1;
  const yTicks: number[] = [];
  for (let m = Math.ceil(midiMin / noteStep) * noteStep; m <= midiMax; m += noteStep) {
    yTicks.push(m);
  }

  const secondsStep = maxTime > 20 ? 5 : maxTime > 8 ? 2 : 1;
  const xTicks: number[] = [];
  for (let s = 0; s <= maxTime; s += secondsStep) xTicks.push(s);

  return (
    <svg
      className="pitch-chart"
      viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
      role="img"
      aria-label="pitch chart comparing reference, your take, and the corrected take"
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
    </svg>
  );
}
