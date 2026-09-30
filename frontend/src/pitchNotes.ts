// Converts MIDI note numbers to note names, for chart axis labels and the
// click-to-inspect readout (C4 = MIDI 60).

import { NOTE_NAMES } from "./api";

export function midiToNoteName(midi: number): string {
  const rounded = Math.round(midi);
  const name = NOTE_NAMES[((rounded % 12) + 12) % 12];
  const octave = Math.floor(rounded / 12) - 1;
  return `${name}${octave}`;
}

export function midiToNoteNameWithCents(midi: number): string {
  const rounded = Math.round(midi);
  const cents = Math.round((midi - rounded) * 100);
  const sign = cents >= 0 ? "+" : "";
  return `${midiToNoteName(midi)} ${sign}${cents}c`;
}
