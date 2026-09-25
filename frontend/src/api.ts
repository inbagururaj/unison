// Thin wrapper around the backend endpoints. Kept separate from App.tsx so
// the component file stays focused on UI state.

export interface PitchTrackData {
  times: number[];
  midi: (number | null)[];
}

export interface SoloWarning {
  warn: boolean;
  reason: string;
  score: number;
}

export type Engine = "autotune" | "retune" | "notes";

export const NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
export const SCALES = ["major", "minor", "harmonic minor", "major pentatonic", "minor pentatonic", "chromatic"];

export interface DetectedKey {
  tonic: string;
  scale: string;
  name: string;
  tuning_cents: number;
  confidence: number;
}

export interface ProcessOptions {
  engine: Engine;
  snapStrength: number;
  retuneSpeedMs: number;
  key: string; // "auto" or a note name
  scale: string; // "auto" or one of SCALES
}

export interface ProcessResult {
  engine: Engine;
  corrected_audio_url: string;
  take_audio_url: string;
  reference_audio_url: string;
  pitch: {
    reference: PitchTrackData;
    before: PitchTrackData;
    after: PitchTrackData;
    scale_notes: number[];
  };
  key: {
    detected: DetectedKey;
    used: { tonic: string; scale: string; name: string };
  };
  solo_warning: SoloWarning;
  timings: Record<string, number>;
}

export class ApiError extends Error {}

async function postForm<T>(url: string, form: FormData): Promise<T> {
  const response = await fetch(url, { method: "POST", body: form });

  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try {
      const body = await response.json();
      if (typeof body.detail === "string") message = body.detail.charAt(0).toUpperCase() + body.detail.slice(1);
    } catch {
      // ignore, use default message
    }
    throw new ApiError(message);
  }

  return (await response.json()) as T;
}

function fileName(blob: File | Blob, fallback: string): string {
  return blob instanceof File ? blob.name : fallback;
}

export async function detectKey(reference: File | Blob): Promise<DetectedKey> {
  const form = new FormData();
  form.append("reference", reference, fileName(reference, "reference.webm"));
  return postForm<DetectedKey>("/api/detect-key", form);
}

export async function processTake(
  reference: File | Blob,
  take: File | Blob,
  options: ProcessOptions,
): Promise<ProcessResult> {
  const form = new FormData();
  form.append("reference", reference, fileName(reference, "reference.webm"));
  form.append("take", take, fileName(take, "take.webm"));
  form.append("snap_strength", String(options.snapStrength));
  form.append("engine", options.engine);
  form.append("retune_speed_ms", String(options.retuneSpeedMs));
  form.append("key", options.key);
  form.append("scale", options.scale);
  return postForm<ProcessResult>("/api/process", form);
}
