// Thin wrapper around the one real backend endpoint. Kept separate from
// App.tsx so the component file stays focused on UI state.

export interface PitchTrackData {
  times: number[];
  midi: (number | null)[];
}

export interface SoloWarning {
  warn: boolean;
  reason: string;
  score: number;
}

export interface ProcessResult {
  corrected_audio_url: string;
  take_audio_url: string;
  reference_audio_url: string;
  pitch: {
    reference: PitchTrackData;
    before: PitchTrackData;
    after: PitchTrackData;
  };
  solo_warning: SoloWarning;
  timings: Record<string, number>;
}

export class ApiError extends Error {}

export async function processTake(
  reference: File | Blob,
  take: File | Blob,
  snapStrength: number,
): Promise<ProcessResult> {
  const form = new FormData();
  form.append("reference", reference, reference instanceof File ? reference.name : "reference.webm");
  form.append("take", take, take instanceof File ? take.name : "take.webm");
  form.append("snap_strength", String(snapStrength));

  const response = await fetch("/api/process", { method: "POST", body: form });

  if (!response.ok) {
    let message = `request failed (${response.status})`;
    try {
      const body = await response.json();
      if (typeof body.detail === "string") message = body.detail;
    } catch {
      // ignore, use default message
    }
    throw new ApiError(message);
  }

  return (await response.json()) as ProcessResult;
}
