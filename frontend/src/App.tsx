import { useRef, useState } from "react";
import {
  ApiError,
  detectKey,
  NOTE_NAMES,
  processTake,
  SCALES,
  type DetectedKey,
  type Engine,
  type ProcessResult,
} from "./api";
import FileSlot from "./FileSlot";
import PitchChart from "./PitchChart";

type Status = "idle" | "processing" | "done" | "error";

const ENGINE_HELP: Record<Engine, string> = {
  autotune: "Snaps each moment of your take to the nearest note in the key. Your timing is never changed.",
  retune: "Stretches your take onto the reference's timing and follows the reference's pitch line.",
  notes: "Cuts your take into notes, then stretches and shifts each one, with tiny segments merged and smooth joins and pitch glides.",
  notes_world:
    "The notes engine's pitch decisions applied to the whole take at once through the vocoder: nothing is cut, stretched or stitched, and your voice's timbre is kept.",
  notes_legacy: "The notes engine exactly as it was before the smoothing changes, for comparing.",
};

function sentenceCase(text: string): string {
  return text.charAt(0).toUpperCase() + text.slice(1);
}

export default function App() {
  const [referenceFile, setReferenceFile] = useState<File | null>(null);
  const [takeBlob, setTakeBlob] = useState<File | Blob | null>(null);
  const [takeLabel, setTakeLabel] = useState<string>("");
  const [snapStrength, setSnapStrength] = useState(70);
  const [engine, setEngine] = useState<Engine>("autotune");
  const [retuneSpeedMs, setRetuneSpeedMs] = useState(120);
  const [keyChoice, setKeyChoice] = useState("auto");
  const [scaleChoice, setScaleChoice] = useState("auto");
  const [detectedKey, setDetectedKey] = useState<DetectedKey | null>(null);
  const [keyStatus, setKeyStatus] = useState<"idle" | "detecting" | "failed">("idle");
  const [status, setStatus] = useState<Status>("idle");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [result, setResult] = useState<ProcessResult | null>(null);

  const [isRecording, setIsRecording] = useState(false);
  const [recordSeconds, setRecordSeconds] = useState(0);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<number | null>(null);
  const secondsRef = useRef(0);

  async function startRecording() {
    setErrorMessage(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      chunksRef.current = [];
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };
      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: "audio/webm" });
        setTakeBlob(blob);
        setTakeLabel(`Recording (${secondsRef.current}s)`);
        stream.getTracks().forEach((t) => t.stop());
      };
      recorder.start();
      mediaRecorderRef.current = recorder;
      setIsRecording(true);
      setRecordSeconds(0);
      secondsRef.current = 0;
      timerRef.current = window.setInterval(() => {
        secondsRef.current += 1;
        setRecordSeconds(secondsRef.current);
      }, 1000);
    } catch {
      setErrorMessage("Could not access the microphone. Check browser permissions.");
    }
  }

  function stopRecording() {
    mediaRecorderRef.current?.stop();
    setIsRecording(false);
    if (timerRef.current !== null) {
      window.clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }

  async function chooseReference(file: File | null) {
    setReferenceFile(file);
    setDetectedKey(null);
    if (!file) {
      setKeyStatus("idle");
      return;
    }
    setKeyStatus("detecting");
    try {
      setDetectedKey(await detectKey(file));
      setKeyStatus("idle");
    } catch {
      setKeyStatus("failed");
    }
  }

  async function handleProcess() {
    if (!referenceFile || !takeBlob) return;
    setStatus("processing");
    setErrorMessage(null);
    setResult(null);
    try {
      const data = await processTake(referenceFile, takeBlob, {
        engine,
        snapStrength,
        retuneSpeedMs,
        key: keyChoice,
        scale: scaleChoice,
      });
      setResult(data);
      setStatus("done");
    } catch (err) {
      const message = err instanceof ApiError ? err.message : "Something went wrong processing the audio.";
      setErrorMessage(message);
      setStatus("error");
    }
  }

  const canProcess = referenceFile !== null && takeBlob !== null && status !== "processing";

  return (
    <main className="page">
      <header className="intro">
        <h1>Unison</h1>
        <p>Upload a reference vocal, sing your own take, and get it corrected toward the reference's pitch and timing.</p>
      </header>

      <section className="step">
        <h2>1. Reference vocal</h2>
        <p className="step-help">A vocals-only recording (audio or video) to sing along to.</p>
        <FileSlot
          label="Reference vocal file"
          fileName={referenceFile ? referenceFile.name : ""}
          onChange={chooseReference}
        />
      </section>

      <section className="step">
        <h2>2. Your take</h2>
        <p className="step-help">Record in the browser, or upload a file.</p>
        <div className="row">
          {!isRecording ? (
            <button type="button" className="button-secondary" onClick={startRecording}>
              Record
            </button>
          ) : (
            <button type="button" className="button-secondary" onClick={stopRecording}>
              Stop ({recordSeconds}s)
            </button>
          )}
          <FileSlot
            label="Your take file"
            fileName={takeLabel}
            onChange={(file) => {
              setTakeBlob(file);
              setTakeLabel(file ? file.name : "");
            }}
          />
        </div>
      </section>

      <section className="step">
        <h2>Engine</h2>
        <p className="step-help">{ENGINE_HELP[engine]}</p>
        <select value={engine} onChange={(e) => setEngine(e.target.value as Engine)}>
          <option value="autotune">Autotune (key/scale)</option>
          <option value="retune">Retune (follow reference)</option>
          <option value="notes">Notes (smoothed)</option>
          <option value="notes_world">Notes (vocoder)</option>
          <option value="notes_legacy">Notes (before smoothing)</option>
        </select>
      </section>

      {engine === "autotune" && (
        <section className="step">
          <h2>Key</h2>
          <p className="step-help">
            {keyStatus === "detecting" && "Detecting the key of the reference..."}
            {keyStatus === "failed" && "Could not detect the key of the reference. Pick one below."}
            {keyStatus === "idle" && !detectedKey && "Detected from the reference once you choose one."}
            {keyStatus === "idle" && detectedKey && (
              <>
                Detected: <strong>{detectedKey.name}</strong>
                {Math.abs(detectedKey.tuning_cents) >= 5 &&
                  ` (tuned ${detectedKey.tuning_cents > 0 ? "+" : ""}${detectedKey.tuning_cents} cents from A440)`}
                {detectedKey.confidence < 0.6 && " (low confidence, check it)"}
              </>
            )}
          </p>
          <div className="row">
            <select value={keyChoice} onChange={(e) => setKeyChoice(e.target.value)} aria-label="Key">
              <option value="auto">Auto{detectedKey ? ` (${detectedKey.tonic})` : ""}</option>
              {NOTE_NAMES.map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
            <select value={scaleChoice} onChange={(e) => setScaleChoice(e.target.value)} aria-label="Scale">
              <option value="auto">Auto{detectedKey ? ` (${sentenceCase(detectedKey.scale)})` : ""}</option>
              {SCALES.map((sc) => (
                <option key={sc} value={sc}>
                  {sentenceCase(sc)}
                </option>
              ))}
            </select>
          </div>
        </section>
      )}

      {engine === "autotune" && (
        <section className="step">
          <h2>Retune speed</h2>
          <p className="step-help">
            How fast notes are pulled onto pitch. Fast sounds robotic; slow keeps your vibrato and slides.
          </p>
          <div className="row">
            <span className="range-end">Fast</span>
            <input
              type="range"
              min={0}
              max={400}
              step={10}
              value={retuneSpeedMs}
              onChange={(e) => setRetuneSpeedMs(Number(e.target.value))}
            />
            <span className="range-end">Slow</span>
            <span className="snap-value">{retuneSpeedMs} ms</span>
          </div>
        </section>
      )}

      <section className="step">
        <h2>Strength</h2>
        <p className="step-help">
          {engine === "autotune"
            ? "How much of the distance to the nearest note is corrected."
            : "How closely your take is pulled toward the reference pitch."}
        </p>
        <div className="row">
          <input
            type="range"
            min={0}
            max={100}
            value={snapStrength}
            onChange={(e) => setSnapStrength(Number(e.target.value))}
          />
          <span className="snap-value">{snapStrength}%</span>
        </div>
      </section>

      <section className="step">
        <button type="button" className="button-primary" disabled={!canProcess} onClick={handleProcess}>
          {status === "processing" ? "Processing..." : "Process"}
        </button>
        {status === "error" && errorMessage && <p className="error-text">{errorMessage}</p>}
      </section>

      {status === "processing" && <p className="loading-text">Analyzing and correcting your take...</p>}

      {result && (
        <section className="results">
          <h2>Results</h2>
          <p className="step-help">
            Engine: {result.engine}
            {result.engine === "autotune" && ` · Key: ${sentenceCase(result.key.used.name)}`}
          </p>

          {result.solo_warning.warn && (
            <p className="warning-text">Warning: {sentenceCase(result.solo_warning.reason)}</p>
          )}

          <div className="players">
            <div className="player">
              <span className="player-label">Your take</span>
              <audio controls src={result.take_audio_url} />
            </div>
            <div className="player">
              <span className="player-label">Corrected</span>
              <audio controls src={result.corrected_audio_url} />
            </div>
            <div className="player">
              <span className="player-label">Reference</span>
              <audio controls src={result.reference_audio_url} />
            </div>
          </div>

          <PitchChart
            reference={result.pitch.reference}
            before={result.pitch.before}
            after={result.pitch.after}
            scaleNotes={result.pitch.scale_notes}
          />
          <div className="chart-legend">
            <span className="legend-item"><i className="legend-swatch legend-swatch-reference" /> Reference</span>
            <span className="legend-item"><i className="legend-swatch legend-swatch-before" /> Your take</span>
            <span className="legend-item"><i className="legend-swatch legend-swatch-after" /> Corrected</span>
            {result.pitch.scale_notes.length > 0 && (
              <span className="legend-item"><i className="legend-swatch legend-swatch-scale" /> Scale notes</span>
            )}
          </div>
        </section>
      )}
    </main>
  );
}
