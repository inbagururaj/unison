import { useRef, useState } from "react";
import { ApiError, processTake, type ProcessResult } from "./api";
import PitchChart from "./PitchChart";

type Status = "idle" | "processing" | "done" | "error";

export default function App() {
  const [referenceFile, setReferenceFile] = useState<File | null>(null);
  const [takeBlob, setTakeBlob] = useState<File | Blob | null>(null);
  const [takeLabel, setTakeLabel] = useState<string>("");
  const [snapStrength, setSnapStrength] = useState(70);
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
        setTakeLabel(`recording (${secondsRef.current}s)`);
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
      setErrorMessage("could not access the microphone. check browser permissions.");
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

  async function handleProcess() {
    if (!referenceFile || !takeBlob) return;
    setStatus("processing");
    setErrorMessage(null);
    setResult(null);
    try {
      const data = await processTake(referenceFile, takeBlob, snapStrength);
      setResult(data);
      setStatus("done");
    } catch (err) {
      const message = err instanceof ApiError ? err.message : "something went wrong processing the audio.";
      setErrorMessage(message);
      setStatus("error");
    }
  }

  const canProcess = referenceFile !== null && takeBlob !== null && status !== "processing";

  return (
    <main className="page">
      <header className="intro">
        <h1>unison</h1>
        <p>Upload a reference vocal, sing your own take, and get it corrected toward the reference's pitch and timing.</p>
      </header>

      <section className="step">
        <h2>1. reference</h2>
        <p className="step-help">A vocals-only recording (audio or video) to sing along to.</p>
        <input
          type="file"
          accept="audio/*,video/*"
          onChange={(e) => setReferenceFile(e.target.files?.[0] ?? null)}
        />
        {referenceFile && <p className="file-chosen">chosen: {referenceFile.name}</p>}
      </section>

      <section className="step">
        <h2>2. your take</h2>
        <p className="step-help">Record in the browser, or upload a file.</p>
        <div className="row">
          {!isRecording ? (
            <button type="button" className="button-secondary" onClick={startRecording}>
              record
            </button>
          ) : (
            <button type="button" className="button-secondary" onClick={stopRecording}>
              stop ({recordSeconds}s)
            </button>
          )}
          <input
            type="file"
            accept="audio/*,video/*"
            onChange={(e) => {
              const file = e.target.files?.[0] ?? null;
              setTakeBlob(file);
              setTakeLabel(file ? file.name : "");
            }}
          />
        </div>
        {takeLabel && <p className="file-chosen">chosen: {takeLabel}</p>}
      </section>

      <section className="step">
        <h2>snap strength</h2>
        <p className="step-help">How closely your take is pulled toward the reference pitch.</p>
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
          {status === "processing" ? "processing..." : "process"}
        </button>
        {status === "error" && errorMessage && <p className="error-text">{errorMessage}</p>}
      </section>

      {status === "processing" && <p className="loading-text">running pitch tracking, alignment, and pitch shift...</p>}

      {result && (
        <section className="results">
          <h2>results</h2>

          {result.solo_warning.warn && (
            <p className="warning-text">warning: {result.solo_warning.reason}</p>
          )}

          <div className="players">
            <div className="player">
              <span className="player-label">your take</span>
              <audio controls src={result.take_audio_url} />
            </div>
            <div className="player">
              <span className="player-label">corrected</span>
              <audio controls src={result.corrected_audio_url} />
            </div>
            <div className="player">
              <span className="player-label">reference</span>
              <audio controls src={result.reference_audio_url} />
            </div>
          </div>

          <PitchChart
            reference={result.pitch.reference}
            before={result.pitch.before}
            after={result.pitch.after}
          />
          <div className="chart-legend">
            <span className="legend-item"><i className="legend-swatch legend-swatch-reference" /> reference</span>
            <span className="legend-item"><i className="legend-swatch legend-swatch-before" /> your take</span>
            <span className="legend-item"><i className="legend-swatch legend-swatch-after" /> corrected</span>
          </div>
        </section>
      )}
    </main>
  );
}
