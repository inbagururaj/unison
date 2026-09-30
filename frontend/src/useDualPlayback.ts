// Decodes the reference, original and corrected audio for a result into
// AudioBuffers once (re-decoding only when the URLs change), then plays all
// three from a chosen time on one AudioContext, scheduled at the same start
// time so they cannot drift relative to each other. Each track goes through
// its own GainNode: checked = 1, unchecked = 0, so toggling a track while
// playing is just a short gain ramp and never restarts anything. Assumes the two tracks share a
// timeline (true for the notes_world/full_lock engines this UI exposes,
// which render the corrected audio on the reference's timeline - see
// backend/app/main.py's comment above where it builds the chart's time
// axes).

import { useEffect, useRef, useState } from "react";

export type TrackKey = "reference" | "before" | "after";
export type CheckedTracks = Record<TrackKey, boolean>;

const TRACK_KEYS: TrackKey[] = ["reference", "before", "after"];
const GAIN_RAMP_S = 0.02;

interface DualPlayback {
  ready: boolean;
  isPlaying: boolean;
  isPaused: boolean;
  togglePause: () => void;
  playheadTime: number | null;
  play: (fromSeconds: number) => void;
  stop: () => void;
}

type Buffers = Record<TrackKey, AudioBuffer>;

interface Node {
  source: AudioBufferSourceNode;
  gain: GainNode;
}

type Sources = Record<TrackKey, Node>;

export function useDualPlayback(
  referenceUrl: string | null,
  takeUrl: string | null,
  correctedUrl: string | null,
  checked: CheckedTracks,
): DualPlayback {
  const contextRef = useRef<AudioContext | null>(null);
  const buffersRef = useRef<Buffers | null>(null);
  const sourcesRef = useRef<Sources | null>(null);
  const rafRef = useRef<number | null>(null);
  const sessionRef = useRef(0);
  const checkedRef = useRef(checked);
  checkedRef.current = checked;

  const [ready, setReady] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [playheadTime, setPlayheadTime] = useState<number | null>(null);

  function getContext(): AudioContext {
    if (!contextRef.current) contextRef.current = new AudioContext();
    return contextRef.current;
  }

  function stopInternal() {
    sessionRef.current += 1;
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    const sources = sourcesRef.current;
    sourcesRef.current = null;
    if (sources) {
      sources.reference.source.onended = null;
      for (const key of TRACK_KEYS) {
        try {
          sources[key].source.stop();
        } catch {
          // already stopped/ended
        }
        sources[key].source.disconnect();
        sources[key].gain.disconnect();
      }
    }
    setIsPlaying(false);
    setIsPaused(false);
    setPlayheadTime(null);
  }

  // re-decode whenever a new result's URLs arrive; drop any stale in-flight decode
  useEffect(() => {
    setReady(false);
    buffersRef.current = null;
    stopInternal();
    if (!referenceUrl || !takeUrl || !correctedUrl) return;

    let cancelled = false;
    const ctx = getContext();

    async function load() {
      const decode = (url: string) =>
        fetch(url)
          .then((r) => r.arrayBuffer())
          .then((b) => ctx.decodeAudioData(b));
      const [refBuf, takeBuf, corrBuf] = await Promise.all([
        decode(referenceUrl as string),
        decode(takeUrl as string),
        decode(correctedUrl as string),
      ]);
      if (cancelled) return;
      buffersRef.current = { reference: refBuf, before: takeBuf, after: corrBuf };
      setReady(true);
    }

    load().catch(() => {
      // leave ready false; caller just won't offer playback
    });

    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [referenceUrl, takeUrl, correctedUrl]);

  useEffect(() => {
    return () => {
      stopInternal();
      contextRef.current?.close();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // checked-state changes while playing: ramp each gain, no restart
  useEffect(() => {
    const sources = sourcesRef.current;
    const ctx = contextRef.current;
    if (!sources || !ctx) return;
    const now = ctx.currentTime;
    for (const key of TRACK_KEYS) {
      const param = sources[key].gain.gain;
      param.cancelScheduledValues(now);
      param.setValueAtTime(param.value, now);
      param.linearRampToValueAtTime(checked[key] ? 1 : 0, now + GAIN_RAMP_S);
    }
  }, [checked]);

  function play(fromSeconds: number) {
    const buffers = buffersRef.current;
    if (!buffers) return;
    stopInternal();

    const ctx = getContext();
    void ctx.resume();

    const duration = Math.min(
      buffers.reference.duration,
      buffers.before.duration,
      buffers.after.duration,
    );
    const offset = Math.min(Math.max(fromSeconds, 0), duration);
    const playFor = duration - offset;
    if (playFor <= 0) return;

    const session = sessionRef.current;
    const startAt = ctx.currentTime + 0.05; // shared future start time so both sources begin in sync

    const nodes = {} as Sources;
    for (const key of TRACK_KEYS) {
      const source = ctx.createBufferSource();
      source.buffer = buffers[key];
      const gain = ctx.createGain();
      gain.gain.value = checkedRef.current[key] ? 1 : 0;
      source.connect(gain);
      gain.connect(ctx.destination);
      nodes[key] = { source, gain };
    }

    nodes.reference.source.onended = () => {
      if (sessionRef.current === session) stopInternal();
    };

    for (const key of TRACK_KEYS) nodes[key].source.start(startAt, offset, playFor);

    sourcesRef.current = nodes;
    setIsPlaying(true);
    setPlayheadTime(offset);

    function tick() {
      if (sessionRef.current !== session) return;
      const elapsed = ctx.currentTime - startAt;
      if (elapsed < 0) {
        rafRef.current = requestAnimationFrame(tick);
        return;
      }
      const t = offset + elapsed;
      if (t >= duration) {
        return; // onended fires next and clears playhead/isPlaying
      }
      setPlayheadTime(t);
      rafRef.current = requestAnimationFrame(tick);
    }
    rafRef.current = requestAnimationFrame(tick);
  }

  function stop() {
    stopInternal();
  }

  function togglePause() {
    const ctx = contextRef.current;
    if (!ctx || !sourcesRef.current) return;
    if (ctx.state === "running") {
      void ctx.suspend();
      setIsPaused(true);
    } else {
      void ctx.resume();
      setIsPaused(false);
    }
  }

  return { ready, isPlaying, isPaused, togglePause, playheadTime, play, stop };
}
