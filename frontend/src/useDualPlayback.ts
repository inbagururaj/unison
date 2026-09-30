// Decodes the reference and corrected audio for a result into AudioBuffers
// once (re-decoding only when the URLs change), then plays both from a
// chosen time on one AudioContext, scheduled at the same start time so they
// cannot drift relative to each other. Assumes the two tracks share a
// timeline (true for the notes_world/full_lock engines this UI exposes,
// which render the corrected audio on the reference's timeline - see
// backend/app/main.py's comment above where it builds the chart's time
// axes).

import { useEffect, useRef, useState } from "react";

interface DualPlayback {
  ready: boolean;
  isPlaying: boolean;
  playheadTime: number | null;
  play: (fromSeconds: number) => void;
  stop: () => void;
}

interface Buffers {
  reference: AudioBuffer;
  corrected: AudioBuffer;
}

interface Sources {
  reference: AudioBufferSourceNode;
  corrected: AudioBufferSourceNode;
}

export function useDualPlayback(referenceUrl: string | null, correctedUrl: string | null): DualPlayback {
  const contextRef = useRef<AudioContext | null>(null);
  const buffersRef = useRef<Buffers | null>(null);
  const sourcesRef = useRef<Sources | null>(null);
  const rafRef = useRef<number | null>(null);
  const sessionRef = useRef(0);

  const [ready, setReady] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
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
      sources.reference.onended = null;
      try {
        sources.reference.stop();
      } catch {
        // already stopped/ended
      }
      try {
        sources.corrected.stop();
      } catch {
        // already stopped/ended
      }
    }
    setIsPlaying(false);
    setPlayheadTime(null);
  }

  // re-decode whenever a new result's URLs arrive; drop any stale in-flight decode
  useEffect(() => {
    setReady(false);
    buffersRef.current = null;
    stopInternal();
    if (!referenceUrl || !correctedUrl) return;

    let cancelled = false;
    const ctx = getContext();

    async function load() {
      const [refBuf, corrBuf] = await Promise.all([
        fetch(referenceUrl as string)
          .then((r) => r.arrayBuffer())
          .then((b) => ctx.decodeAudioData(b)),
        fetch(correctedUrl as string)
          .then((r) => r.arrayBuffer())
          .then((b) => ctx.decodeAudioData(b)),
      ]);
      if (cancelled) return;
      buffersRef.current = { reference: refBuf, corrected: corrBuf };
      setReady(true);
    }

    load().catch(() => {
      // leave ready false; caller just won't offer playback
    });

    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [referenceUrl, correctedUrl]);

  useEffect(() => {
    return () => {
      stopInternal();
      contextRef.current?.close();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function play(fromSeconds: number) {
    const buffers = buffersRef.current;
    if (!buffers) return;
    stopInternal();

    const ctx = getContext();
    void ctx.resume();

    const duration = Math.min(buffers.reference.duration, buffers.corrected.duration);
    const offset = Math.min(Math.max(fromSeconds, 0), duration);
    const playFor = duration - offset;
    if (playFor <= 0) return;

    const session = sessionRef.current;
    const startAt = ctx.currentTime + 0.05; // shared future start time so both sources begin in sync

    const refSource = ctx.createBufferSource();
    refSource.buffer = buffers.reference;
    refSource.connect(ctx.destination);

    const corrSource = ctx.createBufferSource();
    corrSource.buffer = buffers.corrected;
    corrSource.connect(ctx.destination);

    refSource.onended = () => {
      if (sessionRef.current === session) stopInternal();
    };

    refSource.start(startAt, offset, playFor);
    corrSource.start(startAt, offset, playFor);

    sourcesRef.current = { reference: refSource, corrected: corrSource };
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

  return { ready, isPlaying, playheadTime, play, stop };
}
