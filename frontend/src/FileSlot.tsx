// One file slot: a button that opens the hidden native file picker, and the
// chosen file's name beside it. The name is never cut to a fixed length; it
// takes the space available, and CSS ellipsis handles overflow, with the full
// name in the title tooltip. The button reads "Choose file" until this slot
// has a file, then "Replace file"; each slot only knows about its own file.

import { useRef, type ReactNode } from "react";

interface Props {
  label: string; // accessible name of the picker, e.g. "Reference vocal file"
  fileName: string; // "" when the slot is empty
  onChange: (file: File | null) => void;
  extra?: ReactNode; // another way to fill the slot, shown right of the button (e.g. a mic)
}

export function WaveIcon() {
  return (
    <svg className="wave-icon" viewBox="0 0 16 16" width="16" height="16" aria-hidden="true">
      <path
        d="M2 6.5v3M5 4v8M8 2v12M11 5v6M14 6.5v3"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
      />
    </svg>
  );
}

export default function FileSlot({ label, fileName, onChange, extra }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);

  return (
    <div className="file-slot">
      <input
        ref={inputRef}
        type="file"
        accept="audio/*,video/*"
        className="file-input-hidden"
        aria-label={label}
        tabIndex={-1}
        onChange={(e) => {
          onChange(e.target.files?.[0] ?? null);
          e.target.value = ""; // so picking the same file again still fires onChange
        }}
      />
      <button type="button" className="button-secondary" onClick={() => inputRef.current?.click()}>
        <WaveIcon />
        {fileName ? "Replace file" : "Choose file"}
      </button>
      {extra}
      {fileName && (
        <span className="file-chosen" title={fileName}>
          <svg className="file-check" viewBox="0 0 16 16" width="16" height="16" aria-hidden="true">
            <path d="M3 8.5l3.2 3.2L13 4.8" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          <span className="file-name">{fileName}</span>
        </span>
      )}
    </div>
  );
}

export function MicIcon() {
  return (
    <svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true">
      <rect x="6" y="1.5" width="4" height="8" rx="2" fill="currentColor" />
      <path d="M3.5 7.5a4.5 4.5 0 0 0 9 0M8 12v2.5M5.5 14.5h5" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
    </svg>
  );
}
