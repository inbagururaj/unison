// One file slot: a button that opens the hidden native file picker, and the
// chosen file's name beside it. The name is never cut to a fixed length; it
// takes the space available, and CSS ellipsis handles overflow, with the full
// name in the title tooltip. The button reads "Choose file" until this slot
// has a file, then "Replace file"; each slot only knows about its own file.

import { useRef } from "react";

interface Props {
  label: string; // accessible name of the picker, e.g. "Reference vocal file"
  fileName: string; // "" when the slot is empty
  onChange: (file: File | null) => void;
}

export default function FileSlot({ label, fileName, onChange }: Props) {
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
        {fileName ? "Replace file" : "Choose file"}
      </button>
      {fileName && (
        <span className="file-name" title={fileName}>
          {fileName}
        </span>
      )}
    </div>
  );
}
