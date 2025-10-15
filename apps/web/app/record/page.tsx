'use client'

import { useState } from 'react'

export default function RecordPage() {
  const [file, setFile] = useState<File | null>(null)

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0])
    }
  }

  const handleTranscribe = () => {
    if (file) {
      alert(`Transcribing: ${file.name}`)
      // TODO: Implement transcription logic
    }
  }

  return (
    <main style={{ padding: '2rem' }}>
      <h1>Audio Transcription</h1>
      <div style={{ marginTop: '2rem' }}>
        <label htmlFor="audio-upload" style={{ display: 'block', marginBottom: '1rem' }}>
          Upload Audio File:
        </label>
        <input
          id="audio-upload"
          type="file"
          accept="audio/*"
          onChange={handleFileChange}
          style={{ display: 'block', marginBottom: '1rem' }}
        />
        {file && <p>Selected: {file.name}</p>}
        <button
          onClick={handleTranscribe}
          disabled={!file}
          style={{
            padding: '0.5rem 1rem',
            backgroundColor: file ? '#0070f3' : '#ccc',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: file ? 'pointer' : 'not-allowed'
          }}
        >
          Transcribe
        </button>
      </div>
    </main>
  )
}
