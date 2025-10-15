'use client'
import { useState } from 'react'

export default function RecordPage() {
  const [file, setFile] = useState<File | null>(null)
  const [transcript, setTranscript] = useState<string>('')
  const [loading, setLoading] = useState<boolean>(false)
  const [error, setError] = useState<string>('')

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0])
      setTranscript('')
      setError('')
    }
  }

  const handleTranscribe = async () => {
    if (!file) return

    setLoading(true)
    setError('')
    setTranscript('')

    try {
      const formData = new FormData()
      formData.append('file', file)

      const response = await fetch('/api/transcribe', {
        method: 'POST',
        body: formData
      })

      if (!response.ok) {
        throw new Error(`Failed to transcribe: ${response.statusText}`)
      }

      const data = await response.json()
      setTranscript(data.transcript || data.text || JSON.stringify(data))
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to transcribe')
    } finally {
      setLoading(false)
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
          disabled={!file || loading}
          style={{
            padding: '0.5rem 1rem',
            backgroundColor: (file && !loading) ? '#0070f3' : '#ccc',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: (file && !loading) ? 'pointer' : 'not-allowed'
          }}
        >
          {loading ? 'Transcribing...' : 'Transcribe'}
        </button>
      </div>

      {error && (
        <div style={{ marginTop: '2rem', padding: '1rem', backgroundColor: '#fee', border: '1px solid #fcc', borderRadius: '4px' }}>
          <strong>Error:</strong> {error}
        </div>
      )}

      {transcript && (
        <div style={{ marginTop: '2rem', padding: '1rem', backgroundColor: '#f5f5f5', border: '1px solid #ddd', borderRadius: '4px' }}>
          <h2>Transcript:</h2>
          <p style={{ whiteSpace: 'pre-wrap' }}>{transcript}</p>
        </div>
      )}
    </main>
  )
}
