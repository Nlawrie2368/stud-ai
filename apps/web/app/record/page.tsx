'use client'
import { useState } from 'react'

export default function RecordPage() {
  const [file, setFile] = useState<File | null>(null)
  const [transcript, setTranscript] = useState<string>('')
  const [loading, setLoading] = useState<boolean>(false)
  const [error, setError] = useState<string>('')
  const [summarizing, setSummarizing] = useState<boolean>(false)
  const [summary, setSummary] = useState<string>('')
  const [flashcards, setFlashcards] = useState<any[]>([])

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0])
      setTranscript('')
      setError('')
      setSummary('')
      setFlashcards([])
    }
  }

  const handleTranscribe = async () => {
    if (!file) return
    setLoading(true)
    setError('')
    setTranscript('')
    setSummary('')
    setFlashcards([])

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

  const handleSummarize = async () => {
    if (!transcript) return
    setSummarizing(true)
    setError('')

    try {
      const response = await fetch('/api/summarize', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ transcript })
      })

      if (!response.ok) {
        throw new Error(`Failed to summarize: ${response.statusText}`)
      }

      const data = await response.json()
      setSummary(data.summary || '')
      setFlashcards(data.flashcards || [])
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to summarize')
    } finally {
      setSummarizing(false)
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
        <div style={{ marginTop: '1rem', padding: '1rem', backgroundColor: '#fee', border: '1px solid #fcc', borderRadius: '4px' }}>
          Error: {error}
        </div>
      )}

      {transcript && (
        <div style={{ marginTop: '2rem', padding: '1rem', backgroundColor: '#f9f9f9', border: '1px solid #ddd', borderRadius: '4px' }}>
          <h2>Transcript:</h2>
          <p style={{ whiteSpace: 'pre-wrap' }}>{transcript}</p>
          <button
            onClick={handleSummarize}
            disabled={summarizing}
            style={{
              marginTop: '1rem',
              padding: '0.5rem 1rem',
              backgroundColor: summarizing ? '#ccc' : '#0070f3',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: summarizing ? 'not-allowed' : 'pointer'
            }}
          >
            {summarizing ? 'Summarizing...' : 'Summarize'}
          </button>
        </div>
      )}

      {summary && (
        <div style={{ marginTop: '2rem', padding: '1rem', backgroundColor: '#f0f8ff', border: '1px solid #b0d4ff', borderRadius: '4px' }}>
          <h2>Summary:</h2>
          <p style={{ whiteSpace: 'pre-wrap' }}>{summary}</p>
        </div>
      )}

      {flashcards.length > 0 && (
        <div style={{ marginTop: '2rem', padding: '1rem', backgroundColor: '#f0fff0', border: '1px solid #b0ffb0', borderRadius: '4px' }}>
          <h2>Flashcards:</h2>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            {flashcards.map((card, index) => (
              <div key={index} style={{ padding: '1rem', backgroundColor: 'white', border: '1px solid #ddd', borderRadius: '4px' }}>
                <strong>Q: {card.question || card.front || ''}</strong>
                <p>A: {card.answer || card.back || ''}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </main>
  )
}
