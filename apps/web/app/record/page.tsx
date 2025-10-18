'use client'
import { useEffect, useRef, useState } from 'react'

export default function RecordPage() {
  const [file, setFile] = useState<File | null>(null)
  const [transcript, setTranscript] = useState<string>('')
  const [loading, setLoading] = useState<boolean>(false)
  const [error, setError] = useState<string>('')
  const [summarizing, setSummarizing] = useState<boolean>(false)
  const [summary, setSummary] = useState<string>('')
  const [flashcards, setFlashcards] = useState<any[]>([])
  const [isRecording, setIsRecording] = useState<boolean>(false)
  const [recordingSupported, setRecordingSupported] = useState<boolean>(false)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const recordedChunksRef = useRef<Blob[]>([])

  useEffect(() => {
    setRecordingSupported(typeof window !== 'undefined' && !!navigator.mediaDevices?.getUserMedia)
  }, [])

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

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const mediaRecorder = new MediaRecorder(stream)
      recordedChunksRef.current = []
      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) recordedChunksRef.current.push(e.data)
      }
      mediaRecorder.onstop = () => {
        const blob = new Blob(recordedChunksRef.current, { type: 'audio/webm' })
        const recordedFile = new File([blob], `recording-${Date.now()}.webm`, { type: 'audio/webm' })
        setFile(recordedFile)
      }
      mediaRecorderRef.current = mediaRecorder
      mediaRecorder.start()
      setIsRecording(true)
    } catch (err) {
      setError('Failed to access microphone')
    }
  }

  const stopRecording = () => {
    mediaRecorderRef.current?.stop()
    mediaRecorderRef.current = null
    setIsRecording(false)
  }

  const handleSaveNote = async () => {
    try {
      const response = await fetch('/api/notes', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ transcript, summary, flashcards }),
      })
      if (!response.ok) throw new Error('Failed to save note')
      const data = await response.json()
      alert('Saved! ' + (data.title || data.id))
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to save note')
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
        {recordingSupported && (
          <div style={{ display: 'flex', gap: '1rem', marginBottom: '1rem' }}>
            {!isRecording ? (
              <button onClick={startRecording} style={{ padding: '0.5rem 1rem' }}>Start Recording</button>
            ) : (
              <button onClick={stopRecording} style={{ padding: '0.5rem 1rem', backgroundColor: '#d9534f', color: 'white' }}>Stop Recording</button>
            )}
          </div>
        )}
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
          {summary && (
            <button
              onClick={handleSaveNote}
              style={{
                marginLeft: '1rem',
                marginTop: '1rem',
                padding: '0.5rem 1rem',
                backgroundColor: '#28a745',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer'
              }}
            >
              Save Note
            </button>
          )}
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
