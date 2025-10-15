from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import whisper
import tempfile
import os
from typing import Optional
from pydantic import BaseModel
import requests
import json

# Initialize FastAPI app
app = FastAPI(
    title="Audio Transcription API",
    description="FastAPI backend for audio transcription using Whisper (local or Deepgram)",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for Whisper model
whisper_model = None

# Configuration
class TranscriptionConfig:
    USE_DEEPGRAM = os.getenv("USE_DEEPGRAM", "false").lower() == "true"
    DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
    WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")

config = TranscriptionConfig()

# Response models
class TranscriptionResponse(BaseModel):
    text: str
    language: Optional[str] = None
    confidence: Optional[float] = None
    provider: str

class SummarizeRequest(BaseModel):
    transcript: str

class Flashcard(BaseModel):
    question: str
    answer: str

class SummarizeResponse(BaseModel):
    summary: str
    flashcards: list[Flashcard]

# Initialize Whisper model on startup
@app.on_event("startup")
async def startup_event():
    global whisper_model
    if not config.USE_DEEPGRAM:
        print(f"Loading Whisper model: {config.WHISPER_MODEL_SIZE}")
        whisper_model = whisper.load_model(config.WHISPER_MODEL_SIZE)
        print("Whisper model loaded successfully")
    else:
        print("Using Deepgram API for transcription")

# Health check endpoint
@app.get("/")
async def root():
    return {"message": "Audio Transcription API is running", "provider": "Deepgram" if config.USE_DEEPGRAM else "Whisper Local"}

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "provider": "Deepgram" if config.USE_DEEPGRAM else "Whisper Local"}

# Transcription with Deepgram
async def transcribe_with_deepgram(audio_data: bytes, filename: str) -> TranscriptionResponse:
    if not config.DEEPGRAM_API_KEY:
        raise HTTPException(status_code=500, detail="Deepgram API key not configured")
    
    url = "https://api.deepgram.com/v1/listen"
    headers = {
        "Authorization": f"Token {config.DEEPGRAM_API_KEY}",
        "Content-Type": "audio/wav"  # Adjust based on file type
    }
    
    params = {
        "model": "nova-2",
        "language": "en",
        "smart_format": "true"
    }
    
    try:
        response = requests.post(url, headers=headers, params=params, data=audio_data)
        response.raise_for_status()
        
        result = response.json()
        transcript = result["results"]["channels"][0]["alternatives"][0]["transcript"]
        confidence = result["results"]["channels"][0]["alternatives"][0]["confidence"]
        
        return TranscriptionResponse(
            text=transcript,
            confidence=confidence,
            provider="Deepgram"
        )
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Deepgram API error: {str(e)}")
    except (KeyError, IndexError) as e:
        raise HTTPException(status_code=500, detail=f"Error parsing Deepgram response: {str(e)}")

# Transcription with local Whisper
async def transcribe_with_whisper(audio_data: bytes, filename: str) -> TranscriptionResponse:
    global whisper_model
    
    if whisper_model is None:
        raise HTTPException(status_code=500, detail="Whisper model not loaded")
    
    # Save uploaded file to temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp_file:
        tmp_file.write(audio_data)
        tmp_file_path = tmp_file.name
    
    try:
        # Transcribe with Whisper
        result = whisper_model.transcribe(tmp_file_path)
        
        return TranscriptionResponse(
            text=result["text"].strip(),
            language=result.get("language"),
            provider="Whisper Local"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Whisper transcription error: {str(e)}")
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

# Main transcription endpoint
@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Transcribe audio file using either Whisper (local) or Deepgram API
    
    Supported formats: wav, mp3, mp4, m4a, flac, ogg, webm
    """
    
    # Validate file type
    allowed_extensions = {".wav", ".mp3", ".mp4", ".m4a", ".flac", ".ogg", ".webm"}
    file_extension = os.path.splitext(file.filename)[1].lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file format. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Check file size (limit to 25MB)
    max_file_size = 25 * 1024 * 1024  # 25MB
    audio_data = await file.read()
    
    if len(audio_data) > max_file_size:
        raise HTTPException(status_code=400, detail="File size too large. Maximum 25MB allowed.")
    
    if len(audio_data) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded")
    
    try:
        # Choose transcription method based on configuration
        if config.USE_DEEPGRAM:
            return await transcribe_with_deepgram(audio_data, file.filename)
        else:
            return await transcribe_with_whisper(audio_data, file.filename)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

# Batch transcription endpoint
@app.post("/transcribe/batch")
async def transcribe_batch(files: list[UploadFile] = File(...)):
    """
    Transcribe multiple audio files
    """
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files allowed per batch")
    
    results = []
    for i, file in enumerate(files):
        try:
            result = await transcribe_audio(file)
            results.append({
                "filename": file.filename,
                "index": i,
                "transcription": result.dict(),
                "status": "success"
            })
        except HTTPException as e:
            results.append({
                "filename": file.filename,
                "index": i,
                "error": e.detail,
                "status": "failed"
            })
    
    return {"results": results}

# Summarization endpoint
@app.post("/summarize", response_model=SummarizeResponse)
async def summarize_transcript(request: SummarizeRequest):
    """
    Generate a summary and flashcards from transcript text using OpenAI.
    
    NOTE: This is a placeholder implementation. In production, this should call
    OpenAI's API with proper authentication and error handling.
    """
    
    if not request.transcript or len(request.transcript.strip()) == 0:
        raise HTTPException(status_code=400, detail="Transcript text cannot be empty")
    
    # TODO: Replace with actual OpenAI API call
    # Example OpenAI integration:
    # import openai
    # openai.api_key = os.getenv("OPENAI_API_KEY")
    # 
    # response = openai.ChatCompletion.create(
    #     model="gpt-4",
    #     messages=[
    #         {"role": "system", "content": "You are a helpful assistant that summarizes transcripts and creates flashcards."},
    #         {"role": "user", "content": f"Summarize this transcript and create 5 flashcards: {request.transcript}"}
    #     ]
    # )
    
    # Placeholder response
    summary = f"This is a placeholder summary of the transcript (first 100 chars): {request.transcript[:100]}..."
    
    flashcards = [
        Flashcard(question="What is the main topic?", answer="Placeholder answer 1"),
        Flashcard(question="What are the key points?", answer="Placeholder answer 2"),
        Flashcard(question="What is the conclusion?", answer="Placeholder answer 3"),
        Flashcard(question="What are the important details?", answer="Placeholder answer 4"),
        Flashcard(question="What should you remember?", answer="Placeholder answer 5")
    ]
    
    return SummarizeResponse(
        summary=summary,
        flashcards=flashcards
    )

# Configuration endpoint
@app.get("/config")
async def get_config():
    return {
        "provider": "Deepgram" if config.USE_DEEPGRAM else "Whisper Local",
        "whisper_model_size": config.WHISPER_MODEL_SIZE if not config.USE_DEEPGRAM else None,
        "deepgram_configured": bool(config.DEEPGRAM_API_KEY) if config.USE_DEEPGRAM else None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
