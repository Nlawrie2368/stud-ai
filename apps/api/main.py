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
from supabase import create_client, Client
from datetime import datetime

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

# Initialize Supabase client
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = None

if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

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

class SaveNoteRequest(BaseModel):
    transcript: str
    summary: str
    flashcards: list[Flashcard]
    user_id: Optional[str] = None
    title: Optional[str] = None

# Initialize Whisper model on startup
@app.on_event("startup")
async def startup_event():
    global whisper_model
    if not config.USE_DEEPGRAM:
        print(f"Loading Whisper model: {config.WHISPER_MODEL_SIZE}")
        whisper_model = whisper.load_model(config.WHISPER_MODEL_SIZE)
        print("Whisper model loaded successfully")
    else:
        print("Using Deepgram for transcription")

# Health check endpoint
@app.get("/")
async def health_check():
    return {"status": "healthy", "service": "Audio Transcription API"}

# Transcription helper function for Deepgram
async def transcribe_with_deepgram(file_path: str) -> dict:
    """
    Transcribe audio using Deepgram API.
    """
    if not config.DEEPGRAM_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="Deepgram API key not configured"
        )
    
    url = "https://api.deepgram.com/v1/listen"
    headers = {
        "Authorization": f"Token {config.DEEPGRAM_API_KEY}",
        "Content-Type": "audio/wav"
    }
    
    with open(file_path, "rb") as audio_file:
        response = requests.post(url, headers=headers, data=audio_file)
    
    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code,
            detail=f"Deepgram API error: {response.text}"
        )
    
    result = response.json()
    transcript = result["results"]["channels"][0]["alternatives"][0]["transcript"]
    confidence = result["results"]["channels"][0]["alternatives"][0]["confidence"]
    
    return {
        "text": transcript,
        "confidence": confidence,
        "language": "en",
        "provider": "Deepgram"
    }

# Transcription helper function for local Whisper
async def transcribe_with_whisper(file_path: str) -> dict:
    """
    Transcribe audio using local Whisper model.
    """
    if whisper_model is None:
        raise HTTPException(
            status_code=500,
            detail="Whisper model not loaded"
        )
    
    result = whisper_model.transcribe(file_path)
    
    return {
        "text": result["text"],
        "language": result.get("language"),
        "provider": "Whisper Local"
    }

# Main transcription endpoint
@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Transcribe an uploaded audio file using either Deepgram or local Whisper.
    """
    # Validate file type
    allowed_types = ["audio/wav", "audio/mpeg", "audio/mp3", "audio/m4a", "audio/webm"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Supported types: {', '.join(allowed_types)}"
        )
    
    # Save uploaded file to temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_file_path = temp_file.name
    
    try:
        # Choose transcription method based on configuration
        if config.USE_DEEPGRAM:
            result = await transcribe_with_deepgram(temp_file_path)
        else:
            result = await transcribe_with_whisper(temp_file_path)
        
        return TranscriptionResponse(**result)
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Transcription failed: {str(e)}"
        )
    
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

# Batch transcription endpoint
@app.post("/transcribe/batch")
async def transcribe_batch(files: list[UploadFile] = File(...)):
    """
    Transcribe multiple audio files in a batch.
    """
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
        Flashcard(question="What should you remember?", answer="Placeholder answer 5"),
    ]
    
    return SummarizeResponse(
        summary=summary,
        flashcards=flashcards
    )

# Save note endpoint
@app.post("/saveNote")
async def save_note(request: SaveNoteRequest):
    """
    Save transcript, summary, and flashcards to Supabase database.
    """
    if not supabase:
        raise HTTPException(
            status_code=500,
            detail="Supabase client not configured. Please set SUPABASE_URL and SUPABASE_KEY environment variables."
        )
    
    if not request.transcript or len(request.transcript.strip()) == 0:
        raise HTTPException(status_code=400, detail="Transcript cannot be empty")
    
    if not request.summary or len(request.summary.strip()) == 0:
        raise HTTPException(status_code=400, detail="Summary cannot be empty")
    
    try:
        # Prepare note data
        note_data = {
            "transcript": request.transcript,
            "summary": request.summary,
            "flashcards": [flashcard.dict() for flashcard in request.flashcards],
            "user_id": request.user_id,
            "title": request.title or f"Note - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "created_at": datetime.now().isoformat()
        }
        
        # Insert into Supabase
        response = supabase.table("notes").insert(note_data).execute()
        
        return {
            "status": "success",
            "message": "Note saved successfully",
            "note_id": response.data[0]["id"] if response.data else None,
            "data": response.data[0] if response.data else None
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save note: {str(e)}"
        )

# Configuration endpoint
@app.get("/config")
async def get_config():
    return {
        "provider": "Deepgram" if config.USE_DEEPGRAM else "Whisper Local",
        "whisper_model_size": config.WHISPER_MODEL_SIZE if not config.USE_DEEPGRAM else None,
        "deepgram_configured": bool(config.DEEPGRAM_API_KEY) if config.USE_DEEPGRAM else None,
        "supabase_configured": bool(supabase)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
