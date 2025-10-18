from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tempfile
import os
from typing import Optional
from pydantic import BaseModel
import requests
import json
from datetime import datetime
from pathlib import Path
import threading

try:
    # Optional import; we only use it if Supabase is configured
    from supabase import create_client, Client  # type: ignore
except Exception:  # pragma: no cover - optional dependency at runtime
    create_client = None  # type: ignore
    Client = object  # type: ignore

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

# Global variables for optional providers/clients
whisper_model = None  # lazy loaded
_whisper_lock = threading.Lock()

# OpenAI client (lazy init)
_openai_client = None

# Initialize Supabase client (optional)
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = None
if SUPABASE_URL and SUPABASE_KEY and create_client is not None:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception:
        supabase = None

# Configuration
class TranscriptionConfig:
    USE_DEEPGRAM = os.getenv("USE_DEEPGRAM", "false").lower() == "true"
    DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
    WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    # If an OpenAI key is present, prefer OpenAI transcription by default unless explicitly disabled
    USE_OPENAI_TRANSCRIPTION = os.getenv("USE_OPENAI_TRANSCRIPTION", "auto").lower()
    # data storage (fallback when Supabase is not configured)
    NOTES_FILE = os.getenv("NOTES_FILE")

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
    # Do not eagerly load heavy models; we initialize on first use to keep startup fast
    providers = []
    if should_use_openai_transcription():
        providers.append("OpenAI")
    if config.USE_DEEPGRAM:
        providers.append("Deepgram")
    else:
        providers.append("Whisper Local (lazy)")
    print(f"Transcription providers available: {', '.join(providers)}")

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


# Transcription helper function for OpenAI Whisper API
async def transcribe_with_openai(file_path: str) -> dict:
    """
    Transcribe audio using OpenAI's hosted Whisper API.
    """
    client = get_openai_client()
    if client is None:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")

    try:
        # Newer models may be available; whisper-1 is widely supported
        with open(file_path, "rb") as audio_file:
            result = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
            )
        text = getattr(result, "text", None) or result.get("text") if isinstance(result, dict) else None
        if not text:
            # Fallback: try direct attribute access for SDK objects
            text = result.text  # type: ignore[attr-defined]
        return {
            "text": text,
            "language": "en",
            "provider": "OpenAI Whisper",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI transcription failed: {str(e)}")

# Transcription helper function for local Whisper
async def transcribe_with_whisper(file_path: str) -> dict:
    """
    Transcribe audio using local Whisper model.
    """
    global whisper_model
    if whisper_model is None:
        # Lazy import to avoid requiring whisper/torch if not used
        with _whisper_lock:
            if whisper_model is None:
                try:
                    import whisper  # type: ignore
                except Exception as e:  # pragma: no cover - optional
                    raise HTTPException(status_code=500, detail=f"Whisper not available: {str(e)}")
                try:
                    print(f"Loading Whisper model lazily: {config.WHISPER_MODEL_SIZE}")
                    whisper_model = whisper.load_model(config.WHISPER_MODEL_SIZE)
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Failed to load Whisper model: {str(e)}")

    # At this point whisper_model must be available
    result = whisper_model.transcribe(file_path)  # type: ignore[attr-defined]
    
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
        if should_use_openai_transcription():
            result = await transcribe_with_openai(temp_file_path)
        elif config.USE_DEEPGRAM:
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
    
    client = get_openai_client()

    if client is None:
        # Fallback placeholder if OpenAI is not configured
        summary = (
            f"Summary (placeholder): {request.transcript[:300]}..."
            if len(request.transcript) > 300
            else f"Summary (placeholder): {request.transcript}"
        )
        flashcards = [
            Flashcard(question="What is the main topic?", answer="N/A"),
            Flashcard(question="What are two key points?", answer="N/A"),
            Flashcard(question="Any action items?", answer="N/A"),
        ]
        return SummarizeResponse(summary=summary, flashcards=flashcards)

    try:
        system = (
            "You are an expert note-taker. Summarize the transcript in 5-8 bullet points, "
            "then produce 5-8 study flashcards as JSON with 'question' and 'answer'."
        )
        user = (
            "Summarize this transcript and generate flashcards. "
            "Respond strictly in JSON with keys 'summary' (string) and 'flashcards' (array of {question, answer}).\n\nTranscript:\n" + request.transcript
        )
        completion = client.chat.completions.create(
            model=os.getenv("OPENAI_SUMMARY_MODEL", "gpt-4o-mini"),
            temperature=0.2,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
        )
        content = completion.choices[0].message.content  # type: ignore[index]
        data = json.loads(content)
        raw_summary = data.get("summary") or ""
        raw_flashcards = data.get("flashcards") or []
        flashcards = [Flashcard(**fc) for fc in raw_flashcards if isinstance(fc, dict) and fc.get("question")]
        return SummarizeResponse(summary=raw_summary, flashcards=flashcards)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")

# Save note endpoint
@app.post("/saveNote")
async def save_note_compat(request: SaveNoteRequest):
    """
    Backwards-compatible endpoint to save a note. Uses Supabase if configured,
    otherwise falls back to local JSON file storage.
    """
    created = await create_note(request)
    return {
        "status": "success",
        "message": "Note saved successfully",
        "note_id": created.get("id"),
        "data": created,
    }


# -------------------------
# Notes CRUD (Supabase or local file fallback)
# -------------------------

class NoteModel(BaseModel):
    id: str
    transcript: str
    summary: str
    flashcards: list[Flashcard]
    user_id: Optional[str] = None
    title: Optional[str] = None
    created_at: str


def get_notes_file_path() -> Path:
    if config.NOTES_FILE:
        return Path(config.NOTES_FILE)
    # default to repo-root/data/notes.json
    root = Path(__file__).resolve().parents[2]
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / "notes.json"


_notes_lock = threading.Lock()


def _load_local_notes() -> list[dict]:
    notes_path = get_notes_file_path()
    if not notes_path.exists():
        return []
    try:
        with _notes_lock:
            text = notes_path.read_text(encoding="utf-8")
            return json.loads(text) if text.strip() else []
    except Exception:
        return []


def _save_local_notes(notes: list[dict]) -> None:
    notes_path = get_notes_file_path()
    with _notes_lock:
        notes_path.write_text(json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8")


def should_use_openai_transcription() -> bool:
    # auto: use OpenAI when key is present; 'true' or 'false' override
    mode = config.USE_OPENAI_TRANSCRIPTION
    if mode in ("true", "1", "yes"):  # explicit on
        return bool(config.OPENAI_API_KEY)
    if mode in ("false", "0", "no"):  # explicit off
        return False
    # auto
    return bool(config.OPENAI_API_KEY)


def get_openai_client():
    global _openai_client
    if _openai_client is not None:
        return _openai_client
    if not config.OPENAI_API_KEY:
        return None
    try:
        from openai import OpenAI  # lazy import
        _openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
        return _openai_client
    except Exception:
        return None


def _now_iso() -> str:
    return datetime.now().isoformat()


def _new_id() -> str:
    import uuid
    return str(uuid.uuid4())


async def create_note(payload: SaveNoteRequest) -> dict:
    if not payload.transcript or not payload.summary:
        raise HTTPException(status_code=400, detail="Transcript and summary are required")

    note_data = {
        "id": _new_id(),
        "transcript": payload.transcript,
        "summary": payload.summary,
        "flashcards": [fc.dict() for fc in payload.flashcards],
        "user_id": payload.user_id,
        "title": payload.title or f"Note - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "created_at": _now_iso(),
    }

    if supabase:
        try:
            response = supabase.table("notes").insert(note_data).execute()
            saved = response.data[0] if response and response.data else note_data
            # Normalize id to str
            saved["id"] = str(saved.get("id", note_data["id"]))
            return saved
        except Exception as e:
            # Fallback to local on failure
            print(f"Supabase insert failed, falling back to local: {e}")

    # Local file storage
    notes = _load_local_notes()
    notes.append(note_data)
    _save_local_notes(notes)
    return note_data


@app.get("/notes")
async def list_notes():
    if supabase:
        try:
            resp = supabase.table("notes").select("*").order("created_at", desc=True).execute()
            data = resp.data or []
            for n in data:
                n["id"] = str(n.get("id"))
            return {"notes": data}
        except Exception as e:
            print(f"Supabase list failed, falling back to local: {e}")
    notes = list(reversed(_load_local_notes()))  # newest first
    return {"notes": notes}


@app.post("/notes")
async def create_note_endpoint(request: SaveNoteRequest):
    created = await create_note(request)
    return created


@app.get("/notes/{note_id}")
async def get_note(note_id: str):
    if supabase:
        try:
            resp = supabase.table("notes").select("*").eq("id", note_id).single().execute()
            data = resp.data
            if not data:
                raise HTTPException(status_code=404, detail="Note not found")
            data["id"] = str(data.get("id"))
            return data
        except Exception:
            pass
    for n in _load_local_notes():
        if str(n.get("id")) == note_id:
            return n
    raise HTTPException(status_code=404, detail="Note not found")


class UpdateNoteRequest(BaseModel):
    transcript: Optional[str] = None
    summary: Optional[str] = None
    flashcards: Optional[list[Flashcard]] = None
    title: Optional[str] = None


@app.put("/notes/{note_id}")
async def update_note(note_id: str, request: UpdateNoteRequest):
    if supabase:
        try:
            update_data = {}
            if request.transcript is not None:
                update_data["transcript"] = request.transcript
            if request.summary is not None:
                update_data["summary"] = request.summary
            if request.flashcards is not None:
                update_data["flashcards"] = [fc.dict() for fc in request.flashcards]
            if request.title is not None:
                update_data["title"] = request.title
            if not update_data:
                return {"updated": False}
            resp = supabase.table("notes").update(update_data).eq("id", note_id).execute()
            data = resp.data[0] if resp and resp.data else None
            if not data:
                raise HTTPException(status_code=404, detail="Note not found")
            data["id"] = str(data.get("id"))
            return data
        except Exception:
            pass
    # Local
    notes = _load_local_notes()
    for n in notes:
        if str(n.get("id")) == note_id:
            if request.transcript is not None:
                n["transcript"] = request.transcript
            if request.summary is not None:
                n["summary"] = request.summary
            if request.flashcards is not None:
                n["flashcards"] = [fc.dict() for fc in request.flashcards]
            if request.title is not None:
                n["title"] = request.title
            _save_local_notes(notes)
            return n
    raise HTTPException(status_code=404, detail="Note not found")


@app.delete("/notes/{note_id}")
async def delete_note(note_id: str):
    if supabase:
        try:
            resp = supabase.table("notes").delete().eq("id", note_id).execute()
            return {"deleted": True}
        except Exception:
            pass
    notes = _load_local_notes()
    new_notes = [n for n in notes if str(n.get("id")) != note_id]
    if len(new_notes) == len(notes):
        raise HTTPException(status_code=404, detail="Note not found")
    _save_local_notes(new_notes)
    return {"deleted": True}

# Configuration endpoint
@app.get("/config")
async def get_config():
    return {
        "provider": (
            "OpenAI" if should_use_openai_transcription() else ("Deepgram" if config.USE_DEEPGRAM else "Whisper Local")
        ),
        "whisper_model_size": config.WHISPER_MODEL_SIZE if not config.USE_DEEPGRAM else None,
        "deepgram_configured": bool(config.DEEPGRAM_API_KEY) if config.USE_DEEPGRAM else None,
        "openai_configured": bool(config.OPENAI_API_KEY),
        "supabase_configured": bool(supabase),
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
