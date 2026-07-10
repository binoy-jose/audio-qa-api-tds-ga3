from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import base64
import numpy as np
import pandas as pd
import whisper
import os
import tempfile
import re
from scipy.io import wavfile
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class AudioRequest(BaseModel):
    audio_id: str
    audio_base64: str

# ============= MAIN ENDPOINT - HANDLES ROOT PATH =============

@app.post("/")  # THIS IS THE CRITICAL ONE!
@app.post("")   # Also handle empty path
async def process_audio(request: AudioRequest):
    try:
        logger.info(f"Processing audio_id: {request.audio_id}")
        
        # Decode base64 audio
        audio_data = base64.b64decode(request.audio_base64)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_data)
            tmp_path = tmp_file.name
        
        try:
            # Load audio
            sample_rate, audio_samples = wavfile.read(tmp_path)
            
            # Transcribe with Whisper (use tiny model)
            model = whisper.load_model("tiny")
            result = model.transcribe(tmp_path, language="ko")
            transcript = result["text"]
            logger.info(f"Transcript: {transcript[:100]}...")
            
            # Extract numbers from transcript
            numbers = re.findall(r'\d+\.?\d*', transcript)
            
            if numbers:
                data = [{"value": float(num)} for num in numbers]
            else:
                data = [{"sample": float(np.mean(audio_samples))}]
            
            # Compute statistics
            df = pd.DataFrame(data)
            
            stats = {
                "rows": len(df),
                "columns": df.columns.tolist(),
                "mean": df.mean().to_dict(),
                "std": df.std().to_dict(),
                "variance": df.var().to_dict(),
                "min": df.min().to_dict(),
                "max": df.max().to_dict(),
                "median": df.median().to_dict(),
                "mode": df.mode().iloc[0].to_dict() if not df.mode().empty else {},
                "range": (df.max() - df.min()).to_dict(),
                "allowed_values": {},
                "value_range": {},
                "correlation": df.corr().values.tolist() if len(df.columns) > 1 else []
            }
            
            logger.info(f"Returning stats: {stats}")
            return stats
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                
    except Exception as e:
        logger.error(f"Error processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============= ALSO HANDLE OTHER PATHS (JUST IN CASE) =============

@app.post("/answer-audio")
@app.post("/answer")
@app.post("/audio")
async def other_endpoints(request: AudioRequest):
    return await process_audio(request)

# ============= HEALTH CHECK =============

@app.get("/")
@app.get("/health")
async def root():
    return {"message": "Audio QA API is running"}

@app.get("/debug")
async def debug():
    return {
        "message": "Audio QA API is running",
        "endpoints": [
            "POST / (main)",
            "POST /answer-audio",
            "POST /answer",
            "POST /audio",
            "GET /health"
        ]
    }