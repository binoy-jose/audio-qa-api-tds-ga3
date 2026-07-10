from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
import numpy as np
import pandas as pd
import whisper
import os
import tempfile
import re
from scipy.io import wavfile
import io

app = FastAPI()

class AudioRequest(BaseModel):
    audio_id: str
    audio_base64: str

def extract_numbers_from_transcript(transcript: str):
    """
    Extract numeric data from transcribed text.
    This needs to be customized based on your audio content.
    """
    # Look for patterns like "1, 2, 3" or "10.5, 20.3"
    numbers = re.findall(r'\d+\.?\d*', transcript)
    return [float(n) for n in numbers] if numbers else []

def compute_statistics(data):
    """
    Compute all required statistics from the data.
    """
    if not data:
        return {
            "rows": 0,
            "columns": [],
            "mean": {},
            "std": {},
            "variance": {},
            "min": {},
            "max": {},
            "median": {},
            "mode": {},
            "range": {},
            "allowed_values": {},
            "value_range": {},
            "correlation": []
        }
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Basic stats
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
        "allowed_values": {},  # Customize based on your data
        "value_range": {},     # Customize based on your data
        "correlation": df.corr().values.tolist() if len(df.columns) > 1 else []
    }
    
    return stats

@app.post("/answer-audio")
async def process_audio(request: AudioRequest):
    try:
        # 1. Decode base64 audio
        audio_data = base64.b64decode(request.audio_base64)
        
        # 2. Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_data)
            tmp_path = tmp_file.name
        
        try:
            # 3. Load audio to get sample values (for audio statistics)
            sample_rate, audio_samples = wavfile.read(tmp_path)
            
            # 4. Transcribe with Whisper
            model = whisper.load_model("base")
            result = model.transcribe(tmp_path, language="ko")
            transcript = result["text"]
            
            # 5. Extract data from transcript
            # This is the critical part - parse the transcript to get structured data
            numbers = extract_numbers_from_transcript(transcript)
            
            # If we have numbers, convert to structured data
            if numbers:
                # Example: Convert to DataFrame with columns
                # You'll need to adjust this based on actual audio content
                data = [{"value": num} for num in numbers]
            else:
                # Fallback: Use audio sample statistics
                data = [{"sample": float(audio_samples.mean())}]
            
            # 6. Compute statistics
            stats = compute_statistics(data)
            
            return stats
            
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Audio QA API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)