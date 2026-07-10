import requests
import base64

# Your deployed URL
API_URL = "https://audio-qa-api.onrender.com"

# Create a test request
with open("test_audio.wav", "rb") as f:
    audio_base64 = base64.b64encode(f.read()).decode('utf-8')

response = requests.post(
    f"{API_URL}/answer-audio",
    json={
        "audio_id": "test",
        "audio_base64": audio_base64
    }
)

print("Status:", response.status_code)
print("Response:", response.json())