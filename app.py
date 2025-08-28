from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx, os
from dotenv import load_dotenv
import logging
from fastapi.staticfiles import StaticFiles

# after creating app = FastAPI()



# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is required")

app = FastAPI()
# app.mount("/", StaticFiles(directory="static", html=True), name="static")

# Allow frontend (index.html) to call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://aihumanizerforeveryone.netlify.app/"],  # for dev, allow all
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class HumanizeRequest(BaseModel):
    contents: list
    generationConfig: dict = None
    safetySettings: list = None

@app.post("/humanize")
async def humanize(request: HumanizeRequest):
    try:
        logger.info("Received humanize request")
        
        # Extract the prompt from the request structure
        if not request.contents or not request.contents[0].get('parts'):
            raise HTTPException(status_code=400, detail="Invalid request structure")
        
        prompt = request.contents[0]['parts'][0].get('text', '')
        
        if not prompt.strip():
            raise HTTPException(status_code=400, detail="No text provided")

        # Validate prompt length
        if len(prompt) > 50000:  # Reasonable limit
            raise HTTPException(status_code=400, detail="Text too long. Please use content under 50,000 characters.")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={GEMINI_API_KEY}"

        # Prepare the payload for Gemini API
        payload = {
            "contents": request.contents,
            "generationConfig": request.generationConfig or {
                "temperature": 0.8,
                "maxOutputTokens": 4096,
                "topK": 40,
                "topP": 0.9
            }
        }
        
        # Add safety settings if provided
        if request.safetySettings:
            payload["safetySettings"] = request.safetySettings

        logger.info(f"Making request to Gemini API for text of length: {len(prompt)}")

        async with httpx.AsyncClient(timeout=120) as client:  # Increased timeout
            response = await client.post(url, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                logger.info("Successfully received response from Gemini API")
                return result
            elif response.status_code == 429:
                logger.warning("Rate limit hit")
                raise HTTPException(status_code=429, detail="Rate limit exceeded. Please try again later.")
            else:
                logger.error(f"Gemini API error: {response.status_code} - {response.text}")
                raise HTTPException(status_code=response.status_code, detail=f"API error: {response.text}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
