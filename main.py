
# this backend code is working fine for both speech to text and speech to image
import os
import uuid
import tempfile
from fastapi.staticfiles import StaticFiles
import ffmpeg
import openai
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from typing import Any

app = FastAPI()

# Set your OpenAI API key
openai.api_key = ""

# Serve static files (like index.html)
static_folder = os.path.join(os.getcwd(), "app", "static")
app.mount("/static", StaticFiles(directory=static_folder), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    file_path = os.path.join(static_folder, "index.html")
    try:
        with open(file_path, "r") as f:
            return f.read()
    except Exception as e:
        return {"error": str(e)}

@app.post("/transcribe-audio/")
async def process_audio(file: UploadFile = File(...)):
    try:
        print("speech to text")

        # Check if the file is of type .webm
        if not file.filename.endswith(".webm"):
            raise HTTPException(status_code=400, detail="Invalid file type. Only .webm files are allowed.")
        
        # Save the uploaded audio to a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".webm")
        print("Temporary file path:", temp_file.name)

        # Read the file content and save it to the temporary file
        audio_data = await file.read()
        print("Audio data received. Writing to temporary file...")
        with open(temp_file.name, "wb") as f:
            f.write(audio_data)

        # Convert .webm to .wav using ffmpeg
        unique_filename = f"tmp_{uuid.uuid4().hex}.wav"
        print(f"Unique filename for output: {unique_filename}")
        output_path = os.path.join(tempfile.gettempdir(), unique_filename)
        ffmpeg.input(temp_file.name).output(output_path).run()
        print(f"Converted file saved at: {output_path}")

        # Use OpenAI Whisper API for speech-to-text conversion
        with open(output_path, "rb") as audio_file:
            transcription = openai.Audio.transcribe("whisper-1", audio_file)
        print("Transcription received:", transcription)

        # Extract the transcription text
        user_text = transcription['text']
        print(f"User transcription: {user_text}")

        # Use the transcription to decide if it's a text or image response
        intent_response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Determine if the following user input is requesting an image or a text response. Respond with 'text' or 'image' only."},
                {"role": "user", "content": user_text}
            ]
        )

        # Extract the intent (either text or image)
        intent = intent_response['choices'][0]['message']['content'].strip().lower()
        print(f"Intent determined: {intent}")

        if intent == "image":
            # Generate an image if the user requested one
            image_response = openai.Image.create(
                prompt=user_text,
                n=1,
                size="256x256"
            )
            return {"type": "image", "url": image_response['data'][0]['url'],"content":user_text}
        else:
            # Generate a text response if the user requested a text
            chat_response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": user_text}]
            )
            return {"type": "text", "response": chat_response['choices'][0]['message']['content'],"content":user_text}

    except Exception as e:
        # Log the error message for debugging
        print(f"Error occurred: {str(e)}")
        return JSONResponse(status_code=500, content={"error": f"Internal Server Error: {str(e)}"})
