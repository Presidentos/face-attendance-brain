from fastapi import FastAPI, File, UploadFile, Form
from deepface import DeepFace
import shutil
import os

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Face Attendance Brain is awake and running!"}

@app.post("/register/")
async def register_face(name: str = Form(...), file: UploadFile = File(...)):
    # Save the uploaded image temporarily
    file_location = f"temp_{file.filename}"
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(file.file, file_object)
    
    try:
        # Extract the facial embedding using DeepFace
        embedding_objs = DeepFace.represent(img_path=file_location, model_name="Facenet", enforce_detection=False)
        embedding = embedding_objs[0]["embedding"]
        
        # Clean up the temp file
        os.remove(file_location)
        
        # We will connect this to Google Sheets in the next step!
        return {
            "status": "success",
            "name": name,
            "message": "Face registered successfully!",
            "embedding_preview": embedding[:5]
        }
    except Exception as e:
        if os.path.exists(file_location):
            os.remove(file_location)
        return {"status": "error", "message": str(e)}

@app.post("/recognize/")
async def recognize_face(file: UploadFile = File(...)):
    # Save the uploaded image temporarily
    file_location = f"temp_{file.filename}"
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(file.file, file_object)
    
    try:
        # Generate embedding for the new image
        embedding_objs = DeepFace.represent(img_path=file_location, model_name="Facenet", enforce_detection=False)
        new_embedding = embedding_objs[0]["embedding"]
        
        # Clean up the temp file
        os.remove(file_location)
        
        # We will compare this against Google Sheets in the next step!
        return {
            "status": "success",
            "message": "Face analyzed successfully!",
            "embedding_preview": new_embedding[:5]
        }
    except Exception as e:
        if os.path.exists(file_location):
            os.remove(file_location)
        return {"status": "error", "message": str(e)}
