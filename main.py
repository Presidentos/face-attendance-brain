from fastapi import FastAPI, File, UploadFile, Form
from deepface import DeepFace
import shutil
import os
import gspread
import numpy as np
from datetime import datetime

app = FastAPI()

# Connect to Google Sheets using the secret file we set up in Render
SHEET_URL = "https://docs.google.com/spreadsheets/d/1-8NFZYq5ZwOTTAzezRPBYZvP9HdEwz_Modg7tV0gPPw/edit"
gc = gspread.service_account(filename='credentials.json')
sheet = gc.open_by_url(SHEET_URL).sheet1

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
        embedding_str = ",".join(map(str, embedding))
        
        # Create ID, Date, and Time
        face_id = str(int(datetime.now().timestamp()))
        date_str = datetime.now().strftime("%Y-%m-%d")
        time_str = datetime.now().strftime("%H:%M:%S")
        
        # Save to Google Sheet
        sheet.append_row([face_id, name, embedding_str, "Pending Upload", date_str, time_str, "Registered"])
        
        os.remove(file_location)
        return {"status": "success", "message": f"{name} registered successfully!"}
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
        new_embedding = np.array(embedding_objs[0]["embedding"])
        
        # Get all records from the sheet
        records = sheet.get_all_records()
        
        best_match = None
        min_distance = float('inf')
        
        # Compare against everyone in the database
        for row in records:
            if row.get('Embedding'):
                stored_embedding = np.array([float(x) for x in row['Embedding'].split(',')])
                distance = np.linalg.norm(new_embedding - stored_embedding)
                if distance < min_distance:
                    min_distance = distance
                    best_match = row
        
        os.remove(file_location)
        
        # Threshold for matching (10.0 is a safe starting point for Facenet)
        if best_match and min_distance < 10.0:
            # Log attendance
            date_str = datetime.now().strftime("%Y-%m-%d")
            time_str = datetime.now().strftime("%H:%M:%S")
            sheet.append_row([best_match['Face ID'], best_match['Name'], "", "", date_str, time_str, "Present"])
            return {"status": "success", "name": best_match['Name'], "face_id": best_match['Face ID']}
        else:
            return {"status": "error", "message": "Face not recognized."}
            
    except Exception as e:
        if os.path.exists(file_location):
            os.remove(file_location)
        return {"status": "error", "message": str(e)}
