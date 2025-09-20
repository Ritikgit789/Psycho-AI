from fastapi import FastAPI, HTTPException, Form, Depends
import pandas as pd
import joblib
import numpy as np
import google.generativeai as genai
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware


# Load environment variables (Google API Key)
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load Machine Learning model
model = joblib.load("E:/Psycho AI/mental_health_model(LR).pkl")

# Initialize FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or your frontend URL instead of "*"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

AUTH_FILE = "auth.xlsx"

# Ensure auth.xlsx exists
if not os.path.exists(AUTH_FILE):
    df = pd.DataFrame(columns=["username", "password"])
    df.to_excel(AUTH_FILE, index=False)

# Function to authenticate users from Excel
def authenticate_user(username: str, password: str):
    df = pd.read_excel(AUTH_FILE)
    user = df[(df["username"] == username) & (df["password"] == password)]
    return not user.empty

# Function to save new login (optional for signup)
def save_user(username: str, password: str):
    df = pd.read_excel(AUTH_FILE)
    if username in df["username"].values:
        return False  # Username already exists
    new_user = pd.DataFrame([[username, password]], columns=["username", "password"])
    df = pd.concat([df, new_user], ignore_index=True)
    df.to_excel(AUTH_FILE, index=False)
    return True

# Login Endpoint (checks and saves if new)
@app.post("/login/")
def login(username: str = Form(...), password: str = Form(...)):
    if authenticate_user(username, password):
        return {"message": "Login successful"}
    elif save_user(username, password):
        return {"message": "New user registered and logged in"}
    else:
        raise HTTPException(status_code=401, detail="Invalid credentials or username exists")

# API: User Registration
@app.post("/register/")
def register(username: str = Form(...), password: str = Form(...)):
    df = pd.read_excel(AUTH_FILE)
    if username in df["username"].values:
        raise HTTPException(status_code=400, detail="Username already exists")
    new_user = pd.DataFrame([[username, password]], columns=["username", "password"])
    df = pd.concat([df, new_user], ignore_index=True)
    df.to_excel(AUTH_FILE, index=False)
    return {"message": "Registration successful"}

# API: Mental Health Severity Prediction
# Create request body schema using Pydantic
from pydantic import BaseModel
class PredictionRequest(BaseModel):
    age: int
    gender: int
    self_employed: int
    family_history: int
    work_interfere: int
    no_employees: int
    remote_work: int
    tech_company: int
    benefits: int
    care_options: int
    wellness_program: int
    seek_help: int
    anonymity: int
    leave: int
    mental_health_consequence: int
    phys_health_consequence: int
    coworkers: int
    supervisor: int
    mental_health_interview: int
    phys_health_interview: int
    mental_vs_physical: int
    obs_consequence: int
    obs_frequency: int

# Gemini explanation generator
def get_explanation(severity: str) -> str:
    prompt = f"""
You are a compassionate mental health assistant.

A person has been predicted with **'{severity}'** mental health severity.

1. Explain in simple and empathetic language what '{severity}' means.
2. Provide 3–5 personalized coping mechanisms tailored to this level of severity.
3. Suggest self-care routines for stress management and improving mental well-being.
4. Provide them a accurate treatment in 2-3 lines
5. Clearly mention when they should consider seeking professional help.
6. Keep the tone encouraging and supportive.
7. Don't give more paragraphs, to the point and concise.

Format the response using markdown with bullet points and sections.
"""
    model_gemini = genai.GenerativeModel("gemini-2.0-flash")
    response = model_gemini.generate_content(prompt)
    return response.text

# Final predict endpoint (accepts JSON body)
@app.post("/predict/")
def predict(data: PredictionRequest):
    input_features = np.array(list(data.dict().values())).reshape(1, -1)

    prediction = model.predict(input_features)[0]
    severity_map = {0: "Mild", 1: "Moderate", 2: "Severe"}
    severity = severity_map.get(prediction, "Unknown")

    explanation = get_explanation(severity)

    return {
        "severity": severity,
        "coping_mechanisms": explanation
    }

# # API: Gemini Assistant for Mental Health
# from pydantic import BaseModel
# class QuestionInput(BaseModel):
#     question: str

# @app.post("/assistant/")
# def assistant(input: QuestionInput):
#     model_gemini = genai.GenerativeModel("gemini-2.0-flash")  # or "gemini-pro"
    
#     # Compose a structured prompt for better response
#     structured_prompt = f"""
# You are a friendly, empathetic, and professional mental health support assistant.

# User's Question: "{input.question}"

# Your job is to:
# - Understand the user's mental state.
# - Offer thoughtful, clear, and practical suggestions.
# - Recommend 2–3 healthy coping mechanisms (if relevant).
# - Suggest professional help if needed.
# - Be empathetic and supportive, not judgmental.
# - Give the answers freely and accurate.
# - don't say that user is asking that. directly give the answer.

# Begin your answer below:
# """

#     response = model_gemini.generate_content(structured_prompt)
#     return {"response": response.text}


from typing import List, Dict
# Memory: user_id -> list of message dicts {"role": "user"/"assistant", "text": ...}
chat_memory: List[Dict[str, str]] = []

class ChatInput(BaseModel):
    message: str

@app.post("/chatbot/")
def mental_health_chat(input: ChatInput):
    user_msg = input.message

    # Add user's message
    chat_memory.append({"role": "user", "text": user_msg})

    # Build prompt from chat history
    context = """ You are a friendly, empathetic, and professional mental health support assistant. 
    Understand the user's mental state. Ask relevant questions related to user's question. 
    And if user asks any question irrelevant to mental health like (1+1) or any, then 
    always respond - Please I am here to assist you at your mental health problems \n\n"""
    
    for msg in chat_memory:
        role = "User" if msg["role"] == "user" else "Assistant"
        context += f"{role}: {msg['text']}\n"

    context += "Assistant:"

    # Generate response
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(context)

    # Save assistant response
    chat_memory.append({"role": "assistant", "text": response.text})

    return {"response": response.text}






if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

