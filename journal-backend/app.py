from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
from pydantic import BaseModel
import uvicorn
import random
import firebase_admin
from firebase_admin import credentials, firestore

app = FastAPI()

origins = ["http://localhost:5173"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

classifier = pipeline("text-classification", 
                     model='bhadresh-savani/distilbert-base-uncased-emotion', 
                     return_all_scores=True)

# Initialize Firebase
cred = credentials.Certificate("backend/serviceAccountKey.json")
try:
    firebase_admin.initialize_app(cred)
except ValueError as e:
    print("Firebase already initialized:", e)
db = firestore.client()

class JournalEntry(BaseModel):
    text: str
    userId: str

# Prompts and activities
prompts = {
    "joy": ["What made you happy today and why?", "Describe a moment that brought you joy recently."],
    "sadness": ["Write about something you're finding difficult right now.", "What would help you feel better today?"],
    "anger": ["What triggered your frustration today?", "How could you respond differently next time?"],
    "fear": ["What's causing you anxiety right now?", "Write about a time you overcame a similar challenge."],
    "love": ["Who are you grateful for today and why?", "Describe a meaningful connection in your life."],
    "surprise": ["What unexpected event impacted you recently?", "How did this surprise change your perspective?"]
}

activities = {
    "joy": ["Share your happiness with a friend", "Try a new hobby that excites you"],
    "sadness": ["Take a gentle walk in nature", "Practice self-compassion meditation"],
    "anger": ["Try box breathing (4-4-4-4 count)", "Write a letter expressing your feelings (without sending it)"],
    "fear": ["Progressive muscle relaxation", "Make a list of what you can control"],
    "love": ["Reach out to someone you care about", "Practice acts of kindness"],
    "surprise": ["Journal about what you learned from this experience", "Explore the new perspective this gives you"]
}

@app.post("/analyze_emotion")
async def analyze_emotion(entry: JournalEntry):
    result = classifier(entry.text)
    emotions = result[0]
    emotions.sort(key=lambda x: x['score'], reverse=True)
    top_emotion = emotions[0]['label']
    recommended_activities = activities.get(top_emotion, ["Take a deep breath", "Practice mindfulness"])
    writing_prompts = prompts.get(top_emotion, ["How are you feeling right now?"])

    return {
        "emotions": emotions,
        "top_emotion": top_emotion,
        "activities": random.sample(recommended_activities, min(2, len(recommended_activities))),
        "prompts": random.sample(writing_prompts, min(1, len(writing_prompts)))
    }

# Save journal entry to Firebase
@app.post("/save_journal_entry")
async def save_journal_entry(entry: JournalEntry):
    db.collection("entries").add(entry.dict())
    return {"message": "Saved entry to Firebase"}

# Run the app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
