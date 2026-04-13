# utils.py
import os
import json
from datetime import datetime, timezone
from uuid import uuid4
from groq import Groq
from models import db, Call
from app import app # <-- CRITICAL: Pulls in Flask context for background DB saves

# Ensure Groq API key is loaded
from dotenv import load_dotenv
load_dotenv()

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ==========================================
# 1. GROQ AI DATA EXTRACTION
# ==========================================

def analyze_call_transcript(transcript_text: str):
    """Uses Groq (Llama 3) to extract exact JSON data from the transcript."""
    default_data = {"customer_name": "Unknown", "phone_number": "Unknown", "sentiment": 0.5, "outcome": "Neutral"}
    
    if not transcript_text.strip():
        return default_data

    try:
        # Prompting Groq to act as an analyst and return strict JSON
        completion = client.chat.completions.create(
            model="llama3-8b-8192", # Free and fast Groq model
            messages=[
                {
                    "role": "system", 
                    "content": (
                        "You are a QA data extraction assistant. Read the transcript and output ONLY a valid JSON object. "
                        "The JSON object must contain exactly these keys: "
                        "\"customer_name\" (The caller's full name, or \"Unknown\"), "
                        "\"phone_number\" (The 10-digit phone number, or \"Unknown\"), "
                        "\"sentiment_score\" (A float between 0.0 for very negative and 1.0 for very positive), "
                        "\"outcome\" (Must be exactly one of: \"Interested\", \"Neutral\", or \"Not Interested\")."
                    )
                },
                {"role": "user", "content": f"Transcript:\n{transcript_text}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.1 # Low temperature for consistent formatting
        )
        
        # Parse the JSON string returned by Groq into a dictionary
        analysis = json.loads(completion.choices[0].message.content)
        
        return {
            "customer_name": analysis.get("customer_name", "Unknown"),
            "phone_number": analysis.get("phone_number", "Unknown"),
            "sentiment": float(analysis.get("sentiment_score", 0.5)),
            "outcome": analysis.get("outcome", "Neutral")
        }
    except Exception as e:
        print(f"Error analyzing transcript with Groq: {e}")
        return default_data


# ==========================================
# 2. DATABASE HELPERS
# ==========================================

def add_call(status='active'):
    """Creates a new call record when the LiveKit room connects."""
    with app.app_context(): # <-- Tells Flask we are allowed to touch the DB
        call_id = str(uuid4())
        call = Call(
            id=call_id,
            status=status,
            start_time=datetime.now(timezone.utc),
            end_time=None,
            transcript="",
            outcome="Pending",
            sentiment=0.5,
            customer="Unknown",
            phone="N/A",
            duration="N/A",
        )
        db.session.add(call)
        db.session.commit()
        return call_id 

def mark_call_connected(call_id):
    """Updates the status when the user actually starts speaking."""
    with app.app_context():
        call = Call.query.filter_by(id=call_id).first()
        if call:
            call.status = 'connected'
            db.session.commit()

def end_call(call_id, final_transcript, end_time=None):
    """Ends the call, calculates duration, and runs the AI analysis."""
    with app.app_context():
        call = Call.query.filter_by(id=call_id).first()
        if call and call.status != 'ended':
            call.status = 'ended'
            call.end_time = end_time if end_time else datetime.now(timezone.utc)
            call.transcript = final_transcript
            
            # Calculate Duration
            try:
                diff = (call.end_time - call.start_time).total_seconds()
                mins, secs = divmod(int(diff), 60)
                call.duration = f"{mins}m {secs}s"
            except Exception as e:
                print("Duration calc error:", e)
                call.duration = "N/A"

            # Run AI Analysis automatically
            analysis_results = analyze_call_transcript(final_transcript)
            call.customer = analysis_results["customer_name"]
            call.phone = analysis_results["phone_number"]
            call.sentiment = analysis_results["sentiment"]
            call.outcome = analysis_results["outcome"]

            db.session.commit()