# utils.py
import re
import random
import json
from datetime import datetime
from uuid import uuid4
import openai
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from models import db, Call

# Initialize VADER sentiment analyzer
vader_analyzer = SentimentIntensityAnalyzer()

# Conversation memory for AI
conversation = []

def classify_outcome_and_sentiment(transcript):
    scores = vader_analyzer.polarity_scores(transcript)
    compound = scores['compound']  # Value between -1 and 1
    normalized_sentiment = (compound + 1) / 2  # Normalize to 0-1 range

    if normalized_sentiment >= 0.65:
        outcome = "Interested"
    elif normalized_sentiment >= 0.35:
        outcome = "Neutral"
    else:
        outcome = "Not Interested"

    return outcome, normalized_sentiment


def extract_customer_info(transcript):
    name = "Unknown"
    phone = None

    # Try to find name formats: Mr./Ms./Mrs. Name
    match = re.search(r'(Mr\.|Ms\.|Mrs\.)\s+([A-Z][a-z]+)', transcript)
    if match:
        name = f"{match.group(1)} {match.group(2)}"
    else:
        # Try common phrases
        match = re.search(
            r'(my name is|I am|this is)\s+([A-Z][a-z]+\s?[A-Z]?[a-z]*?)',
            transcript,
            re.IGNORECASE,
        )
        if match:
            name = match.group(2).strip()

    # Try to extract a 10-digit number for phone
    phone_match = re.search(r'\b(\d{10})\b', transcript)
    if phone_match:
        phone = phone_match.group(1)
    else:
        # Generate dummy 10-digit number starting with 9
        phone = '9' + ''.join(str(random.randint(0, 9)) for _ in range(9))

    return name, phone


def add_call(status='active'):
    call_id = str(uuid4())
    call = Call(
        id=call_id,
        status=status,
        start_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        end_time=None,
        audio_filename="",
        transcript="",
        entities=json.dumps({}),
        outcome="Pending",
        sentiment=0.5,
        customer="Unknown",
        phone="N/A",
        duration="N/A",
    )
    db.session.add(call)
    db.session.commit()
    return call


def mark_call_connected(call_id):
    call = Call.query.filter_by(id=call_id).first()
    if call:
        call.status = 'connected'
        db.session.commit()


def end_call(call_id, end_time=None):
    call = Call.query.filter_by(id=call_id).first()
    if call and call.status != 'ended':
        call.status = 'ended'
        call.end_time = end_time if end_time else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            start = datetime.strptime(call.start_time, "%Y-%m-%d %H:%M:%S")
            end = datetime.strptime(call.end_time, "%Y-%m-%d %H:%M:%S")
            mins, secs = divmod((end - start).seconds, 60)
            call.duration = f"{mins}m {secs}s"
        except Exception:
            call.duration = "N/A"
        db.session.commit()


def transcribe_audio(file_path):
    try:
        with open(file_path, "rb") as audio_file:
            transcript = openai.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        return transcript.text
    except Exception as e:
        print("Transcription error:", e)
        return None


def ai_response(user_text, system_prompt=None):
    global conversation
    try:
        if not conversation and system_prompt:
            conversation.append({"role": "system", "content": system_prompt})

        conversation.append({"role": "user", "content": user_text})
        completion = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=conversation
        )
        response_text = completion.choices[0].message.content
        conversation.append({"role": "assistant", "content": response_text})
        return response_text
    except Exception as e:
        print("AI response error:", e)
        return "Error generating AI response."
