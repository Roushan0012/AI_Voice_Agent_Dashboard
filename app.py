from flask import Flask, request, render_template, jsonify, Response, send_from_directory, abort
from flask_sqlalchemy import SQLAlchemy
import os
from datetime import datetime
import openai
from uuid import uuid4
import json
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import random

app = Flask(__name__)

RECORDING_DIR = os.path.join(os.path.dirname(__file__), 'recording')
os.makedirs(RECORDING_DIR, exist_ok=True)

# Database setup with SQLite
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///calls.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Initialize VADER sentiment analyzer
vader_analyzer = SentimentIntensityAnalyzer()


# -----------------------------
# Database Model
# -----------------------------
class Call(db.Model):
    id = db.Column(db.String, primary_key=True)
    status = db.Column(db.String)
    start_time = db.Column(db.String)
    end_time = db.Column(db.String)
    audio_filename = db.Column(db.String)
    transcript = db.Column(db.Text)
    entities = db.Column(db.Text)  # JSON string
    outcome = db.Column(db.String)
    sentiment = db.Column(db.Float)
    customer = db.Column(db.String)
    phone = db.Column(db.String)
    duration = db.Column(db.String)

    def to_dict(self):
        return {
            "id": self.id,
            "status": self.status,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "audio_filename": self.audio_filename,
            "transcript": self.transcript,
            "entities": json.loads(self.entities) if self.entities else {},
            "outcome": self.outcome,
            "sentiment": self.sentiment,
            "customer": self.customer,
            "phone": self.phone,
            "duration": self.duration,
        }


with app.app_context():
    db.create_all()


# -----------------------------
# AI Config
# -----------------------------
openai.api_key = os.getenv("OPENAI_API_KEY")

system_prompt = """
You are a professional AI telecalling agent representing Bajaj Finance Limited offering a Flexi Overdraft Facility specially designed for salaried employees.
Your responsibilities include:
- Greeting the customer politely, verifying identity, and stating that the call is recorded for training and quality purposes.
- Introducing the Flexi Overdraft Facility clearly as a non-personal loan financial backup with Flexi terms.
- Collecting comprehensive customer information during the call such as employment status, net take-home salary, company details, PAN card info, date of birth, current city/pincode, outstanding loans, EMIs, and rental or owned housing details.
- Explaining product features: overdraft amount up to 10-24 times net salary, interest rate 1.25% monthly reducing balance on utilized amount, no EMIs, multiple withdrawals, and part repayments allowed without penalties, validity up to 8 years.
- Responding empathetically and professionally to objections or queries about product terms, credit score checks, bureau reports, charging policies, documentation required, and repayment flexibility.
- Summarizing and extracting customer details in a structured manner while maintaining a polite and clear tone.
- Closing conversations by thanking customers and offering further assistance.
"""

conversation = [{"role": "system", "content": system_prompt}]

PROMPT = (
    "Analyze the following transcript of a customer and agent conversation. "
    "Classify the customer outcome as 'Interested', 'Not Interested', or 'Neutral'. "
    "Also, give an overall sentiment score: 1.0 for positive, 0.5 for neutral, or 0.0 for negative.\n\n"
    "Transcript:\n"
)


# -----------------------------
# Utility Functions
# -----------------------------
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


def ai_response(user_text):
    global conversation
    try:
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


# -----------------------------
# Routes
# -----------------------------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/recording/<filename>')
def serve_recording(filename):
    if not (filename.endswith('.webm') or filename.endswith('.mp3') or filename.endswith('.wav')):
        abort(404)

    abs_path = os.path.join(RECORDING_DIR, filename)
    if not os.path.isfile(abs_path):
        abort(404)

    if filename.endswith('.mp3'):
        mimetype = "audio/mpeg"
    elif filename.endswith('.wav'):
        mimetype = "audio/wav"
    else:
        mimetype = "audio/webm"

    return send_from_directory(RECORDING_DIR, filename, mimetype=mimetype)


@app.route('/api/call_summary')
def call_summary():
    active_calls = Call.query.filter_by(status='active').count()
    total_calls = Call.query.filter_by(status='ended').count()
    connected_calls = Call.query.filter(Call.status.in_(['connected', 'ended'])).count()
    success_rate = (connected_calls / total_calls * 100) if total_calls > 0 else 0

    return jsonify({
        "active_calls": active_calls,
        "total_calls": total_calls,
        "connected_calls": connected_calls,
        "success_rate": round(success_rate, 2),
    })


@app.route('/api/call_status_pie')
def call_status_pie():
    active = Call.query.filter_by(status='active').count()
    connected = Call.query.filter_by(status='ended').count()

    return jsonify([
        {"label": "Active", "value": active},
        {"label": "Connected", "value": connected},
    ])


@app.route('/api/call_outcomes_bar')
def call_outcomes_bar():
    outcomes = {}
    for call in Call.query.filter_by(status='ended'):
        outcome = call.outcome or "Unknown"
        outcomes[outcome] = outcomes.get(outcome, 0) + 1

    return jsonify([{"label": k, "value": v} for k, v in outcomes.items()])


@app.route('/process', methods=['POST'])
def process_audio():
    if 'audio_data' not in request.files:
        return jsonify({'error': 'No audio uploaded'}), 400

    call = add_call(status='active')
    audio_file = request.files['audio_data']

    fname = datetime.now().strftime('%Y%m%d%H%M%S') + '.webm'
    audio_path = os.path.join(RECORDING_DIR, fname)
    audio_file.save(audio_path)

    call.audio_filename = fname
    db.session.commit()

    transcription = transcribe_audio(audio_path)
    if not transcription:
        end_call(call.id)
        return jsonify({'error': 'Failed to transcribe audio'}), 500

    call.transcript = transcription
    customer, phone = extract_customer_info(transcription)
    outcome, sentiment = classify_outcome_and_sentiment(transcription)

    call.customer = customer
    call.phone = phone
    call.outcome = outcome
    call.sentiment = sentiment

    mark_call_connected(call.id)
    ai_reply = ai_response(transcription)
    end_call(call.id)
    db.session.commit()

    return jsonify({'transcript': transcription, 'response': ai_reply})


@app.route('/process_text', methods=['POST'])
def process_text():
    data = request.get_json()
    user_text = data.get('text', '')

    if not user_text:
        return jsonify({'response': 'No input text received.'})

    ai_reply = ai_response(user_text)
    return jsonify({'response': ai_reply})


@app.route('/api/calls')
def calls():
    date_from = request.args.get('from')
    date_to = request.args.get('to')
    status = request.args.get('status')
    min_duration = request.args.get('duration', type=int)
    transcript_search = request.args.get('search', '').lower()

    query = Call.query
    if status and status != 'All':
        query = query.filter_by(status=status)
    if date_from:
        query = query.filter(Call.start_time >= date_from)
    if date_to:
        query = query.filter(Call.start_time <= date_to)

    results = []
    for c in query.all():
        if min_duration is not None and c.duration and c.duration != "N/A":
            try:
                mins, secs = c.duration.split()
                dur_sec = int(mins[:-1]) * 60 + int(secs[:-1])
                if dur_sec < min_duration:
                    continue
            except Exception:
                pass

        if transcript_search:
            searched = [
                (c.customer or '').lower(),
                (c.phone or '').lower(),
                (c.outcome or '').lower(),
                (c.transcript or '').lower(),
            ]
            if not any(transcript_search in field for field in searched):
                continue

        results.append({
            "id": c.id,
            "customer": c.customer or "Unknown",
            "phone": c.phone or "N/A",
            "start_time": c.start_time or "",
            "duration": c.duration or "N/A",
            "status": c.status or "",
            "outcome": c.outcome or "Pending",
            "sentiment": c.sentiment or 0.5,
        })

    return jsonify(results)


@app.route('/api/calls/export_csv')
def export_calls_csv():
    si = []
    header = ["Customer", "Phone", "Start Time", "Duration", "Status", "Outcome", "Sentiment", "Audio File"]
    si.append(','.join(header) + '\n')

    for c in Call.query.all():
        row = [
            f'"{c.customer or ""}"',
            f'"{c.phone or ""}"',
            f'"{c.start_time or ""}"',
            f'"{c.duration or ""}"',
            f'"{c.status or ""}"',
            f'"{c.outcome or ""}"',
            str(c.sentiment or ""),
            f'"{c.audio_filename or ""}"',
        ]
        si.append(','.join(row) + '\n')

    output = ''.join(si)
    return Response(
        output,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=call_records.csv"},
    )

@app.route('/save_transcript', methods=['POST'])
def save_transcript():
    data = request.get_json()
    call_id = data.get('call_id')
    transcript = data.get('transcript')

    call = Call.query.get(call_id)
    if not call:
        return jsonify({"error": "Call not found"}), 404

    call.transcript = transcript
    db.session.commit()
    return jsonify({"status": "success"})


@app.route('/api/call/<call_id>')
def call_detail(call_id):
    c = Call.query.filter_by(id=call_id).first()
    if c:
        audio_url = f"/recording/{c.audio_filename}" if c.audio_filename else ""
        return jsonify({
            "id": c.id,
            "customer": c.customer or "Unknown",
            "phone": c.phone or "N/A",
            "start_time": c.start_time or "",
            "duration": c.duration or "N/A",
            "status": c.status or "",
            "outcome": c.outcome or "Pending",
            "sentiment": c.sentiment or 0.5,
            "transcript": c.transcript or "",
            "audio_url": audio_url,
            "entities": json.loads(c.entities) if c.entities else {},
        })
    return jsonify({"error": "Call not found"}), 404


# -----------------------------
# Run App
# -----------------------------
if __name__ == '__main__':
    app.run(debug=True)