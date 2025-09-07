# routes/api_routes.py
import os
import json
from datetime import datetime
from flask import Blueprint, jsonify, request, Response
from models import Call, db
from utils import (
    classify_outcome_and_sentiment,
    extract_customer_info,
    add_call,
    mark_call_connected,
    end_call,
    transcribe_audio,
    ai_response,
)

api_bp = Blueprint("api", __name__)

system_prompt = """
You are a professional AI telecalling agent representing Bajaj Finance Limited...
(keep your full system prompt here)
"""

RECORDING_DIR = os.path.join(os.path.dirname(__file__), '..', 'recording')


@api_bp.route('/api/call_summary')
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


@api_bp.route('/api/call_status_pie')
def call_status_pie():
    active = Call.query.filter_by(status='active').count()
    connected = Call.query.filter_by(status='ended').count()

    return jsonify([
        {"label": "Active", "value": active},
        {"label": "Connected", "value": connected},
    ])


@api_bp.route('/api/call_outcomes_bar')
def call_outcomes_bar():
    outcomes = {}
    for call in Call.query.filter_by(status='ended'):
        outcome = call.outcome or "Unknown"
        outcomes[outcome] = outcomes.get(outcome, 0) + 1

    return jsonify([{"label": k, "value": v} for k, v in outcomes.items()])


@api_bp.route('/process', methods=['POST'])
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
    ai_reply = ai_response(transcription, system_prompt)
    end_call(call.id)
    db.session.commit()

    return jsonify({'transcript': transcription, 'response': ai_reply})


@api_bp.route('/process_text', methods=['POST'])
def process_text():
    data = request.get_json()
    user_text = data.get('text', '')

    if not user_text:
        return jsonify({'response': 'No input text received.'})

    ai_reply = ai_response(user_text, system_prompt)
    return jsonify({'response': ai_reply})


@api_bp.route('/api/calls')
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


@api_bp.route('/api/calls/export_csv')
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


@api_bp.route('/save_transcript', methods=['POST'])
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


@api_bp.route('/api/call/<call_id>')
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
