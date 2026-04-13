# routes/api_routes.py
import os
import uuid
from flask import Blueprint, jsonify, request
from livekit.api import AccessToken, VideoGrants
from models import Call

# This automatically adds /api to the front of all routes below!
api_bp = Blueprint('api', __name__, url_prefix='/api')

@api_bp.route('/call_summary', methods=['GET'])
def get_call_summary():
    """Provides data for the 4 top cards on the dashboard."""
    total_calls = Call.query.count()
    active_calls = Call.query.filter_by(status='active').count()
    connected_calls = Call.query.filter(Call.status.in_(['connected', 'ended'])).count()
    
    # Calculate Success Rate (Calls that connected vs Total)
    success_rate = 0
    if total_calls > 0:
        success_rate = round((connected_calls / total_calls) * 100)

    return jsonify({
        "active_calls": active_calls,
        "total_calls": total_calls,
        "connected_calls": connected_calls,
        "success_rate": success_rate
    })

@api_bp.route('/call_status_pie', methods=['GET'])
def get_call_status_pie():
    """Provides data for the Donut Chart (Status Distribution)."""
    connected = Call.query.filter(Call.status.in_(['connected', 'ended'])).count()
    failed = Call.query.filter_by(status='failed').count()
    
    return jsonify([
        {"name": "Connected", "value": connected},
        {"name": "Failed", "value": failed}
    ])

@api_bp.route('/call_outcomes_bar', methods=['GET'])
def get_call_outcomes_bar():
    """Provides data for the Bar Chart (Interested vs Neutral vs Not Interested)."""
    interested = Call.query.filter_by(outcome='Interested').count()
    neutral = Call.query.filter_by(outcome='Neutral').count()
    not_interested = Call.query.filter_by(outcome='Not Interested').count()

    return jsonify([
        {"name": "Interested", "value": interested},
        {"name": "Neutral", "value": neutral},
        {"name": "Not Interested", "value": not_interested}
    ])

@api_bp.route('/calls', methods=['GET'])
def get_calls():
    """Provides the data for the Detailed Call Records table, with filtering."""
    # Get filter parameters from the URL
    status_filter = request.args.get('status', 'All Status')
    search_query = request.args.get('search', '').lower()

    # Start the database query
    query = Call.query

    # Apply Status Filter
    if status_filter != 'All Status':
        query = query.filter(Call.status == status_filter.lower())

    # Apply Search Filter (searching inside transcripts or customer names)
    if search_query:
        query = query.filter(
            (Call.transcript.ilike(f"%{search_query}%")) | 
            (Call.customer.ilike(f"%{search_query}%"))
        )

    # Order by newest first
    calls = query.order_by(Call.start_time.desc()).all()

    # Convert to dictionary for JSON response
    return jsonify([call.to_dict() for call in calls])

# Notice we changed this from '/api/get_token' to just '/get_token'
@api_bp.route('/get_token', methods=['GET'])
def get_token():
    """Generates a secure LiveKit token for the React web caller."""
    # 1. Grab your keys from the .env file
    api_key = os.getenv("LIVEKIT_API_KEY")
    api_secret = os.getenv("LIVEKIT_API_SECRET")
    livekit_url = os.getenv("LIVEKIT_URL")

    # 2. Create a unique ID for the person clicking the button
    user_id = f"web_caller_{str(uuid.uuid4())[:8]}"

    # 3. Generate the secure token
    token = AccessToken(api_key, api_secret) \
        .with_identity(user_id) \
        .with_name("Web Customer") \
        .with_grants(VideoGrants(room_join=True, room="dashboard-call"))

    # 4. Send the token and the URL back to React
    return jsonify({
        "token": token.to_jwt(),
        "url": livekit_url
    })