# routes/main_routes.py
import os
from flask import Blueprint, render_template, send_from_directory, abort

main_bp = Blueprint("main", __name__)

RECORDING_DIR = os.path.join(os.path.dirname(__file__), '..', 'recording')

@main_bp.route('/')
def index():
    return render_template('index.html')


@main_bp.route('/recording/<filename>')
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
