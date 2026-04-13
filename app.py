from flask import Flask
from flask_cors import CORS
import os  # <-- Add this
from models import db
from routes.api_routes import api_bp
from config import Config  

app = Flask(__name__)

# Allow your React frontend to communicate with this Flask backend
CORS(app) 

# Load configuration from config.py
app.config.from_object(Config)

# Database setup
db.init_app(app)

# <-- Add these two lines to guarantee the folder exists! -->
if not os.path.exists('instance'):
    os.makedirs('instance')

with app.app_context():
    db.create_all()

# Register ONLY the API blueprint
app.register_blueprint(api_bp)

if __name__ == '__main__':
    app.run(debug=True)