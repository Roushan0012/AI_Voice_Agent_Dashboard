from flask import Flask
import os
from models import db
from routes.main_routes import main_bp
from routes.api_routes import api_bp

app = Flask(__name__)

# Database setup
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///calls.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

with app.app_context():
    db.create_all()

# Register blueprints
app.register_blueprint(main_bp)
app.register_blueprint(api_bp)

if __name__ == '__main__':
    app.run(debug=True)
