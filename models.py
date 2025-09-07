# models.py
from flask_sqlalchemy import SQLAlchemy
import json
from datetime import datetime
db = SQLAlchemy()

class Call(db.Model):
    id = db.Column(db.String, primary_key=True)
    status = db.Column(db.String)
    start_time = db.Column(db.DateTime, default=datetime.utcnow)   # ✅ DateTime
    end_time = db.Column(db.DateTime)
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
