# models.py
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timezone

db = SQLAlchemy()

class Call(db.Model):
    id = db.Column(db.String, primary_key=True)
    status = db.Column(db.String)
    # Use timezone-aware datetime for the default
    start_time = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))   
    end_time = db.Column(db.DateTime)
    transcript = db.Column(db.Text)
    outcome = db.Column(db.String)
    sentiment = db.Column(db.Float)
    customer = db.Column(db.String)
    phone = db.Column(db.String)
    duration = db.Column(db.String)

    def to_dict(self):
        return {
            "id": self.id,
            "status": self.status,
            # Format dates to ISO strings so React can read them easily
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "transcript": self.transcript,
            "outcome": self.outcome,
            "sentiment": self.sentiment,
            "customer": self.customer,
            "phone": self.phone,
            "duration": self.duration,
        }