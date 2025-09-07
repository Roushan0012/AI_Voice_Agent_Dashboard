import os

class Config:
    SQLALCHEMY_DATABASE_URI = 'sqlite:///calls.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    RECORDING_DIR = os.path.join(os.path.dirname(__file__), 'recording')
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
