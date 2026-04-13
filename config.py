import os
from dotenv import load_dotenv

# Load environment variables from your .env file
load_dotenv()

# Grab the absolute path of the current directory (AI-Voice-Agent folder)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

class Config:
    # 1. Guarantee the instance folder exists
    INSTANCE_DIR = os.path.join(BASE_DIR, 'instance')
    if not os.path.exists(INSTANCE_DIR):
        os.makedirs(INSTANCE_DIR)

    # 2. Build the absolute path to the database file
    db_path = os.path.join(INSTANCE_DIR, 'calls.db')
    
    # 3. Database Configuration using the absolute path
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL', f'sqlite:///{db_path}')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # 4. Security Key
    SECRET_KEY = os.getenv('SECRET_KEY', 'a_very_secret_random_string_here')