# app/config.py
import os
from datetime import timedelta
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Flask configuration
DEBUG = os.environ.get('APP_ENV') == 'development'
SECRET_KEY = os.environ.get('SECRET_KEY')

# MongoDB configuration
MONGO_URI = os.environ.get('MONGO_URI')

# Redis configuration
REDIS_URL = os.environ.get('REDIS_URL')

# JWT configuration
JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY')
JWT_ACCESS_TOKEN_EXPIRES = timedelta(seconds=int(os.environ.get('JWT_ACCESS_TOKEN_EXPIRES', 3600)))
JWT_REFRESH_TOKEN_EXPIRES = timedelta(seconds=int(os.environ.get('JWT_REFRESH_TOKEN_EXPIRES', 2592000)))

# CORS settings
CORS_HEADERS = 'Content-Type'