# app/__init__.py
from flask import Flask
from flask_cors import CORS
from flask_pymongo import PyMongo
from flask_jwt_extended import JWTManager
from flask_bcrypt import Bcrypt
import redis
import os

# Initialize extensions
mongo = PyMongo()
jwt = JWTManager()
bcrypt = Bcrypt()
redis_client = None

def create_app(config=None):
    """
    Application factory function to create and configure the Flask app
    """
    # Create Flask app instance
    app = Flask(__name__)
    
    # Load default configuration
    app.config.from_object('app.config')
    
    # Override with instance config if exists
    if config:
        app.config.from_object(config)
    
    # Configure MongoDB
    app.config.setdefault('MONGO_URI', 'mongodb://localhost:27017/flask_app')
    
    # Configure Redis
    redis_url = app.config.get('REDIS_URL', 'redis://localhost:6379/0')
    
    # Enable CORS for frontend
    CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})
    
    # Initialize extensions with app
    global redis_client
    mongo.init_app(app)
    jwt.init_app(app)
    bcrypt.init_app(app)
    redis_client = redis.from_url(redis_url)
    
    # Register routes
    with app.app_context():
        from app.routes import register_routes
        register_routes(app)
    
    return app