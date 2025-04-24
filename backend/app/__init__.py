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
    app.config["MONGO_URI"] = f"mongodb://{os.getenv('MONGO_HOST', 'localhost')}:{os.getenv('MONGO_PORT', '27017')}/bkchat"
    mongo.init_app(app)

    # Configure Redis
    redis_url = app.config.get('REDIS_URL', 'redis://localhost:6379/0')

    # Enable CORS for frontend
    CORS(app, resources={
    r"/api/*": {"origins": "http://localhost:3000"},
    r"/": {"origins": "http://localhost:3000"}
    })

    # Initialize extensions with app
    global redis_client
    jwt.init_app(app)
    bcrypt.init_app(app)
    redis_client = redis.from_url(redis_url)

    # Register routes
    with app.app_context():
        from app.routes import register_routes
        register_routes(app)

    return app
