# app/routes/__init__.py
from flask import Blueprint, jsonify

# Create blueprints for different sections of the API
api = Blueprint('api', __name__, url_prefix='/api')
auth_bp = Blueprint('auth', __name__, url_prefix='/api/auth')
chat_bp = Blueprint('chat', __name__, url_prefix='/api/chat')
friends_bp = Blueprint('friends', __name__, url_prefix='/api/friends')
settings_bp = Blueprint('settings', __name__, url_prefix='/api/settings')

def register_routes(app):
    """Register all route blueprints with the Flask app"""
    
    # Root endpoint
    @app.route('/', methods=['GET'])
    def homepage():
        return jsonify({
            "status": "success",
            "message": "BKchat API is running",
            "app_info": {
                "name": "BKchat",
                "version": "1.0.0"
            }
        })
    
    # Import routes from other modules
    from . import auth, chat, friends, settings
    
    # Register blueprints with the app
    app.register_blueprint(api)
    app.register_blueprint(auth_bp)
    app.register_blueprint(chat_bp)
    app.register_blueprint(friends_bp)
    app.register_blueprint(settings_bp)
    
    # Register error handlers
    register_error_handlers(app)
    
    return app

def register_error_handlers(app):
    """Register error handlers for the app"""
    
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            "status": "error",
            "message": "Resource not found",
            "error_code": 404
        }), 404
    
    @app.errorhandler(500)
    def internal_server_error(error):
        return jsonify({
            "status": "error",
            "message": "Internal server error",
            "error_code": 500
        }), 500