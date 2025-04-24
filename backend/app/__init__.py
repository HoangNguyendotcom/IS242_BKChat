from flask import Flask
from flask_cors import CORS

def create_app():
    """Initialize the Flask application"""
    app = Flask(__name__, 
                static_folder='../static', 
                template_folder='../templates')
    
    # Enable CORS
    CORS(app)
    
    # Set secret key
    app.secret_key = "your_secret_key_here"  # Change in production
    
    # Register blueprints
    from app.routes.auth import auth_bp
    from app.routes.chat import chat_bp
    # from app.routes.friends import friends_bp
    # from app.routes.settings import settings_bp
    
    app.register_blueprint(auth_bp)
    app.register_blueprint(chat_bp)
    # app.register_blueprint(friends_bp)
    # app.register_blueprint(settings_bp)
    
    @app.route('/')
    def index():
        from flask import render_template
        return render_template('homepage.html')
    
    return app
