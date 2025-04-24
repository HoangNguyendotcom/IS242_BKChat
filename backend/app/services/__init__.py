# app/services/__init__.py

# Import all services so they are available throughout the app
from .auth_service import AuthService
from .chat_service import ChatService
from .content_filter import ContentFilter

# You can define any common service functionality here