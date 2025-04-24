# app/models/__init__.py
from app import db

# Import all models here so they are registered with SQLAlchemy
from .user import User
from .message import Message
from .conversation import Conversation

# You can define any common model functionality here