# app/models.py
from datetime import datetime
from bson import ObjectId
from app import mongo
from app.models.userstats import UserStats

class Message:
    @staticmethod
    def create(sender_id, receiver_id, text, is_emoji=False):
        message = {
            "senderId": ObjectId(sender_id),
            "receiverId": ObjectId(receiver_id),
            "text": text,
            "timestamp": datetime.utcnow(),
            "isEmoji": is_emoji,
            "isToxic": False,  # Can be updated by a toxicity detection service
            "userFeedback": None
        }
        message_id = mongo.db.messages.insert_one(message).inserted_id
        
        # Update user stats
        UserStats.update_message_count(sender_id, receiver_id)
        
        return str(message_id)
