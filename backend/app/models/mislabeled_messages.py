from datetime import datetime
from bson import ObjectId
from app import mongo

class MislabeledMessages:
    @staticmethod
    def store_toxic_message(message_id, sender_id, text, timestamp):
        try:
            # Get the original message to preserve all fields
            original_message = mongo.db.messages.find_one({"_id": ObjectId(message_id)})
            if not original_message:
                return None

            # Create new message object with all required fields
            new_message = {
                "_id": ObjectId(message_id),
                "senderId": sender_id,
                "receiverId": original_message.get("receiverId"),
                "text": text,
                "timestamp": timestamp,
                "isEmoji": original_message.get("isEmoji", False),
                "isToxic": False,  # Original message was not marked as toxic
                "userFeedback": "Toxic",
                "originalTimestamp": original_message.get("timestamp"),
                "storedAt": datetime.utcnow()
            }

            # Insert into wrong_toxic collection
            result = mongo.db.wrong_toxic.insert_one(new_message)
            return str(result.inserted_id)
        except Exception as e:
            print(f"Error storing toxic message: {str(e)}")
            return None

    @staticmethod
    def store_not_toxic_message(message_id, sender_id, text, timestamp):
        try:
            # Get the original message to preserve all fields
            original_message = mongo.db.messages.find_one({"_id": ObjectId(message_id)})
            if not original_message:
                return None

            # Create new message object with all required fields
            new_message = {
                "_id": ObjectId(message_id),
                "senderId": sender_id,
                "receiverId": original_message.get("receiverId"),
                "text": text,
                "timestamp": timestamp,
                "isEmoji": original_message.get("isEmoji", False),
                "isToxic": True,  # Original message was marked as toxic
                "userFeedback": "Not Toxic",
                "originalTimestamp": original_message.get("timestamp"),
                "storedAt": datetime.utcnow()
            }

            # Insert into wrong_not_toxic collection
            result = mongo.db.wrong_not_toxic.insert_one(new_message)
            return str(result.inserted_id)
        except Exception as e:
            print(f"Error storing not toxic message: {str(e)}")
            return None

    @staticmethod
    def get_toxic_messages():
        """Get all messages incorrectly labeled as not toxic"""
        return list(mongo.db.toxic_messages.find())

    @staticmethod
    def get_not_toxic_messages():
        """Get all messages incorrectly labeled as toxic"""
        return list(mongo.db.not_toxic_messages.find()) 