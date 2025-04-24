# app/models.py
from datetime import datetime
from bson import ObjectId
from app import mongo

class UserStats:
    @staticmethod
    def update_message_count(user_id, friend_id):
        mongo.db.user_stats.update_one(
            {"userId": ObjectId(user_id), "friendId": ObjectId(friend_id)},
            {
                "$inc": {"totalMessages": 1},
                "$set": {"lastInteraction": datetime.utcnow()}
            },
            upsert=True
        )
    
    @staticmethod
    def update_toxic_count(user_id, friend_id):
        mongo.db.user_stats.update_one(
            {"userId": ObjectId(user_id), "friendId": ObjectId(friend_id)},
            {"$inc": {"toxicMessages": 1}},
            upsert=True
        )