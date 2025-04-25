# app/models.py
import random
from datetime import datetime
from bson import ObjectId
from app import mongo

class User:
    @staticmethod
    def create(name, username, email, password_hash, avatar="/avatars/avatar.jpeg"):
        user = {
            "name": name,
            "username": username,
            "email": email,
            "password": password_hash,
            "avatar": avatar,
            "friends": {},
            "created_at": datetime.utcnow()
        }
        user_id = mongo.db.users.insert_one(user).inserted_id
        return str(user_id)

    @staticmethod
    def find_by_username(username):
        user = mongo.db.users.find_one({"username": username})
        if user:
            user['_id'] = str(user['_id'])
        return user

    @staticmethod
    def find_by_id(user_id):
        try:
            user = mongo.db.users.find_one({"_id": ObjectId(user_id)})
            if user:
                user['_id'] = str(user['_id'])
            return user
        except:
            return None

    @staticmethod
    def add_friend(username, friend_username):
        user = mongo.db.users.find_one({"username": username})
        if user:
            user['_id'] = str(user['_id'])
            if 'friends' not in user:
                user['friends'] = {}
            if friend_username not in user['friends']:
                user['friends'][friend_username] = True
                mongo.db.users.update_one(
                    {"username": username}, 
                    {"$set": {"friends": user['friends']}}
                )
                return True
            else:
                return False
        else:
            return False

    @staticmethod
    def find_by_email(email):
        user = mongo.db.users.find_one({"email": email})
        if user:
            user['_id'] = str(user['_id'])
        return user

    @staticmethod
    def update_avatar(username):
        user = mongo.db.users.find_one({"username": username})
        if user:
            avatar_list = ["avatar.jpeg", "avatar1.jpg", "avatar2.png", "avatar3.avif", "avatar4.png", "avatar5.png"]
            new_avatar = random.choice(avatar_list)
            mongo.db.users.update_one({"username": username}, {"$set": {"avatar": new_avatar}})
            return True
        else:
            return False
