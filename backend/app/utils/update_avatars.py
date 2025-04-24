import random
import sys
sys.path.append("/Users/mac/Documents/Master_k24/242_Intelligent_Systems/IS242_BKChat/backend")
from app import mongo
from app.models.user import User
from run import app

def update_all_avatars():
    avatar_list = ["avatar.jpeg", "avatar1.jpg", "avatar2.png", "avatar3.avif", "avatar4.png", "avatar5.png"]
    with app.app_context():
        users = mongo.db.users.find()
        for user in users:
            new_avatar = random.choice(avatar_list)
            mongo.db.users.update_one({"_id": user["_id"]}, {"$set": {"avatar": new_avatar}})
    print("All avatars updated successfully")

if __name__ == "__main__":
    update_all_avatars()
