# app/routes/settings.py
from flask import request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from . import settings_bp
from app import mongo
from app.models.user import User
from app.models.mislabeled_messages import MislabeledMessages
from bson import ObjectId
from datetime import datetime

# Get user settings endpoint
@settings_bp.route('/', methods=['GET'])
def get_settings():
    # Reserved for future implementation
    pass

# Update user settings endpoint
@settings_bp.route('/', methods=['PUT'])
def update_settings():
    # Reserved for future implementation
    pass

# Update profile information endpoint
@settings_bp.route('/profile', methods=['PUT'])
def update_profile():
    # Reserved for future implementation
    pass

# Change password endpoint
@settings_bp.route('/password', methods=['PUT'])
def change_password():
    # Reserved for future implementation
    pass

# Update notification preferences endpoint
@settings_bp.route('/notifications', methods=['PUT'])
def update_notifications():
    # Reserved for future implementation
    pass

# Update avatar endpoint
@settings_bp.route('/avatar', methods=['PUT'])
def update_avatar():
    username = request.args.get('username')
    if not username:
        return jsonify({"message": "Username is required"}), 400
    from app.models.user import User
    if User.update_avatar(username):
        return jsonify({"message": "Avatar updated successfully"}), 200
    else:
        return jsonify({"message": "User not found"}), 404

# Delete account endpoint
@settings_bp.route('/account', methods=['DELETE'])
def delete_account():
    # Reserved for future implementation
    pass

# Update message and toxic counters endpoint
@settings_bp.route('/update-counters', methods=['POST'])
@jwt_required()
def update_counters():
    try:
        # 1. Clear existing mislabeled messages collections
        mongo.db.wrong_toxic.delete_many({})
        mongo.db.wrong_not_toxic.delete_many({})
            
        # 2. Get all users
        all_users = mongo.db.users.find()
        
        # 3. First reset ALL counters to 0 for ALL users
        for user in all_users:
            user_id = user['_id']
            friends = user.get('friends', {})
            
            # Reset all friend counters to 0
            for friend_username in friends.keys():
                mongo.db.users.update_one(
                    {"_id": ObjectId(user_id)},
                    {"$set": {
                        f"friends.{friend_username}": {
                            "messageCounter": 0,
                            "toxicCounter": 0
                        }
                    }}
                )
            
        # 4. Now update counters with actual values
        all_users = mongo.db.users.find()  # Get users again after reset
        for user in all_users:
            user_id = user['_id']
            friends = user.get('friends', {})
            
            # For each friend of the user
            for friend_username, friend_data in friends.items():
                friend = User.find_by_username(friend_username)
                if not friend:
                    continue
                    
                # Count total messages between user and friend
                total_messages = mongo.db.messages.count_documents({
                    "$or": [
                        {"senderId": ObjectId(user_id), "receiverId": ObjectId(friend['_id'])},
                        {"senderId": ObjectId(friend['_id']), "receiverId": ObjectId(user_id)}
                    ]
                })
                
                # Count toxic messages between user and friend
                toxic_messages = mongo.db.messages.count_documents({
                    "$or": [
                        {"senderId": ObjectId(user_id), "receiverId": ObjectId(friend['_id']), "isToxic": True},
                        {"senderId": ObjectId(friend['_id']), "receiverId": ObjectId(user_id), "isToxic": True}
                    ]
                })
                
                # Update counters for both users to ensure they match
                mongo.db.users.update_one(
                    {"_id": ObjectId(user_id)},
                    {"$set": {
                        f"friends.{friend_username}": {
                            "messageCounter": total_messages,
                            "toxicCounter": toxic_messages
                        }
                    }}
                )
                
                mongo.db.users.update_one(
                    {"_id": ObjectId(friend['_id'])},
                    {"$set": {
                        f"friends.{user['username']}": {
                            "messageCounter": total_messages,
                            "toxicCounter": toxic_messages
                        }
                    }}
                )
            
        # 5. Check all messages in the messages db for mislabeling
        all_messages = mongo.db.messages.find()
        
        # 6. Fill out the wrong_toxic and wrong_not_toxic collections
        for message in all_messages:
            if 'userFeedback' in message:
                # If message was marked as toxic but user says it's not toxic
                if message.get('isToxic', False) and message['userFeedback'] == 'Not Toxic':
                    MislabeledMessages.store_not_toxic_message(
                        str(message['_id']),
                        str(message['senderId']),
                        message['text'],
                        message.get('timestamp', datetime.utcnow())
                    )
                # If message was not marked as toxic but user says it is toxic
                elif not message.get('isToxic', False) and message['userFeedback'] == 'Toxic':
                    MislabeledMessages.store_toxic_message(
                        str(message['_id']),
                        str(message['senderId']),
                        message['text'],
                        message.get('timestamp', datetime.utcnow())
                    )
            
        return jsonify({'message': 'Counters updated successfully'}), 200
        
    except Exception as e:
        print(f"Error in update_counters: {str(e)}")  # Debug log
        return jsonify({'message': str(e)}), 500

# Get feedback counts endpoint
@settings_bp.route('/get-feedback-counts', methods=['GET'])
@jwt_required()
def get_feedback_counts():
    try:
        # Count documents in wrong_toxic and wrong_not_toxic collections
        wrong_toxic_count = mongo.db.wrong_toxic.count_documents({})
        wrong_not_toxic_count = mongo.db.wrong_not_toxic.count_documents({})
        
        return jsonify({
            'wrong_toxic_count': wrong_toxic_count,
            'wrong_not_toxic_count': wrong_not_toxic_count
        }), 200
        
    except Exception as e:
        print(f"Error in get_feedback_counts: {str(e)}")  # Debug log
        return jsonify({'message': str(e)}), 500
