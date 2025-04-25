# app/routes/settings.py
from flask import request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from . import settings_bp
from app import mongo
from app.models.user import User
from bson import ObjectId

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
        current_user_id = get_jwt_identity()
        current_user = User.find_by_id(current_user_id)
        
        if not current_user:
            return jsonify({'message': 'User not found'}), 404
            
        # Get all friends
        friends = current_user.get('friends', {})
        
        # For each friend, count messages and toxic messages
        for friend_username, friend_data in friends.items():
            # Skip if this is a nested structure (like 'ndhoang: { sdh241: {...} }')
            if isinstance(friend_data, dict) and any(key.isdigit() for key in friend_data.keys()):
                continue
                
            friend = User.find_by_username(friend_username)
            if not friend:
                continue
                
            # Count total messages between current user and friend
            total_messages = mongo.db.messages.count_documents({
                "$or": [
                    {"senderId": ObjectId(current_user_id), "receiverId": ObjectId(friend['_id'])},
                    {"senderId": ObjectId(friend['_id']), "receiverId": ObjectId(current_user_id)}
                ]
            })
            
            # Count toxic messages between current user and friend
            toxic_messages = mongo.db.messages.count_documents({
                "$or": [
                    {"senderId": ObjectId(current_user_id), "receiverId": ObjectId(friend['_id']), "isToxic": True},
                    {"senderId": ObjectId(friend['_id']), "receiverId": ObjectId(current_user_id), "isToxic": True}
                ]
            })
            
            # Update counters for current user
            mongo.db.users.update_one(
                {"_id": ObjectId(current_user_id)},
                {"$set": {
                    f"friends.{friend_username}": {
                        "messageCounter": total_messages,
                        "toxicCounter": toxic_messages
                    }
                }}
            )
            
            # Get friend's current friends
            friend_doc = mongo.db.users.find_one({"_id": ObjectId(friend['_id'])})
            friend_friends = friend_doc.get('friends', {}) if friend_doc else {}
            
            # Update counters for friend
            mongo.db.users.update_one(
                {"_id": ObjectId(friend['_id'])},
                {"$set": {
                    f"friends.{current_user['username']}": {
                        "messageCounter": total_messages,
                        "toxicCounter": toxic_messages
                    }
                }}
            )
            
        return jsonify({'message': 'Counters updated successfully'}), 200
        
    except Exception as e:
        return jsonify({'message': str(e)}), 500
