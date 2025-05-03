# app/routes/chat.py
from flask import request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from . import chat_bp
from app import mongo
from app.models.message import Message
from app.models.user import User
from bson import ObjectId
import random
import requests

# TEST_MODE: Set to True to always return False for toxicity check (for testing)
# Set to False to use the actual model for toxicity check
TEST_MODE = True

# Get all messages endpoint
@chat_bp.route('/messages', methods=['GET'])
def get_messages():
    # Reserved for future implementation
    pass

# Check toxicity endpoint
@chat_bp.route('/check_toxicity', methods=['POST'])
@jwt_required()
def check_toxicity():
    data = request.get_json()
    text = data.get('text')

    if not text:
        return jsonify({'message': 'Text is required'}), 400

    try:
        # Proxy the request to the external API
        external_url = 'https://bkchat-classifier-92895094743.asia-southeast1.run.app/inference'
        resp = requests.post(external_url, json={"message": text})
        if resp.status_code == 200:
            result = resp.json()
            is_toxic = result.get('result') == '1'
            return jsonify({'isToxic': is_toxic}), 200
        else:
            return jsonify({'message': 'External API error', 'status_code': resp.status_code}), 500
    except Exception as e:
        return jsonify({'message': str(e)}), 500

# Send message endpoint
@chat_bp.route('/messages', methods=['POST'])
@jwt_required()
def send_message():
    data = request.get_json()
    receiver_id = data.get('receiverId')
    text = data.get('text')
    is_emoji = data.get('isEmoji', False)
    is_toxic = data.get('isToxic', False)  # Get is_toxic from request instead of checking

    if not receiver_id or not text:
        return jsonify({'message': 'Receiver ID and text are required'}), 400

    # Get current user ID from JWT token
    current_user_id = get_jwt_identity()
    current_user = User.find_by_id(current_user_id)
    
    if not current_user:
        return jsonify({'message': 'User not found'}), 404

    try:
        message_id = Message.create(current_user_id, receiver_id, text, is_emoji, is_toxic)
        return jsonify({
            'message': 'Message sent successfully', 
            'messageId': message_id,
            'isToxic': is_toxic
        }), 201
    except Exception as e:
        return jsonify({'message': str(e)}), 500

# Get conversation history endpoint
@chat_bp.route('/conversations', methods=['GET'])
def get_conversations():
    # Reserved for future implementation
    pass

# Create new conversation endpoint
@chat_bp.route('/conversations', methods=['POST'])
def create_conversation():
    # Reserved for future implementation
    pass

# Get conversation by ID endpoint
@chat_bp.route('/conversations/<conversation_id>', methods=['GET'])
def get_conversation(conversation_id):
    # Reserved for future implementation
    pass

# Delete conversation endpoint
@chat_bp.route('/conversations/<conversation_id>', methods=['DELETE'])
def delete_conversation(conversation_id):
    # Reserved for future implementation
    pass

import jwt
from app.models.message import Message
from app.models.user import User
from bson import ObjectId

@chat_bp.route('/create_messages', methods=['POST'])
def create_messages():
    data = request.get_json()
    messages = data.get('messages')

    if not messages:
        return jsonify({'message': 'Messages are required'}), 400

    for message in messages:
        senderId = message.get('senderId')
        text = message.get('text')
        timestamp = message.get('timestamp')
        isEmoji = message.get('isEmoji', False)

        # Assuming senderId is tcminh.sdh241 and receiverId is ndhoang.sdh241
        sender_user = User.find_by_username("tcminh.sdh241")
        receiver_user = User.find_by_username("ndhoang.sdh241")

        if not sender_user or not receiver_user:
            return jsonify({'message': 'Sender or receiver not found'}), 400

        sender_id = str(sender_user['_id'])
        receiver_id = str(receiver_user['_id'])

        Message.create(sender_id, receiver_id, text, isEmoji)

    return jsonify({'message': 'Messages created successfully'}), 201

@chat_bp.route('/get_contacts_and_conversations', methods=['GET'])
@jwt_required()
def get_contacts_and_conversations():
    try:
        # Get user ID from JWT token
        user_id = get_jwt_identity()
        print(f"User ID from token: {user_id}")  # Debug log
        
        user = User.find_by_id(user_id)
        print(f"User found: {user is not None}")  # Debug log

        if not user:
            return jsonify({'message': 'User not found'}), 404

        # Get friends from the friends object
        friends = []
        friends_data = user.get('friends', {})
        print(f"Friends data: {friends_data}")  # Debug log
        
        for friend_username in friends_data.keys():
            try:
                friend = User.find_by_username(friend_username)
                if friend:
                    friend['_id'] = str(friend['_id'])
                    friends.append(friend)
            except Exception as e:
                print(f"Error finding friend {friend_username}: {e}")

        print(f"Found {len(friends)} friends")  # Debug log

        conversations = {}
        # Get all messages where user is either sender or receiver, sorted by timestamp descending
        messages = mongo.db.messages.find({
            "$or": [
                {"senderId": ObjectId(user_id)},
                {"receiverId": ObjectId(user_id)}
            ]
        }).sort("timestamp", -1)  # Sort by timestamp descending
        
        # Process messages and keep only last 10 per conversation
        for message in messages:
            message['_id'] = str(message['_id'])
            message['senderId'] = str(message['senderId'])
            message['receiverId'] = str(message['receiverId'])
            senderId = message['senderId']
            receiverId = message['receiverId']
            
            # Create a consistent conversation ID regardless of sender/receiver order
            conversationId = '_'.join(sorted([senderId, receiverId]))
            
            if conversationId not in conversations:
                conversations[conversationId] = []
            
            # Only add message if we haven't reached 10 messages for this conversation
            if len(conversations[conversationId]) < 10:
                conversations[conversationId].append(message)

        # Sort messages within each conversation by timestamp ascending
        for conversationId in conversations:
            conversations[conversationId].sort(key=lambda x: x['timestamp'])

        print(f"Found {len(conversations)} conversations")  # Debug log
        return jsonify({'contacts': friends, 'conversations': conversations}), 200
        
    except Exception as e:
        print(f"Error in get_contacts_and_conversations: {str(e)}")  # Debug log
        return jsonify({'message': str(e)}), 500

@chat_bp.route('/messages/<contact_id>', methods=['GET'])
@jwt_required()
def get_messages_by_contact(contact_id):
    # Get current user ID from JWT token
    current_user_id = get_jwt_identity()
    current_user = User.find_by_id(current_user_id)
    
    if not current_user:
        return jsonify({'message': 'User not found'}), 404

    try:
        # Find messages between current user and contact, sorted by timestamp descending
        # and limited to 10 messages
        messages = mongo.db.messages.find({
            "$or": [
                {"senderId": ObjectId(current_user_id), "receiverId": ObjectId(contact_id)},
                {"senderId": ObjectId(contact_id), "receiverId": ObjectId(current_user_id)}
            ]
        }).sort("timestamp", -1).limit(10)  # Sort by timestamp descending and limit to 10

        formatted_messages = []
        for message in messages:
            message['_id'] = str(message['_id'])
            message['senderId'] = str(message['senderId'])
            message['receiverId'] = str(message['receiverId'])
            formatted_messages.append(message)

        # Sort messages in ascending order before returning
        formatted_messages.sort(key=lambda x: x['timestamp'])

        return jsonify({'messages': formatted_messages}), 200
    except Exception as e:
        return jsonify({'message': str(e)}), 500

# Delete message endpoint
@chat_bp.route('/messages/<message_id>', methods=['DELETE'])
def delete_message(message_id):
    try:
        # Convert string ID to ObjectId
        message_id = ObjectId(message_id)
        
        # Delete the message
        result = mongo.db.messages.delete_one({'_id': message_id})
        
        if result.deleted_count > 0:
            return jsonify({'message': 'Message deleted successfully'}), 200
        else:
            return jsonify({'message': 'Message not found'}), 404
            
    except Exception as e:
        return jsonify({'message': str(e)}), 500

# Update message feedback endpoint
@chat_bp.route('/messages/<message_id>', methods=['PATCH'])
def update_message_feedback(message_id):
    try:
        data = request.get_json()
        user_feedback = data.get('userFeedback')
        
        if not user_feedback:
            return jsonify({'message': 'User feedback is required'}), 400
            
        # Convert string ID to ObjectId
        message_id = ObjectId(message_id)
        
        # Update the message
        result = mongo.db.messages.update_one(
            {'_id': message_id},
            {'$set': {'userFeedback': user_feedback}}
        )
        
        if result.modified_count > 0:
            return jsonify({'message': 'Message feedback updated successfully'}), 200
        else:
            return jsonify({'message': 'Message not found'}), 404
            
    except Exception as e:
        return jsonify({'message': str(e)}), 500
