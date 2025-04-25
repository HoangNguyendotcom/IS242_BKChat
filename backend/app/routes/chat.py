# app/routes/chat.py
from flask import request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from . import chat_bp
from app import mongo
from app.services.toxicity_service import toxicity_service
from app.models.message import Message
from app.models.user import User
from bson import ObjectId

# Get all messages endpoint
@chat_bp.route('/messages', methods=['GET'])
def get_messages():
    # Reserved for future implementation
    pass

# Send message endpoint
@chat_bp.route('/messages', methods=['POST'])
@jwt_required()
def send_message():
    data = request.get_json()
    receiver_id = data.get('receiverId')
    text = data.get('text')
    is_emoji = data.get('isEmoji', False)

    if not receiver_id or not text:
        return jsonify({'message': 'Receiver ID and text are required'}), 400

    # Get current user ID from JWT token
    current_user_id = get_jwt_identity()
    current_user = User.find_by_id(current_user_id)
    
    if not current_user:
        return jsonify({'message': 'User not found'}), 404

    try:
        # Check message toxicity using ML model
        is_toxic = toxicity_service.check_toxicity(text)
        
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
        messages = mongo.db.messages.find({
            "$or": [
                {"senderId": ObjectId(user_id)},
                {"receiverId": ObjectId(user_id)}
            ]
        })
        
        for message in messages:
            message['_id'] = str(message['_id'])
            message['senderId'] = str(message['senderId'])
            message['receiverId'] = str(message['receiverId'])
            senderId = message['senderId']
            receiverId = message['receiverId']
            conversationId = str(senderId) + '_' + str(receiverId)
            if conversationId not in conversations:
                conversations[conversationId] = []
            conversations[conversationId].append(message)

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
        # Find messages between current user and contact
        messages = mongo.db.messages.find({
            "$or": [
                {"senderId": ObjectId(current_user_id), "receiverId": ObjectId(contact_id)},
                {"senderId": ObjectId(contact_id), "receiverId": ObjectId(current_user_id)}
            ]
        }).sort("timestamp", 1)  # Sort by timestamp ascending

        formatted_messages = []
        for message in messages:
            message['_id'] = str(message['_id'])
            message['senderId'] = str(message['senderId'])
            message['receiverId'] = str(message['receiverId'])
            formatted_messages.append(message)

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
