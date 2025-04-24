# app/routes/chat.py
from flask import request, jsonify
from . import chat_bp
from app import mongo

# Get all messages endpoint
@chat_bp.route('/messages', methods=['GET'])
def get_messages():
    # Reserved for future implementation
    pass

# Send message endpoint
@chat_bp.route('/messages', methods=['POST'])
def send_message():
    data = request.get_json()
    receiver_id = data.get('receiverId')
    text = data.get('text')
    is_emoji = data.get('isEmoji', False)

    if not receiver_id or not text:
        return jsonify({'message': 'Receiver ID and text are required'}), 400

    current_user = User.find_by_username("tcminh.sdh241")  # This should be replaced with the actual logged-in user
    if not current_user:
        return jsonify({'message': 'User not found'}), 404

    try:
        message_id = Message.create(str(current_user['_id']), receiver_id, text, is_emoji)
        return jsonify({'message': 'Message sent successfully', 'messageId': message_id}), 201
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
def get_contacts_and_conversations():
    user = User.find_by_username("tcminh.sdh241")

    if not user:
        return jsonify({'message': 'User not found'}), 404

    friend_usernames = user.get('friends', [])
    friends = []
    for friend_username in friend_usernames:
        try:
            friend = User.find_by_username(friend_username)
            if friend:
                friend['_id'] = str(friend['_id'])
                friends.append(friend)
        except Exception as e:
            print(f"Error finding friend {friend_username}: {e}")

    conversations = {}
    for message in mongo.db.messages.find():
        message['_id'] = str(message['_id'])
        message['senderId'] = str(message['senderId'])
        message['receiverId'] = str(message['receiverId'])
        senderId = message['senderId']
        receiverId = message['receiverId']
        conversationId = str(senderId) + '_' + str(receiverId)
        if conversationId not in conversations:
            conversations[conversationId] = []
        conversations[conversationId].append(message)

    return jsonify({'contacts': friends, 'conversations': conversations}), 200

@chat_bp.route('/messages/<contact_id>', methods=['GET'])
def get_messages_by_contact(contact_id):
    current_user = User.find_by_username("tcminh.sdh241")  # This should be replaced with the actual logged-in user
    if not current_user:
        return jsonify({'message': 'User not found'}), 404

    try:
        # Find messages between current user and contact
        messages = mongo.db.messages.find({
            "$or": [
                {"senderId": ObjectId(current_user['_id']), "receiverId": ObjectId(contact_id)},
                {"senderId": ObjectId(contact_id), "receiverId": ObjectId(current_user['_id'])}
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
