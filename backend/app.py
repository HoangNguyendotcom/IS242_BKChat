from flask import Flask, render_template, request, session, jsonify
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_cors import CORS
import redis
import os
import json
import uuid
from datetime import datetime
from pymongo import MongoClient
from bson.objectid import ObjectId
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev_secret_key')
CORS(app)

# Redis setup for real-time messaging
redis_url = os.environ.get('REDIS_URL', 'redis://localhost:6379')
redis_client = redis.from_url(redis_url, ssl=True)

# MongoDB setup
mongo_uri = os.environ.get('MONGO_URI', 'mongodb://localhost:27017/chat_app')
mongo_client = MongoClient(mongo_uri)
db = mongo_client.get_database()

# Collections
users_collection = db.users
messages_collection = db.messages
conversations_collection = db.conversations
friend_requests_collection = db.friend_requests

# Initialize SocketIO with Redis as message queue
socketio = SocketIO(app, cors_allowed_origins="*", message_queue=redis_url)

# User routes
@app.route('/api/users/register', methods=['POST'])
def register():
    data = request.json
    
    # Check if user already exists
    if users_collection.find_one({'email': data['email']}):
        return jsonify({'error': 'Email already registered'}), 400
    
    if users_collection.find_one({'username': data['username']}):
        return jsonify({'error': 'Username already taken'}), 400
    
    # Create new user
    user = {
        'name': data['name'],
        'email': data['email'],
        'username': data['username'],
        'password': generate_password_hash(data['password']),
        'created_at': datetime.utcnow(),
        'avatar': data.get('avatar', ''),
        'bio': data.get('bio', ''),
        'settings': {
            'notifications': {
                'push': True,
                'email': True
            },
            'privacy': {
                'online_status': True,
                'read_receipts': True,
                'typing_indicators': True
            },
            'theme': 'system',
            'language': 'en'
        }
    }
    
    result = users_collection.insert_one(user)
    user['_id'] = str(result.inserted_id)
    
    # Remove password before returning
    user.pop('password', None)
    
    return jsonify(user), 201

@app.route('/api/users/login', methods=['POST'])
def login():
    data = request.json
    
    # Find user by email
    user = users_collection.find_one({'email': data['email']})
    
    if not user or not check_password_hash(user['password'], data['password']):
        return jsonify({'error': 'Invalid credentials'}), 401
    
    # Create session
    session['user_id'] = str(user['_id'])
    
    # Remove password before returning
    user['_id'] = str(user['_id'])
    user.pop('password', None)
    
    return jsonify(user), 200

@app.route('/api/users/logout', methods=['POST'])
def logout():
    session.pop('user_id', None)
    return jsonify({'message': 'Logged out successfully'}), 200

@app.route('/api/users/profile', methods=['GET'])
def get_profile():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    user = users_collection.find_one({'_id': ObjectId(session['user_id'])})
    
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    # Remove password before returning
    user['_id'] = str(user['_id'])
    user.pop('password', None)
    
    return jsonify(user), 200

@app.route('/api/users/profile', methods=['PUT'])
def update_profile():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    data = request.json
    
    # Fields that can be updated
    allowed_fields = ['name', 'username', 'avatar', 'bio', 'settings']
    update_data = {k: v for k, v in data.items() if k in allowed_fields}
    
    result = users_collection.update_one(
        {'_id': ObjectId(session['user_id'])},
        {'$set': update_data}
    )
    
    if result.modified_count == 0:
        return jsonify({'error': 'User not found or no changes made'}), 404
    
    return jsonify({'message': 'Profile updated successfully'}), 200

# Friend routes
@app.route('/api/friends', methods=['GET'])
def get_friends():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    user_id = session['user_id']
    
    # Find user's friends
    user = users_collection.find_one({'_id': ObjectId(user_id)})
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    # Get friends list with online status
    friends = []
    for friend_id in user.get('friends', []):
        friend = users_collection.find_one({'_id': ObjectId(friend_id)})
        if friend:
            friends.append({
                'id': str(friend['_id']),
                'name': friend['name'],
                'username': friend['username'],
                'avatar': friend.get('avatar', ''),
                'online': redis_client.get(f"user:{friend_id}:online") is not None,
                'last_active': friend.get('last_active', None)
            })
    
    return jsonify(friends), 200

@app.route('/api/friends/requests', methods=['GET'])
def get_friend_requests():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    user_id = session['user_id']
    
    # Find pending friend requests
    requests = friend_requests_collection.find({
        'receiver_id': user_id,
        'status': 'pending'
    })
    
    result = []
    for req in requests:
        sender = users_collection.find_one({'_id': ObjectId(req['sender_id'])})
        if sender:
            result.append({
                'request_id': str(req['_id']),
                'sender': {
                    'id': str(sender['_id']),
                    'name': sender['name'],
                    'username': sender['username'],
                    'avatar': sender.get('avatar', ''),
                    'online': redis_client.get(f"user:{req['sender_id']}:online") is not None
                },
                'created_at': req['created_at']
            })
    
    return jsonify(result), 200

@app.route('/api/friends/requests', methods=['POST'])
def send_friend_request():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    data = request.json
    sender_id = session['user_id']
    receiver_id = data['receiver_id']
    
    # Check if users exist
    sender = users_collection.find_one({'_id': ObjectId(sender_id)})
    receiver = users_collection.find_one({'_id': ObjectId(receiver_id)})
    
    if not sender or not receiver:
        return jsonify({'error': 'User not found'}), 404
    
    # Check if already friends
    if receiver_id in sender.get('friends', []):
        return jsonify({'error': 'Already friends'}), 400
    
    # Check if request already exists
    existing_request = friend_requests_collection.find_one({
        'sender_id': sender_id,
        'receiver_id': receiver_id,
        'status': 'pending'
    })
    
    if existing_request:
        return jsonify({'error': 'Friend request already sent'}), 400
    
    # Create friend request
    request_data = {
        'sender_id': sender_id,
        'receiver_id': receiver_id,
        'status': 'pending',
        'created_at': datetime.utcnow()
    }
    
    result = friend_requests_collection.insert_one(request_data)
    
    return jsonify({'message': 'Friend request sent', 'request_id': str(result.inserted_id)}), 201

@app.route('/api/friends/requests/<request_id>', methods=['PUT'])
def respond_to_friend_request(request_id):
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    data = request.json
    user_id = session['user_id']
    action = data['action']  # 'accept' or 'decline'
    
    # Find the request
    friend_request = friend_requests_collection.find_one({
        '_id': ObjectId(request_id),
        'receiver_id': user_id,
        'status': 'pending'
    })
    
    if not friend_request:
        return jsonify({'error': 'Friend request not found'}), 404
    
    if action == 'accept':
        # Update request status
        friend_requests_collection.update_one(
            {'_id': ObjectId(request_id)},
            {'$set': {'status': 'accepted'}}
        )
        
        # Add each user to the other's friends list
        users_collection.update_one(
            {'_id': ObjectId(user_id)},
            {'$addToSet': {'friends': friend_request['sender_id']}}
        )
        
        users_collection.update_one(
            {'_id': ObjectId(friend_request['sender_id'])},
            {'$addToSet': {'friends': user_id}}
        )
        
        return jsonify({'message': 'Friend request accepted'}), 200
    
    elif action == 'decline':
        # Update request status
        friend_requests_collection.update_one(
            {'_id': ObjectId(request_id)},
            {'$set': {'status': 'declined'}}
        )
        
        return jsonify({'message': 'Friend request declined'}), 200
    
    return jsonify({'error': 'Invalid action'}), 400

# Conversation routes
@app.route('/api/conversations', methods=['GET'])
def get_conversations():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    user_id = session['user_id']
    
    # Find conversations where user is a participant
    conversations = conversations_collection.find({
        'participants': user_id
    })
    
    result = []
    for conv in conversations:
        # Get the other participant(s)
        other_participants = []
        for participant_id in conv['participants']:
            if participant_id != user_id:
                user = users_collection.find_one({'_id': ObjectId(participant_id)})
                if user:
                    other_participants.append({
                        'id': str(user['_id']),
                        'name': user['name'],
                        'username': user['username'],
                        'avatar': user.get('avatar', ''),
                        'online': redis_client.get(f"user:{participant_id}:online") is not None
                    })
        
        # Get last message
        last_message = messages_collection.find_one(
            {'conversation_id': str(conv['_id'])},
            sort=[('created_at', -1)]
        )
        
        result.append({
            'id': str(conv['_id']),
            'name': conv.get('name', ''),
            'is_group': conv.get('is_group', False),
            'participants': other_participants,
            'last_message': {
                'content': last_message['content'] if last_message else '',
                'sender_id': last_message['sender_id'] if last_message else '',
                'created_at': last_message['created_at'] if last_message else None
            },
            'unread_count': 0  # To be implemented
        })
    
    return jsonify(result), 200

@app.route('/api/conversations', methods=['POST'])
def create_conversation():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    data = request.json
    user_id = session['user_id']
    
    # Ensure participants list includes the current user
    participants = data.get('participants', [])
    if user_id not in participants:
        participants.append(user_id)
    
    # Check if this is a one-on-one conversation that already exists
    if len(participants) == 2 and not data.get('is_group', False):
        existing_conv = conversations_collection.find_one({
            'participants': {'$all': participants, '$size': 2},
            'is_group': False
        })
        
        if existing_conv:
            return jsonify({
                'message': 'Conversation already exists',
                'conversation_id': str(existing_conv['_id'])
            }), 200
    
    # Create new conversation
    conversation = {
        'name': data.get('name', ''),
        'is_group': data.get('is_group', False),
        'participants': participants,
        'created_at': datetime.utcnow(),
        'created_by': user_id
    }
    
    result = conversations_collection.insert_one(conversation)
    
    return jsonify({
        'message': 'Conversation created',
        'conversation_id': str(result.inserted_id)
    }), 201

@app.route('/api/conversations/<conversation_id>/messages', methods=['GET'])
def get_messages(conversation_id):
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    user_id = session['user_id']
    
    # Check if user is a participant
    conversation = conversations_collection.find_one({
        '_id': ObjectId(conversation_id),
        'participants': user_id
    })
    
    if not conversation:
        return jsonify({'error': 'Conversation not found'}), 404
    
    # Get messages
    messages = messages_collection.find(
        {'conversation_id': conversation_id},
        sort=[('created_at', 1)]
    )
    
    result = []
    for msg in messages:
        result.append({
            'id': str(msg['_id']),
            'content': msg['content'],
            'sender_id': msg['sender_id'],
            'created_at': msg['created_at']
        })
    
    return jsonify(result), 200

# Socket.IO events
@socketio.on('connect')
def handle_connect():
    if 'user_id' not in session:
        return False
    
    user_id = session['user_id']
    
    # Mark user as online
    redis_client.set(f"user:{user_id}:online", 1)
    redis_client.expire(f"user:{user_id}:online", 3600)  # Expire after 1 hour
    
    # Join user's personal room
    join_room(f"user:{user_id}")
    
    # Join rooms for all conversations
    conversations = conversations_collection.find({'participants': user_id})
    for conv in conversations:
        join_room(f"conversation:{str(conv['_id'])}")
    
    emit('user_status', {'user_id': user_id, 'status': 'online'}, broadcast=True)

@socketio.on('disconnect')
def handle_disconnect():
    if 'user_id' not in session:
        return
    
    user_id = session['user_id']
    
    # Mark user as offline
    redis_client.delete(f"user:{user_id}:online")
    
    # Update last active timestamp
    users_collection.update_one(
        {'_id': ObjectId(user_id)},
        {'$set': {'last_active': datetime.utcnow()}}
    )
    
    emit('user_status', {'user_id': user_id, 'status': 'offline'}, broadcast=True)

@socketio.on('message')
def handle_message(data):
    if 'user_id' not in session:
        return
    
    user_id = session['user_id']
    conversation_id = data['conversation_id']
    content = data['content']
    
    # Check if user is a participant
    conversation = conversations_collection.find_one({
        '_id': ObjectId(conversation_id),
        'participants': user_id
    })
    
    if not conversation:
        return
    
    # Create message
    message = {
        'conversation_id': conversation_id,
        'sender_id': user_id,
        'content': content,
        'created_at': datetime.utcnow()
    }
    
    # Check for toxic content (placeholder for ML model integration)
    # This would be replaced with actual ML model call
    is_toxic = False
    if is_toxic:
        message['flagged'] = True
        message['flag_reason'] = 'toxic_content'
    
    # Save message to database
    result = messages_collection.insert_one(message)
    message['id'] = str(result.inserted_id)
    
    # Broadcast to all participants in the conversation
    emit('message', message, room=f"conversation:{conversation_id}")
    
    # Update conversation with last message
    conversations_collection.update_one(
        {'_id': ObjectId(conversation_id)},
        {'$set': {'last_message': message}}
    )

@socketio.on('typing')
def handle_typing(data):
    if 'user_id' not in session:
        return
    
    user_id = session['user_id']
    conversation_id = data['conversation_id']
    is_typing = data['is_typing']
    
    # Check if user is a participant
    conversation = conversations_collection.find_one({
        '_id': ObjectId(conversation_id),
        'participants': user_id
    })
    
    if not conversation:
        return
    
    # Broadcast typing status to conversation participants
    emit('typing', {
        'user_id': user_id,
        'conversation_id': conversation_id,
        'is_typing': is_typing
    }, room=f"conversation:{conversation_id}")

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)
