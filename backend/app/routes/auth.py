# app/routes/auth.py
from flask import request, jsonify
from . import auth_bp
from app.services.auth_service import AuthService
from flask_jwt_extended import jwt_required, get_jwt_identity
from app import mongo
from bson import ObjectId

# Login endpoint
@auth_bp.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({'message': 'Username and password are required'}), 400

    user, token = AuthService.login(username, password)

    if user:
        return jsonify({'user': user, 'access_token': token}), 200
    else:
        return jsonify({'message': 'Invalid username or password'}), 401

# Signup endpoint
@auth_bp.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()
    name = data.get('name')
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')

    if not name or not username or not email or not password:
        return jsonify({'message': 'Name, username, email, and password are required'}), 400

    new_user, error_message = AuthService.signup(name, username, email, password)

    if new_user:
        return jsonify({'message': 'User created successfully'}), 201
    else:
        return jsonify({'message': error_message}), 400

# Register endpoint
@auth_bp.route('/register', methods=['POST'])
def register():
    # Reserved for future implementation
    pass

# Logout endpoint
@auth_bp.route('/logout', methods=['POST'])
def logout():
    # Reserved for future implementation
    pass

# Get all users endpoint
@auth_bp.route('/users', methods=['GET'])
@jwt_required()
def get_all_users():
    try:
        # Get current user's ID
        current_user_id = get_jwt_identity()
        
        # Fetch all users except the current user
        users = list(mongo.db.users.find(
            {"_id": {"$ne": ObjectId(current_user_id)}},
            {"password": 0}  # Exclude password field
        ))
        
        # Convert ObjectId to string for JSON serialization
        for user in users:
            user['_id'] = str(user['_id'])
        
        return jsonify({
            'users': users,
            'message': 'Users fetched successfully'
        }), 200
        
    except Exception as e:
        return jsonify({
            'message': 'Error fetching users',
            'error': str(e)
        }), 500
