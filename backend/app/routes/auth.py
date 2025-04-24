# app/routes/auth.py
from flask import request, jsonify
from . import auth_bp
from app.services.auth_service import AuthService

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
        return jsonify({'access_token': token}), 200
    else:
        return jsonify({'message': 'Invalid username or password'}), 401

# Signup endpoint
@auth_bp.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')

    if not username or not email or not password:
        return jsonify({'message': 'Username, email, and password are required'}), 400

    new_user, error_message = AuthService.signup(username, email, password)

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
