from flask import request, jsonify
from . import friends_bp
from app.models.user import User

@friends_bp.route('/add_friend', methods=['POST'])
def add_friend():
    data = request.get_json()
    username = data.get('username')
    friend_username = data.get('friend_username')

    if not username or not friend_username:
        return jsonify({'message': 'Username and friend_username are required'}), 400

    success = User.add_friend(username, friend_username)

    if success:
        return jsonify({'message': 'Friend added successfully'}), 200
    else:
        return jsonify({'message': 'Could not add friend'}), 400
