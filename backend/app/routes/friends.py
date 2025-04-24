# app/routes/friends.py
from flask import request, jsonify
from . import friends_bp

# Get all friends endpoint
@friends_bp.route('/', methods=['GET'])
def get_friends():
    # Reserved for future implementation
    pass

# Add friend endpoint
@friends_bp.route('/', methods=['POST'])
def add_friend():
    # Reserved for future implementation
    pass

# Remove friend endpoint
@friends_bp.route('/<friend_id>', methods=['DELETE'])
def remove_friend(friend_id):
    # Reserved for future implementation
    pass

# Get friend requests endpoint
@friends_bp.route('/requests', methods=['GET'])
def get_friend_requests():
    # Reserved for future implementation
    pass

# Accept friend request endpoint
@friends_bp.route('/requests/<request_id>/accept', methods=['POST'])
def accept_friend_request(request_id):
    # Reserved for future implementation
    pass

# Reject friend request endpoint
@friends_bp.route('/requests/<request_id>/reject', methods=['POST'])
def reject_friend_request(request_id):
    # Reserved for future implementation
    pass