# app/routes/settings.py
from flask import request, jsonify
from . import settings_bp

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
