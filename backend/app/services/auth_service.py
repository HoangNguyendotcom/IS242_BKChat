from flask_bcrypt import Bcrypt
from flask_jwt_extended import create_access_token
from app.models.user import User

bcrypt = Bcrypt()

class AuthService:
    @staticmethod
    def login(username, password):
        user = User.find_by_username(username)

        if not user:
            return None, None

        if bcrypt.check_password_hash(user['password'], password):
            # Create JWT token
            access_token = create_access_token(identity=user['_id'])
            return user, access_token
        else:
            return None, None

    @staticmethod
    def signup(name, username, email, password):
        existing_user = User.find_by_username(username)
        if existing_user:
            return None, "Username already exists"

        existing_email = User.find_by_email(email)
        if existing_email:
            return None, "Email already exists"

        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        new_user_id = User.create(
            name=name,
            username=username,
            email=email,
            password_hash=hashed_password,
            avatar="/avatars/avatar.jpeg"  # Set default avatar
        )
        new_user = User.find_by_username(username)

        return new_user, None
