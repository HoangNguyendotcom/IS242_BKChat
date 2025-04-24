from flask_bcrypt import Bcrypt
from app.models.user import User

bcrypt = Bcrypt()

class AuthService:
    @staticmethod
    def login(username, password):
        user = User.find_by_username(username)

        if not user:
            return None, None

        if bcrypt.check_password_hash(user['password'], password):
            return user, "test_token"
        else:
            return None, None

    @staticmethod
    def signup(username, email, password):
        existing_user = User.find_by_username(username)
        if existing_user:
            return None, "Username already exists"

        existing_email = User.find_by_email(email)
        if existing_email:
            return None, "Email already exists"

        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        # Assuming name is not required for signup, setting it to username
        new_user_id = User.create(name=username, username=username, email=email, password_hash=hashed_password)
        new_user = User.find_by_username(username)

        return new_user, None
