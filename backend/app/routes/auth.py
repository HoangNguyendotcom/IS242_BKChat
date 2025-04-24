# app/routes/auth.py
from flask import Blueprint, render_template, redirect, url_for

auth_bp = Blueprint('auth', __name__, url_prefix='/auth')

# This route won't match '/login', it matches '/auth/login'
@auth_bp.route('/login')
def login():
    return render_template('index.html')