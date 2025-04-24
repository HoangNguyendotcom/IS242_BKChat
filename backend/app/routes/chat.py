from flask import Blueprint, render_template, session, redirect, url_for

chat_bp = Blueprint('chat', __name__, url_prefix='/chat')

@chat_bp.route('/')
def main():
    if 'user' not in session:
        return redirect(url_for('auth.login'))
    
    return render_template('main.html', username=session['user'])

@chat_bp.route('/send', methods=['POST'])
def send_message():
    # Logic to send a message
    pass

@chat_bp.route('/receive')
def receive_messages():
    # Logic to receive messages
    pass
