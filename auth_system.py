"""
Database-backed Authentication System for DINE MATE
"""

import sqlite3
import hashlib
import streamlit as st
from datetime import datetime
import json

import os
from pathlib import Path

# Database setup - portable path resolution with env var support
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = os.environ.get("FOOD_RECOMMENDER_DB_PATH", str(BASE_DIR / "food_recommender.db"))

def init_database():
    """Initialize SQLite database with users and admins tables."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            email TEXT UNIQUE,
            role TEXT DEFAULT 'user',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP,
            profile_data TEXT
        )
    ''')
    
    # Create admins table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS admins (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            email TEXT UNIQUE,
            admin_code TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP
        )
    ''')
    
    # Insert default admin if not exists
    cursor.execute('SELECT COUNT(*) FROM admins WHERE username = ?', ('admin',))
    if cursor.fetchone()[0] == 0:
        admin_hash = hash_password('admin123')
        cursor.execute('''
            INSERT INTO admins (username, password_hash, email, admin_code)
            VALUES (?, ?, ?, ?)
        ''', ('admin', admin_hash, 'admin@foodrecommender.com', 'ADMIN2024'))
    
    # Insert default users if not exists
    default_users = [
        ('user1', 'password1', 'user1@example.com'),
        ('user2', 'password2', 'user2@example.com'),
        ('demo', 'demo123', 'demo@example.com')
    ]
    
    for username, password, email in default_users:
        cursor.execute('SELECT COUNT(*) FROM users WHERE username = ?', (username,))
        if cursor.fetchone()[0] == 0:
            user_hash = hash_password(password)
            cursor.execute('''
                INSERT INTO users (username, password_hash, email, role)
                VALUES (?, ?, ?, ?)
            ''', (username, user_hash, email, 'user'))
    
    conn.commit()
    conn.close()

def hash_password(password):
    """Hash a password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()

def check_password(hashed_password, user_password):
    """Check if the provided password matches the hashed password."""
    return hashed_password == hash_password(user_password)

def register_user(username, password, email):
    """Register a new user."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        # Check if username already exists
        cursor.execute('SELECT COUNT(*) FROM users WHERE username = ?', (username,))
        if cursor.fetchone()[0] > 0:
            return False, "Username already exists"
        
        # Check if email already exists
        cursor.execute('SELECT COUNT(*) FROM users WHERE email = ?', (email,))
        if cursor.fetchone()[0] > 0:
            return False, "Email already exists"
        
        # Insert new user
        password_hash = hash_password(password)
        cursor.execute('''
            INSERT INTO users (username, password_hash, email, role)
            VALUES (?, ?, ?, ?)
        ''', (username, password_hash, email, 'user'))
        
        conn.commit()
        return True, "User registered successfully"
    
    except Exception as e:
        return False, f"Registration error: {str(e)}"
    finally:
        conn.close()

def register_admin(username, password, email, admin_code):
    """Register a new admin."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        # Check admin code
        if admin_code != 'ADMIN2024':
            return False, "Invalid admin code"
        
        # Check if username already exists
        cursor.execute('SELECT COUNT(*) FROM admins WHERE username = ?', (username,))
        if cursor.fetchone()[0] > 0:
            return False, "Admin username already exists"
        
        # Check if email already exists
        cursor.execute('SELECT COUNT(*) FROM admins WHERE email = ?', (email,))
        if cursor.fetchone()[0] > 0:
            return False, "Admin email already exists"
        
        # Insert new admin
        password_hash = hash_password(password)
        cursor.execute('''
            INSERT INTO admins (username, password_hash, email, admin_code)
            VALUES (?, ?, ?, ?)
        ''', (username, password_hash, email, admin_code))
        
        conn.commit()
        return True, "Admin registered successfully"
    
    except Exception as e:
        return False, f"Admin registration error: {str(e)}"
    finally:
        conn.close()

def login_user(username, password):
    """Login a user."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            SELECT password_hash, role FROM users 
            WHERE username = ?
        ''', (username,))
        
        result = cursor.fetchone()
        if result and check_password(result[0], password):
            # Update last login
            cursor.execute('''
                UPDATE users SET last_login = ? WHERE username = ?
            ''', (datetime.now(), username))
            conn.commit()
            return True, result[1]  # Return success and role
        else:
            return False, "Invalid username or password"
    
    except Exception as e:
        return False, f"Login error: {str(e)}"
    finally:
        conn.close()

def login_admin(username, password):
    """Login an admin."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            SELECT password_hash FROM admins 
            WHERE username = ?
        ''', (username,))
        
        result = cursor.fetchone()
        if result and check_password(result[0], password):
            # Update last login
            cursor.execute('''
                UPDATE admins SET last_login = ? WHERE username = ?
            ''', (datetime.now(), username))
            conn.commit()
            return True, "admin"  # Return success and admin role
        else:
            return False, "Invalid admin username or password"
    
    except Exception as e:
        return False, f"Admin login error: {str(e)}"
    finally:
        conn.close()

def get_user_profile_from_db(username):
    """Get user profile from database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            SELECT profile_data FROM users WHERE username = ?
        ''', (username,))
        
        result = cursor.fetchone()
        if result and result[0]:
            return json.loads(result[0])
        return None
    
    except Exception as e:
        return None
    finally:
        conn.close()

def update_user_profile_in_db(username, profile):
    """Update user profile in database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        profile_json = json.dumps(profile)
        cursor.execute('''
            UPDATE users SET profile_data = ? WHERE username = ?
        ''', (profile_json, username))
        conn.commit()
        return True
    except Exception as e:
        return False
    finally:
        conn.close()

def init_auth_system():
    """Initialize the authentication system with database."""
    # Initialize database
    init_database()
    
    # Initialize session state
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.current_user = None
        st.session_state.user_role = None

def login(username, password, user_type="user"):
    """Authenticate a user or admin."""
    if user_type == "admin":
        success, result = login_admin(username, password)
    else:
        success, result = login_user(username, password)
    
    if success:
        st.session_state.logged_in = True
        st.session_state.current_user = username
        st.session_state.user_role = result
        return True
    return False

def logout():
    """Log out the current user."""
    st.session_state.logged_in = False
    st.session_state.current_user = None
    st.session_state.user_role = None

def is_admin():
    """Check if the current user is an admin."""
    return st.session_state.logged_in and st.session_state.user_role == 'admin'

def is_logged_in():
    """Check if a user is logged in."""
    return st.session_state.logged_in

def get_current_user():
    """Get the current logged-in user."""
    return st.session_state.current_user

def get_user_profile():
    """Get the profile of the current user from database."""
    if is_logged_in():
        return get_user_profile_from_db(st.session_state.current_user)
    return None

def update_user_profile(profile):
    """Update the profile of the current user in database."""
    if is_logged_in():
        return update_user_profile_in_db(st.session_state.current_user, profile)
    return False
