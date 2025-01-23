import sqlite3
import os
from datetime import datetime
import sys
sys.path.append("helpers")
from constants import SQL_DB_NAME


SQL_DB_PATH = os.path.join(os.getcwd(), SQL_DB_NAME)
if not os.path.exists(SQL_DB_PATH):
    os.mkdir(SQL_DB_PATH)


def delete_chat_session(session_id):
    """Delete a chat session and all its messages"""
    conn = get_db_connection()
    try:
        conn.execute('DELETE FROM application_logs WHERE session_id = ?', (session_id,))
        conn.commit()
        return True
    except Exception as e:
        print(f"Error deleting chat session: {e}")
        return False
    finally:
        conn.close()

    
def get_db_connection():
    conn = sqlite3.connect(f'{SQL_DB_PATH}/logs.db')
    conn.row_factory = sqlite3.Row
    return conn

def create_application_logs():
    conn = get_db_connection()
    conn.execute('''CREATE TABLE IF NOT EXISTS application_logs
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     session_id TEXT,
                     user_query TEXT,
                     response TEXT,
                     system_prompt TEXT,
                     retrieved_context TEXT,
                     model TEXT,
                     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.close()

def insert_application_logs(session_id, user_query, response, system_prompt, retrieved_context, model):
    conn = get_db_connection()
    conn.execute('INSERT INTO application_logs (session_id, user_query, response, system_prompt, retrieved_context, model) VALUES (?, ?, ?, ?, ?, ?)',
                 (session_id, user_query, response, system_prompt, retrieved_context, model))
    conn.commit()
    conn.close()

def get_chat_history(session_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT user_query, response FROM application_logs WHERE session_id = ? ORDER BY created_at', (session_id,))
    messages = []
    for row in cursor.fetchall():
        messages.extend([
            {"role": "user", "content": row['user_query']},
            {"role": "assistant", "content": row['response']}
        ])
    conn.close()
    return messages

def create_document_store():
    conn = get_db_connection()
    conn.execute('''CREATE TABLE IF NOT EXISTS document_store
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     filename TEXT,
                     upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.close()

def insert_document_record(filename):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('INSERT INTO document_store (filename) VALUES (?)', (filename,))
    file_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return file_id

def delete_document_record(file_id):
    conn = get_db_connection()
    conn.execute('DELETE FROM document_store WHERE id = ?', (file_id,))
    conn.commit()
    conn.close()
    return True

def get_all_documents():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT id, filename, upload_timestamp FROM document_store ORDER BY upload_timestamp DESC')
    documents = cursor.fetchall()
    conn.close()
    return [dict(doc) for doc in documents]


def get_all_chat_sessions():
    """Retrieve all unique chat sessions with their first message and timestamp"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT DISTINCT 
            session_id,
            first_value(user_query) OVER (PARTITION BY session_id ORDER BY created_at) as first_message,
            first_value(created_at) OVER (PARTITION BY session_id ORDER BY created_at) as started_at
        FROM application_logs
        GROUP BY session_id
        ORDER BY started_at DESC
    ''')
    sessions = cursor.fetchall()
    conn.close()
    return [dict(session) for session in sessions]


# Initialize the database tables
create_application_logs()
create_document_store()