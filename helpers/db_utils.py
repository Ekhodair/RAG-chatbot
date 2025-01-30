import aiosqlite
import os
import sys

sys.path.append("helpers")
from constants import SQL_DB_NAME

SQL_DB_PATH = os.path.join(os.getcwd(), SQL_DB_NAME)
if not os.path.exists(SQL_DB_PATH):
    os.mkdir(SQL_DB_PATH)

async def get_db_connection():
    conn = await aiosqlite.connect(f'{SQL_DB_PATH}/logs.db')
    conn.row_factory = aiosqlite.Row
    return conn

async def delete_chat_session(session_id):
    """Delete a chat session and all its messages"""
    async with await get_db_connection() as conn:
        try:
            await conn.execute('DELETE FROM application_logs WHERE session_id = ?', (session_id,))
            await conn.commit()
            return True
        except Exception as e:
            print(f"Error deleting chat session: {e}")
            return False

async def create_application_logs():
    async with await get_db_connection() as conn:
        await conn.execute('''CREATE TABLE IF NOT EXISTS application_logs
                        (id INTEGER PRIMARY KEY AUTOINCREMENT,
                         session_id TEXT,
                         user_query TEXT,
                         response TEXT,
                         system_prompt TEXT,
                         retrieved_context TEXT,
                         model TEXT,
                         created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        await conn.commit()

async def insert_application_logs(session_id, user_query, response, system_prompt, retrieved_context, model):
    async with await get_db_connection() as conn:
        await conn.execute(
            'INSERT INTO application_logs (session_id, user_query, response, system_prompt, retrieved_context, model) VALUES (?, ?, ?, ?, ?, ?)',
            (session_id, user_query, response, system_prompt, retrieved_context, model))
        await conn.commit()

async def get_chat_history(session_id):
    async with await get_db_connection() as conn:
        cursor = await conn.execute('SELECT user_query, response FROM application_logs WHERE session_id = ? ORDER BY created_at', (session_id,))
        rows = await cursor.fetchall()
        messages = []
        for row in rows:
            messages.extend([
                {"role": "user", "content": row['user_query']},
                {"role": "assistant", "content": row['response']}
            ])
        return messages

async def create_document_store():
    async with await get_db_connection() as conn:
        await conn.execute('''CREATE TABLE IF NOT EXISTS document_store
                        (id INTEGER PRIMARY KEY AUTOINCREMENT,
                         filename TEXT,
                         upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        await conn.commit()

async def insert_document_record(filename):
    async with await get_db_connection() as conn:
        cursor = await conn.execute('INSERT INTO document_store (filename) VALUES (?)', (filename,))
        await conn.commit()
        return cursor.lastrowid

async def delete_document_record(file_id):
    async with await get_db_connection() as conn:
        await conn.execute('DELETE FROM document_store WHERE id = ?', (file_id,))
        await conn.commit()
        return True

async def get_all_documents():
    async with await get_db_connection() as conn:
        cursor = await conn.execute('SELECT id, filename, upload_timestamp FROM document_store ORDER BY upload_timestamp DESC')
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

async def get_all_chat_sessions():
    """Retrieve all unique chat sessions with their first message and timestamp"""
    async with await get_db_connection() as conn:
        cursor = await conn.execute('''
            SELECT DISTINCT 
                session_id,
                first_value(user_query) OVER (PARTITION BY session_id ORDER BY created_at) as first_message,
                first_value(created_at) OVER (PARTITION BY session_id ORDER BY created_at) as started_at
            FROM application_logs
            GROUP BY session_id
            ORDER BY started_at DESC
        ''')
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

async def initialize_db():
    await create_application_logs()
    await create_document_store()

# Run initialization
import asyncio
asyncio.run(initialize_db())
