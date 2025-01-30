import os
from typing import Dict, Any
import json
from dotenv import load_dotenv

# Define environment variables file path and load them.
dotenv_path = os.path.join(os.getcwd(), '.env')
config_path = os.path.join(os.getcwd(), 'config.json')

load_dotenv(dotenv_path)

def load_config(config_path: str) -> Dict[str, Any]:
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON in configuration file {config_path}")
# Auth
HF_TOKEN = os.getenv('HF_TOKEN')
# DB
CHROMA_DB_NAME = "chroma_db"
SQL_DB_NAME = "session_logs_db"
LOGGING_LEVEL = os.getenv("LOGGING_LEVEL", "DEBUG")
# configs
GLOBAL_CONFIG = load_config(config_path)
# Prompts
SYSTEM_PROMPT = "You are a helpful AI assistant. Use the given context to answer the user's question."
PROMPT = """Answer the question below from the following context:
### Context: {context}
### User's question: {question}
"""
