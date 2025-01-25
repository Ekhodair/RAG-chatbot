# RAG-chatbot
***

# Overview
a service focused on serving Document chatbot assistant to enable user to chat with their documents, it supports both pdf and .doc file extensions.

***
## Install

### Python Version
```sh
python-3.10.12
```

***

### Virtual environment 
**Please make sure to initialize a virtual environment before installing any requirements:**

    $ python3 -m venv .venv
    $ source .venv/bin/activate
    
### Requirements

    $ pip install -r requirements.txt


## RUN

**Set all required environment variables in the shell as shown in .env-example file:**

    $ source .env
 
   
 ### Inference
  
To run backend API and the UI, you can run the following commands

    $ uvicorn api:app --host 0.0.0.0 --port 8083
    $ streamlit run frontend/streamlit_app.py 
or you can use docker-compose 

    $ docker-compose up -d
***

## Input / Output

### Chat Endpoint

**POST** `/chat`

#### Request Payload
```json
{
    "question": string,
    "session_id": string | null,
    "model": string (default: "llama3.3")
}
```

#### Response
Server-Sent Events (SSE) stream with the following format for each chunk:
```json
{
    "token": string,
    "session_id": string
}
```

### Upload Document Endpoint

**POST** `/upload-doc`

#### Request
Multipart form data with file upload (Supported formats: .pdf, .docx)

#### Response
```json
{
    "message": string,
    "file_id": integer
}
```

#### Error Response
```json
{
    "detail": string
}
```

### List Documents Endpoint

**GET** `/list-docs`

#### Response
```json
[
    {
        "id": integer,
        "filename": string,
        "upload_timestamp": string (ISO format)
    }
]
```

### Delete Document Endpoint

**POST** `/delete-doc`

#### Request Payload
```json
{
    "file_id": integer
}
```

#### Success Response
```json
{
    "message": string
}
```

#### Error Response
```json
{
    "error": string
}
```


**You can reach and interact with the system through http://0.0.0.0:8501**

**The documentation for api can be accessed through http://0.0.0.0:8083/docs**
