import os
import uuid
import json
import logging
import os
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from helpers.db_utils import (
    insert_application_logs,
    get_chat_history,
    get_all_documents,
    insert_document_record,
    delete_document_record,
)
from helpers.chroma_utils import index_document_to_chroma, delete_doc_from_chroma
from helpers.constants import SYSTEM_PROMPT
from schemas import QueryInput, DocumentInfo, DeleteFileRequest
from core import DocQA
from constants import LOGS_DIR


LOGS_PATH = os.path.join(os.getcwd(), LOGS_DIR)
if not os.path.exists(LOGS_PATH):
    os.mkdir(LOGS_PATH)

logging.basicConfig(filename=f"{LOGS_PATH}/app.log", level=logging.INFO)

doc_qa = DocQA()
app = FastAPI()


@app.post("/chat")
async def chat(query_input: QueryInput):
    session_id = query_input.session_id
    model_name = query_input.model
    logging.info(
        f"Session ID: {session_id}, Model: {model_name}, User Query: {query_input.question}"
    )
    if not session_id:
        session_id = str(uuid.uuid4())
        messages = []
    else:
        messages = get_chat_history(session_id)

    def generate_response():
        full_response = ""
        for token, current_full_response, retrieved_doc in doc_qa(
            query_input.question, model_name, messages
        ):
            full_response = current_full_response
            docs_content = retrieved_doc
            yield f"data: {json.dumps({'token': token, 'session_id': session_id})}\n\n"

        # Log after streaming is complete
        insert_application_logs(
            session_id=session_id,
            user_query=query_input.question,
            response=full_response,
            system_prompt=SYSTEM_PROMPT,
            retrieved_context=docs_content,
            model=model_name,
        )
        logging.info(f"Session ID: {session_id}, AI Response: {full_response}")

    return StreamingResponse(generate_response(), media_type="text/event-stream")


@app.post("/upload-doc")
def upload_and_index_document(file: UploadFile = File(...)):
    allowed_extensions = [".pdf", ".docx"]
    file_extension = os.path.splitext(file.filename)[1].lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed types are: {', '.join(allowed_extensions)}",
        )

    temp_file_path = f"temp_{file.filename}"

    try:
        # Save the uploaded file to a temporary file
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        file_id = insert_document_record(file.filename)
        success = index_document_to_chroma(temp_file_path, file_id)

        if success:
            return {
                "message": f"File {file.filename} has been successfully uploaded and indexed.",
                "file_id": file_id,
            }
        else:
            delete_document_record(file_id)
            raise HTTPException(
                status_code=500, detail=f"Failed to index {file.filename}."
            )
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@app.get("/list-docs", response_model=list[DocumentInfo])
def list_documents():
    return get_all_documents()


@app.post("/delete-doc")
def delete_document(request: DeleteFileRequest):
    # Delete from Chroma
    chroma_delete_success = delete_doc_from_chroma(request.file_id)

    if chroma_delete_success:
        # If successfully deleted from Chroma, delete from our database
        db_delete_success = delete_document_record(request.file_id)
        if db_delete_success:
            return {
                "message": f"Successfully deleted document with file_id {request.file_id} from the system."
            }
        else:
            return {
                "error": f"Deleted from Chroma but failed to delete document with file_id {request.file_id} from the database."
            }
    else:
        return {
            "error": f"Failed to delete document with file_id {request.file_id} from Chroma."
        }