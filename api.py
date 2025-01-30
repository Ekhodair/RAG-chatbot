import os
import uuid
import json
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
from helpers.logger import create_logger
from schemas import QueryInput, DocumentInfo, DeleteFileRequest
from core.run_vllm import DocQA


logger = create_logger(__name__)

doc_qa = DocQA()
app = FastAPI()

@app.post("/chat")
async def chat(query_input: QueryInput):
    session_id = query_input.session_id
    model_name = query_input.model
    logger.info(
        f"Session ID: {session_id}, Model: {model_name}, User Query: {query_input.question}"
    )
    if not session_id:
        session_id = str(uuid.uuid4())
        messages = []
    else:
        messages = await get_chat_history(session_id)
    result_gen, docs_content = await doc_qa(
            query_input.question, model_name, messages
        )
    async def generate_response():
        previous_response = ""
        async for req_output in result_gen:
            full_response = req_output.outputs[0].text
            token = full_response[len(previous_response):]
            previous_response = full_response
            yield f"data: {json.dumps({'token': token, 'session_id': session_id})}\n\n"

        await insert_application_logs(
            session_id=session_id,
            user_query=query_input.question,
            response=full_response,
            system_prompt=SYSTEM_PROMPT,
            retrieved_context=docs_content,
            model=model_name,
        )
        logger.info(f"Session ID: {session_id}, AI Response: {full_response}")

    return StreamingResponse(generate_response(), media_type="text/event-stream")



@app.post("/upload-doc")
async def upload_and_index_document(file: UploadFile = File(...)):
    allowed_extensions = [".pdf", ".docx", '.md', '.csv', '.html']
    file_extension = os.path.splitext(file.filename)[1].lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed types are: {', '.join(allowed_extensions)}",
        )

    temp_file_path = f"temp_{file.filename}"
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        file_id = await insert_document_record(file.filename)
        success = await index_document_to_chroma(temp_file_path, file_id)

        if success:
            return {
                "message": f"File {file.filename} has been successfully uploaded and indexed.",
                "file_id": file_id,
            }
        else:
            await delete_document_record(file_id)
            raise HTTPException(
                status_code=500, detail=f"Failed to index {file.filename}."
            )
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@app.get("/list-docs", response_model=list[DocumentInfo])
async def list_documents():
    return await get_all_documents()


@app.post("/delete-doc")
async def delete_document(request: DeleteFileRequest):
    # Delete from Chroma
    chroma_delete_success = await delete_doc_from_chroma(request.file_id)

    if chroma_delete_success:
        # If successfully deleted from Chroma, delete from our database
        db_delete_success = await delete_document_record(request.file_id)
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