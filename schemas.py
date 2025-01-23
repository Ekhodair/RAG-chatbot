from pydantic import BaseModel, Field
from datetime import datetime


class QueryInput(BaseModel):
    question: str
    session_id: str = Field(default=None)
    model: str = Field(default='llama3.3') # default model


class DocumentInfo(BaseModel):
    id: int
    filename: str
    upload_timestamp: datetime


class DeleteFileRequest(BaseModel):
    file_id: int