from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, Response
from query import langGraph_chat, retrieve_conversation
from dbLogic import save_document_to_db, remove_document
from pydantic import BaseModel
import os

app = FastAPI()
DIRECTORY = "data/pdfs"

class ChatRequest(BaseModel):
    message: str
    thread_id: str = "test"

# retrieve the conversation upon loading in or refreshing the page
@app.get("/conversation/{thread_id}")
async def get_conversation(thread_id: str = "test"):
    conversation = retrieve_conversation(thread_id)
    return JSONResponse(content=conversation, status_code=200)

# post a message to the LLM and recieve a response
@app.post("/chat")
async def chat_endpoint(message: ChatRequest):
    try:
        response = langGraph_chat(message.message, thread_id=message.thread_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Something went wrong: {str(e)}")

    return JSONResponse(content=response, status_code=200)

# document to be uploaded 
@app.post("/upload/pdf")
async def add_document_endpoint(files: list[UploadFile] = File(...)):
    try:
        for file in files:
            if file.filename is None or not file.filename.lower().endswith('.pdf'):
                raise HTTPException(status_code=400, detail="Invalid file type. Please upload a PDF file.")

            completePath = os.path.join(DIRECTORY, file.filename)
            if os.path.exists(completePath):
                continue  # Skip existing files to avoid overwriting

        
            # write the file to the directory
            try:
                with open(completePath, 'wb') as f:
                    # read the file in chunks to reduce memory usage
                    while contents := file.file.read(1024 * 1024):
                        f.write(contents)

                # convert the pdf into a document and save this to the DB
                save_document_to_db(completePath)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error uploading file {file.filename}: {str(e)}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Something went wrong: {str(e)}")

    return Response(status_code=200)

# document to be removed
@app.delete("/remove/pdf/{pdfName}")
async def remove_document_endpoint(pdfName: str):
    pdfPath = os.path.join(DIRECTORY, pdfName)
    if not os.path.exists(pdfPath):
        raise HTTPException(status_code=400, detail=f"File {pdfName} does not exist.")

    try:
        remove_document(pdfPath)
        os.remove(pdfPath)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error removing document {pdfName}: {str(e)}")

    return Response(status_code=200)
