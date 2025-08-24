from fastapi import FastAPI, UploadFile, File, HTTPException
from RAG_Model.RAG_model.model_logic.dbLogic import *
from RAG_Model.RAG_model.model_logic.query import langGraph_chat
import os

app = FastAPI()
DIRECTORY = "../data/pdfs"

# post a message to the LLM and recieve a response
@app.post("/chat")
async def chat_endpoint(message: str):
    try:
        response = langGraph_chat(message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Something went wrong: {str(e)}")

    return {"response": response}

# document to be uploaded 
@app.post("/upload/pdf")
async def add_document_endpoint(files: list[UploadFile] = File(...)):
    
    for file in files:
        if file.filename is None or not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload a PDF file.")
        
        completePath = os.path.join(DIRECTORY, file.filename)
        if os.path.exists(completePath):
            continue  # Skip existing files to avoid overwriting

        try:
            # write the file to the directory
            with open(completePath, 'wb') as f:
                # read the file in chunks to reduce memory usage
                while contents := file.file.read(1024 * 1024):
                    f.write(contents)

            # convert the pdf into a document and save this to the DB
            save_document_to_db(completePath)

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error uploading file {file.filename}: {str(e)}")

    return {"message": f"Files uploaded successfully."}

# document to be removed
@app.delete("/remove/pdf/{fileName}")
async def remove_document_endpoint(fileName: str):
    completePath = os.path.join(DIRECTORY, fileName)
    if not os.path.exists(completePath):
        raise HTTPException(status_code=404, detail=f"File {fileName} does not exist.")
    
    try:
        remove_document(fileName)
        os.remove(completePath)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error removing document {fileName}: {str(e)}")

    return {"message": f"Document {fileName} removed successfully."}
