from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from typing import List

import openai 
from dotenv import load_dotenv
import os
import shutil

load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']

CHROMA_PATH = 'chromaPDF'
DATA_PATH = 'data/pdfs'

def main():
    documents = load_document()
    chunks = split_documents(documents)
    save_chroma(chunks)

def load_document():
    loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = loader.load()
    return documents

def textSplitter():
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=40,
        length_function=len,
        is_separator_regex=False,
    )

    return text_splitter 

def split_documents(documents: List[Document]):
    text_splitter = textSplitter()
    chunks = text_splitter.split_documents(documents)
    print(f"documents: {len(documents)} split into {len(chunks)} chunks\n")

    return chunks

def save_chroma(chunks: List[Document]):
    # get the db
    db = Chroma(
        persist_directory=CHROMA_PATH, 
        embedding_function=OpenAIEmbeddings(),
    )

    chunks_with_ids = getChunkIds(chunks) # generates ids for chunks 

    currentEntries = db.get(include=[]) # gets all current entries within the db
    currentDB_ids = set(currentEntries['ids']) # converts it into a set for faster lookup
    print(f"current documents in the database are: {len(currentDB_ids)}\n")

    # goes through current db to see if we have any documents to add
    documentsToAdd = []
    documentsToAdd_id = []
    for chunk in chunks_with_ids:
        if chunk.metadata['ID'] not in currentDB_ids:
            documentsToAdd.append(chunk)
            documentsToAdd_id.append(chunk.metadata["ID"])


    if len(documentsToAdd):
        print(f"new documents to be added: {len(documentsToAdd)}\n")
        db.add_documents(documents=documentsToAdd, ids=documentsToAdd_id)
        print("Documents added!\n")
    else:
        print("No documents to add\n")

def getChunkIds(chunks : List[Document]) -> List[Document]:
    res = []

    section = 1
    prevID = ""
    for c in chunks:
        source, page = c.metadata['source'], c.metadata['page']
        identifier = f"{source}-{page}"

        if identifier == prevID:
            section += 1
        else:
            section = 1

        prevID = identifier
        finalID = f"{identifier}-{section}"

        c.metadata['ID'] = finalID
        res.append(c)
    
    return res

def destroy_db():
    if os.path.exists(CHROMA_PATH):
        print("clearing db...")
        shutil.rmtree(CHROMA_PATH)
    else:
        print("DB could not be destroyed.")

if __name__ == "__main__":
    main()