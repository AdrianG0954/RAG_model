from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import Chroma
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

def split_documents(documents: List[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"documents: {len(documents)} split into {len(chunks)} chunks")

    return chunks

def save_chroma(chunks: List[Document]):
    # if db already exists, delete it
    if os.path.exists(CHROMA_PATH):
        print("Database already exists...\nCreating a new one.")
        shutil.rmtree(CHROMA_PATH)
    
    # create a new DB from the documents
    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings()
        ,persist_directory=CHROMA_PATH
    )
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


if __name__ == "__main__":
    main()