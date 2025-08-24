import os
import shutil
from typing import List

from dotenv import load_dotenv
import openai

from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables and set OpenAI API key
load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']
CHROMA_PATH = '../chromaPDF'

def load_document(file_path: str) -> List[Document]:
    """
    Loads a PDF file and returns a list of Document objects.
    """
    loader = PyPDFLoader(file_path=file_path, mode="single")
    return loader.load()

def split_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Splits documents into chunks using RecursiveCharacterTextSplitter.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def generate_chunk_ids(chunks: List[Document]) -> List[Document]:
    """
    Assigns unique IDs to each chunk based on source and page.
    """
    res = []
    section = 1
    prev_id = ""
    for c in chunks:
        source, page = c.metadata['source'], c.metadata.get('page', 0)
        identifier = f"{source}-{page}"

        if identifier == prev_id:
            section += 1
        else:
            section = 1

        prev_id = identifier
        final_id = f"{identifier}-{section}"

        c.metadata['ID'] = final_id
        res.append(c)
    return res

def save_to_chromaDB(chunks: List[Document]) -> None:
    """
    Saves chunks to the Chroma vector database, avoiding duplicates.
    """
    db = Chroma(
        persist_directory=CHROMA_PATH, 
        embedding_function=OpenAIEmbeddings(),
    )

    chunks_with_ids = generate_chunk_ids(chunks)
    current_entries = db.get(include=[])
    current_db_ids = set(current_entries['ids'])

    documents_to_add = []
    documents_to_add_id = []
    documents_to_add_source = []
    for chunk in chunks_with_ids:
        if chunk.metadata['ID'] not in current_db_ids:
            documents_to_add.append(chunk)
            documents_to_add_id.append(chunk.metadata["ID"])
            documents_to_add_source.append(chunk.metadata["source"])

    if documents_to_add:
        db.add_documents(
            documents=documents_to_add,
            ids=documents_to_add_id,
            source=documents_to_add_source
        )

def remove_document(source: str) -> None:
    """
    Removes all chunks from the DB with the given source.
    """
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=OpenAIEmbeddings(),
    )

    current_entries = db.get(include=["metadatas", "ids"])
    to_delete = []
    for meta, doc_id in zip(current_entries["metadatas"], current_entries["ids"]):
        if meta.get("source") == source:
            to_delete.append(doc_id)

    if to_delete:
        db.delete(to_delete)

def destroy_db() -> None:
    """
    Deletes the entire Chroma database directory.
    """
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print("Chroma DB destroyed.")
    else:
        print("Chroma DB does not exist.")

def save_document_to_db(file_path: str) -> None:
    """
    Loads, splits, and saves a PDF document to the Chroma DB.
    """
    documents = load_document(file_path)
    chunks = split_documents(documents)
    save_to_chromaDB(chunks)

if __name__ == "__main__":
    # Example usage
   save_document_to_db("../data/pdfs/operating_systems_three_easy_pieces.pdf")