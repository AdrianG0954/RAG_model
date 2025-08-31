import tempfile
import shutil
import pytest
from pprint import pprint
from langchain_chroma import Chroma
from langchain.schema import Document
from dbLogic import generate_chunk_ids, save_to_chromaDB, remove_document, clearDb

# Mock Embedding function for testing
@pytest.fixture
def dummy_embeddings():
    class DummyEmbeddings:
        def embed_documents(self, texts):
            return [[0.1, 0.2, 0.3] for _ in texts]
    return DummyEmbeddings()

# Temporary Chroma DB for testing
@pytest.fixture
def temp_chroma_db(dummy_embeddings):
    temp_dir = tempfile.mkdtemp()
    tempDb = Chroma(
        persist_directory=temp_dir,
        embedding_function=dummy_embeddings,
    )
    yield tempDb
    shutil.rmtree(temp_dir)

# Test adding and retrieving a document
def test_add_and_get_document(temp_chroma_db: Chroma):
    doc = Document(page_content="Hello, world!", metadata={"source": "test.pdf"})

    # Add document to the DB
    save_to_chromaDB([doc], externalDb=temp_chroma_db)
    current_entries = temp_chroma_db.get(include=["documents", "metadatas"])
    assert len(current_entries["documents"]) == 1
    assert current_entries["documents"][0] == "Hello, world!"
    assert current_entries["metadatas"][0]["source"] == "test.pdf"
    assert current_entries["ids"][0] == "test.pdf-0-1"

    # Add duplicate and ensure it is not stored
    save_to_chromaDB([doc], externalDb=temp_chroma_db)

    current_entries = temp_chroma_db.get(include=[])
    assert len(current_entries["ids"]) == 1

# Test generating chunk IDs
def test_generate_chunk_ids():
    docs = [
        Document(page_content="Doc 1 Page 1", metadata={"source": "data/pdfs/doc1.pdf", "page": 1}),
        Document(page_content="Doc 1 Page 2", metadata={"source": "data/pdfs/doc1.pdf", "page": 2}),
        Document(page_content="Doc 2 Page 1", metadata={"source": "data/pdfs/doc2.pdf", "page": 1}),
        Document(page_content="Doc 3 Page 1", metadata={"source": "data/pdfs/doc3.pdf", "page": 1}),
    ]
    expected = {
        "data/pdfs/doc1.pdf": ["data/pdfs/doc1.pdf-1-1", "data/pdfs/doc1.pdf-2-1"],
        "data/pdfs/doc2.pdf": ["data/pdfs/doc2.pdf-1-1", "data/pdfs/doc2.pdf-1-2"],
        "data/pdfs/doc3.pdf": ["data/pdfs/doc3.pdf-1-1"],
    }

    chunk_ids = generate_chunk_ids(docs)
    assert len(chunk_ids) == 4
    for d in chunk_ids:
        source = d.metadata["source"]
        i = d.metadata["page"] - 1
        assert expected[source][i] == d.metadata["ID"]

# Test removing a document
def test_remove_document(temp_chroma_db: Chroma):
    doc = Document(page_content="Hello, world!", metadata={"source": "test.pdf"})
    save_to_chromaDB([doc], externalDb=temp_chroma_db)

    current_entries = temp_chroma_db.get(include=["documents", "metadatas"])
    assert len(current_entries["ids"]) == 1
    assert current_entries["metadatas"][0]["source"] == "test.pdf"

    remove_document(doc.metadata["source"], externalDb=temp_chroma_db)
    current_entries = temp_chroma_db.get(include=["documents", "metadatas"])
    assert len(current_entries["ids"]) == 0

# Test clearing db
def test_clear_db(temp_chroma_db: Chroma):
    docs = [
        Document(page_content="Hello, world!", metadata={"source": "test.pdf"}),
        Document(page_content="Bye, world!", metadata={"source": "test2.pdf"}),
        Document(page_content="Hello again!", metadata={"source": "test3.pdf"})
    ]
    save_to_chromaDB(docs, externalDb=temp_chroma_db)
    current_entries = temp_chroma_db.get(include=["documents", "metadatas"])
    assert len(current_entries["ids"]) == 3

    clearDb(externalDb=temp_chroma_db)
    current_entries = temp_chroma_db.get(include=["documents", "metadatas"])
    assert len(current_entries["ids"]) == 0