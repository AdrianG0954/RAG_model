import tempfile
import shutil
import pytest
from pprint import pprint
from langchain_chroma import Chroma
from langchain.schema import Document
from dbLogic import generate_chunk_ids, save_to_chromaDB

@pytest.fixture
def dummy_embeddings():
    class DummyEmbeddings:
        def embed_documents(self, texts):
            return [[0.1, 0.2, 0.3] for _ in texts]
        def embed_query(self, _text):
            return [0.1, 0.2, 0.3]
    return DummyEmbeddings()

@pytest.fixture
def temp_chroma_db(dummy_embeddings):
    temp_dir = tempfile.mkdtemp()
    tempDb = Chroma(
        persist_directory=temp_dir,
        embedding_function=dummy_embeddings,
    )
    yield tempDb
    shutil.rmtree(temp_dir)

def test_add_and_get_document(temp_chroma_db: Chroma):
    doc = Document(page_content="Hello, world!", metadata={"source": "test.pdf"})
    
    # add documents to db
    save_to_chromaDB([doc], externalDb=temp_chroma_db)
    current_entries = temp_chroma_db.get(include=["documents", "metadatas"])
    pprint(current_entries)
    assert len(current_entries["documents"]) == 1
    assert current_entries["documents"][0] == "Hello, world!"
    assert current_entries["metadatas"][0]["source"] == "test.pdf"
    assert current_entries["ids"][0] == "test.pdf-0-1"

    # add duplicate and ensure it is not stored
    save_to_chromaDB([doc], externalDb=temp_chroma_db)

    current_entries = temp_chroma_db.get(include=[])
    assert len(current_entries["ids"]) == 1


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
    for d in chunk_ids:
        source = d.metadata["source"]
        i = d.metadata["page"] - 1
        assert expected[source][i] == d.metadata["ID"]

    assert len(chunk_ids) == 4

# TODO: test removing document


# TODO: test clearing db