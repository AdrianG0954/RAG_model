# TODO: Add tests for all endpoints
from fastapi.testclient import TestClient
from ragEndpoints import app

client = TestClient(app)


