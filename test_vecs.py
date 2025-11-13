import tempfile
import pytest
from pathlib import Path
from ankivec import VectorEmbeddingManager

class MockAnkiDB:
    def __init__(self, conn):
        self.conn = conn
        self.cursor = conn.cursor()

    def first(self, query, *args):
        return self.cursor.execute(query, args).fetchone()

    def scalar(self, query, *args):
        return self.cursor.execute(query, args).fetchone()[0]

    def execute(self, query, *args):
        return self.cursor.execute(query, args)

@pytest.fixture
def manager():
    import sqlite3
    db_path = "/Users/sam/Library/Application Support/Anki2/TestUser/collection.anki2"
    conn = sqlite3.connect(db_path)
    with tempfile.TemporaryDirectory() as temp_dir:
        yield VectorEmbeddingManager("nomic-embed-text", temp_dir, db = MockAnkiDB(conn))
    conn.close()

def test_manager(manager):
    notes = manager.db.execute("SELECT id, flds FROM notes order by mod desc LIMIT 20")
    for (nid, flds) in notes:
        assert nid == manager.search(flds, 1)[0]
