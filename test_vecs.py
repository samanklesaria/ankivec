import sqlite3
import tempfile
import pytest
from ankivec import VectorEmbeddingManager

@pytest.fixture
def synth_db():
    with tempfile.NamedTemporaryFile(suffix=".db") as f:
        conn = sqlite3.connect(f.name)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE notes (
                id INTEGER PRIMARY KEY,
                flds TEXT,
                mod INTEGER
            )
        """)
        cursor.execute("INSERT INTO notes (id, flds, mod) VALUES (1, 'cat gato', 1234567890)")
        cursor.execute("INSERT INTO notes (id, flds, mod) VALUES (2, 'fig leaf', 1234567890)")
        conn.commit()
        conn.close()
        yield f.name

@pytest.fixture
def manager():
    yield VectorEmbeddingManager("nomic-embed-text", 768, "/Users/sam/Library/Application Support/Anki2/User 1/collection.anki2")

def test_search_first_note_syth(synth_db):
    manager = VectorEmbeddingManager("nomic-embed-text", 768, synth_db)
    results = manager.search("cat", n_results=1)
    assert len(results) == 1
    assert results[0] == 1

def test_manager(manager):
    notes = manager.db.execute("SELECT id, flds FROM notes order by mod desc LIMIT 20").fetchall()
    for (nid, flds) in notes:
        assert nid == manager.search(flds, 1)[0]
