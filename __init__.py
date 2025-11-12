import sys
import os
import sqlite3
from pathlib import Path
from typing import List, Tuple
from typing_extensions import Iterator
import itertools
from types import MethodType

try:
    from aqt import mw, gui_hooks
    from aqt.qt import *
    from aqt.utils import showInfo, tooltip
    from aqt.browser import Browser
    from anki import hooks
    IN_ANKI = True
except ImportError:
    IN_ANKI = False

if IN_ANKI:
    # Hack to make anki use the virtual environment
    ADDON_ROOT_DIR = Path(__file__).parent
    sys.path.append(os.path.join(ADDON_ROOT_DIR, ".venv/lib/python3.13/site-packages/"))

import sqlite_vec
from sqlite_vec import serialize_float32
import ollama

class VectorEmbeddingManager:
    def __init__(self, model_name: str, db_path: str):
        self.model_name = model_name

        # Use direct sqlite3 connection to Anki's database
        self.conn = sqlite3.connect(db_path)
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self.conn.enable_load_extension(False)
        self.db = self.conn.cursor()

        self._init_tables()
        self._sync()

    def _init_tables(self):
        self.db.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS ankivec_vec USING vec0(
                note_id INTEGER PRIMARY KEY,
                embedding float[768]
            )
        """)
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS ankivec_metadata (
                key TEXT PRIMARY KEY,
                value INTEGER
            )
        """)
        self.conn.commit()

    def _sync(self):
        # Check if notes table has been modified since last sync
        notes_max_mod = self.db.execute("SELECT MAX(mod) FROM notes").fetchone()[0]
        if notes_max_mod is None: return
        stored_max_mod_row = self.db.execute("SELECT value FROM ankivec_metadata WHERE key = ?", ("notes_max_mod",)).fetchone()
        stored_max_mod = stored_max_mod_row[0] if stored_max_mod_row else 0
        if notes_max_mod <= stored_max_mod: return

        total = self.db.execute("SELECT COUNT() FROM notes").fetchone()[0]
        if total == 0: return

        if IN_ANKI:
            progress = QProgressDialog("Syncing cards to vector database...", "Cancel", 0, total, mw)
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.setWindowTitle("AnkiVec Sync")
            progress.setValue(0)
        else:
            progress = None

        notes_to_add = self.db.execute("SELECT id, flds FROM notes WHERE id NOT IN (SELECT note_id FROM ankivec_vec)")

        self.add_cards(notes_to_add, progress)
        if progress: progress.close()

        # Store the latest modification time
        self.db.execute("INSERT OR REPLACE INTO ankivec_metadata (key, value) VALUES (?, ?)",
                       ("notes_max_mod", notes_max_mod))
        self.conn.commit()

    def add_cards(self, notes, progress=None):
        processed = 0
        for batch in itertools.batched(notes, 128):
            note_ids, card_text = zip(*batch)
            embeddings = ollama.embed(model=self.model_name, input=card_text)["embeddings"]

            for note_id, embedding in zip(note_ids, embeddings):
                self.db.execute(
                    "INSERT OR REPLACE INTO ankivec_vec (note_id, embedding) VALUES (?, ?)",
                    (note_id, serialize_float32(embedding))
                )

            if progress:
                processed += len(batch)
                progress.setValue(processed)
                if progress.wasCanceled():
                    break
        self.conn.commit()

    def search(self, query: str, n_results: int = 20) -> List[int]:
        query_embedding = ollama.embed(model=self.model_name, input=query)["embeddings"][0]
        query_str = str(query_embedding)

        self.db.execute(
            "SELECT note_id FROM ankivec_vec WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
            (query_str, n_results)
        )
        results = self.db.fetchall()

        return [row[0] for row in results]

    def note_ids_to_card_ids(self, note_ids: List[int]) -> List[int]:
        if not note_ids:
            return []
        placeholders = ','.join('?' * len(note_ids))
        self.db.execute(f"SELECT id FROM cards WHERE nid IN ({placeholders})", note_ids)
        return [row[0] for row in self.db.fetchall()]

def wrap_vec_search(txt):
    if not isinstance(txt, str):
        return txt
    parts = txt.split("vec:", 1)
    regular_query = parts[0].strip()
    if len(parts) > 1:
        vec_query = parts[1].strip()
        note_ids = manager.search(vec_query, n_results=100)
        card_ids = manager.note_ids_to_card_ids(note_ids)
        return f"{regular_query} (" + " OR ".join(f"cid:{cid}" for cid in card_ids) + ")"
    return regular_query

if IN_ANKI:
    _original_table_search = None

    def patched_table_search(self, txt: str) -> None:
        global manager
        transformed_txt = wrap_vec_search(txt)
        return _original_table_search(self, transformed_txt)

    def init_hook():
        global manager, config, _original_table_search

        config = mw.addonManager.getConfig("ankivec")
        manager = VectorEmbeddingManager(config["model_name"], mw.col.path)

    def browser_did_init(browser):
        global _original_table_search
        from aqt.browser.table.table import Table
        if _original_table_search is None:
            _original_table_search = Table.search
            Table.search = patched_table_search

    gui_hooks.main_window_did_init.append(init_hook)
    gui_hooks.browser_will_show.append(browser_did_init)

    def handle_deleted(_, note_ids):
        pass

    hooks.notes_will_be_deleted.append(handle_deleted)
else:
    VectorEmbeddingManager("nomic-embed-text", "/Users/sam/Library/Application Support/Anki2/User 1/collection.anki2")
