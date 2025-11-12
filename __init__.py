import sys
import os
from pathlib import Path

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

import sqlite3
import sqlite_vec
from sqlite_vec import serialize_float32
import ollama
import itertools

class VectorEmbeddingManager:
    def __init__(self, model_name: str, embedding_size: int, db_path: str):
        self.model_name = model_name

        # Use direct sqlite3 connection to Anki's database
        self.conn = sqlite3.connect(db_path)
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self.conn.enable_load_extension(False)
        self.db = self.conn.cursor()
        self._init_tables(embedding_size)
        self._sync()

    def __del__(self):
        self.conn.close()

    def _init_tables(self, embedding_size: int):
        self.db.execute("drop table if exists ankivec_vec")
        self.db.execute("drop table if exists ankivec_metadata")
        self.conn.commit()

        self.db.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS ankivec_vec USING vec0(
                note_id INTEGER PRIMARY KEY,
                embedding float[{embedding_size}]
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
        notes_to_add = self.db.execute("SELECT id, flds FROM notes")

        if IN_ANKI:
            progress = QProgressDialog("Syncing cards to vector database...", "Cancel", 0, total, mw)
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.setWindowTitle("AnkiVec Sync")
            progress.setValue(0)
        else:
            progress = None
        self.add_cards(notes_to_add, progress)

        # Store the latest modification time
        self.db.execute("INSERT OR REPLACE INTO ankivec_metadata (key, value) VALUES (?, ?)",
                       ("notes_max_mod", notes_max_mod))
        self.conn.commit()

    def add_cards(self, notes, progress):
        processed = 0
        writer = self.conn.cursor()
        if not progress:
            sys.stdout.write("Syncing Cards: ")
            sys.stdout.flush()
        for batch in itertools.batched(notes, 1):
            note_ids, card_text = zip(*batch)
            joined_text = ["search_document: " + " ".join(c.split(chr(0x1f))) for c in card_text]
            try:
                embeddings = ollama.embed(model=self.model_name, input=joined_text)["embeddings"]
            except:
                print("Trying to embed ", note_ids, card_text)
                raise
            writer.executemany(
                "INSERT OR REPLACE INTO ankivec_vec (note_id, embedding) VALUES (?, ?)",
                list(zip(note_ids, map(serialize_float32, embeddings)))
            )
            processed += len(batch)
            if progress:
                progress.setValue(processed)
                if progress.wasCanceled():
                    break
            else:
                sys.stdout.write(".")
                sys.stdout.flush()
        if not progress:
            print("")


    def search(self, query: str, n_results: int = 20) -> list[int]:
        embedding = ollama.embed(model=self.model_name, input='search_query: ' + query)["embeddings"][0]

        self.db.execute(
            "SELECT note_id FROM ankivec_vec WHERE embedding MATCH ? and k = ?",
            (serialize_float32(embedding), n_results)
        )
        results = self.db.fetchall()

        return [row[0] for row in results]

if IN_ANKI:
    _original_table_search = None

    def wrap_vec_search(txt, n):
        if not isinstance(txt, str):
            return txt
        parts = txt.split("vec:", 1)
        regular_query = parts[0].strip()
        if len(parts) > 1:
            vec_query = parts[1].strip()
            note_ids = manager.search(vec_query, n_results=n)
            return f"{regular_query} (" + " OR ".join(f"nid:{nid}" for nid in note_ids) + ")"
        return regular_query

    def patched_table_search(self, txt: str) -> None:
        global manager
        transformed_txt = wrap_vec_search(txt, config['search_results_limit'])
        return _original_table_search(self, transformed_txt)

    def init_hook():
        global manager, config, _original_table_search
        config = mw.addonManager.getConfig("ankivec")
        manager = VectorEmbeddingManager(config["model_name"], config["embedding_size"], mw.col.path)

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
