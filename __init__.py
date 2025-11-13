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
    import platform
    import subprocess
    system = platform.system()
    ADDON_ROOT_DIR = Path(__file__).parent

    # Install dependencies with uv
    if system == "Windows":
        raise NotImplementedError("Windows is not yet supported")
    elif system == "Darwin":
        uv_path = "/Applications/Anki.app/Contents/MacOS/uv"
    elif system == "Linux":
        raise NotImplementedError("Linux is not yet supported")
    else:
        raise NotImplementedError("Unknown system")
    subprocess.check_call([uv_path, "sync", "--project", str(ADDON_ROOT_DIR)], cwd=str(ADDON_ROOT_DIR))

    # Hack to make anki use the virtual environment
    sys.path.append(os.path.join(ADDON_ROOT_DIR, ".venv/lib/python3.13/site-packages/"))


import sqlite3
import sqlite_vec
from sqlite_vec import serialize_float32
import ollama
import itertools

class VectorEmbeddingManager:
    def __init__(self, model_name: str, embedding_size: int, db_path: str):
        self.model_name = model_name
        self.embedding_size = embedding_size

        # Use direct sqlite3 connection to Anki's database
        self.conn = sqlite3.connect(db_path)
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self.conn.enable_load_extension(False)
        self.db = self.conn.cursor()
        self._init_tables()
        self._sync()

    def __del__(self):
        self.conn.close()

    def _init_tables(self):
        self.db.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS ankivec_vec USING vec0(
                note_id INTEGER PRIMARY KEY,
                embedding float[{self.embedding_size}]
            )
        """)
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS ankivec_metadata (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                mod INTEGER,
                model_name TEXT
            )
        """)
        self.conn.commit()

    def _sync(self):
        stored_model_name, stored_mod = (self.db.execute("SELECT model_name, mod FROM ankivec_metadata WHERE id = 1").fetchone()
             or (None, 0))

        if self.model_name != stored_model_name:
            if IN_ANKI:
                tooltip("Model changed. Reindexing all cards...", parent=mw)
            stored_mod = 0

        # Check if notes table has been modified since last sync
        total, notes_mod = self.db.execute("SELECT COUNT(), max(mod) FROM notes where mod > ?", (stored_mod,)).fetchone()
        if total == 0: return
        self.db.execute("delete from ankivec_vec where note_id in (SELECT id FROM notes where mod > ?)", (stored_mod,))
        notes_to_add = self.db.execute("SELECT id, flds FROM notes where mod > ?", (stored_mod,))

        if IN_ANKI:
            progress = QProgressDialog("Syncing cards to vector database...", "Cancel", 0, total, mw)
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.setWindowTitle("AnkiVec Sync")
            progress.setValue(0)
        else:
            progress = None
            sys.stdout.write(f"Syncing {total} Cards: ")
            sys.stdout.flush()
        self.add_cards(notes_to_add, progress)

        # Store the latest modification time
        self.db.execute("INSERT OR REPLACE INTO ankivec_metadata (id, mod, model_name) VALUES (?, ?, ?)",
                       (1, notes_mod, self.model_name))
        self.conn.commit()

    def add_cards(self, notes, progress):
        processed = 0
        writer = self.conn.cursor()
        for batch in itertools.batched(notes, 128):
            note_ids, card_text = zip(*batch)
            joined_text = ["search_document: " + " ".join(c.split(chr(0x1f))) for c in card_text]
            try:
                embeddings = ollama.embed(model=self.model_name, input=joined_text)["embeddings"]
            except:
                print("FAILED TO EMBED\n:", card_text)
                processed += len(batch)
                continue
            writer.executemany(
                "INSERT INTO ankivec_vec (note_id, embedding) VALUES (?, ?)",
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
        if progress:
            progress.cancel()
        else:
            print("")

    def search(self, query: str, n_results: int = 20) -> list[int]:
        embedding = ollama.embed(model=self.model_name, input='search_query: ' + query)["embeddings"][0]
        self.db.execute(
            "SELECT note_id FROM ankivec_vec WHERE embedding MATCH ? and k = ?",
            (serialize_float32(embedding), n_results)
        )
        results = self.db.fetchall()
        return [row[0] for row in results]

    def delete_notes(self, note_ids: list[int]) -> None:
        self.db.execute("DELETE FROM ankivec_vec WHERE note_id IN ?", (tuple(note_ids),))
        self.db.commit()

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

    def handle_config_update():
        global manager, config
        manager.reindex_all()
        tooltip("Model changed. Reindexing all cards...", parent=mw)

    gui_hooks.main_window_did_init.append(init_hook)
    gui_hooks.browser_will_show.append(browser_did_init)

    def handle_deleted(_, note_ids):
        manager.delete_notes(note_ids)

    def handle_saved(note):
        card_text = " ".join(note.fields)
        joined_text = "search_document: " + card_text
        try:
            embedding = ollama.embed(model=manager.model_name, input=joined_text)["embeddings"][0]
            manager.db.execute(
                "INSERT INTO ankivec_vec (note_id, embedding) VALUES (?, ?)",
                (note.id, serialize_float32(embedding))
            )
            manager.conn.commit()
        except:
            print(f"FAILED TO EMBED note {note.id}")

    hooks.notes_will_be_deleted.append(handle_deleted)
    hooks.note_will_be_added.append(handle_saved)
    hooks.note_will_flush.append(handle_saved)
