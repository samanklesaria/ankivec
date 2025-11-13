import sys
import os
from pathlib import Path

try:
    from aqt import mw, gui_hooks
    from aqt.qt import *
    from aqt.utils import tooltip
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

import chromadb
import ollama
import itertools

class VectorEmbeddingManager:
    def __init__(self, model_name: str, collection_path: str, db):
        self.model_name = model_name
        self.db = db

        self.client = chromadb.PersistentClient(path=collection_path)
        self.collection = self.client.get_or_create_collection(
            name="ankivec",
            metadata={"model_name": model_name}
        )
        self._sync()

    def _sync(self):
        stored_model_name = self.collection.metadata.get("model_name")
        stored_mod = int(self.collection.metadata.get("mod", 0))

        if self.model_name != stored_model_name:
            if IN_ANKI:
                tooltip("Model changed. Reindexing all cards...", parent=mw)
            self.client.delete_collection("ankivec")
            self.collection = self.client.create_collection(
                name="ankivec",
                metadata={"model_name": self.model_name}
            )
            stored_mod = 0

        # Check if notes table has been modified since last sync
        total, notes_mod = self.db.first("SELECT COUNT(), max(mod) FROM notes where mod > ?", stored_mod)
        if total == 0: return

        total = self.db.scalar("SELECT count() FROM notes where mod > ?", stored_mod)
        notes_to_add = self.db.execute("SELECT id, flds FROM notes where mod > ?", stored_mod)

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
        self.collection.modify(metadata={"model_name": self.model_name, "mod": str(notes_mod or 0)})

    def add_cards(self, notes, progress):
        processed = 0
        for batch in itertools.batched(notes, 128):
            note_ids, card_text = zip(*batch)
            joined_text = ["search_document: " + " ".join(c.split(chr(0x1f))) for c in card_text]
            try:
                embeddings = ollama.embed(model=self.model_name, input=joined_text)["embeddings"]
            except:
                print("FAILED TO EMBED\n:", card_text)
                processed += len(batch)
                continue

            self.collection.upsert(
                ids=[str(i) for i in note_ids],
                embeddings=embeddings
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
        embeddings = ollama.embed(model=self.model_name, input='search_query: ' + query)["embeddings"]
        results = self.collection.query(
            query_embeddings=embeddings,
            n_results=n_results
        )
        return [int(id) for id in results["ids"][0]]

    def delete_notes(self, note_ids: list[int]) -> None:
        self.collection.delete(ids=[str(i) for i in note_ids])

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
        collection_path = str(Path(mw.col.path).parent / "ankivec_chromadb")
        manager = VectorEmbeddingManager(config["model_name"], collection_path, mw.col.db)

    def browser_did_init(browser):
        global _original_table_search
        from aqt.browser.table.table import Table
        if _original_table_search is None:
            _original_table_search = Table.search
            Table.search = patched_table_search

    gui_hooks.main_window_did_init.append(init_hook)
    gui_hooks.browser_will_show.append(browser_did_init)

    def handle_deleted(_, note_ids):
        manager.delete_notes(note_ids)

    def handle_saved(note):
        card_text = " ".join(note.fields)
        joined_text = "search_document: " + card_text
        embedding = ollama.embed(model=manager.model_name, input=joined_text)["embeddings"][0]
        manager.collection.upsert(
            ids=[str(note.id)],
            embeddings=[embedding],
            metadatas=[{"note_id": note.id}]
        )

    hooks.notes_will_be_deleted.append(handle_deleted)
    hooks.note_will_be_added.append(handle_saved)
    hooks.note_will_flush.append(handle_saved)
