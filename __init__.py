import sys
import os
from pathlib import Path
from typing import List, Tuple
from aqt import mw, gui_hooks
from aqt.qt import *
from aqt.utils import showInfo, tooltip

# Hack to make anki use the virtual environment
ADDON_ROOT_DIR = Path(__file__).parent
sys.path.append(os.path.join(ADDON_ROOT_DIR, ".venv/lib/python3.13/site-packages/"))

import chromadb
from chromadb.config import Settings
import ollama

class VectorEmbeddingManager:
    def __init__(self, collection_path: str, model_name: str):
        self.model_name = model_name
        self.client = chromadb.PersistentClient(
            path=collection_path,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name="anki_cards",
            metadata={"hnsw:space": "cosine"}
        )

    def get_embedding(self, text: str) -> List[float]:
        return ollama.embed(model=self.model_name, text=text)["embeddings"]

    def index_card(self, card_id: int, front: str, back: str):
        combined_text = f"{front} {back}"
        embedding = self.get_embedding(combined_text)
        self.collection.upsert(
            ids=[card_id],
            embeddings=[embedding],
        )

    def search(self, query: str, n_results: int = 20) -> List[Tuple[int, float, str, str]]:
        return self.collection.query(
            query_embeddings= self.get_embedding(query)
            n_results=n_results
        )["ids"][0]


def init_hook():
    global suggestion_window, manager, config

    config = mw.addonManager.getConfig("ankivec")

    suggestion_window = SuggestionWindow()
    db_path = os.path.join(os.path.dirname(mw.col.path), "ankivec_db")
    manager = VectorEmbeddingManager(db_path, config["model_name"])



gui_hooks.main_window_did_init.append(init_hook)

def handle_deleted(_, note_ids):
    pass

hooks.notes_will_be_deleted.append(handle_deleted)
