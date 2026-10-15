import os
import numpy as np
import faiss
import pickle
from collections import defaultdict

from search_config import (
    CACHE_PATH,
    INDEX_PATH,
    META_PATH,
    MODEL_NAME,
    OCR_ENABLED,
    PRETRAINED,
)

LEGACY_INDEX_PATH = "data/index.faiss"
LEGACY_META_PATH = "data/meta.npy"
LEGACY_CACHE_PATH = "cache/embedding_cache.pkl"


def create_search_index(dimension):
    index = faiss.IndexHNSWFlat(dimension, 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 200
    index.hnsw.efSearch = 256
    return index

class IndexManager:
    def __init__(self):
        self.index = None
        self.meta = []
        self.cache = {}
        self.item_rows = {}
        self.item_images = {}
        self.item_texts = {}
        self.load()

    def load(self):
        index_path = INDEX_PATH
        meta_path = META_PATH
        cache_path = CACHE_PATH
        if not os.path.exists(index_path):
            if (
                MODEL_NAME == "ViT-L-14"
                and PRETRAINED == "laion2b_s32b_b82k"
                and not OCR_ENABLED
                and os.path.exists(LEGACY_INDEX_PATH)
                and os.path.exists(LEGACY_META_PATH)
            ):
                index_path = LEGACY_INDEX_PATH
                meta_path = LEGACY_META_PATH
                cache_path = LEGACY_CACHE_PATH
            else:
                if os.path.exists(meta_path):
                    self.meta = list(np.load(meta_path, allow_pickle=True))
                if self.meta:
                    raise RuntimeError(
                        "Search metadata exists without its FAISS index. Rebuild the index."
                    )
                self._build_lookup()
                return

        self.index = faiss.read_index(index_path)
        if os.path.exists(meta_path):
            self.meta = list(np.load(meta_path, allow_pickle=True))
        if self.index is None and self.meta:
            raise RuntimeError(
                "Search metadata exists without its FAISS index. Rebuild the index."
            )
        if self.index is not None and self.index.ntotal != len(self.meta):
            raise RuntimeError(
                f"Search index/metadata mismatch: {self.index.ntotal} vectors, "
                f"{len(self.meta)} metadata rows. Rebuild the index."
            )
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                self.cache = pickle.load(f)
        self._build_lookup()

    def _build_lookup(self):
        rows = defaultdict(list)
        images = defaultdict(list)
        texts = defaultdict(set)
        for row, entry in enumerate(self.meta):
            item_id = entry["item_id"]
            rows[item_id].append(row)
            if entry["path"] not in images[item_id]:
                images[item_id].append(entry["path"])
            if entry.get("recognized_text"):
                texts[item_id].add(entry["recognized_text"])
        self.item_rows = dict(rows)
        self.item_images = dict(images)
        self.item_texts = {
            item_id: " ".join(sorted(item_texts))
            for item_id, item_texts in texts.items()
        }

    def save(self):
        os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
        os.makedirs(os.path.dirname(META_PATH), exist_ok=True)
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        if self.index is not None:
            faiss.write_index(self.index, INDEX_PATH)
        np.save(META_PATH, np.array(self.meta, dtype=object))
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(self.cache, f)
        self._build_lookup()

    def remove_item(self, item_id):
        if self.index is None:
            return

        removed_rows = self.item_rows.get(item_id, [])
        if not removed_rows:
            return

        removed = set(removed_rows)
        keep_rows = [row for row in range(self.index.ntotal) if row not in removed]
        dimension = self.index.d
        removed_entries = [self.meta[row] for row in removed]
        vectors = self.index.reconstruct_n(0, self.index.ntotal)
        removed_paths = {
            entry.get("source_path")
            for entry in removed_entries
            if entry.get("source_path")
        }
        self.index = create_search_index(dimension)
        if keep_rows:
            self.index.add(vectors[keep_rows])
        self.meta = [entry for row, entry in enumerate(self.meta) if row not in removed]
        for source_path in removed_paths:
            self.cache.pop(source_path, None)
        self.save()


_shared_manager = None


def get_manager():
    global _shared_manager
    if _shared_manager is None:
        _shared_manager = IndexManager()
    return _shared_manager