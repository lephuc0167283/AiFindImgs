import os
import numpy as np
import faiss
import pickle

INDEX_PATH = "data/index.faiss"
META_PATH = "data/meta.npy"
CACHE_PATH = "cache/embedding_cache.pkl"

class IndexManager:
    def __init__(self):
        self.index = None
        self.meta = []
        self.cache = {}
        self.load()

    def load(self):
        if os.path.exists(INDEX_PATH):
            self.index = faiss.read_index(INDEX_PATH)
            self.meta = list(np.load(META_PATH, allow_pickle=True))
        if os.path.exists(CACHE_PATH):
            with open(CACHE_PATH, "rb") as f:
                self.cache = pickle.load(f)

    def save(self):
        faiss.write_index(self.index, INDEX_PATH)
        np.save(META_PATH, np.array(self.meta, dtype=object))
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(self.cache, f)

    def remove_item(self, item_id):
        if self.index is None:
            return

        new_meta = []
        keep_indices = []

        for i, m in enumerate(self.meta):
            if m["item_id"] != item_id:
                new_meta.append(m)
                keep_indices.append(i)

        if not keep_indices:
            return

        vectors = self.index.reconstruct_n(0, self.index.ntotal)
        vectors = vectors[keep_indices]

        d = vectors.shape[1]
        self.index = faiss.IndexFlatIP(d)
        self.index.add(vectors)

        self.meta = new_meta
        self.save()