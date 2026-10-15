import os
import re


MODEL_NAME = os.environ.get("AIFIND_MODEL_NAME", "ViT-L-14")
PRETRAINED = os.environ.get("AIFIND_PRETRAINED", "laion2b_s32b_b82k")
OCR_ENABLED = os.environ.get("AIFIND_ENABLE_OCR", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
SEARCH_CANDIDATES = max(50, int(os.environ.get("AIFIND_SEARCH_CANDIDATES", "256")))

MODEL_TAG = re.sub(r"[^a-zA-Z0-9_-]+", "_", f"{MODEL_NAME}_{PRETRAINED}").strip("_")
OCR_TAG = "ocr1" if OCR_ENABLED else "ocr0"

INDEX_PATH = f"data/index_{MODEL_TAG}_mv1_{OCR_TAG}.faiss"
META_PATH = f"data/meta_{MODEL_TAG}_mv1_{OCR_TAG}.npy"
CACHE_PATH = f"cache/embedding_cache_{MODEL_TAG}_mv1_{OCR_TAG}.pkl"
