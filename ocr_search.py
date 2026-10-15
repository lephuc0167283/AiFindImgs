import re
import threading
import unicodedata

import numpy as np
from PIL import Image

from search_config import OCR_ENABLED

_engine = None
_engine_lock = threading.Lock()


def extract_text(image: Image.Image) -> str:
    global _engine
    if not OCR_ENABLED:
        return ""

    if _engine is None:
        with _engine_lock:
            if _engine is None:
                try:
                    from rapidocr_onnxruntime import RapidOCR
                except ImportError as exc:
                    raise RuntimeError(
                        "OCR is enabled but rapidocr-onnxruntime is not installed. "
                        "Install requirements.txt or set AIFIND_ENABLE_OCR=0."
                    ) from exc
                _engine = RapidOCR()

    results, _ = _engine(np.asarray(image.convert("RGB")))
    if not results:
        return ""
    return " ".join(
        text.strip()
        for _, text, confidence in results
        if confidence >= 0.35 and text.strip()
    )


def text_similarity(left: str, right: str) -> float:
    def tokens(value):
        normalized = unicodedata.normalize("NFKC", value).casefold()
        return set(re.findall(r"\w+", normalized, flags=re.UNICODE))

    left_tokens = tokens(left)
    right_tokens = tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = len(left_tokens & right_tokens)
    return (2 * overlap) / (len(left_tokens) + len(right_tokens))
