import os
import re
import threading
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
from PIL import Image, ImageOps

from image_preprocess import retrieval_views
from index_manager import create_search_index, get_manager
from model_loader import get_model
from ocr_search import extract_text, text_similarity
from search_config import OCR_ENABLED, SEARCH_CANDIDATES

device = "cuda" if torch.cuda.is_available() else "cpu"
_model = None
_preprocess = None
_model_lock = threading.Lock()


def _get_model():
    global _model, _preprocess
    if _model is None:
        with _model_lock:
            if _model is None:
                model, preprocess = get_model()
                model = model.to(device)
                model.eval()
                _model = model
                _preprocess = preprocess
    return _model, _preprocess


def get_index():
    manager = get_manager()
    return manager.index, manager.meta


def _encode_tensors(tensors):
    model, _ = _get_model()
    try:
        with torch.inference_mode():
            with torch.amp.autocast(device_type=device, enabled=device == "cuda"):
                vectors = model.encode_image(tensors.to(device))
                vectors = vectors / vectors.norm(
                    dim=-1, keepdim=True
                ).clamp_min(1e-12)
        return vectors.float().cpu()
    except RuntimeError as exc:
        if device != "cuda" or "out of memory" not in str(exc).lower():
            raise
        if len(tensors) == 1:
            raise
        torch.cuda.empty_cache()
        midpoint = len(tensors) // 2
        return torch.cat((
            _encode_tensors(tensors[:midpoint]),
            _encode_tensors(tensors[midpoint:]),
        ))


def encode_img(path):
    _, preprocess = _get_model()
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    tensors = torch.stack([preprocess(view) for view in retrieval_views(image)])
    return _encode_tensors(tensors).numpy()


def search_multi(file_paths, page=1, page_size=20):
    page = max(1, page)
    page_size = max(1, page_size)
    manager = get_manager()
    index, meta = manager.index, manager.meta
    if index is None or index.ntotal == 0 or not file_paths:
        return {"results": [], "total": 0, "has_more": False}

    query_vectors = []
    query_texts = []
    candidate_ids = set()
    candidate_count = min(SEARCH_CANDIDATES, index.ntotal)
    for path in file_paths:
        vectors = encode_img(path)
        query_vectors.append(vectors)
        if OCR_ENABLED:
            with Image.open(path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
            query_texts.append(extract_text(image))
        else:
            query_texts.append("")
        for vector in vectors:
            _, rows = index.search(vector.reshape(1, -1), candidate_count)
            candidate_ids.update(
                meta[row]["item_id"] for row in rows[0] if row != -1
            )

    if not candidate_ids:
        return {"results": [], "total": 0, "has_more": False}

    item_ids = sorted(candidate_ids)
    candidate_rows = [
        row
        for item_id in item_ids
        for row in manager.item_rows[item_id]
    ]
    candidate_vectors = index.reconstruct_batch(
        np.asarray(candidate_rows, dtype=np.int64)
    )
    row_weights = np.asarray(
        [1.0 if meta[row].get("view_id", 0) == 0 else 0.97 for row in candidate_rows],
        dtype=np.float32,
    )
    row_item_ids = [meta[row]["item_id"] for row in candidate_rows]

    scores_by_item = defaultdict(list)
    for vectors in query_vectors:
        query_weights = np.asarray(
            [1.0 if view_id == 0 else 0.97 for view_id in range(len(vectors))],
            dtype=np.float32,
        )
        similarities = vectors @ candidate_vectors.T
        similarities *= query_weights[:, None]
        similarities *= row_weights[None, :]

        max_by_item = {}
        for column, item_id in enumerate(row_item_ids):
            score = float(similarities[:, column].max())
            if item_id not in max_by_item or score > max_by_item[item_id]:
                max_by_item[item_id] = score
        for item_id, score in max_by_item.items():
            scores_by_item[item_id].append(score)

    ranked = []
    for item_id, scores in scores_by_item.items():
        visual_score = max(scores) * 0.6 + sum(scores) / len(scores) * 0.4
        text_score = max(
            (
                text_similarity(query_text, manager.item_texts.get(item_id, ""))
                for query_text in query_texts
            ),
            default=0.0,
        )
        final_score = visual_score + 0.04 * text_score
        ranked.append((item_id, final_score))
    ranked.sort(key=lambda result: result[1], reverse=True)

    total_results = len(ranked)
    start = (page - 1) * page_size
    end = start + page_size
    results = []
    for item_id, score in ranked[start:end]:
        images = manager.item_images[item_id]
        file_path = images[0]
        results.append({
            "item_id": item_id,
            "score": float(score),
            "image_url": f"/{file_path}",
            "images": [f"/{path}" for path in images],
            "modified_date": get_file_modified_date(file_path),
            "is_confident": score > 0.9,
        })

    return {
        "results": results,
        "total": total_results,
        "has_more": end < total_results,
    }


def index_images(paths):
    paths = list(paths)
    if not paths:
        return

    manager = get_manager()
    indexed_vectors = []
    new_entries = []
    cache_updates = {}
    for path in paths:
        vectors = encode_img(path).astype("float32")
        if manager.index is not None and manager.index.d != vectors.shape[-1]:
            raise RuntimeError(
                "Image embedding dimension does not match the active index."
            )
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        recognized_text = extract_text(image)
        source_path = os.path.normcase(os.path.abspath(path))
        relative_path = os.path.relpath(source_path).replace("\\", "/")
        filename = os.path.splitext(os.path.basename(path))[0]
        item_id = re.split(r"[-_]", filename)[0].strip()
        indexed_vectors.append(vectors)
        new_entries.extend({
            "item_id": item_id,
            "path": relative_path,
            "source_path": source_path,
            "view_id": view_id,
            "recognized_text": recognized_text,
        } for view_id in range(len(vectors)))
        stat = os.stat(source_path)
        cache_updates[source_path] = {
            "mtime_ns": stat.st_mtime_ns,
            "size": stat.st_size,
        }

    vectors = np.concatenate(indexed_vectors)
    if manager.index is None:
        manager.index = create_search_index(vectors.shape[-1])
    manager.index.add(vectors)
    manager.meta.extend(new_entries)
    manager.cache.update(cache_updates)
    manager.save()


def index_image(path):
    index_images([path])


def detect_duplicates(threshold: float = 0.95):
    manager = get_manager()
    index = manager.index
    if index is None or index.ntotal == 0:
        return []

    representative_rows = [
        rows[0]
        for rows in manager.item_rows.values()
        if rows
    ]
    representative_ids = [manager.meta[row]["item_id"] for row in representative_rows]
    vectors = index.reconstruct_batch(
        np.asarray(representative_rows, dtype=np.int64)
    )
    duplicates = []
    visited = set()
    neighbors = min(200, len(representative_rows))

    for row, item_id in enumerate(representative_ids):
        if item_id in visited:
            continue
        scores, indices = index.search(vectors[row:row + 1], neighbors)
        group = {item_id}
        for score, candidate_row in zip(scores[0], indices[0]):
            if candidate_row == -1:
                continue
            candidate_id = manager.meta[candidate_row]["item_id"]
            if candidate_id != item_id and score >= threshold:
                group.add(candidate_id)
        if len(group) > 1:
            duplicates.append(sorted(group))
            visited.update(group)
        else:
            visited.add(item_id)

    return duplicates


def get_file_modified_date(file_path):
    modified_at = os.path.getmtime(file_path)
    return datetime.fromtimestamp(modified_at).strftime("%d-%m-%Y %H:%M")
