import numpy as np
import torch
import open_clip
import os
import re
from datetime import datetime
from PIL import Image, ImageOps
from collections import defaultdict
from index_manager import IndexManager
from model_loader import get_model
model, preprocess = get_model()
device = "cuda" if torch.cuda.is_available() else "cpu"

def get_index():
    manager = IndexManager()
    return manager.index, manager.meta

# model, _, preprocess = open_clip.create_model_and_transforms(
#     "ViT-L-14",
#     pretrained="laion2b_s32b_b82k"
# )
model = model.to(device)
model.eval()

def encode_img(path):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    img_t = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        with torch.amp.autocast("cuda"):
            vec = model.encode_image(img_t)
            vec = vec / vec.norm(dim=-1, keepdim=True)

    return vec.float().cpu().numpy()

def search_multi(file_paths, page=1, page_size=20):

    all_scores = defaultdict(list)
    index, meta = get_index()
    if index is None or index.ntotal == 0:
        return {"results": [], "total": 0, "has_more": False}
    for path in file_paths:
        q_vec = encode_img(path)    

        scores, indices = index.search(q_vec, 100)

        for s, idx in zip(scores[0], indices[0]):
            if idx != -1:
                item_id = meta[idx]["item_id"]
                all_scores[item_id].append(float(s))

    refined = []

    for item_id, scores in all_scores.items():
        max_score = max(scores)
        avg_score = sum(scores) / len(scores)
        boost = len(scores) * 0.02
        final_score = (max_score * 0.6 + avg_score * 0.4) + boost
        refined.append((item_id, final_score))

    refined.sort(key=lambda x: x[1], reverse=True)

    total_results = len(refined)

    # ✅ Pagination
    start = (page - 1) * page_size
    end = start + page_size
    paged = refined[start:end]

    results = []

    for item_id, score in paged:
        images = [m["path"] for m in meta if m["item_id"] == item_id]

        file_path = images[0]

        modified_date = get_date_from_filename(file_path)

        results.append({
            "item_id": item_id,
            "score": float(score),
            "image_url": f"/{images[0]}",
            "images": [f"/{p}" for p in images],
            "modified_date": modified_date,
            "is_confident": score > 0.9
        })

    return {
        "results": results,
        "total": total_results,
        "has_more": end < total_results
    }
def detect_duplicates(threshold: float = 0.95):

    index, meta = get_index()

    if index is None or index.ntotal == 0:
        return []

    # Lấy toàn bộ vectors
    vectors = index.reconstruct_n(0, index.ntotal)

    duplicates = []
    visited = set()

    for i in range(len(vectors)):
        if i in visited:
            continue

        v = vectors[i:i+1]
        scores, indices = index.search(v, 10)

        group = []

        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and score >= threshold and idx != i:
                group.append(meta[idx]["item_id"])
                visited.add(idx)

        if group:
            group.append(meta[i]["item_id"])
            visited.add(i)
            duplicates.append(list(set(group)))

    return duplicates
def get_date_from_filename(file_path):
    filename = os.path.basename(file_path)

    match = re.search(r"\((\d{2}-\d{2}-\d{4})\)", filename)

    if match:
        return match.group(1)

    return "N/A"