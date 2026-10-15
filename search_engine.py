import torch
import os
import re
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
    width, height = img.size
    views = [img]

    for ratio in (0.8, 0.6):
        crop_width = max(1, int(width * ratio))
        crop_height = max(1, int(height * ratio))
        left = (width - crop_width) // 2
        top = (height - crop_height) // 2
        views.append(img.crop((left, top, left + crop_width, top + crop_height)))

    img_t = torch.stack([preprocess(view) for view in views]).to(device)

    with torch.inference_mode():
        with torch.amp.autocast(device_type=device, enabled=device == "cuda"):
            vec = model.encode_image(img_t)
            vec = vec / vec.norm(dim=-1, keepdim=True)

    return vec.float().cpu().numpy()

def search_multi(file_paths, page=1, page_size=20):
    page = max(1, page)
    page_size = max(1, page_size)
    index, meta = get_index()
    if index is None or index.ntotal == 0:
        return {"results": [], "total": 0, "has_more": False}

    query_vectors = []
    query_ids = []
    view_weights = []
    for query_id, path in enumerate(file_paths):
        vectors = encode_img(path)
        for view_id, vector in enumerate(vectors):
            query_vectors.append(vector)
            query_ids.append(query_id)
            view_weights.append(1.0 if view_id == 0 else 0.97)

    scores_by_item = defaultdict(lambda: [float("-inf")] * len(file_paths))
    for vector, query_id, weight in zip(query_vectors, query_ids, view_weights):
        scores, indices = index.search(vector.reshape(1, -1), index.ntotal)
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            item_id = meta[idx]["item_id"]
            scores_by_item[item_id][query_id] = max(
                scores_by_item[item_id][query_id],
                float(score) * weight
            )

    refined = []
    for item_id, query_scores in scores_by_item.items():
        matched_scores = [score for score in query_scores if score != float("-inf")]
        max_score = max(matched_scores)
        avg_score = sum(matched_scores) / len(matched_scores)
        final_score = max_score * 0.6 + avg_score * 0.4
        refined.append((item_id, final_score))

    refined.sort(key=lambda x: x[1], reverse=True)

    total_results = len(refined)

    start = (page - 1) * page_size
    end = start + page_size
    paged = refined[start:end]

    images_by_item = defaultdict(list)
    for entry in meta:
        images_by_item[entry["item_id"]].append(entry["path"])

    results = []

    for item_id, score in paged:
        images = images_by_item[item_id]

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