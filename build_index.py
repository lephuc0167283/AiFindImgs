import os

os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import argparse
import pickle
import re
import time
from concurrent.futures import ThreadPoolExecutor

import faiss
import numpy as np
import torch
from PIL import Image, ImageOps

from image_preprocess import retrieval_views
from index_manager import create_search_index
from model_loader import get_model
from ocr_search import extract_text
from search_config import CACHE_PATH, INDEX_PATH, META_PATH

IMAGE_FOLDER = "products_active"
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".jxl")
BATCH_SIZE = 8
NUM_WORKERS = 6


def extract_item_id(filename):
    name = os.path.splitext(os.path.basename(filename))[0]
    return re.split(r"[-_]", name)[0].strip()


def normalized_path(path):
    return os.path.normcase(os.path.abspath(path))


def load_views(path, preprocess):
    try:
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        return torch.stack([preprocess(view) for view in retrieval_views(image)]), None
    except Exception as exc:
        return None, exc


def encode_batch(model, tensors, device):
    batch_size, view_count = tensors.shape[:2]
    inputs = tensors.flatten(0, 1).to(device, non_blocking=device == "cuda")
    with torch.inference_mode():
        if device == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                features = model.encode_image(inputs)
        else:
            features = model.encode_image(inputs)
        features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return features.float().reshape(batch_size, view_count, -1).cpu().numpy()


def encode_batch_with_fallback(model, tensors, device):
    try:
        return encode_batch(model, tensors, device)
    except RuntimeError as exc:
        if device != "cuda" or "out of memory" not in str(exc).lower():
            raise
        if len(tensors) == 1:
            raise
        torch.cuda.empty_cache()
        midpoint = len(tensors) // 2
        return np.concatenate((
            encode_batch_with_fallback(model, tensors[:midpoint], device),
            encode_batch_with_fallback(model, tensors[midpoint:], device),
        ))


def remove_rows(index, meta, removed_paths):
    if index is None or not removed_paths:
        return index, meta

    removed_rows = {
        row
        for row, entry in enumerate(meta)
        if entry.get("source_path") in removed_paths
    }
    if not removed_rows:
        return index, meta

    vectors = index.reconstruct_n(0, index.ntotal)
    keep_rows = [row for row in range(index.ntotal) if row not in removed_rows]
    rebuilt = create_search_index(index.d)
    if keep_rows:
        rebuilt.add(vectors[keep_rows])
    remaining_meta = [
        entry for row, entry in enumerate(meta) if row not in removed_rows
    ]
    return rebuilt, remaining_meta


def save_index(index, meta, cache):
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(META_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)

    temporary_index = INDEX_PATH + ".tmp"
    temporary_meta = META_PATH + ".tmp"
    temporary_cache = CACHE_PATH + ".tmp"
    try:
        faiss.write_index(index, temporary_index)
        with open(temporary_meta, "wb") as output:
            np.save(output, np.array(meta, dtype=object))
        with open(temporary_cache, "wb") as output:
            pickle.dump(cache, output)
        os.replace(temporary_meta, META_PATH)
        os.replace(temporary_cache, CACHE_PATH)
        os.replace(temporary_index, INDEX_PATH)
    finally:
        for path in (temporary_index, temporary_meta, temporary_cache):
            if os.path.exists(path):
                os.remove(path)


def main():
    parser = argparse.ArgumentParser(
        description="Build or incrementally update the multi-view image index."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-encode all active images using the selected model and OCR settings.",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU unavailable; building on CPU.")

    model, preprocess = get_model()
    model = model.to(device).eval()

    if args.force:
        index = None
        meta = []
        cache = {}
    else:
        index = faiss.read_index(INDEX_PATH) if os.path.exists(INDEX_PATH) else None
        meta = (
            list(np.load(META_PATH, allow_pickle=True))
            if os.path.exists(META_PATH)
            else []
        )
        if index is not None and index.ntotal != len(meta):
            raise RuntimeError(
                f"Index/metadata mismatch ({index.ntotal} vectors, {len(meta)} rows); "
                "use --force to rebuild."
            )
        if index is None and meta:
            raise RuntimeError(
                "Metadata exists but the FAISS index is missing; use --force to rebuild."
            )
        if os.path.exists(CACHE_PATH):
            with open(CACHE_PATH, "rb") as cache_file:
                cache = pickle.load(cache_file)
        else:
            cache = {}

    image_paths = []
    for root, dirs, files in os.walk(IMAGE_FOLDER):
        dirs.sort()
        image_paths.extend(
            normalized_path(os.path.join(root, filename))
            for filename in sorted(files)
            if filename.lower().endswith(SUPPORTED_EXTENSIONS)
        )

    active_paths = set(image_paths)
    indexed_paths = {entry.get("source_path") for entry in meta}
    deleted_paths = indexed_paths - active_paths
    if deleted_paths:
        index, meta = remove_rows(index, meta, deleted_paths)
        for path in deleted_paths:
            cache.pop(path, None)

    changed_paths = []
    for path in image_paths:
        try:
            stat = os.stat(path)
        except OSError as exc:
            print(f"Skip unavailable image {path}: {exc}")
            continue
        signature = {"mtime_ns": stat.st_mtime_ns, "size": stat.st_size}
        if args.force or index is None or cache.get(path) != signature:
            changed_paths.append((path, signature))

    if not changed_paths:
        if deleted_paths:
            if index is None:
                raise RuntimeError("Deleted indexed images but the FAISS index is missing.")
            save_index(index, meta, cache)
            print(f"Removed {len(deleted_paths)} deleted images from the index.")
            return
        if index is None:
            print("No supported product images found; no index was created.")
            return
        print(f"Index is current: {len(meta)} vectors across {len(indexed_paths)} images.")
        return

    changed_set = {path for path, _ in changed_paths}
    index, meta = remove_rows(index, meta, changed_set)
    for path in changed_set:
        cache.pop(path, None)

    print(
        f"Encoding {len(changed_paths)} images into six views each "
        f"({device.upper()}); unchanged entries are retained."
    )
    started = time.perf_counter()
    encoded_images = 0
    errors = 0
    pending_tensors = []
    pending_paths = []

    def flush_batch():
        nonlocal index, encoded_images, errors
        if not pending_paths:
            return

        features = encode_batch_with_fallback(
            model, torch.stack(pending_tensors), device
        )
        if index is None:
            index = create_search_index(features.shape[-1])
        index.add(features.reshape(-1, features.shape[-1]))

        for (path, signature), vectors in zip(
            pending_paths, features
        ):
            with Image.open(path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
            recognized_text = extract_text(image)

            relative_path = os.path.relpath(path).replace("\\", "/")
            item_id = extract_item_id(path)
            for view_id in range(len(vectors)):
                meta.append({
                    "item_id": item_id,
                    "path": relative_path,
                    "source_path": path,
                    "view_id": view_id,
                    "recognized_text": recognized_text,
                })
            cache[path] = signature
            encoded_images += 1

        pending_tensors.clear()
        pending_paths.clear()
        elapsed = max(time.perf_counter() - started, 1e-6)
        rate = encoded_images / elapsed
        print(
            f"\rEncoded {encoded_images}/{len(changed_paths)} images "
            f"({rate:.1f} images/s)",
            end="",
            flush=True,
        )

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        loaded_images = executor.map(
            lambda path_signature: (
                path_signature,
                load_views(path_signature[0], preprocess),
            ),
            changed_paths,
        )
        for (path, signature), (tensors, error) in loaded_images:
            if error is not None:
                print(f"\nSkip unreadable image {path}: {error}")
                errors += 1
                continue
            pending_tensors.append(tensors)
            pending_paths.append((path, signature))
            if len(pending_paths) >= BATCH_SIZE:
                flush_batch()
        flush_batch()

    if index is None:
        raise RuntimeError("No valid images were available to build the index.")
    save_index(index, meta, cache)
    elapsed = time.perf_counter() - started
    print(
        f"\nIndex ready: {index.ntotal} vectors, {len(meta)} metadata rows, "
        f"{len({entry['source_path'] for entry in meta})} images; "
        f"skipped {errors} unreadable images in {elapsed:.1f}s."
    )


if __name__ == "__main__":
    main()
