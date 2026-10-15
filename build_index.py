import os

os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import argparse
import pickle
import re
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import faiss
import numpy as np
import torch
from PIL import Image, ImageOps

from image_preprocess import retrieval_views
from index_manager import create_search_index
from model_loader import get_model
from ocr_search import extract_text
from search_config import CACHE_PATH, INDEX_PATH, META_PATH, OCR_ENABLED

IMAGE_FOLDER = "products_active"
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".jxl")
GPU_BATCH_SIZE = 16
CPU_BATCH_SIZE = 8
CHECKPOINT_INTERVAL = 3000
NUM_WORKERS = 6
MAX_PENDING_LOADS = NUM_WORKERS * 2


def extract_item_id(filename):
    name = os.path.splitext(os.path.basename(filename))[0]
    return re.split(r"[-_]", name)[0].strip()


def normalized_path(path):
    return os.path.normcase(os.path.abspath(path))


def load_views(path, preprocess):
    decode_started = time.perf_counter()
    try:
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        decode_seconds = time.perf_counter() - decode_started
    except Exception as exc:
        return None, exc, time.perf_counter() - decode_started, 0.0

    preprocess_started = time.perf_counter()
    try:
        tensors = torch.stack([preprocess(view) for view in retrieval_views(image)])
    except Exception as exc:
        return None, exc, decode_seconds, time.perf_counter() - preprocess_started
    return (
        tensors,
        None,
        decode_seconds,
        time.perf_counter() - preprocess_started,
    )


def load_changed_images(executor, changed_paths, preprocess):
    paths = iter(changed_paths)
    pending = deque()

    for _ in range(MAX_PENDING_LOADS):
        path_signature = next(paths, None)
        if path_signature is None:
            break
        future = executor.submit(load_views, path_signature[0], preprocess)
        pending.append((path_signature, future))

    while pending:
        path_signature, future = pending.popleft()
        yield path_signature, future.result()

        next_path_signature = next(paths, None)
        if next_path_signature is not None:
            next_future = executor.submit(
                load_views, next_path_signature[0], preprocess
            )
            pending.append((next_path_signature, next_future))


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
    build_started = time.perf_counter()

    index_load_started = time.perf_counter()
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
    index_load_seconds = time.perf_counter() - index_load_started

    directory_scan_started = time.perf_counter()
    image_paths = []
    for root, dirs, files in os.walk(IMAGE_FOLDER):
        dirs.sort()
        image_paths.extend(
            normalized_path(os.path.join(root, filename))
            for filename in sorted(files)
            if filename.lower().endswith(SUPPORTED_EXTENSIONS)
        )
    directory_scan_seconds = time.perf_counter() - directory_scan_started

    active_paths = set(image_paths)
    indexed_paths = {entry.get("source_path") for entry in meta}
    deleted_paths = indexed_paths - active_paths
    deleted_cleanup_started = time.perf_counter()
    if deleted_paths:
        index, meta = remove_rows(index, meta, deleted_paths)
        for path in deleted_paths:
            cache.pop(path, None)
    deleted_cleanup_seconds = time.perf_counter() - deleted_cleanup_started

    signature_scan_started = time.perf_counter()
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
    signature_scan_seconds = time.perf_counter() - signature_scan_started

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
    index_cleanup_started = time.perf_counter()
    index, meta = remove_rows(index, meta, changed_set)
    for path in changed_set:
        cache.pop(path, None)
    index_cleanup_seconds = time.perf_counter() - index_cleanup_started

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = GPU_BATCH_SIZE if device == "cuda" else CPU_BATCH_SIZE
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU unavailable; building on CPU.")

    model_started = time.perf_counter()
    model, preprocess = get_model()
    model = model.to(device).eval()
    model_load_seconds = time.perf_counter() - model_started

    print(
        f"Preparation: model {model_load_seconds:.1f}s, index/cache load "
        f"{index_load_seconds:.1f}s, directory scan {directory_scan_seconds:.1f}s, "
        f"change detection {signature_scan_seconds:.1f}s, "
        f"removed-image cleanup {deleted_cleanup_seconds:.1f}s, "
        f"changed-image cleanup {index_cleanup_seconds:.1f}s "
        f"(total startup {time.perf_counter() - build_started:.1f}s).",
        flush=True,
    )
    print(
        f"Encoding {len(changed_paths)} images into six views each "
        f"({device.upper()}); unchanged entries are retained."
    )
    started = time.perf_counter()
    encoded_images = 0
    processed_images = 0
    errors = 0
    next_checkpoint = CHECKPOINT_INTERVAL
    pending_tensors = []
    pending_paths = []
    decode_seconds = 0.0
    preprocess_seconds = 0.0
    encode_seconds = 0.0
    faiss_add_seconds = 0.0
    ocr_seconds = 0.0
    checkpoint_save_seconds = 0.0

    def report_progress():
        elapsed = max(time.perf_counter() - started, 1e-6)
        rate = encoded_images / elapsed
        percentage = processed_images / len(changed_paths) * 100
        print(
            f"\rProgress {percentage:6.2f}% | "
            f"Processed {processed_images}/{len(changed_paths)} | "
            f"Indexed {encoded_images} | Skipped {errors} | "
            f"{rate:.1f} images/s | "
            f"worker decode {decode_seconds:.1f}s | "
            f"worker preprocess {preprocess_seconds:.1f}s | "
            f"CLIP {encode_seconds:.1f}s | FAISS {faiss_add_seconds:.1f}s | "
            f"OCR {ocr_seconds:.1f}s | save {checkpoint_save_seconds:.1f}s",
            end="",
            flush=True,
        )

    def flush_batch():
        nonlocal index, encoded_images, processed_images, next_checkpoint
        nonlocal encode_seconds, faiss_add_seconds, ocr_seconds
        nonlocal checkpoint_save_seconds
        if not pending_paths:
            return

        stage_started = time.perf_counter()
        features = encode_batch_with_fallback(
            model, torch.stack(pending_tensors), device
        )
        encode_seconds += time.perf_counter() - stage_started
        if index is None:
            index = create_search_index(features.shape[-1])
        stage_started = time.perf_counter()
        index.add(features.reshape(-1, features.shape[-1]))
        faiss_add_seconds += time.perf_counter() - stage_started

        for (path, signature), vectors in zip(
            pending_paths, features
        ):
            recognized_text = ""
            if OCR_ENABLED:
                stage_started = time.perf_counter()
                with Image.open(path) as source:
                    image = ImageOps.exif_transpose(source).convert("RGB")
                recognized_text = extract_text(image)
                ocr_seconds += time.perf_counter() - stage_started

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

        processed_images += len(pending_paths)
        pending_tensors.clear()
        pending_paths.clear()
        report_progress()

        if encoded_images >= next_checkpoint:
            print(
                f"\nSaving checkpoint after {encoded_images} indexed images...",
                flush=True,
            )
            stage_started = time.perf_counter()
            save_index(index, meta, cache)
            checkpoint_save_seconds += time.perf_counter() - stage_started
            print("Checkpoint saved; continuing build.", flush=True)
            while next_checkpoint <= encoded_images:
                next_checkpoint += CHECKPOINT_INTERVAL

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        loaded_images = load_changed_images(executor, changed_paths, preprocess)
        for (path, signature), (
            tensors,
            error,
            image_decode_seconds,
            image_preprocess_seconds,
        ) in loaded_images:
            decode_seconds += image_decode_seconds
            preprocess_seconds += image_preprocess_seconds
            if error is not None:
                processed_images += 1
                errors += 1
                print(f"\nSkip unreadable image {path}: {error}")
                report_progress()
                continue
            pending_tensors.append(tensors)
            pending_paths.append((path, signature))
            if len(pending_paths) >= batch_size:
                flush_batch()
        flush_batch()

    if index is None:
        raise RuntimeError("No valid images were available to build the index.")
    print()
    final_save_started = time.perf_counter()
    save_index(index, meta, cache)
    final_save_seconds = time.perf_counter() - final_save_started
    elapsed = time.perf_counter() - started
    indexed_paths_after_build = {
        entry["source_path"] for entry in meta if entry.get("source_path")
    }
    missing_paths = active_paths - indexed_paths_after_build
    if index.ntotal != len(meta):
        raise RuntimeError(
            f"Index/metadata mismatch after build ({index.ntotal} vectors, "
            f"{len(meta)} rows)."
        )
    print(
        f"\nIndex ready: {index.ntotal} vectors, {len(meta)} metadata rows, "
        f"{len(indexed_paths_after_build & active_paths)}/{len(active_paths)} active images; "
        f"skipped {errors} unreadable images in {elapsed:.1f}s."
    )
    print(
        "Stage times: "
        f"worker decode {decode_seconds:.1f}s, "
        f"worker preprocess {preprocess_seconds:.1f}s, "
        f"CLIP {encode_seconds:.1f}s, FAISS add {faiss_add_seconds:.1f}s, "
        f"OCR {ocr_seconds:.1f}s, checkpoints {checkpoint_save_seconds:.1f}s, "
        f"final save {final_save_seconds:.1f}s; total run "
        f"{time.perf_counter() - build_started:.1f}s. Worker times overlap "
        "other stages."
    )
    if missing_paths:
        print(
            f"WARNING: {len(missing_paths)} active images are not indexed. "
            "Check skipped-image errors and run the builder again after fixing them."
        )


if __name__ == "__main__":
    main()
