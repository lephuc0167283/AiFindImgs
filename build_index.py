import os
import re
import time
import pickle
import numpy as np
import torch
import faiss
from PIL import Image, ImageOps
import open_clip
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import threading
import os
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
import warnings
warnings.filterwarnings("ignore")
# ================= CONFIG =================
IMAGE_FOLDER = "products_active"
INDEX_PATH = "data/index.faiss"
META_PATH = "data/meta.npy"
CACHE_PATH = "cache/embedding_cache.pkl"

# GTX 3050 = 4GB VRAM → batch 64-96 là ngọt nhất
BATCH_SIZE = 64
NUM_WORKERS = 6          # i5 gen11 = 4C8T → 6 workers IO
PREFETCH_QUEUE = 4       # số batch prefetch trước
SAVE_INTERVAL = 500

# ================= DEVICE CHECK =================
if torch.cuda.is_available():
    device = "cuda"
    torch.backends.cudnn.benchmark = True      # tối ưu conv cho fixed input
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"🔥 GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    device = "cpu"
    print("⚠️  CPU mode - GPU không detect được!")

os.makedirs("data", exist_ok=True)
os.makedirs("cache", exist_ok=True)

# ================= MODEL =================
print("\n📦 Loading CLIP model...")
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-L-14",
    pretrained="laion2b_s32b_b82k"
)
model = model.to(device)
model.eval()

# Compile model nếu PyTorch >= 2.0 (tăng ~20%)
if hasattr(torch, "compile") and device == "cuda":
    try:
        model = torch.compile(model, mode="reduce-overhead")
        print("✅ torch.compile() active")
    except Exception:
        pass

print(f"✅ Model ready on {device.upper()}\n")

# ================= PREFETCH PIPELINE =================
# Load ảnh song song trên CPU trong khi GPU đang encode batch trước
class ImagePrefetcher:
    def __init__(self, paths, batch_size, num_workers):
        self.paths = paths
        self.batch_size = batch_size
        self.queue = Queue(maxsize=PREFETCH_QUEUE)
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self._thread = threading.Thread(target=self._fill, daemon=True)
        self._thread.start()

    def _load_one(self, path):
        try:
            img = Image.open(path)
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")
            tensor = preprocess(img)
            return tensor, path
        except Exception as e:
            print(f"⚠️  Skip {path}: {e}")
            return None, path

    def _fill(self):
        batch_tensors = []
        batch_paths = []

        futures = []
        for path in self.paths:
            futures.append(self.executor.submit(self._load_one, path))

        for future in futures:
            tensor, path = future.result()
            if tensor is None:
                continue

            batch_tensors.append(tensor)
            batch_paths.append(path)

            if len(batch_tensors) >= self.batch_size:
                self.queue.put((batch_tensors[:], batch_paths[:]))
                batch_tensors.clear()
                batch_paths.clear()

        if batch_tensors:
            self.queue.put((batch_tensors, batch_paths))

        self.queue.put(None)  # sentinel

    def __iter__(self):
        while True:
            item = self.queue.get()
            if item is None:
                break
            yield item

# ================= ENCODE =================
def encode_batch(tensors):
    """Encode một batch tensor đã preprocess sẵn"""
    stacked = torch.stack(tensors).to(device, non_blocking=True)

    with torch.no_grad():
        if device == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                feats = model.encode_image(stacked)
        else:
            feats = model.encode_image(stacked)

        feats = feats / feats.norm(dim=-1, keepdim=True)

    return feats.float().cpu().numpy()

def extract_item_id(filename):
    name = os.path.splitext(os.path.basename(filename))[0]
    name = re.split(r"[-_]", name)[0]
    return name.strip()

# ================= LOAD CACHE =================
if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
    print(f"📦 Cache loaded: {len(cache)} files")
else:
    cache = {}
    print("📦 No cache found")

# ================= LOAD INDEX =================
if os.path.exists(INDEX_PATH):
    index = faiss.read_index(INDEX_PATH)
    meta = list(np.load(META_PATH, allow_pickle=True))
    print(f"📦 Existing index loaded ({len(meta)} entries)\n")
else:
    index = None
    meta = []
    print("📦 Creating new index\n")

# ================= SCAN FILES =================
image_paths = []
for root, dirs, files in os.walk(IMAGE_FOLDER):
    for f in sorted(files):  # sorted để consistent
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
            image_paths.append(os.path.join(root, f))

new_images = []
for path in image_paths:
    try:
        mtime = os.path.getmtime(path)
        if path not in cache or cache[path]["mtime"] != mtime:
            new_images.append(path)
    except OSError:
        continue

print(f"🖼️  Total images : {len(image_paths)}")
print(f"🆕 New/changed  : {len(new_images)}")

if not new_images:
    print("✅ Nothing new to encode")
    exit()

# ================= ENCODE STREAMING =================
print(f"\n🚀 Start encoding {len(new_images)} images...")
print(f"   Batch size : {BATCH_SIZE}")
print(f"   Workers    : {NUM_WORKERS}")
print(f"   Device     : {device.upper()}\n")

start_time = time.time()
encoded_count = 0
total_new = len(new_images)
errors = 0

# Warm up GPU
if device == "cuda":
    dummy = torch.zeros(1, 3, 224, 224).to(device)
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            model.encode_image(dummy)
    torch.cuda.synchronize()
    print("🔥 GPU warmed up\n")

prefetcher = ImagePrefetcher(new_images, BATCH_SIZE, NUM_WORKERS)

for batch_tensors, batch_paths in prefetcher:
    try:
        vecs = encode_batch(batch_tensors).astype("float32")
    except RuntimeError as e:
        if "out of memory" in str(e):
            torch.cuda.empty_cache()
            # Fallback: chia đôi batch
            print(f"⚠️  OOM! Splitting batch...")
            mid = len(batch_tensors) // 2
            v1 = encode_batch(batch_tensors[:mid]).astype("float32")
            v2 = encode_batch(batch_tensors[mid:]).astype("float32")
            vecs = np.vstack([v1, v2])
        else:
            raise

    # Init index
    if index is None:
        d = vecs.shape[1]
        index = faiss.IndexFlatIP(d)

    index.add(vecs)

    for i, path in enumerate(batch_paths):
        item_id = extract_item_id(path)
        meta.append({
            "item_id": item_id,
            "path": path.replace("\\", "/")
        })
        try:
            cache[path] = {"mtime": os.path.getmtime(path)}
        except OSError:
            pass

    encoded_count += len(batch_paths)

    # Save periodically
    if encoded_count % SAVE_INTERVAL < BATCH_SIZE:
        print("💾 Auto-saving...")
        faiss.write_index(index, INDEX_PATH)
        np.save(META_PATH, np.array(meta, dtype=object))
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(cache, f)

    # Progress
    elapsed = time.time() - start_time
    speed = encoded_count / elapsed
    remaining = total_new - encoded_count
    eta = remaining / speed if speed > 0 else 0
    pct = encoded_count / total_new * 100

    bar_len = 30
    filled = int(bar_len * encoded_count / total_new)
    bar = "█" * filled + "░" * (bar_len - filled)

    print(f"[{bar}] {pct:.1f}% | "
          f"{encoded_count}/{total_new} | "
          f"{speed:.1f} img/s | "
          f"ETA: {eta:.0f}s", end="\r")

print()  # newline sau progress bar

# ================= FINAL SAVE =================
print("\n💾 Final saving...")
faiss.write_index(index, INDEX_PATH)
np.save(META_PATH, np.array(meta, dtype=object))
with open(CACHE_PATH, "wb") as f:
    pickle.dump(cache, f)

total_time = time.time() - start_time
print(f"\n✅ BUILD COMPLETE")
print(f"   Encoded : {encoded_count} images")
print(f"   Time    : {total_time:.1f}s")
print(f"   Speed   : {encoded_count/total_time:.1f} img/s avg")
print(f"   Index   : {len(meta)} total entries")