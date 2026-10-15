# Image-search upgrade

The upgraded index stores six CLIP embeddings per product image (the full image,
a centered crop, and four overlapping corner crops). FAISS HNSW retrieves likely
matches; the search code then compares all stored views for those candidates and
reranks products. The index is specific to the selected model and OCR setting,
so the original `data/index.faiss` is left untouched.

## Build and start

From the project directory:

```powershell
.\.venv\Scripts\python.exe build_index.py
.\.venv\Scripts\python.exe -m uvicorn app:app --reload
```

The server loads the CLIP model on the first image-search request rather than
at startup. The FAISS index is still loaded into memory when the server starts.

The builder updates only new, changed, or removed images and skips loading the
CLIP model when no image needs encoding. Run it again after changing the model
or OCR setting. Use `build_index.py --force` to re-encode all active images for
the current settings.
GPU builds batch 16 images at a time (CPU builds retain batches of 8) and
automatically split a GPU batch if memory is insufficient. When OCR is disabled,
each image is decoded only once during indexing.
Build progress reports the processed percentage, indexed/skipped counts, and
stage timings. Image loading is pipelined with a bounded queue to limit RAM use.
A checkpoint is saved after every 3,000 successfully indexed images; the
completed index is saved again at the end. The final report checks that indexed
images, vectors, and metadata agree and warns if any active images remain out
of the index.

JPEG XL (`.jxl`) images are supported in `products_active`; the builder reads
them directly without converting the originals. Install `requirements.txt`
before indexing, then run `build_index.py` to add them to search. Browsers that
support JPEG XL receive the original image, and other browsers receive a cached
WebP preview generated on first request.

## Compare embedding models

Model settings are read by both the builder and the server. Each configuration
gets a separate index, so you can switch back without overwriting another
model's index. For example, in PowerShell:

```powershell
$env:AIFIND_MODEL_NAME = "ViT-B-32"
$env:AIFIND_PRETRAINED = "laion2b_s34b_b79k"
.\.venv\Scripts\python.exe build_index.py
.\.venv\Scripts\python.exe benchmark_search.py .\queries\manifest.json
```

The first use of a model may download its OpenCLIP checkpoint. The default
remains `ViT-L-14` with `laion2b_s32b_b82k`.

SigLIP 2 is also available through OpenCLIP; to evaluate it, set:

```powershell
$env:AIFIND_MODEL_NAME = "ViT-B-16-SigLIP2"
$env:AIFIND_PRETRAINED = "webli"
.\.venv\Scripts\python.exe build_index.py
.\.venv\Scripts\python.exe benchmark_search.py .\queries\manifest.json
```

## OCR

OCR is optional because running it across a large inventory makes the initial
index build slower. Enable it before building and keep the same setting for the
server:

```powershell
$env:AIFIND_ENABLE_OCR = "1"
.\.venv\Scripts\python.exe build_index.py
.\.venv\Scripts\python.exe -m uvicorn app:app --reload
```

OCR text is stored with the image metadata and used as a small reranking signal
when query and product images share recognized text. Install the updated
`requirements.txt` before enabling OCR.

## Measure ranking quality

Create a JSON manifest of real query images and the correct product IDs. Keep
query images separate from the catalog images to avoid testing exact duplicates:

```json
[
  {"image": "queries/shoe-side-view.jpg", "item_id": "1771646202717"},
  {"image": "queries/bag-logo.jpg", "item_id": "1771646202727"}
]
```

Image paths in the manifest are relative to that manifest. Run:

```powershell
.\.venv\Scripts\python.exe benchmark_search.py .\queries\manifest.json
```

Compare Recall@1/5/10, mean reciprocal rank, and latency across models using the
same manifest. Without labeled real-world queries, no setting can be claimed to
be more accurate than another.
Keep a representative, held-out query manifest and compare it before changing
the model, number of retrieval views, FAISS candidate count, or reranking
weights; those settings can change search results.
