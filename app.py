import os
import shutil
import uuid
import time
import os
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
import warnings
warnings.filterwarnings("ignore")
from typing import List

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from search_engine import search_multi
from index_manager import IndexManager
from search_engine import detect_duplicates
app = FastAPI()
manager = IndexManager()

# ================= MIDDLEWARE =================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ================= FOLDERS =================
for folder in [
    "uploads",
    "products_active",
    "products_sold",
    "HangTonQ4",
    "static",
    "data",
    "cache"
]:
    os.makedirs(folder, exist_ok=True)

# ================= STATIC =================
app.mount("/products_active", StaticFiles(directory="products_active"), name="active")
app.mount("/products_sold", StaticFiles(directory="products_sold"), name="sold")
app.mount("/HangTonQ4", StaticFiles(directory="HangTonQ4"), name="q4")
app.mount("/static", StaticFiles(directory="static"), name="static")

# ================= UI =================
@app.get("/")
def serve_home():
    return FileResponse("static/index.html")

@app.get("/HangTonQ4")
def serve_q4():
    return FileResponse("static/Q4HT.html")

# ================= SEARCH WITH PAGINATION =================
@app.post("/search-multi")
async def api_search(
    files: List[UploadFile] = File(...),
    page: int = Form(1),
    page_size: int = Form(20)
):

    start_total = time.time()
    tmp_paths = []

    for f in files:
        if not f.content_type or not f.content_type.startswith("image/"):
            continue

        ext = os.path.splitext(f.filename)[1] if f.filename else ".jpg"
        path = f"uploads/{uuid.uuid4().hex}{ext}"

        with open(path, "wb") as buffer:
            buffer.write(await f.read())

        tmp_paths.append(path)

    if not tmp_paths:
        return {"results": [], "total": 0, "has_more": False}

    try:
        data = search_multi(tmp_paths, page=page, page_size=page_size)
        print(f"✅ Search done in {time.time()-start_total:.3f}s")
        return data
    finally:
        for p in tmp_paths:
            if os.path.exists(p):
                os.remove(p)

# ================= MOVE TO SOLD =================
@app.post("/move-to-sold/{item_id}")
def move_to_sold(item_id: str):

    moved = False

    for file in os.listdir("products_active"):
        if file.startswith(item_id):
            shutil.move(
                f"products_active/{file}",
                f"products_sold/{file}"
            )
            moved = True

    if moved:
        manager.remove_item(item_id)

    return {"status": "moved" if moved else "not_found"}

# ================= RESTORE =================
@app.post("/restore-item/{item_id}")
def restore_item(item_id: str):

    restored = False

    for file in os.listdir("products_sold"):
        if file.startswith(item_id):
            shutil.move(
                f"products_sold/{file}",
                f"products_active/{file}"
            )
            restored = True

    return {"status": "restored" if restored else "not_found"}

# ================= DELETE =================
@app.post("/delete-item/{item_id}")
def delete_item(item_id: str):

    deleted = False

    for folder in ["products_active", "products_sold"]:
        for file in os.listdir(folder):
            if file.startswith(item_id):
                os.remove(f"{folder}/{file}")
                deleted = True

    if deleted:
        manager.remove_item(item_id)

    return {"status": "deleted" if deleted else "not_found"}

# ================= STATS =================
@app.get("/stats")
def stats():

    active = len(os.listdir("products_active"))
    sold = len(os.listdir("products_sold"))

    return {
        "active_files": active,
        "sold_files": sold,
        "total_files": active + sold
    } 
@app.get("/detect-duplicates")
def api_detect_duplicates(threshold: float = 0.95):

    start = time.time()
    data = detect_duplicates(threshold)
    print(f"🔍 Duplicate scan done in {time.time()-start:.2f}s")

    return {
        "duplicates": data,
        "count": len(data)
    }