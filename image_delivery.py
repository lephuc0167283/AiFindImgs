import hashlib
import os
import uuid
from pathlib import Path

import pillow_jxl
from fastapi import HTTPException
from PIL import Image, ImageOps


def resolve_product_image(image_path: str) -> Path:
    parts = image_path.replace("\\", "/").split("/")
    if (
        len(parts) < 2
        or parts[0] not in {"products_active", "products_sold"}
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise HTTPException(status_code=404, detail="Image not found")

    root = Path(parts[0]).resolve()
    source = Path(*parts).resolve()
    if (
        os.path.normcase(os.path.commonpath((str(root), str(source))))
        != os.path.normcase(str(root))
        or not source.is_file()
    ):
        raise HTTPException(status_code=404, detail="Image not found")
    return source


def jxl_webp_preview(image_path: str, source: Path) -> Path:
    stat = source.stat()
    cache_key = hashlib.sha256(
        f"{image_path}\0{stat.st_mtime_ns}\0{stat.st_size}".encode("utf-8")
    ).hexdigest()
    cache_dir = Path("cache/image_previews")
    cache_dir.mkdir(parents=True, exist_ok=True)
    preview = cache_dir / f"{cache_key}.webp"

    if not preview.is_file():
        temporary = cache_dir / f"{cache_key}.{uuid.uuid4().hex}.tmp"
        try:
            with Image.open(source) as original:
                image = ImageOps.exif_transpose(original)
                if image.mode not in {"RGB", "RGBA"}:
                    image = image.convert("RGBA" if "A" in image.getbands() else "RGB")
                image.save(temporary, format="WEBP", quality=88, method=6)
            os.replace(temporary, preview)
        finally:
            if temporary.exists():
                temporary.unlink()

    return preview
