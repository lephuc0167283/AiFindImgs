import io
from PIL import Image, ImageOps

try:
    from rembg import remove
    REMBG_AVAILABLE = True
except Exception:
    REMBG_AVAILABLE = False


def load_and_fix_image(path: str) -> Image.Image:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    return img


def center_crop_pil(img: Image.Image, ratio: float = 0.9) -> Image.Image:
    w, h = img.size
    new_w = max(1, int(w * ratio))
    new_h = max(1, int(h * ratio))
    left = max((w - new_w) // 2, 0)
    top = max((h - new_h) // 2, 0)
    right = left + new_w
    bottom = top + new_h
    return img.crop((left, top, right, bottom))


def retrieval_views(img: Image.Image) -> list[Image.Image]:
    width, height = img.size
    views = [img, center_crop_pil(img, 0.8)]
    crop_width = max(1, int(width * 0.72))
    crop_height = max(1, int(height * 0.72))
    offsets = (
        (0, 0),
        (width - crop_width, 0),
        (0, height - crop_height),
        (width - crop_width, height - crop_height),
    )
    views.extend(
        img.crop((left, top, left + crop_width, top + crop_height))
        for left, top in offsets
    )
    return views


def remove_background_pil(img: Image.Image) -> Image.Image:
    if not REMBG_AVAILABLE:
        return img

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    output = remove(buf.getvalue())
    out = Image.open(io.BytesIO(output)).convert("RGBA")

    white_bg = Image.new("RGBA", out.size, (255, 255, 255, 255))
    merged = Image.alpha_composite(white_bg, out).convert("RGB")
    return merged