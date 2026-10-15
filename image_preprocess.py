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
    new_w = int(w * ratio)
    new_h = int(h * ratio)
    left = max((w - new_w) // 2, 0)
    top = max((h - new_h) // 2, 0)
    right = left + new_w
    bottom = top + new_h
    return img.crop((left, top, right, bottom))


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