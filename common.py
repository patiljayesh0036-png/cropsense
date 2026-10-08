"""
common.py - settings and image helpers shared by train_models.py AND app.py.

Keeping the image preparation in ONE place guarantees that the images the
models see during prediction are prepared exactly like the ones used in training.
"""
import os

import numpy as np
from PIL import Image, ImageOps

# ----------------------------------------------------------------------------
# Folders
# ----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "datasets")
MODEL_DIR = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(MODEL_DIR, "reports")

# Each model -> list of possible dataset folders (first one that exists is used)
DATA_PATHS = {
    "rice_water_rgb": ["rice/water_stress"],
    "rice_water_thermal": ["rice/water_stress_thermal"],
    "rice_nutrients": ["rice/nutrients"],
    "maize_water": ["maize/water_stress_rbg", "maize/water_stress"],
    "maize_nutrients": ["maize/nutrients"],
}

# ----------------------------------------------------------------------------
# Image settings
# ----------------------------------------------------------------------------
IMG_SIZE = 160            # input size for the image-folder models
MAIZE_WATER_SIZE = 48     # the .npy arrays are 48x48
MAIZE_WATER_UPSCALE = 96  # the model upsamples 48 -> 96 internally
TILE_ASPECT = 1.8         # images wider/taller than this ratio get cut into tiles
MAX_TILES = 6             # max tiles taken from one long image
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

# ----------------------------------------------------------------------------
# Class names
# ----------------------------------------------------------------------------
# !! ASSUMPTION - please verify against your maize dataset's documentation !!
# The .npy labels are the numbers 0, 1, 2. Change the names/order here if wrong.
MAIZE_WATER_LABELS = ["Healthy", "Mild stress", "Severe stress"]

# Maps messy folder names to clean display names
CLEAN_NAMES = {
    "healthy": "Healthy",
    "helathy": "Healthy",
    "mild_stress": "Mild stress",
    "severe_stress": "Severe stress",
    "harvest stage": "Harvest stage",
    "nitrogen": "Nitrogen",
    "nitrogen(n)": "Nitrogen",
    "phosphorus": "Phosphorus",
    "phosphorous": "Phosphorus",
    "phosphorus(p)": "Phosphorus",
    "potassium": "Potassium",
    "pottasium": "Potassium",
    "potassium(k)": "Potassium",
    "magnesium": "Magnesium",
    "magnessium": "Magnesium",
}


def clean_name(folder_name):
    key = folder_name.strip().lower()
    if key in CLEAN_NAMES:
        return CLEAN_NAMES[key]
    return folder_name.replace("_", " ").strip().capitalize()


def resolve_dataset(key):
    """Return the real dataset folder for a model key."""
    for rel in DATA_PATHS[key]:
        path = os.path.join(DATASET_DIR, *rel.split("/"))
        if os.path.isdir(path):
            return path
    raise FileNotFoundError(
        f"Dataset folder for '{key}' not found. Looked for: {DATA_PATHS[key]} "
        f"inside {DATASET_DIR}"
    )


# ----------------------------------------------------------------------------
# Image helpers
# ----------------------------------------------------------------------------
def open_image(source):
    """Open a file path or file-like object as an RGB PIL image (EXIF-rotated)."""
    img = Image.open(source)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def _resize(img, size):
    return np.asarray(img.resize((size, size), Image.Resampling.BILINEAR), dtype=np.uint8)


def whole_image(img, size):
    """Resize the whole image to size x size. Returns array (1, size, size, 3)."""
    return _resize(img, size)[None, ...]


def tile_image(img, size, max_tiles=MAX_TILES, aspect=TILE_ASPECT):
    """
    Long strip images (e.g. 326x4110) are cut into square tiles instead of being
    squashed. Normal-shaped images are just resized.
    Returns uint8 array (n_tiles, size, size, 3).
    """
    w, h = img.size
    long_side, short_side = max(w, h), min(w, h)
    if long_side / short_side <= aspect:
        return whole_image(img, size)

    n_fit = max(1, long_side // short_side)
    k = min(n_fit, max_tiles)
    offsets = np.linspace(0, long_side - short_side, k).astype(int)

    tiles = []
    for off in offsets:
        if w >= h:   # wide strip -> slide horizontally
            box = (int(off), 0, int(off) + short_side, short_side)
        else:        # tall strip -> slide vertically
            box = (0, int(off), short_side, int(off) + short_side)
        tiles.append(_resize(img.crop(box), size))
    return np.stack(tiles)


def colour_features(img, bins=8):
    """
    Colour histograms (RGB + HSV) plus mean/std for one PIL image.
    Used by the rice water-stress colour model (training AND prediction).
    Returns a 1-D float array of length 54.
    """
    small = img.resize((128, 128))
    rgb = np.asarray(small, dtype=np.float32)
    hsv = np.asarray(small.convert("HSV"), dtype=np.float32)
    feats = []
    for arr in (rgb, hsv):
        for ch in range(3):
            h, _ = np.histogram(arr[..., ch], bins=bins, range=(0, 255))
            feats.append(h / arr[..., ch].size)
    flat = rgb.reshape(-1, 3) / 255.0
    feats += [flat.mean(0), flat.std(0)]
    return np.concatenate(feats)
