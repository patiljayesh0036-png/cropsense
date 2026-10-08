"""
Run this from inside your project/ folder:

    python inspect_datasets.py > dataset_report.txt

Then paste the contents of dataset_report.txt back into the chat.
It only READS your data. It does not change or delete anything.
"""
import os
import random
from collections import Counter

ROOT = "datasets"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

try:
    from PIL import Image
except ImportError:
    Image = None
    print("[!] Pillow not installed. Image sizes will be skipped. (pip install pillow)")

try:
    import numpy as np
except ImportError:
    np = None
    print("[!] NumPy not installed. .npy files will be skipped. (pip install numpy)")


def describe_images(folder, files):
    """Print extension counts, sample names and sample image sizes."""
    exts = Counter(os.path.splitext(f)[1].lower() for f in files)
    print(f"      extensions : {dict(exts)}")
    samples = sorted(files)[:6]
    print(f"      first files: {samples}")
    if Image is not None and files:
        picks = random.sample(files, min(3, len(files)))
        info = []
        for f in picks:
            try:
                with Image.open(os.path.join(folder, f)) as im:
                    info.append(f"{im.size[0]}x{im.size[1]} {im.mode}")
            except Exception as e:
                info.append(f"unreadable({e.__class__.__name__})")
        print(f"      sample size: {info}")


def inspect_folder_tree(base):
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames.sort()
        depth = dirpath[len(base):].count(os.sep)
        indent = "  " * (depth + 1)
        imgs = [f for f in filenames if os.path.splitext(f)[1].lower() in IMG_EXT]
        others = [f for f in filenames if f not in imgs]
        print(f"{indent}[{os.path.basename(dirpath) or base}]  images={len(imgs)}  other_files={len(others)}")
        if imgs:
            describe_images(dirpath, imgs)
        if others:
            print(f"      other files: {others[:6]}")


def inspect_npy(folder):
    if np is None:
        return
    for f in sorted(os.listdir(folder)):
        if not f.endswith(".npy"):
            continue
        path = os.path.join(folder, f)
        try:
            arr = np.load(path, allow_pickle=True)
            line = f"      {f}: shape={arr.shape} dtype={arr.dtype}"
            if arr.dtype != object and arr.size > 0:
                line += f" min={arr.min():.3f} max={arr.max():.3f}"
            print(line)
            if f.lower().endswith("label.npy") and arr.dtype != object:
                flat = arr if arr.ndim == 1 else arr.argmax(axis=1) if arr.ndim == 2 else arr
                if flat.ndim == 1:
                    print(f"          class counts: {dict(Counter(flat.tolist()))}")
        except Exception as e:
            print(f"      {f}: could not load ({e})")


def main():
    if not os.path.isdir(ROOT):
        print(f"Folder '{ROOT}' not found. Run this script from the project/ folder.")
        return

    for crop in sorted(os.listdir(ROOT)):
        crop_path = os.path.join(ROOT, crop)
        if not os.path.isdir(crop_path):
            continue
        for module in sorted(os.listdir(crop_path)):
            module_path = os.path.join(crop_path, module)
            if not os.path.isdir(module_path):
                continue
            print("=" * 70)
            print(f"{crop.upper()} / {module}")
            print("=" * 70)
            inspect_folder_tree(module_path)
            inspect_npy(module_path)
            print()


if __name__ == "__main__":
    main()
