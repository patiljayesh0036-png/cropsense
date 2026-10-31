"""
check_predictions.py - runs sample dataset images through the SAME code the website uses.

    python check_predictions.py

It does not start the server and changes nothing. Most sampled images were seen
during training, so scores should look HIGH. If a model scores low here, the
problem is in the pipeline (a bug), not in the model's limits.
"""
import os
import random
from collections import Counter, defaultdict

import numpy as np
from PIL import Image

import common as C
import app as A      # the website's prediction functions (the server is NOT started)

random.seed(0)
PER_CLASS = 15


def folder_images(key):
    root = C.resolve_dataset(key)
    data = {}
    for d in sorted(os.listdir(root)):
        p = os.path.join(root, d)
        if os.path.isdir(p):
            files = [os.path.join(p, f) for f in os.listdir(p) if f.lower().endswith(C.IMAGE_EXTS)]
            if files:
                data[C.clean_name(d)] = files
    return data


def report(title, rows):
    """rows: list of (actual, predicted, confidence, source)"""
    if not rows:
        print(f"\n=== {title}: nothing to test")
        return
    correct = sum(1 for a, p, _, _ in rows if a == p)
    print(f"\n=== {title}: {correct}/{len(rows)} correct ({correct / len(rows) * 100:.1f}%)")
    table = defaultdict(Counter)
    for a, p, _, _ in rows:
        table[a][p] += 1
    for actual, counts in table.items():
        print(f"  {actual:15s} -> {dict(counts)}")
    wrong = [r for r in rows if r[0] != r[1]]
    for a, p, conf, src in wrong[:8]:
        print(f"  WRONG: {src}   actual {a}, predicted {p} ({conf * 100:.0f}%)")
    if len(wrong) > 8:
        print(f"  ... and {len(wrong) - 8} more wrong")


def test_folder_model(key, predict):
    try:
        data = folder_images(key)
        rows = []
        for label, files in data.items():
            for path in random.sample(files, min(PER_CLASS, len(files))):
                classes, probs, _, _ = predict(C.open_image(path))
                i = int(np.argmax(probs))
                rows.append((label, classes[i], float(probs[i]), os.path.relpath(path, C.BASE_DIR)))
        report(key, rows)
    except FileNotFoundError as e:
        print(f"\n=== {key}: skipped ({e})")


def test_maize_water():
    try:
        root = C.resolve_dataset("maize_water")
        X = np.load(os.path.join(root, "testData.npy"))
        y = np.load(os.path.join(root, "testLabel.npy")).astype(int)
        rows = []
        for idx, label in enumerate(C.MAIZE_WATER_LABELS):
            pool = list(np.where(y == idx)[0])
            for i in random.sample(pool, min(PER_CLASS * 3, len(pool))):
                img = Image.fromarray(X[i].astype("uint8"))
                classes, probs, _, _ = A.predict_maize_water(img)
                k = int(np.argmax(probs))
                rows.append((label, classes[k], float(probs[k]), f"testData[{i}]"))
        report("maize_water (test set, expect about 70%)", rows)
    except FileNotFoundError as e:
        print(f"\n=== maize_water: skipped ({e})")


test_folder_model("rice_nutrients", lambda img: A.predict_nutrients("rice", img))
test_folder_model("maize_nutrients", lambda img: A.predict_nutrients("maize", img))
test_folder_model("rice_water_rgb", lambda img: A.predict_rice_water(img, None))
test_maize_water()
print("\nDone. Paste everything above into the chat.")