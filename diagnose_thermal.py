"""
diagnose_thermal.py - does the rice THERMAL (and RGB) data carry a usable signal?
Read-only. Takes about a minute.

    python diagnose_thermal.py
"""
import os

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import common as C


def colour_features(img, bins=8):
    """Colour histograms (RGB + HSV) plus mean/std. No deep learning."""
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


def load(key):
    root = C.resolve_dataset(key)
    X, y, names = [], [], []
    for d in sorted((d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))), key=str.lower):
        files = [f for f in sorted(os.listdir(os.path.join(root, d))) if f.lower().endswith(C.IMAGE_EXTS)]
        if not files:
            continue
        idx = len(names)
        names.append(C.clean_name(d))
        for f in files:
            X.append(colour_features(C.open_image(os.path.join(root, d, f))))
            y.append(idx)
    return np.array(X), np.array(y), names


for key in ("rice_water_rgb", "rice_water_thermal"):
    X, y, names = load(key)
    counts = np.bincount(y)
    print("=" * 66)
    print(f"{key}: {len(y)} images, classes {dict(zip(names, counts.tolist()))}")
    print(f"  always-guess-biggest-class baseline: {counts.max() / counts.sum() * 100:.1f}%")
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=4, random_state=0)
    models = {
        "random forest": RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                random_state=0, n_jobs=-1),
        "logistic regression": make_pipeline(StandardScaler(),
                                             LogisticRegression(max_iter=2000, class_weight="balanced")),
    }
    for name, model in models.items():
        s = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        print(f"  colour features + {name:20s}: {s.mean() * 100:5.1f}%  (+/- {s.std() * 100:.1f})")
print()
print("Done. Copy everything above and paste it into the chat.")
