"""
diagnose_maize.py - checks WHY the maize water-stress model generalises poorly.
Read-only: it does not change any data or models. Takes about a minute.

    python diagnose_maize.py
"""
import os

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import common as C

root = C.resolve_dataset("maize_water")
names = C.MAIZE_WATER_LABELS


def load(split):
    x = np.load(os.path.join(root, f"{split}Data.npy")).astype("float32")
    y = np.load(os.path.join(root, f"{split}Label.npy")).astype(int)
    return x, y


data = {s: load(s) for s in ("train", "val", "test")}

print("=" * 70)
print("1) Average colour per split and class  (R, G, B on a 0-255 scale)")
print("=" * 70)
for split, (x, y) in data.items():
    for c, name in enumerate(names):
        m = x[y == c]
        if len(m) == 0:
            continue
        r, g, b = m[..., 0].mean(), m[..., 1].mean(), m[..., 2].mean()
        print(f"  {split:5s} {name:14s} n={len(m):5d}  R={r:6.1f} G={g:6.1f} B={b:6.1f}  brightness std={m.mean(axis=(1,2,3)).std():5.1f}")
    print()


def features(x):
    """Simple colour features: per-channel mean, std and an 8-bin histogram."""
    n = len(x)
    flat = x.reshape(n, -1, 3)
    feats = [flat.mean(1), flat.std(1)]
    for ch in range(3):
        hist = np.stack([np.histogram(img[:, ch], bins=8, range=(0, 255))[0] for img in flat])
        feats.append(hist / flat.shape[1])
    return np.concatenate(feats, axis=1)


print("=" * 70)
print("2) Simple colour-only model (random forest, no deep learning)")
print("=" * 70)
Ftr, ytr = features(data["train"][0]), data["train"][1]
rf = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1).fit(Ftr, ytr)
for split in ("train", "val", "test"):
    f, y = (Ftr, ytr) if split == "train" else (features(data[split][0]), data[split][1])
    print(f"  {split:5s} accuracy: {accuracy_score(y, rf.predict(f)) * 100:5.1f}%")

print()
print("=" * 70)
print("3) Train on VAL instead, test on TEST (are val and test similar to each other?)")
print("=" * 70)
Fva, yva = features(data["val"][0]), data["val"][1]
Fte, yte = features(data["test"][0]), data["test"][1]
rf2 = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1).fit(Fva, yva)
print(f"  train-on-val -> test accuracy: {accuracy_score(yte, rf2.predict(Fte)) * 100:5.1f}%")
print("\nDone. Copy everything above and paste it into the chat.")
