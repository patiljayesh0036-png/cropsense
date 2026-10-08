"""
train_models.py - trains the 5 MobileNetV2 models for the crop stress project.

Models:
  rice_water_rgb      Rice water stress, RGB photo      (5-fold cross-validation)
  rice_water_thermal  Rice water stress, thermal photo  (5-fold cross-validation)
  rice_nutrients      Rice N / P / K deficiency         (long strips cut into tiles)
  maize_water         Maize water stress from .npy      (48x48 arrays)
  maize_nutrients     Maize 5-class nutrient status     (Healthy, Mg, N, P, K)

Usage (run from the project folder, with the venv active):
  python train_models.py maize_water --quick     # 1-epoch smoke test
  python train_models.py maize_water             # real training
  python train_models.py all                     # everything, one after another

Outputs go to models/ (.keras model + _meta.json) and models/reports/ (plots).
"""
import argparse
import hashlib
import json
import os
import random
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import numpy as np
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers

import common as C

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

BATCH = 32
FINE_TUNE_LAYERS = 40   # how many of the last MobileNetV2 layers to unfreeze
QUICK = False           # set by --quick


# ============================================================================
# Model building
# ============================================================================
def build_model(num_classes, input_size, resize_to=None, brightness=False):
    """
    MobileNetV2 transfer-learning model.
    Input: float pixels in the range 0..255 (any RGB image).
    Augmentation (training only) and MobileNetV2 preprocessing are INSIDE the
    model, so the Flask app only has to feed raw 0..255 pixels.
    """
    inputs = keras.Input(shape=(input_size, input_size, 3), name="image")

    aug_layers = [
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.15),
        layers.RandomContrast(0.15),
    ]
    if brightness:   # helps when lighting differs between training and real photos
        aug_layers.append(layers.RandomBrightness(0.15, value_range=(0, 255)))
    augment = keras.Sequential(aug_layers, name="augmentation")
    x = augment(inputs)
    if resize_to:
        x = layers.Resizing(resize_to, resize_to)(x)
        base_size = resize_to
    else:
        base_size = input_size
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="mobilenet_preprocess")(x)

    base = keras.applications.MobileNetV2(
        input_shape=(base_size, base_size, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False
    x = base(x, training=False)          # keeps BatchNorm in inference mode
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="probs")(x)
    return keras.Model(inputs, outputs), base


def _compile(model, lr):
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def make_ds(x, y, batch=BATCH, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(min(len(x), 4096), seed=SEED)
    ds = ds.batch(batch)
    ds = ds.map(lambda a, b: (tf.cast(a, tf.float32), b), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def class_weights(y, n):
    w = compute_class_weight("balanced", classes=np.arange(n), y=y)
    return {i: float(v) for i, v in enumerate(w)}


def train_two_stage(model, base, train_ds, val_ds, cw, epochs1, epochs2):
    """Stage 1: train only the new head. Stage 2: fine-tune the top of MobileNetV2."""
    if QUICK:
        epochs1, epochs2 = 1, 1
    callbacks = []
    if val_ds is not None:
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=3, restore_best_weights=True
            )
        )

    history = {}

    def merge(h):
        for k, v in h.history.items():
            history.setdefault(k, []).extend(v)

    print("  -- stage 1: training classifier head")
    _compile(model, 1e-3)
    merge(model.fit(train_ds, validation_data=val_ds, epochs=epochs1,
                    class_weight=cw, callbacks=callbacks, verbose=2))

    print("  -- stage 2: fine-tuning top layers")
    base.trainable = True
    for layer in base.layers[:-FINE_TUNE_LAYERS]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    _compile(model, 1e-5)
    merge(model.fit(train_ds, validation_data=val_ds, epochs=epochs2,
                    class_weight=cw, callbacks=callbacks, verbose=2))
    return history


def predict_probs(model, x):
    return model.predict(np.asarray(x, dtype="float32"), batch_size=64, verbose=0)


# ============================================================================
# Reporting helpers
# ============================================================================
def save_confusion_matrix(cm, class_names, title, filename):
    os.makedirs(C.REPORT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(1.2 * len(class_names) + 3, 1.0 * len(class_names) + 3))
    ax.imshow(cm, cmap="Greens")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(os.path.join(C.REPORT_DIR, filename), dpi=150)
    plt.close(fig)


def save_history(history, title, filename):
    if not history:
        return
    os.makedirs(C.REPORT_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, label in ((axes[0], "accuracy", "Accuracy"), (axes[1], "loss", "Loss")):
        ax.plot(history.get(key, []), label="train")
        if f"val_{key}" in history:
            ax.plot(history[f"val_{key}"], label="validation")
        ax.set_title(f"{title} - {label}")
        ax.set_xlabel("epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(C.REPORT_DIR, filename), dpi=150)
    plt.close(fig)


def evaluate(name, probs, y, class_names, groups=None):
    """Prints metrics, saves confusion matrix. Returns a dict of results."""
    n = len(class_names)
    pred = probs.argmax(1)
    acc = accuracy_score(y, pred)
    print(f"\n  [{name}] accuracy = {acc * 100:.2f}%  (n={len(y)})")
    print(classification_report(y, pred, labels=list(range(n)),
                                target_names=class_names, zero_division=0))
    cm = confusion_matrix(y, pred, labels=list(range(n)))
    save_confusion_matrix(cm, class_names, f"{name} (acc {acc * 100:.1f}%)",
                          f"{name}_confusion_matrix.png")
    result = {"accuracy": float(acc), "n": int(len(y))}

    if groups is not None:   # average tile probabilities per original image
        g_true, g_pred = [], []
        for g in np.unique(groups):
            m = groups == g
            g_true.append(y[m][0])
            g_pred.append(probs[m].mean(0).argmax())
        g_acc = accuracy_score(g_true, g_pred)
        print(f"  [{name}] image-level accuracy (tiles averaged) = {g_acc * 100:.2f}%  (n={len(g_true)})")
        result["image_level_accuracy"] = float(g_acc)
    return result


def save_model(model, name, class_names, input_size, extra=None):
    os.makedirs(C.MODEL_DIR, exist_ok=True)
    model.save(os.path.join(C.MODEL_DIR, f"{name}.keras"))
    meta = {"name": name, "classes": class_names, "input_size": input_size}
    meta.update(extra or {})
    with open(os.path.join(C.MODEL_DIR, f"{name}_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"  saved models/{name}.keras  +  {name}_meta.json")


# ============================================================================
# Data loading
# ============================================================================
def load_class_folders(root):
    """Reads  root/<class_folder>/<images>  -> file list, label array, clean class names."""
    dirs = sorted(
        (d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))),
        key=str.lower,
    )
    files, labels, kept = [], [], []
    for d in dirs:
        imgs = [f for f in sorted(os.listdir(os.path.join(root, d)))
                if f.lower().endswith(C.IMAGE_EXTS)]
        if not imgs:
            continue
        idx = len(kept)
        kept.append(d)
        files += [os.path.join(root, d, f) for f in imgs]
        labels += [idx] * len(imgs)
        print(f"    {C.clean_name(d):15s} <- folder '{d}': {len(imgs)} images")
    return files, np.array(labels), [C.clean_name(d) for d in kept]


# ============================================================================
# 1 + 2. Rice water stress (RGB and thermal) - tiny datasets -> cross-validation
# ============================================================================
def run_rice_water(kind):
    """
    Rice water stress (tiny datasets -> 5-fold cross-validation).
    For RGB we ALSO train a colour-histogram random forest and average it with
    the CNN, because on this data the simple colour model is very competitive.
    """
    key = f"rice_water_{kind}"
    print(f"\n=== {key} ===")
    root = C.resolve_dataset(key)
    files, y, classes = load_class_folders(root)
    X_list, F_list = [], []
    for f in files:
        img = C.open_image(f)
        X_list.append(C.whole_image(img, C.IMG_SIZE))
        if kind == "rgb":
            F_list.append(C.colour_features(img))
    X = np.concatenate(X_list)
    F = np.stack(F_list) if F_list else None
    n = len(classes)
    print(f"  total {len(y)} images, {n} classes")

    def new_rf():
        return RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                      random_state=SEED, n_jobs=-1)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros((len(y), n), dtype="float32")      # CNN out-of-fold probabilities
    oof_rf = np.zeros((len(y), n), dtype="float32")   # colour-model out-of-fold probabilities
    fold_acc = []
    for fold, (tr, te) in enumerate(skf.split(X, y), 1):
        print(f"\n  --- fold {fold}/5 ---")
        model, base = build_model(n, C.IMG_SIZE)
        train_two_stage(model, base, make_ds(X[tr], y[tr], batch=16, shuffle=True),
                        None, class_weights(y[tr], n), epochs1=15, epochs2=12)
        probs = predict_probs(model, X[te])
        oof[te] = probs
        fold_acc.append(float(accuracy_score(y[te], probs.argmax(1))))
        print(f"  fold {fold} CNN accuracy: {fold_acc[-1] * 100:.1f}%")
        if F is not None:
            rf = new_rf().fit(F[tr], y[tr])
            oof_rf[te] = rf.predict_proba(F[te])
            print(f"  fold {fold} colour-model accuracy: "
                  f"{accuracy_score(y[te], oof_rf[te].argmax(1)) * 100:.1f}%")
        keras.backend.clear_session()

    cnn_res = evaluate(f"{key}_cnn_cv", oof, y, classes)
    print(f"  CNN cross-validated accuracy: {np.mean(fold_acc) * 100:.1f}% "
          f"(+/- {np.std(fold_acc) * 100:.1f})")
    cv_acc = cnn_res["accuracy"]
    extra = {"cnn_cv_accuracy": cnn_res["accuracy"], "cv_fold_accuracies": fold_acc}

    w_cnn = 1.0
    if F is not None:
        evaluate(f"{key}_colour_cv", oof_rf, y, classes)
        print("\n  CNN weight in the CNN + colour-model average (cross-validated):")
        scores = {}
        for w in (0.0, 0.25, 0.5, 0.75, 1.0):
            acc = accuracy_score(y, (w * oof + (1 - w) * oof_rf).argmax(1))
            scores[w] = acc
            print(f"    CNN weight {w:.2f} -> accuracy {acc * 100:.1f}%")
        w_cnn = max(scores, key=lambda w: (scores[w], -abs(w - 0.5)))
        print(f"  chosen CNN weight: {w_cnn:.2f}")
        ens = w_cnn * oof + (1 - w_cnn) * oof_rf
        ens_res = evaluate(f"{key}_ensemble_cv", ens, y, classes)
        cv_acc = ens_res["accuracy"]
        extra.update({"ensemble_cnn_weight": w_cnn, "colour_model": f"{key}_colour.joblib",
                      "colour_cv_accuracy": float(accuracy_score(y, oof_rf.argmax(1)))})

    print("\n  --- final model(s) on ALL images ---")
    model, base = build_model(n, C.IMG_SIZE)
    hist = train_two_stage(model, base, make_ds(X, y, batch=16, shuffle=True),
                           None, class_weights(y, n), epochs1=15, epochs2=12)
    save_history(hist, key, f"{key}_history.png")
    if F is not None:
        os.makedirs(C.MODEL_DIR, exist_ok=True)
        joblib.dump(new_rf().fit(F, y), os.path.join(C.MODEL_DIR, f"{key}_colour.joblib"))
        print(f"  saved models/{key}_colour.joblib")
    extra["cv_accuracy"] = cv_acc
    save_model(model, key, classes, C.IMG_SIZE, extra)
    write_fusion_config()


def _majority_baseline(key):
    """Accuracy you would get by always guessing the biggest class."""
    root = C.resolve_dataset(key)
    counts = []
    for d in os.listdir(root):
        p = os.path.join(root, d)
        if os.path.isdir(p):
            counts.append(len([f for f in os.listdir(p) if f.lower().endswith(C.IMAGE_EXTS)]))
    return max(counts) / sum(counts)


def write_fusion_config():
    """
    Late-fusion weights for rice water stress (RGB model + thermal model).
    Each model is weighted by how much it beats the 'always guess the biggest
    class' baseline, so a model that is no better than guessing gets almost no vote.
    """
    paths = [os.path.join(C.MODEL_DIR, f"rice_water_{k}_meta.json") for k in ("rgb", "thermal")]
    if not all(os.path.exists(p) for p in paths):
        print("  fusion: train rice_water_rgb and rice_water_thermal first")
        return
    rgb, thermal = (json.load(open(p, encoding="utf-8")) for p in paths)
    if rgb["classes"] != thermal["classes"]:
        print("  WARNING: RGB and thermal class lists differ - fusion not written")
        return
    base_rgb = _majority_baseline("rice_water_rgb")
    base_th = _majority_baseline("rice_water_thermal")
    skill_rgb = max(rgb["cv_accuracy"] - base_rgb, 0.02)
    skill_th = max(thermal["cv_accuracy"] - base_th, 0.02)
    total = skill_rgb + skill_th
    cfg = {
        "classes": rgb["classes"],
        "weight_rgb": skill_rgb / total,
        "weight_thermal": skill_th / total,
        "cv_accuracy_rgb": rgb["cv_accuracy"],
        "cv_accuracy_thermal": thermal["cv_accuracy"],
        "baseline_rgb": base_rgb,
        "baseline_thermal": base_th,
    }
    with open(os.path.join(C.MODEL_DIR, "rice_water_fusion.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print(f"  fusion weights written: RGB {cfg['weight_rgb']:.2f}, thermal {cfg['weight_thermal']:.2f}")


# ============================================================================
# 3 + 5. Nutrient deficiency (rice and maize) - image folders, tiled
# ============================================================================
def run_nutrients(crop):
    key = f"{crop}_nutrients"
    print(f"\n=== {key} ===")
    root = C.resolve_dataset(key)
    files, y, classes = load_class_folders(root)
    n = len(classes)

    # Split by ORIGINAL image first, then tile -> no tile of a test image is ever seen in training
    idx = np.arange(len(files))
    tr_i, tmp_i = train_test_split(idx, test_size=0.30, stratify=y, random_state=SEED)
    va_i, te_i = train_test_split(tmp_i, test_size=0.50, stratify=y[tmp_i], random_state=SEED)

    def build_split(ids):
        tiles, labels, groups = [], [], []
        for i in ids:
            t = C.tile_image(C.open_image(files[i]), C.IMG_SIZE)
            tiles.append(t)
            labels += [y[i]] * len(t)
            groups += [i] * len(t)
        return np.concatenate(tiles), np.array(labels), np.array(groups)

    print("  cutting images into tiles ...")
    Xtr, ytr, _ = build_split(tr_i)
    Xva, yva, _ = build_split(va_i)
    Xte, yte, gte = build_split(te_i)
    print(f"  images  train/val/test = {len(tr_i)}/{len(va_i)}/{len(te_i)}")
    print(f"  samples train/val/test = {len(ytr)}/{len(yva)}/{len(yte)}")

    model, base = build_model(n, C.IMG_SIZE)
    hist = train_two_stage(
        model, base,
        make_ds(Xtr, ytr, shuffle=True), make_ds(Xva, yva),
        class_weights(ytr, n), epochs1=8, epochs2=10,
    )
    save_history(hist, key, f"{key}_history.png")
    res = evaluate(f"{key}_test", predict_probs(model, Xte), yte, classes, groups=gte)
    save_model(model, key, classes, C.IMG_SIZE, {"test": res})


# ============================================================================
# 4. Maize water stress - .npy arrays
# ============================================================================
def _hashes(arr):
    return [hashlib.md5(a.astype(np.uint8).tobytes()).hexdigest() for a in arr]


def run_maize_water():
    """
    diagnose_maize.py showed that VAL and TEST images look alike but TRAIN images
    do not (different colour/brightness spread). So we train on train + most of
    val (val repeated, so the model focuses on that image style) and keep 25% of
    val aside for early stopping. The TEST set is only used once, at the very end.
    """
    key = "maize_water"
    print(f"\n=== {key} ===")
    root = C.resolve_dataset(key)

    def load(name):
        return np.load(os.path.join(root, name))

    Xtr, ytr = load("trainData.npy").astype("float32"), load("trainLabel.npy").astype(int)
    Xva, yva = load("valData.npy").astype("float32"), load("valLabel.npy").astype(int)
    Xte, yte = load("testData.npy").astype("float32"), load("testLabel.npy").astype(int)
    classes = C.MAIZE_WATER_LABELS
    n = len(classes)
    print(f"  train {Xtr.shape}, val {Xva.shape}, test {Xte.shape}; classes = {classes}")

    # Honesty check: are test images exact copies of training images?
    train_set = set(_hashes(Xtr))
    unique_mask = np.array([h not in train_set for h in _hashes(Xte)])
    print(f"  test images that also appear in train: {(~unique_mask).sum()} of {len(unique_mask)}")

    fit_i, hold_i = train_test_split(np.arange(len(yva)), test_size=0.25,
                                     stratify=yva, random_state=SEED)
    repeat = 3
    Xfit = np.concatenate([Xtr] + [Xva[fit_i]] * repeat)
    yfit = np.concatenate([ytr] + [yva[fit_i]] * repeat)
    Xhold, yhold = Xva[hold_i], yva[hold_i]
    print(f"  training samples: {len(yfit)}  (train {len(ytr)} + val {len(fit_i)} x{repeat})")
    print(f"  early-stopping hold-out (from val): {len(yhold)}")

    model, base = build_model(n, C.MAIZE_WATER_SIZE, resize_to=C.MAIZE_WATER_UPSCALE,
                              brightness=True)
    hist = train_two_stage(
        model, base,
        make_ds(Xfit, yfit, shuffle=True), make_ds(Xhold, yhold),
        class_weights(yfit, n), epochs1=6, epochs2=8,
    )
    save_history(hist, key, f"{key}_history.png")

    evaluate(f"{key}_holdout_val", predict_probs(model, Xhold), yhold, classes)
    probs = predict_probs(model, Xte)
    res = evaluate(f"{key}_test", probs, yte, classes)
    if unique_mask.any() and (~unique_mask).any():
        res["unique_only"] = evaluate(f"{key}_test_unique_only", probs[unique_mask],
                                      yte[unique_mask], classes)
    save_model(model, key, classes, C.MAIZE_WATER_SIZE, {"test": res})


# ============================================================================
# Main
# ============================================================================
RUNNERS = {
    "rice_water_rgb": lambda: run_rice_water("rgb"),
    "rice_water_thermal": lambda: run_rice_water("thermal"),
    "rice_nutrients": lambda: run_nutrients("rice"),
    "maize_water": run_maize_water,
    "maize_nutrients": lambda: run_nutrients("maize"),
    "fusion": write_fusion_config,   # recompute rice water-stress fusion weights only
}


def main():
    global QUICK
    parser = argparse.ArgumentParser(description="Train the crop stress models")
    parser.add_argument("targets", nargs="+", choices=list(RUNNERS) + ["all"],
                        help="which model(s) to train")
    parser.add_argument("--quick", action="store_true",
                        help="1 epoch per stage - only to test that everything runs")
    args = parser.parse_args()
    QUICK = args.quick

    print("TensorFlow", tf.__version__, "| GPUs:", tf.config.list_physical_devices("GPU") or "none (CPU)")
    targets = list(RUNNERS) if "all" in args.targets else args.targets
    for t in targets:
        start = time.time()
        RUNNERS[t]()
        print(f"\n  {t} finished in {(time.time() - start) / 60:.1f} min")
    print("\nAll requested models are done. Files are in the models/ folder.")


if __name__ == "__main__":
    main()
