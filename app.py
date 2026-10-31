"""
app.py - Flask server for CropSense (security-hardened version).

Run:   python app.py
Open:  http://127.0.0.1:5000

Optional settings (environment variables):
  APP_USER, APP_PASSWORD_HASH   turn on password protection (HTTP Basic Auth)
  APP_HOST, APP_PORT            where to listen (default 127.0.0.1 and 5000)
  TRUST_PROXY=1                 ONLY when running behind exactly one trusted reverse proxy
"""
import base64
import hmac
import io
import json
import os
import threading
import time
import traceback
import warnings
from collections import deque
from urllib.parse import urlsplit

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import numpy as np
from flask import Flask, Response, jsonify, render_template, request
from PIL import Image
from tensorflow import keras
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.security import check_password_hash

import common as C

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024    # max 16 MB per request
app.config["MAX_FORM_MEMORY_SIZE"] = 1024 * 1024        # max 1 MB of non-file form fields
if os.environ.get("TRUST_PROXY") == "1":
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
LOW_CONFIDENCE = 0.60        # below this the app warns the user
MODEL_NAMES = ["rice_water_rgb", "rice_water_thermal", "rice_nutrients",
               "maize_water", "maize_nutrients"]

# Image safety
MAX_IMAGE_PIXELS = 64_000_000                      # refuse huge images ("decompression bombs")
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
warnings.simplefilter("error", Image.DecompressionBombWarning)
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP", "TIFF"}

# Optional password protection (off unless BOTH variables are set)
AUTH_USER = os.environ.get("APP_USER", "")
AUTH_HASH = os.environ.get("APP_PASSWORD_HASH", "")
AUTH_ENABLED = bool(AUTH_USER and AUTH_HASH)

# Rate limits per client address: (window in seconds, max requests in that window)
PREDICT_LIMITS = [(60, 10), (3600, 100)]
AUTH_FAIL_LIMITS = [(600, 10)]

_lock = threading.RLock()    # TensorFlow predict calls are done one at a time
_cache = {}

DISCLAIMER = ("These are AI-generated guidelines. Doses are typical ranges, not prescriptions. "
              "Confirm with a soil test and your local agriculture extension officer before applying.")

# Browser security headers
CSP = "; ".join([
    "default-src 'self'",
    "script-src 'self'",
    "style-src 'self' https://fonts.googleapis.com",
    "font-src https://fonts.gstatic.com",
    "img-src 'self' data: blob:",
    "connect-src 'self'",
    "object-src 'none'",
    "base-uri 'self'",
    "form-action 'self'",
    "frame-ancestors 'none'",
])


# ----------------------------------------------------------------------------
# Rate limiter (in memory, per client address)
# ----------------------------------------------------------------------------
class RateLimiter:
    def __init__(self, rules):
        self.rules = rules
        self.horizon = max(window for window, _ in rules)
        self.hits = {}
        self.lock = threading.Lock()

    def retry_after(self, key, record):
        """Return 0 if allowed (and note the hit when record=True), else seconds to wait."""
        now = time.monotonic()
        with self.lock:
            q = self.hits.setdefault(key, deque())
            while q and now - q[0] > self.horizon:
                q.popleft()
            for window, limit in self.rules:
                recent = [t for t in q if now - t <= window]
                if len(recent) >= limit:
                    return int(window - (now - recent[0])) + 1
            if record:
                q.append(now)
            if len(self.hits) > 5000:      # keep memory bounded
                for k in [k for k, v in self.hits.items() if not v or now - v[-1] > self.horizon]:
                    del self.hits[k]
            return 0


predict_limiter = RateLimiter(PREDICT_LIMITS)
auth_limiter = RateLimiter(AUTH_FAIL_LIMITS)


RECOMMENDATIONS = {
    ("rice", "water_stress"): {
        "Healthy": {
            "level": "good", "title": "No water stress detected",
            "summary": "The plant looks adequately watered.",
            "actions": [
                "Keep the current irrigation schedule. Maintain about 2-5 cm of standing water from tillering to flowering, unless you follow alternate wetting and drying.",
                "Check soil moisture every few days, especially in hot, dry or windy weather.",
                "Never let the field dry out at panicle initiation and flowering. These are the most sensitive stages.",
            ]},
        "Mild stress": {
            "level": "warn", "title": "Mild water stress",
            "summary": "Early signs of water shortage. Act within a day or two.",
            "actions": [
                "Irrigate soon and bring the water level back to about 3-5 cm.",
                "Look for leaf rolling in the afternoon and drying leaf tips; they confirm the stress.",
                "Hold off on top-dressing fertilizer until the field has enough water again.",
                "Check that bunds are not leaking and that water reaches the whole field evenly.",
            ]},
        "Severe stress": {
            "level": "alert", "title": "Severe water stress",
            "summary": "Serious water shortage. Irrigate immediately.",
            "actions": [
                "Give a shallow irrigation (about 3-5 cm) at once and keep the water for several days.",
                "If the crop is flowering, act the same day. Even a short dry spell at this stage can reduce grain filling and yield.",
                "Delay urea or other nitrogen top-dressing until the plants have recovered.",
                "Ask your local agriculture officer about a foliar potassium spray to support recovery.",
                "Re-check after 3-4 days. Scorched leaves will not recover, but new growth should be green.",
            ]},
    },
    ("rice", "nutrients"): {
        "Nitrogen": {
            "level": "warn", "title": "Likely nitrogen (N) deficiency",
            "summary": "Older leaves turn pale yellow from the tip, and plants grow slowly with fewer tillers.",
            "actions": [
                "Top-dress with urea (neem-coated urea loses less nitrogen). Rice typically needs about 100-120 kg N/ha in total, split into basal, tillering and panicle-initiation doses. The exact amount depends on variety and soil.",
                "Use a leaf colour chart (LCC) to decide when to give the next dose instead of guessing.",
                "Apply on moist soil or in thin water, and avoid applying just before heavy rain.",
            ]},
        "Phosphorus": {
            "level": "warn", "title": "Likely phosphorus (P) deficiency",
            "summary": "Plants are stunted, with narrow, dark green or purplish leaves and few tillers.",
            "actions": [
                "Apply a phosphorus fertilizer such as single super phosphate (SSP) or DAP. Phosphorus moves slowly in soil, so put it near the roots, mostly as a basal dose. A typical range is 40-60 kg P2O5/ha.",
                "If the soil is acidic, check its pH. Liming can free up locked phosphorus.",
                "DAP also contains nitrogen, so count it in your nitrogen plan.",
            ]},
        "Potassium": {
            "level": "warn", "title": "Likely potassium (K) deficiency",
            "summary": "Older leaves show yellow-brown edges and tips, with brown spots. Plants may lodge.",
            "actions": [
                "Apply muriate of potash (MOP). A typical range is 40-60 kg K2O/ha, split between basal and panicle initiation.",
                "Return crop straw to the field where possible; it recycles potassium.",
                "Potassium-deficient plants get diseases more easily, so watch for leaf spots.",
            ]},
    },
    ("maize", "water_stress"): {
        "Healthy": {
            "level": "good", "title": "No water stress detected",
            "summary": "The crop looks adequately watered.",
            "actions": [
                "Keep the current irrigation schedule and check soil moisture regularly.",
                "Water is most critical at knee-high stage, tasselling and silking, and grain filling.",
                "Mulching helps hold soil moisture in hot weather.",
            ]},
        "Mild stress": {
            "level": "warn", "title": "Mild water stress",
            "summary": "Early signs of water shortage. Irrigate within a day or two.",
            "actions": [
                "Irrigate soon with light, frequent applications rather than waiting for a heavy watering.",
                "Midday leaf rolling that does not recover by evening confirms stress.",
                "If the crop is near tasselling or silking, make water the top priority.",
                "Delay nitrogen side-dressing until the soil is moist again.",
            ]},
        "Severe stress": {
            "level": "alert", "title": "Severe water stress",
            "summary": "Serious water shortage. Irrigate immediately.",
            "actions": [
                "Irrigate at once by furrow or drip, and repeat according to soil moisture.",
                "At tasselling and silking, stress can sharply reduce kernel set, so act the same day.",
                "After recovery, scout for pests and diseases that follow drought, such as spider mites and stalk rot.",
                "Apply mulch to reduce further moisture loss.",
            ]},
    },
    ("maize", "nutrients"): {
        "Healthy": {
            "level": "good", "title": "No nutrient deficiency detected",
            "summary": "The leaf does not show the common deficiency patterns.",
            "actions": [
                "Continue balanced fertilization based on a soil test.",
                "Photos can miss early deficiencies, so re-check if you see yellowing, purple tints or leaf-edge scorching.",
            ]},
        "Magnesium": {
            "level": "warn", "title": "Likely magnesium (Mg) deficiency",
            "summary": "Yellow stripes between the veins of older leaves, with the veins staying green.",
            "actions": [
                "A foliar spray of magnesium sulphate (Epsom salt) at about 1-2% gives a quick correction.",
                "On acidic soils, dolomitic lime supplies magnesium over time.",
                "Too much potassium can block magnesium uptake, so keep K and Mg in balance.",
            ]},
        "Nitrogen": {
            "level": "warn", "title": "Likely nitrogen (N) deficiency",
            "summary": "Older leaves turn yellow in a V-shape from the tip along the midrib.",
            "actions": [
                "Side-dress urea around the 6-8 leaf stage (about knee-high). A typical amount is 40-60 kg N/ha at this stage; total nitrogen for the season is often 120-150 kg N/ha depending on hybrid and soil.",
                "Apply on moist soil, a few centimetres from the stem, and cover it lightly.",
                "Avoid applying just before heavy rain, which washes nitrogen away.",
            ]},
        "Phosphorus": {
            "level": "warn", "title": "Likely phosphorus (P) deficiency",
            "summary": "Young plants show purple-red leaf edges and stunted growth.",
            "actions": [
                "Apply DAP or SSP in a band near the seed row at sowing (not touching the seed). A typical range is 40-60 kg P2O5/ha.",
                "Cold, wet soil makes the symptom worse; it may ease as the weather warms.",
                "If the soil is acidic, check its pH, since acid soils lock up phosphorus.",
            ]},
        "Potassium": {
            "level": "warn", "title": "Likely potassium (K) deficiency",
            "summary": "Yellowing and scorching along the edges of older leaves; stalks may be weak.",
            "actions": [
                "Apply muriate of potash (MOP) as a basal dose or early side-dress. A typical range is 30-50 kg K2O/ha.",
                "Weak stalks lodge easily, so earth up the rows and avoid late heavy nitrogen.",
                "Confirm with a soil test, because potassium needs vary a lot between soils.",
            ]},
    },
}


# ----------------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------------
def load_bundle(name):
    """Load (and cache) a trained model plus its metadata."""
    with _lock:
        if name in _cache:
            return _cache[name]
        model_path = os.path.join(C.MODEL_DIR, f"{name}.keras")
        meta_path = os.path.join(C.MODEL_DIR, f"{name}_meta.json")
        if not (os.path.exists(model_path) and os.path.exists(meta_path)):
            raise FileNotFoundError(
                f"The '{name}' model is not trained yet. "
                f"Run: python train_models.py {name}")
        model = keras.models.load_model(model_path, compile=False)
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        bundle = {"model": model, "meta": meta, "colour": None}
        colour_file = meta.get("colour_model")
        if colour_file:
            colour_path = os.path.join(C.MODEL_DIR, colour_file)
            if os.path.exists(colour_path):
                bundle["colour"] = joblib.load(colour_path)
        _cache[name] = bundle
        return bundle


def load_fusion():
    """Weights for combining the rice RGB and thermal results."""
    path = os.path.join(C.MODEL_DIR, "rice_water_fusion.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
        return float(cfg["weight_rgb"]), float(cfg["weight_thermal"])
    return 0.9, 0.1


def cnn_probs(bundle, batch):
    with _lock:
        return bundle["model"].predict(np.asarray(batch, dtype="float32"),
                                       batch_size=32, verbose=0)


def as_dict(classes, probs):
    return {c: round(float(p), 4) for c, p in zip(classes, probs)}


# ----------------------------------------------------------------------------
# Predictors - each returns (classes, probabilities, details, warnings)
# ----------------------------------------------------------------------------
def predict_nutrients(crop, img):
    bundle = load_bundle(f"{crop}_nutrients")
    size = int(bundle["meta"].get("input_size", C.IMG_SIZE))   # tile size the model was trained with
    tiles = C.tile_image(img, size)                   # same tiling as training
    probs = cnn_probs(bundle, tiles).mean(axis=0)     # average over tiles
    warns = []
    if crop == "rice":
        warns.append("This rice model only knows nitrogen, phosphorus and potassium "
                     "deficiency. It cannot tell you that a plant is healthy.")
    return bundle["meta"]["classes"], probs, {"tiles_used": int(len(tiles))}, warns


def predict_maize_water(img):
    bundle = load_bundle("maize_water")
    arr = C.whole_image(img, C.MAIZE_WATER_SIZE)
    probs = cnn_probs(bundle, arr)[0]
    warns = ["This model was trained on small 48x48 pixel images, so fine detail is "
             "lost. Treat the result as a screening aid."]
    return bundle["meta"]["classes"], probs, {}, warns


def predict_rice_water(img_rgb, img_thermal):
    bundle = load_bundle("rice_water_rgb")
    classes = bundle["meta"]["classes"]
    warns, details = [], {}

    cnn = cnn_probs(bundle, C.whole_image(img_rgb, C.IMG_SIZE))[0]
    details["rgb_cnn"] = as_dict(classes, cnn)
    rgb = cnn
    if bundle["colour"] is not None:
        colour = bundle["colour"].predict_proba(C.colour_features(img_rgb)[None, :])[0]
        w = float(bundle["meta"].get("ensemble_cnn_weight", 0.5))
        rgb = w * cnn + (1.0 - w) * colour
        details["rgb_colour_model"] = as_dict(classes, colour)
        details["rgb_cnn_weight"] = w
    details["rgb_combined"] = as_dict(classes, rgb)

    final = rgb
    if img_thermal is not None:
        try:
            tbundle = load_bundle("rice_water_thermal")
            thermal = cnn_probs(tbundle, C.whole_image(img_thermal, C.IMG_SIZE))[0]
            w_rgb, w_th = load_fusion()
            final = (w_rgb * rgb + w_th * thermal) / (w_rgb + w_th)
            details["thermal"] = as_dict(tbundle["meta"]["classes"], thermal)
            details["fusion_weights"] = {"rgb": round(w_rgb, 3), "thermal": round(w_th, 3)}
            warns.append(
                f"The thermal image was used with a small weight ({w_th * 100:.0f}%). On our "
                "test data the thermal model was not reliably better than guessing, so the "
                "result is driven mainly by the RGB photo.")
        except FileNotFoundError as e:
            warns.append(f"Thermal image ignored: {e}")
    else:
        warns.append("Result is based on the RGB image only. A thermal image is optional.")
    return classes, final, details, warns


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def error(message, code=400, headers=None):
    resp = jsonify({"ok": False, "error": message})
    resp.status_code = code
    if headers:
        resp.headers.update(headers)
    return resp


def too_many(wait):
    return error(f"Too many requests. Please wait {wait} seconds and try again.", 429,
                 {"Retry-After": str(wait)})


def read_image(file_storage):
    """Open an uploaded file safely. Raises ValueError with a user-friendly message."""
    ext = os.path.splitext(file_storage.filename or "")[1].lower()
    if ext not in C.IMAGE_EXTS:
        raise ValueError("Unsupported file type. Please upload a JPG, PNG or WEBP image.")
    try:
        with Image.open(file_storage.stream) as probe:        # reads only the header
            if probe.format not in ALLOWED_FORMATS:
                raise ValueError("Unsupported image format. Please upload a JPG, PNG or WEBP image.")
            if probe.width * probe.height > MAX_IMAGE_PIXELS:
                raise ValueError("That image has too many pixels. Please upload a smaller photo.")
        file_storage.stream.seek(0)
        return C.open_image(file_storage.stream)
    except ValueError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise ValueError("That image has too many pixels. Please upload a smaller photo.")
    except Exception:
        raise ValueError("That file could not be read as an image. Please try another photo.")


def make_preview(img):
    """Small JPEG as a data URL, so no uploaded photo is ever written to disk."""
    preview = img.copy()
    preview.thumbnail((640, 640))
    buffer = io.BytesIO()
    preview.save(buffer, "JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def credentials_ok(auth):
    if auth is None or auth.type != "basic":
        return False
    user_ok = hmac.compare_digest((auth.username or "").encode("utf-8"), AUTH_USER.encode("utf-8"))
    pass_ok = check_password_hash(AUTH_HASH, auth.password or "")   # always checked: same timing
    return user_ok and pass_ok


# ----------------------------------------------------------------------------
# Request guards and security headers
# ----------------------------------------------------------------------------
@app.before_request
def guard():
    ip = request.remote_addr or "unknown"

    # 1. Optional password protection
    if AUTH_ENABLED:
        wait = auth_limiter.retry_after(ip, record=False)
        if wait:
            return too_many(wait)
        auth = request.authorization
        if not credentials_ok(auth):
            if auth is not None:                      # wrong credentials: count the failure
                auth_limiter.retry_after(ip, record=True)
            return Response("Authentication required.", 401,
                            {"WWW-Authenticate": 'Basic realm="CropSense", charset="UTF-8"'})

    # 2. Block cross-site requests that send data
    if request.method not in ("GET", "HEAD", "OPTIONS"):
        origin = request.headers.get("Origin")
        if origin is not None and urlsplit(origin).netloc != request.host:
            return error("Cross-site request blocked.", 403)

    # 3. Rate limit the expensive endpoint
    if request.method == "POST" and request.path == "/predict":
        wait = predict_limiter.retry_after(ip, record=True)
        if wait:
            return too_many(wait)
    return None


@app.after_request
def add_security_headers(resp):
    resp.headers["Content-Security-Policy"] = CSP
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "no-referrer"
    resp.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=(), payment=()"
    resp.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    resp.headers["Cross-Origin-Resource-Policy"] = "same-origin"
    resp.headers["Server"] = "CropSense"
    if request.is_secure:
        resp.headers["Strict-Transport-Security"] = "max-age=31536000"
    if not request.path.startswith("/static/"):
        resp.headers["Cache-Control"] = "no-store"
    return resp


# ----------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------
@app.get("/")
def index():
    return render_template("index.html")


@app.get("/health")
def health():
    ready = all(os.path.exists(os.path.join(C.MODEL_DIR, f"{name}.keras")) for name in MODEL_NAMES)
    return jsonify({"ok": True, "models_ready": ready})


@app.post("/predict")
def predict():
    crop = request.form.get("crop", "").strip().lower()
    module = request.form.get("module", "").strip().lower()
    if crop not in ("rice", "maize"):
        return error("Please choose a crop: rice or maize.")
    if module not in ("water_stress", "nutrients"):
        return error("Please choose an analysis type: water stress or nutrient deficiency.")

    file = request.files.get("image")
    if file is None or not file.filename:
        return error("Please choose an image to analyse.")

    try:
        img = read_image(file)
        thermal_img = None
        if crop == "rice" and module == "water_stress":
            tfile = request.files.get("thermal_image")
            if tfile is not None and tfile.filename:
                thermal_img = read_image(tfile)

        if module == "nutrients":
            classes, probs, details, warnings_list = predict_nutrients(crop, img)
        elif crop == "maize":
            classes, probs, details, warnings_list = predict_maize_water(img)
        else:
            classes, probs, details, warnings_list = predict_rice_water(img, thermal_img)
    except ValueError as e:
        return error(str(e))
    except FileNotFoundError as e:
        return error(str(e), 503)
    except Exception:
        traceback.print_exc()
        return error("Something went wrong while analysing the image. Please try again.", 500)

    ranking = sorted(({"label": c, "prob": round(float(p), 4)} for c, p in zip(classes, probs)),
                     key=lambda r: r["prob"], reverse=True)
    top = ranking[0]
    if top["prob"] < LOW_CONFIDENCE:
        second = ranking[1]
        warnings_list.insert(0, f"The model is not sure ({top['prob'] * 100:.0f}% for the top answer). "
                                f"The second most likely answer is {second['label']} "
                                f"({second['prob'] * 100:.0f}%). Try a clearer, closer photo in good light.")

    images = {"image": make_preview(img)}
    if thermal_img is not None:
        images["thermal_image"] = make_preview(thermal_img)

    recommendation = RECOMMENDATIONS.get((crop, module), {}).get(top["label"])
    return jsonify({
        "ok": True,
        "crop": crop,
        "module": module,
        "prediction": top["label"],
        "confidence": top["prob"],
        "low_confidence": top["prob"] < LOW_CONFIDENCE,
        "probabilities": ranking,
        "recommendation": recommendation,
        "warnings": warnings_list,
        "details": details,
        "images": images,
        "disclaimer": DISCLAIMER,
    })


@app.errorhandler(413)
def too_large(_):
    return error("The upload is too large. Please use files under 16 MB in total.", 413)


@app.errorhandler(500)
def server_error(_):
    return error("Something went wrong on the server. Please try again.", 500)


if __name__ == "__main__":
    host = os.environ.get("APP_HOST", "127.0.0.1")
    port = int(os.environ.get("APP_PORT", "5000"))
    print("Looking for trained models in:", C.MODEL_DIR)
    for model_name in MODEL_NAMES:
        try:
            load_bundle(model_name)
            print(f"  loaded  {model_name}")
        except FileNotFoundError:
            print(f"  MISSING {model_name}  (train it with: python train_models.py {model_name})")
    print("\nPassword protection:", "ON" if AUTH_ENABLED else "OFF")
    if host not in ("127.0.0.1", "localhost") and not AUTH_ENABLED:
        print("WARNING: listening on a network address without password protection.")
    print(f"Open http://{host}:{port} in your browser")
    app.run(host=host, port=port, debug=False, use_reloader=False)