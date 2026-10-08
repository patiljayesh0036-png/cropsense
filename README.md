# CropSense

AI-powered crop stress and nutrient deficiency detection for rice and maize. A final year BE project.

A farmer picks a crop (rice or maize) and a check (water stress or nutrient deficiency), uploads a photo, and gets a prediction, a confidence score and practical advice on irrigation or fertilizer.

## What it does

| Check | Method | Measured accuracy |
|---|---|---|
| Rice, water stress (photo) | Colour-histogram random forest (also trained: MobileNetV2, 75.6%) | 83.3% (5-fold cross-validation, 90 images) |
| Rice, water stress (thermal) | MobileNetV2 | 47.0% (about chance; used with a 5% weight) |
| Rice, nutrient deficiency (N, P, K) | MobileNetV2 on image tiles | 81.0% per photo (174 test photos) |
| Maize, water stress | MobileNetV2 | 72.5% (6,065 test images) |
| Maize, nutrient deficiency (5 classes) | MobileNetV2 on image tiles | 82.9% per photo (70 test photos) |

Rice water stress can take an optional thermal image, combined with the photo result by weighted averaging. Maize water stress uses RGB only.

## Limitations

- Several datasets are small, so the accuracies are rough estimates.
- The rice RGB and thermal images are not paired, so the combined accuracy cannot be measured. The thermal model is no better than guessing on this data.
- The maize water stress images are crops from only 656 original photos, so the test accuracy may be optimistic for new fields. Upload a close-up of a single leaf.
- The rice nutrient model cannot recognise a healthy leaf.
- Results are screening aids, not prescriptions. Confirm with a soil test and an agriculture officer.

## Project structure

```
app.py               Flask server (upload, prediction, advice, security)
common.py            shared image helpers and settings
train_models.py      trains the models (MobileNetV2 transfer learning)
security_check.py    read-only security audit
templates/           index.html
static/              style.css, app.js
models/              trained models and accuracy reports
datasets/            NOT included (see below)
```

## Setup

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```
Then open http://127.0.0.1:5000

To retrain (needs the datasets, placed in `datasets/` with the same folder layout):
```
python train_models.py all
```

## Optional password protection

Set `APP_USER` and `APP_PASSWORD_HASH` (a hash made with `werkzeug.security.generate_password_hash`) before starting the app. Use it only over HTTPS. The built-in Flask server is for demos, not production.

## Datasets and credits

- Maize water stress: *A maize leaf image dataset with three water treatment levels*, Shuo Zhuang, Mendeley Data, 2019, DOI 10.17632/w77b87fvct.1. Check the licence on the dataset page.
- Rice datasets: ADD THE SOURCE AND LICENCE OF YOUR RICE DATASETS HERE.
- Maize nutrient dataset: ADD THE SOURCE HERE.

The datasets are not included in this repository.