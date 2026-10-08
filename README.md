# Animal Sound Classifier

A machine learning project that classifies short audio clips into one of ten animals, comparing several
feature-extraction and model combinations, plus a small web app for playing a clip and seeing what each model predicts.

**Classes:** bird, cat, chicken, cow, dog, donkey, frog, lion, monkey, sheep

## Contents

| File | Purpose |
|---|---|
| `animal_audio_classification_code.ipynb` | The full pipeline: data checks, preprocessing, feature extraction, training and evaluation |
| `app.py` | Gradio web app: play a clip, see each model's prediction and confidence |
| `audio_pipeline.py` | Preprocessing and model code used by the app (mirrors the notebook exactly) |
| `requirements.txt` | Python dependencies |

## Dataset

[Animal Sound Dataset by Yash Nita](https://github.com/YashNita/Animal-Sound-Dataset): 875 `.wav` files.
The dataset is **not included** in this repository. To rerun the notebook, download it and place the audio
files in a folder called `animals_sounds/` next to the notebook (the label is read from the start of each filename, e.g. `cat (12).wav`).

| Animal | Files | Animal | Files |
|---|---|---|---|
| bird | 200 | lion | 45 |
| cat | 200 | sheep | 40 |
| dog | 200 | frog | 35 |
| cow | 75 | chicken | 30 |
| donkey | 25 | monkey | 25 |

The classes are heavily imbalanced, and clip lengths range from about 0.2 s to 40 s.

## Method

**Preprocessing.** Every clip is resampled to 22,050 Hz, centre-cropped or zero-padded to exactly 4 s
(88,200 samples) and peak-normalised. Data is split 80/10/10 into train/validation/test (stratified, `random_state=42`).

**Pipelines compared**

| Features | Model |
|---|---|
| none (mean of the signal) | Dummy classifier (baseline) |
| Classical spectral features (centroid, bandwidth, contrast, rolloff, zero-crossing rate, RMS) via librosa | Logistic Regression |
| Mel spectrogram (128 mels) | Small CNN (PyTorch) |
| VGGish embeddings (128-d, averaged over time) via torchvggish | Logistic Regression |

## Results

| Pipeline | Test accuracy |
|---|---|
| Dummy classifier (validation accuracy) | 23.0% |
| Spectral features + Logistic Regression | 59.1% |
| VGGish embeddings + Logistic Regression | 79.5% |
| **Mel spectrogram + CNN** | **85.2%** |

- The **CNN on mel spectrograms** was the best performer and was confused least often.
- VGGish + Logistic Regression reached 100% training accuracy, which points to overfitting, and it was confused more often than the CNN.
- All models did best on bird, cat and dog, the three classes with the most data, and worse on the rarer animals.
- The test set is only 88 clips, so these numbers are rough, and the CNN is not seeded, so retraining can shift its score slightly.

**Limitations and ideas for improvement:** the dataset is imbalanced and many clips are very short, so padding to 4 s leaves
some clips mostly silence. Filtering out very short clips and merging in another dataset to balance the classes would likely help. 
Train on another dataset as the models only perform well for birds, cats and dogs as they were the most populated in the dataset.

## Running the notebook

```bash
python -m venv venv
# Windows: venv\Scripts\activate      Mac/Linux: source venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --user --name animal-audio
```

Open the notebook, select the `animal-audio` kernel and run all cells. Python 3.11 was used originally.
The first time VGGish is created, `torchvggish` downloads its pretrained weights (about 275 MB).

## Trying the models in the web app

1. Run the notebook top to bottom
   This creates `models/` containing `classes.json`, `logreg_spectral.joblib`, `logreg_vggish.joblib` and `cnn_mel.pt`.
2. Start the app from the same environment:

```bash
python app.py
```

3. Open http://127.0.0.1:7860, upload or record a clip, press **Classify**, and optionally select the real animal to see each model marked right or wrong.

The app uses the same preprocessing as the notebook, tells you when a clip was cropped or padded to 4 s,
and shows the waveform and mel spectrogram of what the models actually heard. Any `.wav` files you put in a `samples/` folder appear as one-click examples.

Use the same scikit-learn version for training and for the app, otherwise the saved `.joblib` files may fail to load.
Only load model files you trust, since `.joblib` files can execute code when opened.
