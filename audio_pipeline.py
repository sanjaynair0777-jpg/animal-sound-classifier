"""
Inference-side copy of the preprocessing + models from the IAML notebook.
Everything here mirrors the notebook exactly (sr=22050, 4 s centre crop/pad,
peak normalisation, same feature extractors) so predictions match training.
"""
import json
from pathlib import Path

import librosa
import numpy as np
import torch
from torch import nn

SR = 22050
DURATION = 4.0
MODELS_DIR = Path(__file__).parent / "models"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------- preprocessing (notebook cell 6) ----------
def preprocess_audio(filepath, target_sr=SR, target_duration=DURATION):
    y, _ = librosa.load(filepath, sr=target_sr)
    target_length = int(target_sr * target_duration)
    if len(y) > target_length:
        start = (len(y) - target_length) // 2
        y = y[start:start + target_length]
    elif len(y) < target_length:
        pad_left = (target_length - len(y)) // 2
        pad_right = target_length - len(y) - pad_left
        y = np.pad(y, (pad_left, pad_right), mode="constant")
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    return y


# ---------- features (notebook cells 11 and 17) ----------
def extract_spectral_features(y, sr=SR):
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)
    return np.hstack([
        centroid.mean(), bandwidth.mean(), contrast.mean(axis=1),
        rolloff.mean(), zcr.mean(), rms.mean(),
    ])


def make_melspec_image(y, sr=SR, n_mels=128):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    S_dB = librosa.power_to_db(S, ref=np.max)
    S_norm = (S_dB - S_dB.min()) / (S_dB.max() - S_dB.min())
    return S_norm.astype(np.float32)


# ---------- CNN (notebook cell 20) ----------
class AudioCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        with torch.no_grad():
            flat_dim = self.features(torch.zeros(1, 1, 128, 173)).numel()
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(flat_dim, 128), nn.ReLU(), nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ---------- model registry ----------
SPEC = "Logistic Regression (spectral features)"
CNN = "CNN (mel-spectrogram)"
VGG = "VGGish + Logistic Regression"

_cache = {}


def load_classes():
    p = MODELS_DIR / "classes.json"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Run the export cell at the end of your notebook first."
        )
    return json.loads(p.read_text())


def _vggish():
    if "vggish" not in _cache:
        from torchvggish import vggish
        m = vggish()
        m.to(DEVICE).eval()
        _cache["vggish"] = m
    return _cache["vggish"]


def available_models():
    out = []
    if (MODELS_DIR / "logreg_spectral.joblib").exists():
        out.append(SPEC)
    if (MODELS_DIR / "cnn_mel.pt").exists():
        out.append(CNN)
    if (MODELS_DIR / "logreg_vggish.joblib").exists():
        out.append(VGG)
    return out


def _load(name):
    if name in _cache:
        return _cache[name]
    import joblib
    if name == SPEC:
        _cache[name] = joblib.load(MODELS_DIR / "logreg_spectral.joblib")
    elif name == VGG:
        _cache[name] = joblib.load(MODELS_DIR / "logreg_vggish.joblib")
    elif name == CNN:
        m = AudioCNN(num_classes=len(load_classes()))
        m.load_state_dict(torch.load(MODELS_DIR / "cnn_mel.pt", map_location=DEVICE))
        m.to(DEVICE).eval()
        _cache[name] = m
    return _cache[name]


def _vggish_embedding(y, sr=SR):
    from torchvggish import vggish_input
    patches = vggish_input.waveform_to_examples(np.array(y).astype(float).flatten(), sr)
    if isinstance(patches, torch.Tensor):
        patches = patches.float().to(DEVICE)
    else:
        patches = torch.from_numpy(patches).float().to(DEVICE)
    with torch.no_grad():
        emb = _vggish()(patches)
    return emb.mean(dim=0).cpu().numpy()


def _sk_probs(model, X, n_classes):
    """predict_proba mapped onto the full class list via model.classes_."""
    p = model.predict_proba(X)[0]
    full = np.zeros(n_classes)
    full[np.asarray(model.classes_, dtype=int)] = p
    return full


def predict_probs(name, y):
    """Return a probability vector over all classes for one model."""
    classes = load_classes()
    model = _load(name)
    if name == SPEC:
        return _sk_probs(model, extract_spectral_features(y).reshape(1, -1), len(classes))
    if name == VGG:
        return _sk_probs(model, _vggish_embedding(y).reshape(1, -1), len(classes))
    if name == CNN:
        x = torch.from_numpy(make_melspec_image(y)[None, None]).to(DEVICE)
        with torch.no_grad():
            return torch.softmax(model(x), dim=1)[0].cpu().numpy()
    raise ValueError(name)
