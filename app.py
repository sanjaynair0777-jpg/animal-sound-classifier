"""Animal sound classifier: play a clip, see what each of your models thinks it is."""
import glob
import os

import gradio as gr
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import audio_pipeline as ap

CLASSES = ap.load_classes()
MODELS = ap.available_models()
if not MODELS:
    raise SystemExit("No trained models found in ./models. Run the export cell in the notebook first.")

EMOJI = {"bird": "🐦", "cat": "🐱", "chicken": "🐔", "cow": "🐮", "dog": "🐶",
         "donkey": "🫏", "frog": "🐸", "lion": "🦁", "monkey": "🐵", "sheep": "🐑"}
label = lambda c: f"{EMOJI.get(c, '')} {c}".strip()


def make_plot(y):
    fig, ax = plt.subplots(2, 1, figsize=(7, 4.6), constrained_layout=True)
    t = np.arange(len(y)) / ap.SR
    ax[0].plot(t, y, lw=0.6)
    ax[0].set(title="Waveform (the 4 s window the models actually hear)", xlabel="seconds", xlim=(0, ap.DURATION))
    S = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=ap.SR, n_mels=128), ref=np.max)
    img = librosa.display.specshow(S, sr=ap.SR, x_axis="time", y_axis="mel", ax=ax[1])
    ax[1].set(title="Mel spectrogram")
    fig.colorbar(img, ax=ax[1], format="%+2.0f dB")
    return fig


def classify(audio_path, chosen_models, true_animal):
    if audio_path is None:
        raise gr.Error("Upload, record or pick an audio clip first.")
    if not chosen_models:
        raise gr.Error("Pick at least one model.")

    y = ap.preprocess_audio(audio_path)
    raw_dur = librosa.get_duration(path=audio_path)

    table = {"animal": [label(c) for c in CLASSES]}
    lines = []
    for name in chosen_models:
        p = ap.predict_probs(name, y)
        table[name] = [round(float(v) * 100, 1) for v in p]
        top = int(np.argmax(p))
        verdict = ""
        if true_animal and true_animal != "(unknown)":
            verdict = "  ✅ correct" if CLASSES[top] == true_animal else f"  ❌ (actual: {true_animal})"
        lines.append(f"**{name}** → {label(CLASSES[top])} ({p[top]*100:.1f}%){verdict}")

    note = ""
    if raw_dur > ap.DURATION:
        note = f"\n\n_Clip is {raw_dur:.1f} s long: only the middle {ap.DURATION:.0f} s was classified, as in training._"
    elif raw_dur < ap.DURATION:
        note = f"\n\n_Clip is {raw_dur:.1f} s long: padded with silence to {ap.DURATION:.0f} s, as in training._"

    df = pd.DataFrame(table)
    fig = make_plot(y)
    plt.close(fig)
    return "\n\n".join(lines) + note, df, fig


samples = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "samples", "*.wav")))
guess = lambda p: os.path.basename(p).split("_")[0]

with gr.Blocks(title="Animal sound classifier") as demo:
    gr.Markdown("# 🐾 Animal sound classifier\nPlay a clip, then see whether each model can tell which animal it is.")
    with gr.Row():
        with gr.Column():
            audio = gr.Audio(type="filepath", sources=["upload", "microphone"], label="Audio clip (play it here)")
            models = gr.CheckboxGroup(MODELS, value=MODELS, label="Models to run")
            truth = gr.Dropdown(["(unknown)"] + CLASSES, value="(unknown)",
                                label="Actual animal (optional, to mark right/wrong)")
            btn = gr.Button("Classify", variant="primary")
            if samples:
                gr.Examples([[s, MODELS, guess(s) if guess(s) in CLASSES else "(unknown)"] for s in samples],
                            inputs=[audio, models, truth], label="Sample clips")
        with gr.Column():
            verdict = gr.Markdown()
            probs = gr.Dataframe(label="Confidence per animal (%)", interactive=False)
            plot = gr.Plot(label="What the model sees")
    btn.click(classify, [audio, models, truth], [verdict, probs, plot])

if __name__ == "__main__":
    demo.launch()
