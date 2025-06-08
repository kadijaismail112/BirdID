import os
import glob
import numpy as np
import librosa
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# =============================================================================
# 1. Config
# =============================================================================
DATA_DIR    = "preprocessed_audio"                # expects subfolders per class: data/<label>/*.wav
LABELS_PATH = "new_model2_Labels.txt"
MODEL_PATH  = "new_model2.tflite"

SAMPLE_RATE = 48000
DURATION    = 3.0                   # seconds
NUM_SAMPLES = int(SAMPLE_RATE * DURATION)

TEST_SIZE   = 0.1                   # 10% hold-out
RANDOM_SEED = 42

# =============================================================================
# 2. Load labels
# =============================================================================
with open(LABELS_PATH) as f:
    LABELS = [line.strip() for line in f if line.strip()]
if len(LABELS) == 0:
    raise ValueError("No labels found in " + LABELS_PATH)
print("Classes:", LABELS)

# =============================================================================
# 3. Discover all audio files + true labels
# =============================================================================
filepaths = []
y_true    = []
for label in LABELS:
    pattern = os.path.join(DATA_DIR, label, "*.wav")
    for wav in glob.glob(pattern):
        filepaths.append(wav)
        y_true.append(label)

if not filepaths:
    raise RuntimeError(f"No .wav files found under {DATA_DIR}/<label>/")

# =============================================================================
# 4. Split into train / test
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    filepaths, y_true,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    stratify=y_true
)
print(f"Total files: {len(filepaths)} → Test set: {len(X_test)} files")

# =============================================================================
# 5. Load TFLite model
# =============================================================================
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Sanity check
_, in_samples = input_details[0]["shape"]   # e.g. [1, 144000]
_, n_classes  = output_details[0]["shape"]  # e.g. [1, 12]
assert in_samples == NUM_SAMPLES, f"Model expects {in_samples} samples, script uses {NUM_SAMPLES}"
assert n_classes == len(LABELS), f"Model outputs {n_classes} classes, but found {len(LABELS)} labels"

# =============================================================================
# 6. Inference on the test set
# =============================================================================
y_pred = []
for path in X_test:
    # 6.1 Load & preprocess
    wav, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True, duration=DURATION)
    if len(wav) < NUM_SAMPLES:
        wav = np.pad(wav, (0, NUM_SAMPLES - len(wav)), mode="constant")
    else:
        wav = wav[:NUM_SAMPLES]

    # 6.2 Run TFLite
    inp = wav.astype(np.float32)[np.newaxis, :]
    interpreter.set_tensor(input_details[0]["index"], inp)
    interpreter.invoke()
    out = interpreter.get_tensor(output_details[0]["index"])[0]

    # 6.3 Decode prediction
    idx = int(np.argmax(out))
    y_pred.append(LABELS[idx])

# =============================================================================
# 7. Compute & display confusion matrix
# =============================================================================
cm = confusion_matrix(y_test, y_pred, labels=LABELS)
print("\n" + classification_report(y_test, y_pred, target_names=LABELS))

# 7.1 Plot
fig, ax = plt.subplots(figsize=(8, 8))
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
ax.set(
    xticks=np.arange(len(LABELS)),
    yticks=np.arange(len(LABELS)),
    xticklabels=LABELS,
    yticklabels=LABELS,
    ylabel="True label",
    xlabel="Predicted label",
    title="Confusion Matrix (10% hold-out)"
)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(
            j, i, f"{cm[i, j]}",
            ha="center", va="center",
            color="white" if cm[i, j] > thresh else "black"
        )

plt.tight_layout()
plt.show()
