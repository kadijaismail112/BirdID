#!/usr/bin/env python3
# train_binary_head.py
# -----------------------------------------------------------
#  Run:
#    pip install birdnetlib tensorflow==2.16 tqdm
#
#    python train_binary_head.py \
#         --pos_dir clt_bird_call/processed_chunks \
#         --neg_dir other_bird_call \
#         --out_dir model_out \
#         --epochs 30
# -----------------------------------------------------------

import argparse, json, random
from pathlib import Path
import numpy as np, tensorflow as tf
from tqdm import tqdm
from birdnetlib.analyzer import Analyzer   # uses TFLite runtime, no TF deps
import librosa  # for loading audio files

SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)


# ---------- helpers ----------------------------------------------------------
def collect_wavs(root: Path, label: int):
    """Return [(Path,label), …] for every *.wav under *root* (recursive)."""
    return [(p, label) for p in root.rglob("*.wav")]

def train_val_test(data, val=0.1, test=0.1):
    random.shuffle(data)
    n = len(data); n_test = int(n*test); n_val = int(n*val)
    return data[: n-n_val-n_test], data[n-n_val-n_test: n-n_test], data[n-n_test:]

def embeddings(an: Analyzer, paths):
    embs = []
    # BirdNET expects 3-second chunks at 48kHz
    CHUNK_SIZE = 144000  # 48000 samples/sec * 3 seconds
    
    for p in tqdm(paths, desc="embeddings", unit="file"):
        # Load audio file using librosa
        audio, sr = librosa.load(str(p), sr=48000)  # BirdNET expects 48kHz
        
        # Ensure audio is in correct format (float32)
        audio = audio.astype(np.float32)
        
        # Split audio into 3-second chunks
        chunk_embeddings = []
        
        # If audio is shorter than CHUNK_SIZE, pad it
        if len(audio) < CHUNK_SIZE:
            # Pad with zeros to reach CHUNK_SIZE
            padded_audio = np.zeros(CHUNK_SIZE, dtype=np.float32)
            padded_audio[:len(audio)] = audio
            
            # Process the padded chunk
            data = np.array([padded_audio], dtype="float32")
            features = an._return_embeddings(data)
            chunk_embeddings.append(features[0])
        else:
            # Process each full chunk
            for i in range(0, len(audio) - CHUNK_SIZE + 1, CHUNK_SIZE // 2):  # 50% overlap
                chunk = audio[i:i + CHUNK_SIZE]
                
                # Skip chunks that are too short
                if len(chunk) < CHUNK_SIZE:
                    continue
                    
                data = np.array([chunk], dtype="float32")
                features = an._return_embeddings(data)
                chunk_embeddings.append(features[0])
        
        # If we got any valid embeddings, average them
        if chunk_embeddings:
            file_embedding = np.mean(chunk_embeddings, axis=0)
            embs.append(file_embedding)
        else:
            print(f"Warning: No valid embeddings for {p}")
    
    return np.stack(embs)

def build_head():
    m = tf.keras.Sequential([
        tf.keras.layers.Input((1024,)),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="binary_crossentropy",
              metrics=["accuracy"])
    return m


def main(pos_dir: Path, neg_dir: Path, out_dir: Path, epochs: int):
    out_dir.mkdir(parents=True, exist_ok=True)

    data = collect_wavs(pos_dir, 1) + collect_wavs(neg_dir, 0)
    train, val, test = train_val_test(data)

    an = Analyzer()                           # frozen BirdNET backbone
    X_tr = embeddings(an, [p for p,_ in train]); y_tr = np.array([y for _,y in train])
    X_va = embeddings(an, [p for p,_ in val]);   y_va = np.array([y for _,y in val])
    X_te = embeddings(an, [p for p,_ in test]);  y_te = np.array([y for _,y in test])

    model = build_head()
    model.fit(X_tr, y_tr,
              validation_data=(X_va, y_va),
              epochs=epochs,
              batch_size=64,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=5,
                                                          restore_best_weights=True)])

    loss, acc = model.evaluate(X_te, y_te, verbose=0)
    print(f"test accuracy = {acc:.3f}")

    # save metrics
    (out_dir / "metrics.json").write_text(json.dumps({"test_accuracy": float(acc)}, indent=2))

    # Save model in Keras format
    print("Saving model in Keras format...")
    model.save(out_dir / "classifier.keras")
    print("Model saved successfully")

    # Save label list
    (out_dir / "labels.txt").write_text("other\nclt\n")
    print("saved:", out_dir / "classifier.keras", "and labels.txt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos_dir", required=True, type=Path)
    ap.add_argument("--neg_dir", required=True, type=Path,
                    help="top folder that contains sub-folders for each negative species")
    ap.add_argument("--out_dir", default="model_out", type=Path)
    ap.add_argument("--epochs", type=int, default=30)
    main(**vars(ap.parse_args()))
