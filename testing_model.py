import time
import numpy as np
import sounddevice as sd
sd.default.device = 0 
import tensorflow as tf
import matplotlib.pyplot as plt

# ======================================
# 1. Load species (class) names from labels.txt
# ======================================
LABELS_PATH = "new_model2_Labels.txt"
with open(LABELS_PATH, "r") as f:
    LABELS = [line.strip() for line in f.read().splitlines() if line.strip()]

if len(LABELS) != 12:
    raise ValueError(f"Expected 12 labels, but found {len(LABELS)} in '{LABELS_PATH}'.")

print(f"Loaded {len(LABELS)} labels:")
for idx, name in enumerate(LABELS):
    print(f"  {idx:>2} → {name}")

# ======================================
# 2. Model & Recording Parameters
# ======================================
MODEL_PATH = "new_model2.tflite"  # TFLite file
SAMPLE_RATE = 48000
DURATION_SEC = 3.0
NUM_SAMPLES = int(SAMPLE_RATE * DURATION_SEC)  # 144000

# ======================================
# 3. Function to record exactly 3 seconds of audio
# ======================================
def record_waveform() -> np.ndarray:
    """
    Record exactly 3 seconds (144000 samples) of mono audio at 48 kHz.
    Returns a float32 NumPy array of shape (144000,) with values in [-1.0, +1.0].
    """
    audio = sd.rec(
        frames=NUM_SAMPLES,
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )   # shape: (144000, 1)
    print(sd.query_devices())
    sd.wait()  # block until recording is done
    
    return audio[:, 0]  # flatten to shape (144000,)

# ======================================
# 4. Load the TFLite model
# ======================================
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Check that the interpreter’s input shape matches [1, 144000]:
inp_shape = input_details[0]["shape"]         # e.g. [1, 144000]
out_shape = output_details[0]["shape"]        # e.g. [1, 11]

print(f"\nTFLite input shape: {inp_shape}")
print(f"TFLite output shape: {out_shape}\n")

# Sanity‐check: out_shape should be [1, 11]
_, num_classes = out_shape
if num_classes != len(LABELS):
    raise ValueError(
        f"Model outputs {num_classes} scores, but you have {len(LABELS)} labels."
    )

# ======================================
# 5. Real‐time inference loop
# ======================================
def run_realtime_inference():
    devices = sd.query_devices()
    #for idx, dev in enumerate(devices):
        #if dev['max_input_channels'] > 0:
        #    print(f"Input #{idx}: {dev['name']} (Channels: {dev['max_input_channels']})")
    
    print("Starting real‐time inference. Press Ctrl+C to stop.\n")
    try:
        while True:
            t0 = time.time()
            
            # Record raw waveform
            waveform = record_waveform()  # shape: (144000,)

            # Convert to shape [1, 144000] (batch dimension)
            input_tensor = waveform[np.newaxis, :].astype(np.float32)  
            # dtype float32 to align with TFLite model, expects float32
            #(f"Waveform min/max: {np.min(waveform)}, {np.max(waveform)}")
            #print(f"Input tensor shape: {input_tensor.shape}, dtype: {input_tensor.dtype}")
            # Feed into TFLite interpreter
            interpreter.set_tensor(input_details[0]["index"], input_tensor)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]["index"])[0]
            # output_data is now a length‐11 float32 array of class logits/probabilities

            # 5.4 Postprocess: pick top‐1
            top_idx = int(np.argmax(output_data))
            top_conf = float(output_data[top_idx])
            predicted_label = LABELS[top_idx]

            elapsed = time.time() - t0
            timestamp = time.strftime("%H:%M:%S")

            print(
                f"[{timestamp}] Detected: {predicted_label:<20} "
                f"(confidence={top_conf:.3f})  "
                f"Latency={elapsed:.2f}s"
            )

    except KeyboardInterrupt:
        print("\n⏹ Real‐time inference stopped by user.\n")

if __name__ == "__main__":
    run_realtime_inference()
