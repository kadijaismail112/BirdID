import sys, pathlib, numpy as np, tensorflow as tf

if len(sys.argv) != 2:
    sys.exit("usage: python check_shape.py model.tflite")

model_path = pathlib.Path(sys.argv[1])
interpreter = tf.lite.Interpreter(model_path=str(model_path))
interpreter.allocate_tensors()

inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]

def dname(d):  # robust dtype string
    return np.dtype(d).name

print("MODEL :", model_path)
print("INPUT :", inp["shape"], "dtype", dname(inp["dtype"]))
print("OUTPUT:", out["shape"], "dtype", dname(out["dtype"]))