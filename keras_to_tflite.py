import tensorflow as tf
import os

try:
    # Load the Keras model
    print("Loading Keras model...")
    model = tf.keras.models.load_model('model_out/classifier.keras')

    # Export to SavedModel format using Keras 3 API
    saved_model_dir = 'model_out/saved_model'
    print(f"Exporting model as SavedModel to {saved_model_dir} ...")
    model.export(saved_model_dir)

    # Convert the SavedModel to TFLite
    print("Converting SavedModel to TFLite format...")
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float32]
    tflite_model = converter.convert()

    # Save the TFLite model
    tflite_path = 'model_out/classifier_from_savedmodel.tflite'
    print(f"Saving TFLite model to {tflite_path} ...")
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)

    print(f"Model successfully converted to TFLite format and saved as '{tflite_path}'")

except Exception as e:
    print(f"An error occurred during conversion: {str(e)}")
