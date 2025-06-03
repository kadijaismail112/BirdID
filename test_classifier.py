#!/usr/bin/env python3
"""
test_classifier.py
-----------------
Small utility script to verify the classifier model can be loaded properly.
"""

import tensorflow as tf
import sys

# Custom DepthwiseConv2D class to handle incompatible 'groups' parameter
class CustomDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    @classmethod
    def from_config(cls, config):
        # Remove the 'groups' parameter if it exists
        if 'groups' in config:
            config = config.copy()
            config.pop('groups')
        return super(CustomDepthwiseConv2D, cls).from_config(config)

def main():
    # Define the path to the classifier model
    classifier_path = "model_out/classifier.keras"
    
    # Try to load the model with custom objects
    try:
        print(f"Attempting to load classifier model from: {classifier_path}")
        custom_objects = {"DepthwiseConv2D": CustomDepthwiseConv2D}
        model = tf.keras.models.load_model(classifier_path, custom_objects=custom_objects)
        
        # Print model summary
        print("\nModel successfully loaded!")
        print(f"Model type: {type(model)}")
        
        # Print input and output shapes
        if hasattr(model, 'input_shape'):
            print(f"Input shape: {model.input_shape}")
        if hasattr(model, 'output_shape'):
            print(f"Output shape: {model.output_shape}")
        
        # Print model summary
        print("\nModel summary:")
        model.summary()
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

