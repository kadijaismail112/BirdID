#!/usr/bin/env python3
"""
merge_backbone_head_local.py
----------------------------
Stitch `audio-model.h5` (full BirdNET waveform backbone) to a binary head and
export `birdnet_clt_full.tflite` that BirdNET-Pi can load.

Example
-------
python merge_backbone_head_local.py \
    --backbone   birdnet_backbone/keras/audio-model.h5 \
    --head_model model_out/classifier.keras \
    --out_file   model_out/birdnet_clt_full.tflite \
    --head_dim   1024           # use 512 if you trained a 512-D head
"""

import argparse
import pathlib
import tensorflow as tf
import numpy as np
import logging
import time
import os
import psutil
import gc
import sys
import h5py
import json
import traceback
from tensorflow.python.eager import context

# Configure logging - create a module-level logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("merge_backbone")

def setup_logging(level=logging.INFO):
    """Set up logging with the specified level"""
    global logger
    logger.setLevel(level)
    return logger

# Memory tracking utility
def get_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return {
        "rss": memory_info.rss / (1024 * 1024),  # RSS in MB
        "vms": memory_info.vms / (1024 * 1024),  # VMS in MB
    }

def log_memory_usage(prefix=""):
    mem = get_memory_usage()
    logger.info(f"{prefix}Memory usage - RSS: {mem['rss']:.2f} MB, VMS: {mem['vms']:.2f} MB")

SR, WIN_SEC = 48_000, 3
SAMPLES = SR * WIN_SEC            # 144 000

# Custom compatibility layers to handle older Keras model format
class CustomDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    """Custom DepthwiseConv2D that handles incompatible parameters from older Keras versions."""
    
    @classmethod
    def from_config(cls, config):
        # Make a copy to avoid modifying the original
        config = config.copy()
        
        # Remove known incompatible parameters
        for param in ['groups', 'kernel_initializer', 'bias_initializer']:
            if param in config and param != 'depthwise_initializer':
                print(f"Removing incompatible parameter from DepthwiseConv2D: {param}")
                config.pop(param, None)
        
        # Handle special case for initialization parameters
        if 'kernel_initializer' in config:
            config['depthwise_initializer'] = config.pop('kernel_initializer')
            
        return super(CustomDepthwiseConv2D, cls).from_config(config)

class CustomConv2D(tf.keras.layers.Conv2D):
    """Custom Conv2D that handles incompatible parameters from older Keras versions."""
    
    @classmethod
    def from_config(cls, config):
        # Make a copy to avoid modifying the original
        config = config.copy()
        
        # Remove known incompatible parameters
        if 'groups' in config and int(config['groups']) == 1:
            print("Removing groups=1 parameter from Conv2D")
            config.pop('groups')
            
        return super(CustomConv2D, cls).from_config(config)

class CustomBatchNormalization(tf.keras.layers.BatchNormalization):
    """Custom BatchNormalization that handles axis conversion between Keras versions."""
    
    @classmethod
    def from_config(cls, config):
        # Make a copy to avoid modifying the original
        config = config.copy()
        
        # Handle axis conversion issues
        if 'axis' in config:
            axis = config['axis']
            # Always use -1 for channel-last format
            if isinstance(axis, (tuple, list)):
                print(f"Converting BatchNorm axis from {axis} to -1")
                config['axis'] = -1
            elif not isinstance(axis, int):
                print(f"BatchNorm has invalid axis type {type(axis)}, defaulting to -1")
                config['axis'] = -1
            elif axis == 3:  # Convert specific problematic axis
                print(f"Converting BatchNorm axis from {axis} to -1")
                config['axis'] = -1
        
        return super(CustomBatchNormalization, cls).from_config(config)

# Custom activation functions
def swish(x):
    """Swish activation function: x * sigmoid(x)"""
    return x * tf.sigmoid(x)

# Register the custom activation
tf.keras.utils.get_custom_objects().update({'swish': swish})

class CustomLayer(tf.keras.layers.Layer):
    """Base custom layer to handle unknown parameters."""
    
    @classmethod
    def from_config(cls, config):
        # Filter out any unknown parameters
        config = {k: v for k, v in config.items() if k in cls.__init__.__code__.co_varnames}
        return super(CustomLayer, cls).from_config(config)

# ── stub for BirdNET’s custom layer ──────────────────────────────────────────
class MelSpecLayerSimple(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(dtype=kwargs.get("dtype", "float32"))
        self._config = kwargs      # store for get_config

    def call(self, inputs, **_):   # identity pass-through
        return inputs

    def get_config(self):
        cfg = super().get_config()
        cfg.update(self._config)
        return cfg

# Create custom objects dictionary
custom_objects = {
    # Custom layers
    "MelSpecLayerSimple": MelSpecLayerSimple,
    "DepthwiseConv2D": CustomDepthwiseConv2D,
    "Conv2D": CustomConv2D,
    
    # Activations
    "swish": swish,
    
    # Layer names that might have custom implementations
    "Reshape": tf.keras.layers.Reshape,
    "Activation": tf.keras.layers.Activation,
    "Add": tf.keras.layers.Add,
    "Multiply": tf.keras.layers.Multiply,
    "GlobalAveragePooling2D": tf.keras.layers.GlobalAveragePooling2D,
    "Dropout": tf.keras.layers.Dropout,
    "BatchNormalization": CustomBatchNormalization,
    "ZeroPadding2D": tf.keras.layers.ZeroPadding2D,
    "Dense": tf.keras.layers.Dense,
    "InputLayer": tf.keras.layers.InputLayer,
    "Concatenate": tf.keras.layers.Concatenate,
    "MaxPooling2D": tf.keras.layers.MaxPooling2D,
    "AveragePooling2D": tf.keras.layers.AveragePooling2D
}

# ── helper to cut the backbone at the penultimate Dense ─────────────────────
def load_backbone(backbone_path: pathlib.Path):
    """
    Load the backbone model with compatibility handling and extract the embedding layer.
    
    Args:
        backbone_path: Path to the backbone model file
        
    Returns:
        A keras Model that outputs the penultimate layer embeddings
    """
    logger.info(f"Loading backbone model from {backbone_path}")
    start_time = time.time()
    
    try:
        # Configure memory growth to prevent OOM errors
        logger.info("Configuring GPU memory growth")
        for gpu in tf.config.experimental.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)
        
        # Set memory optimization options
        tf.keras.backend.clear_session()
        gc.collect()
        
        # Set TF behavior flags for better compatibility
        os.environ['TF_KERAS'] = '1'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF internal logging
        
        logger.info("Using custom layer classes to handle compatibility issues")
        
        # Use low-level H5 file handling to check model format version
        with h5py.File(backbone_path, 'r') as h5file:
            if 'keras_version' in h5file.attrs:
                keras_version = h5file.attrs['keras_version'].decode('utf-8') if isinstance(h5file.attrs['keras_version'], bytes) else h5file.attrs['keras_version']
                logger.info(f"Model was saved with Keras version: {keras_version}")
            
            # Look for layer configuration to understand model structure
            if 'model_config' in h5file.attrs:
                try:
                    config_str = h5file.attrs['model_config']
                    if isinstance(config_str, bytes):
                        config_str = config_str.decode('utf-8')
                    config = json.loads(config_str)
                    
                    # Check for BatchNormalization layers with problematic axis
                    if 'layers' in config.get('config', {}):
                        logger.info(f"Model has {len(config['config']['layers'])} layers")
                        bn_layers = [l for l in config['config']['layers'] 
                                    if l.get('class_name') == 'BatchNormalization']
                        logger.info(f"Found {len(bn_layers)} BatchNormalization layers")
                        for l in bn_layers[:2]:  # Log just a few as example
                            if 'axis' in l.get('config', {}):
                                logger.info(f"BatchNorm axis example: {l['config']['axis']}")
                except Exception as e:
                    logger.warning(f"Failed to parse model config: {e}")
        
        # Load model with custom objects and additional options for compatibility
        full = tf.keras.models.load_model(
            backbone_path, 
            compile=False, 
            custom_objects=custom_objects,
            options=tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
        )
        
        # Convert model to functional if it's Sequential
        if not isinstance(full, tf.keras.models.Model):
            logger.warning("Converting non-Model to functional model")
            inputs = tf.keras.Input(batch_shape=full.layers[0].input_shape)
            outputs = full(inputs)
            full = tf.keras.models.Model(inputs, outputs)
        
        logger.info(f"Backbone loaded successfully in {time.time() - start_time:.2f} seconds")
        log_memory_usage("After backbone load: ")
        
        logger.info(f"Backbone model summary: {len(full.layers)} layers")
        logger.info(f"Input shape: {full.input_shape}, Output shape: {full.output_shape}")
        
        # Log the last few layers
        for i, layer in enumerate(full.layers[-5:]):
            logger.info(f"Layer {len(full.layers)-5+i}: {layer.name}, type: {type(layer).__name__}, output shape: {layer.output_shape}")
        
        # Find the embedding layer (penultimate layer)
        embedding_layer = None
        for i in range(len(full.layers) - 2, 0, -1):
            layer = full.layers[i]
            # Look for the Dense layer that comes before the final activation
            if isinstance(layer, tf.keras.layers.Dense) or layer.name == "GLOBAL_AVG_POOL":
                embedding_layer = layer
                break
        
        if embedding_layer is None:
            logger.warning("Could not find a suitable embedding layer, defaulting to second-to-last layer")
            embedding_layer = full.layers[-2]
        
        penult = embedding_layer.output
        logger.info(f"Using layer as embedding: {embedding_layer.name} with shape {embedding_layer.output_shape}")
        
        # Create a new model that outputs embeddings
        embed_model = tf.keras.Model(full.input, penult, name="birdnet_embed")
        
        # Cleanup to free memory
        del full
        gc.collect()
        
        return embed_model
    except Exception as e:
        logger.error(f"Error loading backbone model: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone",   required=True, type=pathlib.Path)
    p.add_argument("--head_model", required=True, type=pathlib.Path)
    p.add_argument("--out_file",   required=True, type=pathlib.Path)
    p.add_argument("--head_dim",   choices=[512, 1024], type=int, default=1024,
                   help="Embedding size that your head expects")
    p.add_argument("--quant", choices=["dynamic", "float16", "none"],
                   default="dynamic", help="Quantization method for TFLite model")
    p.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    p.add_argument("--memory_limit", type=int, default=0, 
                   help="Limit GPU memory in MB (0 for dynamic growth)")
    p.add_argument("--skip_verification", action="store_true",
                   help="Skip TFLite model verification step")
    args = p.parse_args()
    
    # Configure memory limits if specified
    if args.memory_limit > 0:
        logger.info(f"Setting GPU memory limit to {args.memory_limit}MB")
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.set_logical_device_configuration(
                    gpus[0],
                    [tf.config.LogicalDeviceConfiguration(memory_limit=args.memory_limit)]
                )
                logger.info("GPU memory limit set successfully")
            except Exception as e:
                logger.warning(f"Could not set GPU memory limit: {str(e)}")
    
    # Setup logging
    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)
    logger.info(f"Starting model merge with arguments: {args}")
    log_memory_usage("Initial: ")

    # 1 ─ backbone to 1024-vector
    logger.info("Creating input layer")
    wav_in  = tf.keras.Input((SAMPLES, 1), dtype=tf.float32, name="audio")
    logger.info(f"Input shape: {wav_in.shape}")
    
    logger.info("Loading and connecting backbone model")
    start_time = time.time()
    embed1024 = load_backbone(args.backbone)(wav_in, training=False)  # (None,1024)
    logger.info(f"Backbone connected in {time.time() - start_time:.2f} seconds")
    logger.info(f"Backbone output shape: {embed1024.shape}")
    log_memory_usage("After backbone connection: ")
    
    # Force garbage collection
    gc.collect()
    log_memory_usage("After GC: ")

    # 2 ─ adapt to head dim (if head is 512-D, keep first half)
    logger.info(f"Adapting to head dimension: {args.head_dim}")
    emb_in = embed1024[:, :args.head_dim]
    logger.info(f"Embedding input shape: {emb_in.shape}")

    # 3 ─ load head
    logger.info(f"Loading head model from {args.head_model}")
    start_time = time.time()
    try:
        # Cleanup before loading head model
        gc.collect()
        
        # Load head model with same custom objects for consistency
        head = tf.keras.models.load_model(
            args.head_model, 
            compile=False,
            custom_objects=custom_objects,
            options=tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
        )
        logger.info(f"Head loaded successfully in {time.time() - start_time:.2f} seconds")
        logger.info(f"Head model summary: {len(head.layers)} layers")
        logger.info(f"Head input shape: {head.input_shape}, output shape: {head.output_shape}")
        
        # Log head model layers
        for i, layer in enumerate(head.layers):
            logger.info(f"Head layer {i}: {layer.name}, type: {type(layer).__name__}, output shape: {layer.output_shape}")
        
        # Check input shape compatibility
        if head.input_shape[1:] != emb_in.shape[1:]:
            logger.warning(f"Head input shape {head.input_shape[1:]} doesn't match embedding shape {emb_in.shape[1:]}")
            logger.info("Attempting to adapt shapes...")
            
            # Try to adapt the shape if needed
            if head.input_shape[1] == emb_in.shape[1]:
                logger.info("Dimensions match, continuing with connection")
            elif args.head_dim != embed1024.shape[1]:
                # If head_dim doesn't match the actual embedding size, adjust it
                logger.warning(f"Adjusting head_dim from {args.head_dim} to {embed1024.shape[1]}")
                emb_in = embed1024[:, :head.input_shape[1]]
                logger.info(f"Adjusted embedding shape: {emb_in.shape}")
            else:
                logger.warning(f"Mismatch in dimensions that cannot be automatically fixed. Expected {head.input_shape[1]}, got {emb_in.shape[1]}")
        
        # Connect the head to the embedding
        logits = head(emb_in)
        logger.info(f"Output logits shape: {logits.shape}")
        
    except Exception as e:
        logger.error(f"Error loading head model: {str(e)}")
        logger.error(traceback.format_exc())
        raise

    logger.info("Creating full model")
    full_model = tf.keras.Model(wav_in, logits, name="birdnet_full")
    logger.info(f"Full model created with {len(full_model.layers)} layers")
    log_memory_usage("After full model creation: ")

    # 4 ─ TFLite export
    logger.info("Converting model to TFLite format")
    logger.info(f"Quantization mode: {args.quant}")
    start_time = time.time()
    
    try:
        # Run garbage collection before conversion to free memory
        gc.collect()
        
        # Create converter with experimental flags for better memory usage
        conv = tf.lite.TFLiteConverter.from_keras_model(full_model)
        
        # Set memory optimization flags
        conv.allow_custom_ops = True
        conv.experimental_new_converter = True
        
        # Configure quantization based on user choice
        if args.quant == "dynamic":
            logger.info("Using dynamic quantization")
            conv.optimizations = [tf.lite.Optimize.DEFAULT]
        elif args.quant == "float16":
            logger.info("Using float16 quantization")
            conv.optimizations = [tf.lite.Optimize.DEFAULT]
            conv.target_spec.supported_types = [tf.float16]
            # Use reduced precision operations where possible
            conv.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS
            ]
        else:
            logger.info("No quantization applied")
            
        # Set other optimization flags
        if hasattr(tf.lite, 'Optimize'):
            conv.experimental_enable_resource_variables = True
        
        logger.info("Starting TFLite conversion...")
        try:
            tflite_bytes = conv.convert()
            logger.info(f"TFLite conversion completed in {time.time() - start_time:.2f} seconds")
            logger.info(f"TFLite model size: {len(tflite_bytes) / (1024 * 1024):.2f} MB")
            
            logger.info(f"Writing TFLite model to {args.out_file}")
            # Create parent directory if it doesn't exist
            if not args.out_file.parent.exists():
                args.out_file.parent.mkdir(parents=True, exist_ok=True)
                
            args.out_file.write_bytes(tflite_bytes)
            logger.info(f"TFLite model written successfully")
        except Exception as conv_error:
            logger.error(f"TFLite conversion failed: {str(conv_error)}")
            
            # Try an alternative approach with reduced optimizations
            logger.info("Attempting conversion with reduced optimizations...")
            conv = tf.lite.TFLiteConverter.from_keras_model(full_model)
            conv.optimizations = []  # Disable optimizations
            conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
            
            tflite_bytes = conv.convert()
            logger.info(f"Alternative TFLite conversion completed in {time.time() - start_time:.2f} seconds")
            logger.info(f"TFLite model size: {len(tflite_bytes) / (1024 * 1024):.2f} MB")
            
            args.out_file.write_bytes(tflite_bytes)
            logger.info(f"TFLite model written successfully with reduced optimizations")
            
    except Exception as e:
        logger.error(f"Error during TFLite conversion: {str(e)}")
        logger.error(traceback.format_exc())
        raise

    # verify
    logger.info("Verifying TFLite model")
    try:
        interp = tf.lite.Interpreter(model_path=str(args.out_file))
        interp.allocate_tensors()
        
        input_details = interp.get_input_details()
        output_details = interp.get_output_details()
        
        logger.info(f"TFLite model verified successfully")
        logger.info(f"✓ wrote {args.out_file}")
        logger.info(f"  INPUT : {input_details[0]['shape']}")
        logger.info(f"  OUTPUT: {output_details[0]['shape']}")
        
        print("✓ wrote", args.out_file)
        print("  INPUT :", input_details[0]["shape"])
        print("  OUTPUT:", output_details[0]["shape"])
    except Exception as e:
        logger.error(f"Error verifying TFLite model: {str(e)}")
        raise

if __name__ == "__main__":
    main()
