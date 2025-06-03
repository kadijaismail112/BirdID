#!/usr/bin/env python3
"""
inspect_model.py
---------------
Examines the structure of a Keras h5 model file without fully loading it.
Specifically looks for DepthwiseConv2D layers and their configurations.
"""

import h5py
import json
import argparse
import sys
import gc
import os
import traceback
from pathlib import Path

def print_attrs(name, obj):
    """Print attributes of an h5py object"""
    print(f"\nAttributes for {name}:")
    for key, val in obj.attrs.items():
        if key == "model_config" and isinstance(val, bytes):
            # Pretty print model config as JSON
            try:
                config = json.loads(val.decode('utf-8'))
                print(f"  {key}: <model config - see detailed output below>")
                return config
            except json.JSONDecodeError:
                print(f"  {key}: <invalid JSON>")
        else:
            print(f"  {key}: {val}")
    return None

def analyze_layer_configs(config):
    """Analyze layer configurations in the model config"""
    if not config or 'config' not in config:
        return
    
    layers = config.get('config', {}).get('layers', [])
    print(f"\nFound {len(layers)} layers in model config")
    
    depthwise_layers = []
    
    # First pass - collect basic info about all layers
    print("\nLayer Summary:")
    for i, layer in enumerate(layers):
        layer_class = layer.get('class_name', 'Unknown')
        layer_name = layer.get('config', {}).get('name', 'unnamed')
        print(f"  {i+1}. {layer_class}: {layer_name}")
        
        # Track DepthwiseConv2D layers for detailed analysis
        if layer_class == 'DepthwiseConv2D':
            depthwise_layers.append((i, layer))
    
    # Second pass - detailed analysis of DepthwiseConv2D layers
    if depthwise_layers:
        print(f"\nFound {len(depthwise_layers)} DepthwiseConv2D layers:")
        for idx, layer in depthwise_layers:
            layer_config = layer.get('config', {})
            print(f"\n  Layer {idx+1}: {layer_config.get('name')}")
            print("  Configuration:")
            for key, value in layer_config.items():
                print(f"    {key}: {value}")
            
            # Highlight problematic 'groups' parameter
            if 'groups' in layer_config:
                print(f"    *** ISSUE FOUND: 'groups' parameter present with value {layer_config['groups']} ***")
    else:
        print("\nNo DepthwiseConv2D layers found in the model")

def explore_group(group, prefix=''):
    """Recursively explore an h5py group"""
    for key in group.keys():
        item = group[key]
        path = f"{prefix}/{key}" if prefix else key
        
        if isinstance(item, h5py.Group):
            print(f"Group: {path}")
            explore_group(item, path)
        elif isinstance(item, h5py.Dataset):
            shape_info = f"shape={item.shape}" if hasattr(item, 'shape') else ""
            dtype_info = f"dtype={item.dtype}" if hasattr(item, 'dtype') else ""
            print(f"Dataset: {path} {shape_info} {dtype_info}")

def inspect_model(model_path):
    """Inspect a Keras h5 model file"""
    try:
        print(f"Opening model file: {model_path}")
        with h5py.File(model_path, 'r') as f:
            # Print basic file info
            print(f"File type: {type(f)}")
            print(f"Keys: {list(f.keys())}")
            
            # Explore the file structure
            print("\n--- File Structure ---")
            explore_group(f)
            
            # Look for model config
            print("\n--- Model Configuration ---")
            model_config = None
            
            # First look in standard locations
            if 'model_weights' in f and 'model_config' in f.attrs:
                print("Found model_config in root attributes")
                model_config = print_attrs('/', f)
            
            # Analyze layer configurations if model_config found
            if model_config:
                analyze_layer_configs(model_config)
            else:
                print("Could not find model configuration in expected locations")
            
            # Cleanup
            gc.collect()
            
    except Exception as e:
        print(f"Error inspecting model: {str(e)}")
        traceback.print_exc()
        return 1
    
    return 0

def main():
    parser = argparse.ArgumentParser(description='Inspect Keras h5 model file structure')
    parser.add_argument('model_path', type=Path, help='Path to the Keras h5 model file')
    args = parser.parse_args()
    
    if not args.model_path.exists():
        print(f"Error: Model file {args.model_path} does not exist")
        return 1
    
    return inspect_model(args.model_path)

if __name__ == '__main__':
    sys.exit(main())

