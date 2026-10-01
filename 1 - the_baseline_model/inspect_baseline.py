#!/usr/bin/env python3
"""Inspect Baseline Seed Model and Visualize Tensor Flow."""

import os
import sys
import numpy as np
import tensorflow as tf

from program import build_model

# Terminal styling
CYAN = "\033[0;36m"
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
BOLD = "\033[1m"
RESET = "\033[0m"


def print_ascii_digit(image_2d):
    """Render a 28x28 grayscale image as ASCII art in terminal."""
    chars = " .:-=+*#%@"
    print("\n" + f"{CYAN}--- Sample 28x28 Handwritten Digit (Input Image) ---{RESET}")
    for row in image_2d:
        line = "".join(chars[int(val * (len(chars) - 1))] for val in row)
        print(f"  {line}")


def main():
    print("=" * 75)
    print(f"{BOLD}Step 1: Inspecting the Baseline Seed Model (program.py){RESET}")
    print("=" * 75)

    print(f"\n{BOLD}1. The Mission: MNIST Handwritten Digit Classification{RESET}")
    print("   Input  : A 28x28 grayscale image (pixel values from 0.0 to 1.0)")
    print("   Output : A probability vector of length 10 (digits 0 through 9)")

    # Load MNIST sample
    print("   Loading MNIST dataset (one-time download on first run)...")
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    sample_idx = 0
    sample_image = x_train[sample_idx] / 255.0
    sample_label = y_train[sample_idx]

    print_ascii_digit(sample_image)
    print(f"  True Digit Label: {GREEN}{BOLD}{sample_label}{RESET}")

    print(f"\n{BOLD}2. Instantiating the Baseline Architecture (Lego Blocks){RESET}")
    tf.keras.backend.clear_session()
    model = build_model()

    print(f"\n{BOLD}3. Model Architecture Summary:{RESET}")
    model.summary()

    param_count = model.count_params()
    print(f"\nTotal Trainable Parameters: {YELLOW}{BOLD}{param_count:,}{RESET}")

    print(f"\n{BOLD}4. Tensor Flow & Dimensions Walkthrough:{RESET}")
    print(f"  [Input]       Shape: (batch, 28, 28, 1)  -> Raw pixel grid")
    print(f"  [Conv2D 1]    Shape: (batch, 26, 26, 8)  -> Only 8 filters (rudimentary feature detection)")
    print(f"  [MaxPool 1]   Shape: (batch, 13, 13, 8)  -> Downsampled by 2x")
    print(f"  [Flatten]     Shape: (batch, 1352)       -> Unrolled into a 1D vector (13*13*8)")
    print(f"  [Dense 1]     Shape: (batch, 16)         -> Only 16 decision neurons (very narrow!)")
    print(f"  [Output]      Shape: (batch, 10)         -> 10 class probabilities (Softmax)")

    print(f"\n{BOLD}5. Testing Forward Pass on Untrained Model:{RESET}")
    input_tensor = np.expand_dims(sample_image, axis=(0, -1)).astype(np.float32)
    predictions = model(input_tensor, training=False).numpy()[0]

    print("  Output probabilities across classes [0..9]:")
    for digit, prob in enumerate(predictions):
        bar = "█" * int(prob * 40)
        print(f"    Digit {digit}: {prob*100:5.1f}% | {bar}")

    print("\n" + "=" * 75)
    print(f"{GREEN}{BOLD}✨ Baseline Model verified!{RESET}")
    print(f"Next: Learn how AlphaEvolve's contract works in Step 2:")
    print(f"{BOLD}cd '../2 - the_alphaevolve_contract' && ./run.sh{RESET}")
    print("=" * 75)


if __name__ == "__main__":
    main()
