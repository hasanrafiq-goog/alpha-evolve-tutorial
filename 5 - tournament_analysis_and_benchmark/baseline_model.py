"""Seed model program for AlphaEvolve MNIST Architecture Search."""

import tensorflow as tf

# EVOLVE-BLOCK-START
"""Candidate model definition. AlphaEvolve mutates and evolves this function."""


def build_model() -> tf.keras.Model:
    """Builds a naive baseline CNN for MNIST classification (28x28x1 -> 10 classes).

    Returns:
        tf.keras.Model: Model instance with input shape (28, 28, 1) and output shape (10,).
    """
    inputs = tf.keras.Input(shape=(28, 28, 1), name="image_input")

    # A single, rudimentary convolution (only 8 filters) with ReLU
    x = tf.keras.layers.Conv2D(
        filters=8,
        kernel_size=(3, 3),
        activation="relu",
        name="conv1",
    )(inputs)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)

    # Simple classification head (only 16 neurons, no batchnorm, no dropout)
    x = tf.keras.layers.Flatten(name="flatten")(x)
    x = tf.keras.layers.Dense(16, activation="relu", name="dense1")(x)
    outputs = tf.keras.layers.Dense(
        10, activation="softmax", name="classification_output"
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="mnist_naive_baseline")
    return model
# EVOLVE-BLOCK-END
