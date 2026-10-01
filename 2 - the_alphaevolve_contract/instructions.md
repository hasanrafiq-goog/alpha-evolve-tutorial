# Problem: Neural Architecture Search for MNIST Classification

You are tasked with discovering and evolving high-performing Neural Network Architectures for MNIST digit classification using TensorFlow / Keras.

The goal is to evolve the `build_model()` function inside `program.py` to achieve the highest possible validation accuracy (`val_accuracy`), while maintaining computational efficiency and training stability.

## Target & Constraints

1. **Input Shape**: The model receives grayscale image batches with shape `(batch_size, 28, 28, 1)` and pixel values normalized to `[0.0, 1.0]`.
2. **Output Shape**: The model must output classification probabilities with shape `(batch_size, 10)` corresponding to digits `0-9`.
3. **Function Signature**:
   ```python
   def build_model() -> tf.keras.Model:
       """Constructs and returns a tf.keras.Model instance."""
   ```
4. **Primary Metric**: `val_accuracy` (higher is better).
5. **Secondary Metrics**: `param_count` (lower/efficient is better), `val_loss` (lower is better).

## Ideas for Evolution & Exploration

Feel free to explore modern deep learning architectural primitives, including:
- **Topology**: Residual / skip connections, dense connections, multi-branch feature aggregators.
- **Convolutions**: Depthwise-separable convolutions (`tf.keras.layers.SeparableConv2D`), dilated convolutions, multi-scale kernel filters (e.g. 3x3, 5x5).
- **Normalization & Regularization**: Batch Normalization (`tf.keras.layers.BatchNormalization`), Spatial Dropout, Layer Normalization.
- **Activations**: Advanced activation functions like GELU, Swish/SiLU (`tf.nn.silu` or `activation="swish"`), LeakyReLU, or Mish.
- **Attention & Pooling**: Squeeze-and-Excitation channel attention blocks, Global Average Pooling vs Flattening.

## Rules
- Do NOT hardcode training epochs, optimizers, or loss functions inside `build_model()`; only define the architecture topology.
- Keep the total parameter count reasonable (under 500,000 parameters) so it trains fast and avoids overfitting.
- The returned object MUST be an instantiated `tf.keras.Model` or `tf.keras.Sequential`.
