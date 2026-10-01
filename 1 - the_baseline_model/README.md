# Step 1: Demystifying the Baseline Model (The Seed Recipe)

Welcome to **Step 1**! In this step, we will examine the seed program (`program.py`) that AlphaEvolve begins with.

Even if you have **zero background in machine learning**, this guide will explain the architecture in plain English.

---

## 🎯 Learning Objectives
1. Understand the MNIST problem: 28x28 handwritten digit recognition.
2. Demystify Neural Networks using the **Lego Blocks** analogy.
3. Inspect `program.py` and identify the `# EVOLVE-BLOCK` mutation boundary.
4. Run `inspect_baseline.py` to see the layer summary and a real handwritten digit rendered directly in your terminal.

---

## 🧩 The Non-AI Intuition: Lego Blocks

Think of a Convolutional Neural Network (CNN) as assembling Lego blocks:

```
  [ 🖼️ 28x28 Image ]
          │
          ▼
   [ 🧱 Conv2D (8) ] ──> Rudimentary filter: Only 8 tiny feature detectors!
          │
          ▼
   [ 🧱 MaxPool ]    ──> Compressor: Shrinks image to 13x13
          │
          ▼
   [ 🧱 Flatten ]    ──> Unrolls the 2D grid into a 1D list of 1,352 numbers
          │
          ▼
   [ 🧱 Dense (16) ] ──> Tiny bottleneck: Only 16 decision neurons (no dropout, no batchnorm)
          │
          ▼
   [ 🧱 Output (10) ]──> The voting committee: Outputs probability for digits 0 to 9
```

* **Why is this baseline intentionally weak?**
  It only uses 1 small convolution layer (8 filters) and a cramped 16-neuron dense layer. In 2 proxy epochs it only scores **~87%**, leaving massive headroom for AlphaEvolve to discover deeper convolutions, pooling, normalization, and modern activations!

---

## 📜 The Code: `program.py`

Open `program.py`. Notice these special comment tags:

```python
# EVOLVE-BLOCK-START
def build_model() -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(28, 28, 1), name="image_input")
    ...
    return model
# EVOLVE-BLOCK-END
```

* Everything **between** `# EVOLVE-BLOCK-START` and `# EVOLVE-BLOCK-END` is the code that **Gemini will mutate**.
* Everything **outside** is immutable.

---

## 🚀 How to Run Step 1

Execute the single runner script:
```bash
./run.sh
```

### What happens when you run `./run.sh`:
1. Activates your virtual environment.
2. Loads a real handwritten digit from MNIST.
3. Renders the digit in your terminal using ASCII characters (`@`, `#`, `.`).
4. Prints the full Keras `model.summary()` showing layer names and parameter counts (295,930 parameters).
5. Runs a forward pass on the untrained network showing raw probability predictions across digits 0–9.

---

## 🎤 Presenter Speaking Notes (For PSO Trainers)
* *"Notice how small the code inside `build_model()` is. It's just 25 lines of Python."*
* *"In traditional NAS, you had to define complex search spaces with JSON/YAML schemas like `min_filters: 16, max_filters: 64`. With AlphaEvolve, the search space is the Python language itself!"*
* *"Gemini can replace this entire function with stacked convolutions, batch normalization, skip connections, or Swish activations."*
