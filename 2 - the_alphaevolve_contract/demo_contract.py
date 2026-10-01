#!/usr/bin/env python3
"""Interactive Demonstration of the 3 AlphaEvolve Pillars:
1. System Prompt (instructions.md)
2. Mutation Boundary (# EVOLVE-BLOCK)
3. Insight-driven Error Feedback Loop
"""

import os
import sys
import traceback
import numpy as np
import tensorflow as tf

# Styling
CYAN = "\033[0;36m"
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"


def demonstrate_pillar_1():
    print(f"\n{CYAN}{BOLD}--- PILLAR 1: The System Prompt (instructions.md) ---{RESET}")
    instructions_path = os.path.join(os.path.dirname(__file__), "instructions.md")
    with open(instructions_path, "r") as f:
        lines = f.readlines()
    print("Gemini receives these exact constraints and architectural ideas:")
    for line in lines[8:28]:
        print(f"  {line.rstrip()}")


def demonstrate_pillar_2():
    print(f"\n{CYAN}{BOLD}--- PILLAR 2: The Mutation Boundary (# EVOLVE-BLOCK) ---{RESET}")
    print("AlphaEvolve uses comment markers to restrict where the LLM can write code:")
    print(f"""
  {YELLOW}# Code OUTSIDE the block is IMMUTABLE (imports, helpers, evaluation harness):{RESET}
  import tensorflow as tf

  {GREEN}{BOLD}# EVOLVE-BLOCK-START{RESET}
  {GREEN}# Gemini is ONLY allowed to mutate code inside this block!{RESET}
  def build_model() -> tf.keras.Model:
      inputs = tf.keras.Input(shape=(28, 28, 1))
      ...
      return model
  {GREEN}{BOLD}# EVOLVE-BLOCK-END{RESET}
    """)


def demonstrate_pillar_3():
    print(f"\n{CYAN}{BOLD}--- PILLAR 3: The Insight-Driven Error Feedback Loop ---{RESET}")
    print("What happens when Gemini proposes a broken architecture?")
    print("In traditional NAS, a crash gives score = 0.0, wasting the trial.")
    print("In AlphaEvolve, the crash traceback is captured and sent back to Gemini as an 'insight'!\n")

    # Candidate A: Valid Mutation (Adds BatchNorm + Dropout)
    print(f"{BOLD}[Candidate A] Gemini proposes: BatchNorm + Dropout addition{RESET}")
    code_a = """
def build_model() -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(28, 28, 1))
    x = tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(10, activation="softmax")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)
"""
    try:
        ns = {"tf": tf}
        exec(code_a, ns)
        model = ns["build_model"]()
        out = model(np.zeros((1, 28, 28, 1)), training=False)
        assert out.shape == (1, 10)
        print(f"  Result: {GREEN}[✓] Compiled and shape-verified successfully!{RESET}\n")
    except Exception as e:
        print(f"  Result: {RED}[✗] Failed: {e}{RESET}\n")

    # Candidate B: Broken Dimension Mismatch (Attempting invalid skip-connection)
    print(f"{BOLD}[Candidate B] Gemini attempts an invalid skip-connection (mismatched spatial shapes){RESET}")
    code_b = """
def build_model() -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(28, 28, 1))
    x1 = tf.keras.layers.Conv2D(16, (3, 3), padding="same")(inputs)  # shape: (28, 28, 16)
    x2 = tf.keras.layers.MaxPooling2D((2, 2))(x1)                     # shape: (14, 14, 16)
    # BUG: Trying to add 28x28 tensor to 14x14 tensor!
    skip = tf.keras.layers.Add()([x1, x2])
    x = tf.keras.layers.Flatten()(skip)
    outputs = tf.keras.layers.Dense(10, activation="softmax")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)
"""
    try:
        ns = {"tf": tf}
        exec(code_b, ns)
        model = ns["build_model"]()
        model(np.zeros((1, 28, 28, 1)), training=False)
    except Exception as e:
        err_msg = traceback.format_exc().strip().split("\n")[-1]
        print(f"  Caught Exception: {RED}{err_msg}{RESET}")
        print(f"  {YELLOW}Formatted Insight sent back to Gemini:{RESET}")
        insight_payload = {
            "insights": [
                {
                    "label": "architecture_error",
                    "text": f"Shape dimension mismatch during forward pass: {err_msg}. Ensure skip-connection tensors have identical spatial dimensions."
                }
            ]
        }
        print(f"  Payload: {insight_payload}")
        print(f"  {GREEN}--> Gemini reads this error and automatically adds a 1x1 stride-2 conv to fix the dimensions on the next turn!{RESET}\n")

    # Candidate C: Output shape mismatch (outputs 5 classes instead of 10)
    print(f"{BOLD}[Candidate C] Gemini outputs 5 classes instead of 10 (SLA Violation){RESET}")
    code_c = """
def build_model() -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(28, 28, 1))
    x = tf.keras.layers.Flatten()(inputs)
    outputs = tf.keras.layers.Dense(5, activation="softmax")(x) # BUG: 5 classes instead of 10!
    return tf.keras.Model(inputs=inputs, outputs=outputs)
"""
    try:
        ns = {"tf": tf}
        exec(code_c, ns)
        model = ns["build_model"]()
        out = model(np.zeros((1, 28, 28, 1)), training=False)
        if out.shape[-1] != 10:
            raise ValueError(f"Output shape has {out.shape[-1]} classes, but MNIST requires exactly 10 classes.")
    except Exception as e:
        print(f"  Caught Assertion: {RED}{e}{RESET}")
        print(f"  {YELLOW}Formatted Insight sent back to Gemini:{RESET}")
        print(f"  Payload: {{'label': 'shape_assertion', 'text': '{e}'}}")
        print(f"  {GREEN}--> Gemini corrects the output layer back to Dense(10, activation='softmax').{RESET}\n")


def main():
    print("=" * 75)
    print(f"{BOLD}Step 2: The AlphaEvolve Contract (Prompt, Code Block, & Insights){RESET}")
    print("=" * 75)

    demonstrate_pillar_1()
    demonstrate_pillar_2()
    demonstrate_pillar_3()

    print("=" * 75)
    print(f"{GREEN}{BOLD}✨ The 3 Pillars of AlphaEvolve verified!{RESET}")
    print(f"Next: See how the Local Evaluator trains architectures in Step 3:")
    print(f"{BOLD}cd '../3 - local_evaluation_sandbox' && ./run.sh{RESET}")
    print("=" * 75)


if __name__ == "__main__":
    main()
