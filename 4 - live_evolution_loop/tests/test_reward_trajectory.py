"""Unit tests verifying reward trajectory mapping and milestone dot calculation."""

import pytest


def test_milestone_dot_calculation():
    """Verify that only candidates establishing a new high-water mark receive green milestone dots."""
    candidates = [
        {"run_number": 1, "nas_reward": 0.35},  # New high-water mark -> Green
        {"run_number": 2, "nas_reward": 0.20},  # Lower score -> Blue
        {"run_number": 3, "nas_reward": 0.55},  # New high-water mark -> Green
        {"run_number": 4, "nas_reward": 0.40},  # Lower score -> Blue
        {"run_number": 5, "nas_reward": 0.58},  # New high-water mark -> Green
    ]

    running_max = -1.0
    point_colors = []
    exact_rewards = []

    for c in candidates:
        rew = c["nas_reward"]
        exact_rewards.append(rew)
        if rew > running_max:
            running_max = rew
            point_colors.append("green")
        else:
            point_colors.append("blue")

    assert exact_rewards == [0.35, 0.20, 0.55, 0.40, 0.58]
    assert point_colors == ["green", "blue", "green", "blue", "green"]
