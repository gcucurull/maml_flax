from typing import Tuple, Optional

import numpy as np


def generate_sinusoids(
    batch_size: int,
    num_points: int,
    amp: Optional[float] = None,
    phase: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return data generated from sinusoids with a random amplitude and phase
    y = A * sin(x + phase)
    """
    if amp is not None:
        assert phase is not None
    else:
        amp = np.random.uniform(0.1, 5.0, size=(batch_size, 1, 1))
        phase = np.random.uniform(0, np.pi, size=(batch_size, 1, 1))

    samples_x = np.random.uniform(-5, 5, size=(batch_size, num_points, 1))
    samples_y = amp * np.sin(samples_x + phase)

    return samples_x, samples_y, amp, phase


def generate_sin_tasks(
    batch_size: int, num_points: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x, y, _, _ = generate_sinusoids(batch_size, num_points * 2)
    meta_train_x, meta_val_x = np.split(x, 2, axis=1)
    meta_train_y, meta_val_y = np.split(y, 2, axis=1)
    return meta_train_x, meta_train_y, meta_val_x, meta_val_y
