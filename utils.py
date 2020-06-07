import matplotlib.pyplot as plt
import numpy as np


def plot_sinusoid(
    x: np.ndarray, y: np.ndarray, amp: float = 1, phase: float = 1
) -> None:
    """
    Plot the predicted sinusoid (x, y) on top of the ground truth sinusoid
    defined by its amplitude and phase.
    """
    time = np.arange(-5, 5, 0.1)
    res = amp * np.sin(time + phase)

    plt.plot(time, res)
    plt.scatter(x, y)
    plt.show()
