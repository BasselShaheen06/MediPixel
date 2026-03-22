"""
Noise generation for medical images.

All functions accept and return numpy uint8 arrays.
They are pure functions — no side effects, no UI state.
"""

import numpy as np


def add_gaussian(image: np.ndarray, sigma: float = 25.0) -> np.ndarray:
    """
    Add zero-mean Gaussian noise to an image.

    Args:
        image: 2-D uint8 numpy array.
        sigma: Standard deviation of the noise (0–255 scale). Higher = noisier.

    Returns:
        Noisy image clipped to uint8 [0, 255].
    """
    noise = np.random.normal(0, sigma, image.shape)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def add_salt_and_pepper(image: np.ndarray, prob: float = 0.05) -> np.ndarray:
    """
    Add salt-and-pepper (impulse) noise.

    Args:
        image: 2-D uint8 numpy array.
        prob:  Total corrupted-pixel fraction. Half become 255 (salt),
               half become 0 (pepper).

    Returns:
        Noisy image as uint8.
    """
    out = image.copy()
    rng = np.random.random(image.shape)
    out[rng < prob / 2] = 255       # salt
    out[rng > 1 - prob / 2] = 0     # pepper
    return out


def add_poisson(image: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """
    Add Poisson (shot) noise — models photon-counting noise in imaging.

    Args:
        image: 2-D uint8 numpy array.
        scale: Scales the lambda of the Poisson distribution.
               Higher scale = less noise (more photons).

    Returns:
        Noisy image clipped to uint8 [0, 255].
    """
    scaled = image.astype(np.float32) * scale
    noisy = np.random.poisson(np.maximum(scaled, 0)).astype(np.float32) / scale
    return np.clip(noisy, 0, 255).astype(np.uint8)
