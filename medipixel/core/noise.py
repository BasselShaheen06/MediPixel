"""
Noise generation for medical images.

All functions accept numpy arrays (uint8) and return uint8 arrays.
Supports both grayscale (H, W) and color (H, W, 3) images.
Pure functions — no side effects, no UI state.
"""

import numpy as np


def _is_color(image: np.ndarray) -> bool:
    return image.ndim == 3 and image.shape[2] == 3


def add_gaussian(image: np.ndarray, sigma: float = 25.0) -> np.ndarray:
    """
    Add zero-mean Gaussian noise.

    Args:
        image: uint8 array, shape (H, W) or (H, W, 3).
        sigma: Standard deviation on the 0-255 scale.

    Returns:
        Noisy image clipped to uint8 [0, 255].
    """
    noise = np.random.normal(0, sigma, image.shape)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def add_salt_and_pepper(image: np.ndarray, prob: float = 0.05) -> np.ndarray:
    """
    Add salt-and-pepper (impulse) noise.

    For color images, each corrupted pixel has ALL channels set to 255 or 0
    simultaneously — this models a dead pixel, not channel-level corruption.

    Args:
        image: uint8 array, shape (H, W) or (H, W, 3).
        prob:  Total corrupted-pixel fraction.

    Returns:
        Noisy image as uint8.
    """
    out = image.copy()
    h, w = image.shape[:2]
    rng = np.random.random((h, w))

    salt_mask   = rng < prob / 2
    pepper_mask = rng > 1 - prob / 2

    if _is_color(image):
        out[salt_mask]   = 255
        out[pepper_mask] = 0
    else:
        out[salt_mask]   = 255
        out[pepper_mask] = 0

    return out


def add_poisson(image: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """
    Add Poisson (shot) noise — signal-dependent, models photon counting.

    Args:
        image: uint8 array, shape (H, W) or (H, W, 3).
        scale: Dose factor. Higher = fewer photons = more noise.

    Returns:
        Noisy image clipped to uint8 [0, 255].
    """
    scaled = image.astype(np.float32) * scale
    noisy  = np.random.poisson(np.maximum(scaled, 0)).astype(np.float32) / scale
    return np.clip(noisy, 0, 255).astype(np.uint8)