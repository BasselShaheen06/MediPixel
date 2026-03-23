"""
Spatial denoising filters for medical images.

All functions accept numpy arrays (uint8) and return uint8 arrays.
Supports both grayscale (H, W) and color (H, W, 3) images.
Pure functions — no side effects, no UI state.
"""

import cv2
import numpy as np


def _is_color(image: np.ndarray) -> bool:
    return image.ndim == 3 and image.shape[2] == 3


def median(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Median filter — effective against salt-and-pepper noise.
    Preserves edges better than Gaussian blur.

    OpenCV medianBlur works on both grayscale and color natively.

    Args:
        image:       uint8 array, shape (H, W) or (H, W, 3).
        kernel_size: Must be odd (3, 5, 7…).

    Returns:
        Filtered image as uint8.
    """
    return cv2.medianBlur(image, kernel_size)


def bilateral(
    image: np.ndarray,
    d: int = 9,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0,
) -> np.ndarray:
    """
    Bilateral filter — edge-preserving smoothing.

    OpenCV bilateralFilter works on both grayscale and color natively.
    For color images, filtering is done in the BGR color space.

    Args:
        image:       uint8 array, shape (H, W) or (H, W, 3).
        d:           Diameter of the pixel neighbourhood.
        sigma_color: Intensity range — large = blur across bigger differences.
        sigma_space: Spatial range — large = consider farther pixels.

    Returns:
        Filtered image as uint8.
    """
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def non_local_means(
    image: np.ndarray,
    h: float = 10.0,
    template_window: int = 7,
    search_window: int = 21,
) -> np.ndarray:
    """
    Non-Local Means denoising.

    For grayscale: uses fastNlMeansDenoising.
    For color: uses fastNlMeansDenoisingColored which filters luminance
    and chrominance separately in LAB space — this avoids color bleeding
    artefacts that occur when filtering RGB channels independently.

    Args:
        image:           uint8 array, shape (H, W) or (H, W, 3).
        h:               Filter strength for luminance. Match to noise sigma.
        template_window: Patch size for comparison (odd, pixels).
        search_window:   Search area size (odd, pixels).

    Returns:
        Denoised image as uint8.
    """
    if _is_color(image):
        # fastNlMeansDenoisingColored expects BGR
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        result = cv2.fastNlMeansDenoisingColored(
            bgr, None, h, h, template_window, search_window
        )
        return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    else:
        return cv2.fastNlMeansDenoising(
            image, None, h, template_window, search_window
        )