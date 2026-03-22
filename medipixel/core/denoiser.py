"""
Spatial denoising filters for medical images.

All functions accept and return numpy uint8 arrays.
They are pure functions — no side effects, no UI state.
"""

import cv2
import numpy as np


def median(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Median filter — effective against salt-and-pepper noise.
    Preserves edges better than a Gaussian blur.

    Args:
        image:       2-D uint8 numpy array.
        kernel_size: Must be odd and positive (3, 5, 7 …).

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

    Blurs pixels that are spatially close AND similar in intensity.
    Leaves sharp edges intact because cross-edge pixels have very
    different intensities and therefore low color-domain weight.

    Args:
        image:       2-D uint8 numpy array.
        d:           Diameter of the pixel neighbourhood.
        sigma_color: Intensity range — large value = blur across bigger
                     intensity differences.
        sigma_space: Spatial range — large value = consider farther pixels.

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
    Non-Local Means (NLM) denoising.

    Instead of only using nearby pixels, NLM finds similar patches
    anywhere in the image and averages them. This recovers fine
    structural detail that local filters destroy.

    Args:
        image:           2-D uint8 numpy array.
        h:               Filter strength. Higher = smoother but blurrier.
        template_window: Patch size used for comparison (odd, pixels).
        search_window:   Search area size (odd, pixels).

    Returns:
        Denoised image as uint8.
    """
    return cv2.fastNlMeansDenoising(image, None, h, template_window, search_window)
