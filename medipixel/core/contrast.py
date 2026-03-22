"""
Contrast enhancement for medical images.

All functions accept and return numpy uint8 arrays.
They are pure functions — no side effects, no UI state.
"""

import cv2
import numpy as np
from skimage import exposure


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_uint8(image: np.ndarray) -> np.ndarray:
    """Normalise any numeric array to uint8 [0, 255]."""
    image = image.astype(np.float32)
    mn, mx = image.min(), image.max()
    if mx > mn:
        return ((image - mn) * (255.0 / (mx - mn))).astype(np.uint8)
    return np.zeros_like(image, dtype=np.uint8)


# ── Public API ────────────────────────────────────────────────────────────────

def histogram_equalization(image: np.ndarray) -> np.ndarray:
    """
    Global histogram equalization.

    Redistributes pixel intensities so the histogram is approximately flat.
    Maximises overall contrast but can over-enhance noise in uniform regions.

    Args:
        image: 2-D uint8 numpy array.

    Returns:
        Contrast-enhanced uint8 image.
    """
    return cv2.equalizeHist(_to_uint8(image))


def clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Contrast Limited Adaptive Histogram Equalization (CLAHE).

    Divides the image into tiles and equalises each tile independently.
    The clip_limit cap prevents noise amplification — the key improvement
    over plain histogram equalization for medical images.

    Args:
        image:          2-D uint8 numpy array.
        clip_limit:     Contrast cap per tile (1.0 = no amplification).
        tile_grid_size: (rows, cols) tile grid. Larger grid = more local.

    Returns:
        Contrast-enhanced uint8 image.
    """
    engine = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return engine.apply(_to_uint8(image))


def adaptive_gamma(image: np.ndarray) -> np.ndarray:
    """
    Adaptive gamma correction based on mean image brightness.

    Gamma < 1 brightens a dark image; gamma > 1 darkens a bright one.
    The gamma value is derived from the image itself rather than set manually.

    Args:
        image: 2-D uint8 numpy array.

    Returns:
        Gamma-corrected uint8 image.
    """
    img_norm = image.astype(np.float32)
    mn, mx = img_norm.min(), img_norm.max()
    if mx > mn:
        img_norm = (img_norm - mn) / (mx - mn)
    else:
        return image.copy()

    mean_brightness = float(np.mean(img_norm))

    # Dark image (mean < 0.5) → gamma < 1 → brighter
    # Bright image (mean ≥ 0.5) → gamma > 1 → darker
    if mean_brightness < 0.5:
        gamma = 0.5 + mean_brightness        # range: [0.5, 1.0)
    else:
        gamma = 1.0 + (mean_brightness - 0.5)  # range: [1.0, 1.5]

    enhanced = exposure.adjust_gamma(img_norm, gamma)
    return (enhanced * 255).astype(np.uint8)


def frequency_filter(
    image: np.ndarray,
    kind: str,
    cutoff: float,
    order: int,
) -> np.ndarray:
    """
    Butterworth frequency-domain filter (lowpass or highpass).

    Works in the Fourier domain:
      1. FFT → shift DC to centre
      2. Multiply by a Butterworth mask
      3. Inverse FFT → clip to uint8

    The Butterworth shape avoids the ringing artefacts of an ideal brick-wall
    filter while still having a controllable roll-off via `order`.

    Args:
        image:  2-D uint8 numpy array.
        kind:   "lowpass" or "highpass".
        cutoff: Normalised cutoff frequency in (0, 1].
                0.1 = low cutoff (blurry/only fine edges),
                0.9 = high cutoff (almost all frequencies pass).
        order:  Butterworth order — higher = steeper roll-off.

    Returns:
        Filtered image as uint8.

    Raises:
        ValueError: If `kind` is not "lowpass" or "highpass".
    """
    if kind not in ("lowpass", "highpass"):
        raise ValueError(f"kind must be 'lowpass' or 'highpass', got {kind!r}")

    img_f = image.astype(np.float32)
    f_shift = np.fft.fftshift(np.fft.fft2(img_f))

    rows, cols = img_f.shape
    u = np.linspace(-0.5, 0.5, cols)
    v = np.linspace(-0.5, 0.5, rows)
    U, V = np.meshgrid(u, v)
    D = np.sqrt(U ** 2 + V ** 2)

    # Butterworth transfer function
    butter = 1.0 / (1.0 + (D / cutoff) ** (2 * order))
    mask = butter if kind == "lowpass" else (1.0 - butter)

    filtered = np.abs(np.fft.ifft2(np.fft.ifftshift(f_shift * mask)))

    mn, mx = filtered.min(), filtered.max()
    if mx > mn:
        return ((filtered - mn) * (255.0 / (mx - mn))).astype(np.uint8)
    return np.zeros_like(image, dtype=np.uint8)
