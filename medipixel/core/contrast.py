"""
Contrast enhancement and frequency filtering for medical images.

All functions accept numpy arrays (uint8) and return uint8 arrays.
Supports both grayscale (H, W) and color (H, W, 3) images.

Color strategy:
  - Histogram equalization and CLAHE operate on the L channel in LAB
    color space. This enhances luminance contrast without shifting hues.
  - Adaptive gamma operates on luminance (LAB L channel).
  - Frequency filter is applied per-channel independently.
  - All operations preserve the original color appearance.

Pure functions — no side effects, no UI state.
"""

import cv2
import numpy as np
from skimage import exposure


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_color(image: np.ndarray) -> bool:
    return image.ndim == 3 and image.shape[2] == 3


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    """Normalise any numeric array to uint8 [0, 255]."""
    arr = arr.astype(np.float32)
    mn, mx = arr.min(), arr.max()
    if mx > mn:
        return ((arr - mn) * (255.0 / (mx - mn))).astype(np.uint8)
    return np.zeros_like(arr, dtype=np.uint8)


def _apply_to_luminance(image: np.ndarray, fn) -> np.ndarray:
    """
    Apply fn (grayscale → grayscale) to the L channel of a color image.
    Converts RGB → LAB, applies fn to L, converts back to RGB.
    This preserves hue and saturation — only luminance contrast changes.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    l_enhanced = fn(l_ch)
    merged = cv2.merge([l_enhanced, a_ch, b_ch])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)


# ── Public API ────────────────────────────────────────────────────────────────

def histogram_equalization(image: np.ndarray) -> np.ndarray:
    """
    Global histogram equalization.

    For grayscale: equalises the single channel directly.
    For color: equalises only the luminance (L) channel in LAB space.
    This avoids the hue shifts that occur when equalising RGB separately.

    Args:
        image: uint8 array, shape (H, W) or (H, W, 3).

    Returns:
        Contrast-enhanced uint8 image, same shape as input.
    """
    if _is_color(image):
        return _apply_to_luminance(image, cv2.equalizeHist)
    return cv2.equalizeHist(_to_uint8(image))


def clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Contrast Limited Adaptive Histogram Equalization (CLAHE).

    For grayscale: applies directly.
    For color: applies to L channel in LAB space only.

    Args:
        image:          uint8 array, shape (H, W) or (H, W, 3).
        clip_limit:     Contrast cap per tile. 1.0 = no amplification.
        tile_grid_size: (rows, cols) tile grid.

    Returns:
        Contrast-enhanced uint8 image.
    """
    engine = cv2.createCLAHE(
        clipLimit=clip_limit, tileGridSize=tile_grid_size
    )
    if _is_color(image):
        return _apply_to_luminance(image, engine.apply)
    return engine.apply(_to_uint8(image))


def adaptive_gamma(image: np.ndarray) -> np.ndarray:
    """
    Adaptive gamma correction based on mean image brightness.

    For grayscale: corrects the single channel.
    For color: corrects luminance only, preserving hue and saturation.

    Gamma < 1 brightens a dark image; gamma > 1 darkens a bright one.

    Args:
        image: uint8 array, shape (H, W) or (H, W, 3).

    Returns:
        Gamma-corrected uint8 image.
    """
    def _gamma_on_gray(gray: np.ndarray) -> np.ndarray:
        img_norm = gray.astype(np.float32)
        mn, mx = img_norm.min(), img_norm.max()
        if mx <= mn:
            return gray.copy()
        img_norm = (img_norm - mn) / (mx - mn)
        mean_brightness = float(np.mean(img_norm))
        if mean_brightness < 0.5:
            gamma = 0.5 + mean_brightness        # [0.5, 1.0)
        else:
            gamma = 1.0 + (mean_brightness - 0.5)  # [1.0, 1.5]
        enhanced = exposure.adjust_gamma(img_norm, gamma)
        return (enhanced * 255).astype(np.uint8)

    if _is_color(image):
        return _apply_to_luminance(image, _gamma_on_gray)
    return _gamma_on_gray(image)


def frequency_filter(
    image: np.ndarray,
    kind: str,
    cutoff: float,
    order: int,
) -> np.ndarray:
    """
    Butterworth frequency-domain filter (lowpass or highpass).

    For grayscale: filters the single channel.
    For color: filters each RGB channel independently then recombines.
    This is valid because R, G, B channels share the same spatial
    frequency content — the filter acts identically on each.

    Args:
        image:  uint8 array, shape (H, W) or (H, W, 3).
        kind:   "lowpass" or "highpass".
        cutoff: Normalised cutoff frequency in (0, 1].
        order:  Butterworth order — higher = steeper roll-off.

    Returns:
        Filtered image as uint8.

    Raises:
        ValueError: If kind is not "lowpass" or "highpass".
    """
    if kind not in ("lowpass", "highpass"):
        raise ValueError(
            f"kind must be 'lowpass' or 'highpass', got {kind!r}"
        )

    def _filter_channel(ch: np.ndarray) -> np.ndarray:
        img_f  = ch.astype(np.float32)
        f_shift = np.fft.fftshift(np.fft.fft2(img_f))
        rows, cols = img_f.shape
        u = np.linspace(-0.5, 0.5, cols)
        v = np.linspace(-0.5, 0.5, rows)
        U, V = np.meshgrid(u, v)
        D    = np.sqrt(U ** 2 + V ** 2)
        butter = 1.0 / (1.0 + (D / cutoff) ** (2 * order))
        mask   = butter if kind == "lowpass" else (1.0 - butter)
        filtered = np.abs(np.fft.ifft2(np.fft.ifftshift(f_shift * mask)))
        mn, mx = filtered.min(), filtered.max()
        if mx > mn:
            return ((filtered - mn) * (255.0 / (mx - mn))).astype(np.uint8)
        return np.zeros_like(ch, dtype=np.uint8)

    if _is_color(image):
        channels = [_filter_channel(image[:, :, c]) for c in range(3)]
        return np.stack(channels, axis=2)
    return _filter_channel(image)


# ── Luminance extraction (for SNR/CNR in color mode) ─────────────────────────

def to_luminance(image: np.ndarray) -> np.ndarray:
    """
    Convert a color image to a grayscale luminance map.

    Uses the ITU-R BT.601 coefficients — the standard for converting
    RGB to perceptual luminance:
        Y = 0.299 R + 0.587 G + 0.114 B

    These coefficients reflect human visual sensitivity: we are most
    sensitive to green, less to red, least to blue.

    Args:
        image: uint8 array, shape (H, W) or (H, W, 3).

    Returns:
        uint8 array, shape (H, W). If already grayscale, returns a copy.
    """
    if not _is_color(image):
        return image.copy()
    return np.dot(
        image.astype(np.float32),
        [0.299, 0.587, 0.114]
    ).astype(np.uint8)