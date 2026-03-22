# core/denoiser.py

Spatial denoising filters. All functions accept and return numpy `uint8` arrays. No UI dependencies.

---

## `median`

```python
def median(image: np.ndarray, kernel_size: int = 3) -> np.ndarray
```

Median filter. Best against salt & pepper noise. Non-linear — preserves edges better than a Gaussian blur.

**Parameters:**

| Name | Type | Default | Description |
|---|---|---|---|
| `image` | `np.ndarray` | — | 2-D uint8 array |
| `kernel_size` | `int` | `3` | Must be odd (3, 5, 7…) |

---

## `bilateral`

```python
def bilateral(
    image: np.ndarray,
    d: int = 9,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0,
) -> np.ndarray
```

Bilateral filter. Edge-preserving Gaussian blur — pixels across edges have very different intensities and therefore low colour-domain weight.

**Parameters:**

| Name | Type | Default | Description |
|---|---|---|---|
| `image` | `np.ndarray` | — | 2-D uint8 array |
| `d` | `int` | `9` | Diameter of the pixel neighbourhood |
| `sigma_color` | `float` | `75.0` | Intensity range — large = blur across bigger intensity differences |
| `sigma_space` | `float` | `75.0` | Spatial range — large = consider farther pixels |

---

## `non_local_means`

```python
def non_local_means(
    image: np.ndarray,
    h: float = 10.0,
    template_window: int = 7,
    search_window: int = 21,
) -> np.ndarray
```

Non-Local Means denoising. Finds similar patches anywhere in the image and averages them. Best detail preservation, slowest.

**Parameters:**

| Name | Type | Default | Description |
|---|---|---|---|
| `image` | `np.ndarray` | — | 2-D uint8 array |
| `h` | `float` | `10.0` | Filter strength — higher = smoother but blurrier |
| `template_window` | `int` | `7` | Patch size for comparison (odd, pixels) |
| `search_window` | `int` | `21` | Search area size (odd, pixels) |