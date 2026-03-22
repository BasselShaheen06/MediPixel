# core/contrast.py

Contrast enhancement and frequency filtering. All functions accept and return numpy `uint8` arrays. No UI dependencies.

---

## `histogram_equalization`

```python
def histogram_equalization(image: np.ndarray) -> np.ndarray
```

Global histogram equalization. Redistributes pixel intensities so the histogram is approximately flat. Maximises overall contrast but can over-enhance noise.

---

## `clahe`

```python
def clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray
```

Contrast Limited Adaptive Histogram Equalization. Equalises each tile independently, with a clip limit to prevent noise amplification.

**Parameters:**

| Name | Type | Default | Description |
|---|---|---|---|
| `image` | `np.ndarray` | — | 2-D uint8 array |
| `clip_limit` | `float` | `2.0` | Contrast cap per tile. `1.0` = no amplification |
| `tile_grid_size` | `tuple[int, int]` | `(8, 8)` | Grid of tiles. Larger = more localised enhancement |

---

## `adaptive_gamma`

```python
def adaptive_gamma(image: np.ndarray) -> np.ndarray
```

Gamma correction with gamma derived from the image's mean brightness. Brightens dark images, darkens bright ones, without manual parameter selection.

---

## `frequency_filter`

```python
def frequency_filter(
    image: np.ndarray,
    kind: str,
    cutoff: float,
    order: int,
) -> np.ndarray
```

Butterworth filter in the Fourier domain. Avoids the ringing artefacts of an ideal brick-wall filter.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `image` | `np.ndarray` | 2-D uint8 array |
| `kind` | `str` | `"lowpass"` or `"highpass"` |
| `cutoff` | `float` | Normalised cutoff in `(0, 1]`. `0.1` = blurry, `0.9` = near-original |
| `order` | `int` | Butterworth order — higher = steeper roll-off |

**Raises:** `ValueError` if `kind` is not `"lowpass"` or `"highpass"`.

**Example:**

```python
from medipixel.core.contrast import frequency_filter

# Remove noise with a gentle lowpass
smooth = frequency_filter(image, kind="lowpass", cutoff=0.3, order=2)

# Extract edges with a highpass
edges = frequency_filter(image, kind="highpass", cutoff=0.2, order=4)
```