# core/contrast.py

Contrast enhancement and frequency filtering. All functions accept and return
numpy `uint8` arrays. No UI dependencies.

---

## Choosing a method

| Method | Global / Local | Noise amplification | Best for |
|---|---|---|---|
| Histogram equalization | Global | High | Images with heavily skewed histograms |
| CLAHE | Local (per tile) | Controlled | Medical images with non-uniform illumination |
| Adaptive gamma | Global | None | Quick brightness correction |
| Frequency filter | Frequency domain | Depends on cutoff | Targeted noise removal or edge enhancement |

---

## `histogram_equalization`

```python
def histogram_equalization(image: np.ndarray) -> np.ndarray
```

Redistributes pixel intensities so the output histogram is approximately flat.
Uses the CDF of the input as the mapping function:

$$T(r) = \lfloor (L-1) \cdot \text{CDF}(r) \rfloor$$

**Returns** — uint8 array, same shape as input.

**Example**

```python
from medipixel.core.contrast import histogram_equalization

enhanced = histogram_equalization(image)
```

!!! warning "When not to use this"
    Global HE treats the whole image identically. In medical images with
    large uniform regions (air, background), the CDF is dominated by those
    regions. Soft tissue contrast — where it matters clinically — may actually
    worsen. Use CLAHE instead for medical images.

---

## `clahe`

```python
def clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray
```

Contrast Limited Adaptive Histogram Equalization. Divides the image into tiles
and equalises each tile independently. The `clip_limit` caps the histogram
before computing the CDF — excess counts are redistributed uniformly —
preventing the noise amplification that plain AHE causes in uniform regions.

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `image` | `np.ndarray` | — | 2-D uint8 array |
| `clip_limit` | `float` | `2.0` | Histogram cap per tile. `1.0` = no amplification, `4.0` = strong enhancement. |
| `tile_grid_size` | `tuple[int, int]` | `(8, 8)` | Grid of tiles. Larger grid = more localised, more edge artefacts. |

**Example**

```python
from medipixel.core.contrast import clahe

# Standard medical imaging setting
standard = clahe(image, clip_limit=2.0, tile_grid_size=(8, 8))

# More aggressive — better contrast, more noise
aggressive = clahe(image, clip_limit=4.0, tile_grid_size=(16, 16))
```

!!! tip "CLAHE is the standard"
    CLAHE is used in retinal vessel segmentation, lung nodule detection,
    and mammography CAD pipelines. When in doubt, use CLAHE over plain
    histogram equalization for any medical image.

!!! info "clip_limit intuition"
    `clip_limit=1.0` — no contrast enhancement, just redistribution.
    `clip_limit=2.0` — standard, balanced.
    `clip_limit=8.0` — approaches unconstrained AHE, aggressive noise amplification.

---

## `adaptive_gamma`

```python
def adaptive_gamma(image: np.ndarray) -> np.ndarray
```

Applies gamma correction with gamma derived from the image's own mean brightness.
No manual parameter required.

- Mean brightness < 0.5 → gamma ∈ [0.5, 1.0) → **brightens** the image
- Mean brightness ≥ 0.5 → gamma ∈ [1.0, 1.5] → **darkens** the image

**Example**

```python
from medipixel.core.contrast import adaptive_gamma

corrected = adaptive_gamma(dark_image)   # automatically brightens
corrected = adaptive_gamma(bright_image) # automatically darkens
```

!!! tip "When to use"
    Useful as a quick normalisation step before other processing.
    Not a substitute for CLAHE when local contrast matters.

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

Butterworth filter in the Fourier domain. The Butterworth transfer function:

$$H(f) = \frac{1}{1 + (f / f_c)^{2n}}$$

provides a smooth roll-off that avoids the ringing artefacts of an ideal
brick-wall filter, while still having a controllable transition via `order`.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `image` | `np.ndarray` | 2-D uint8 array |
| `kind` | `str` | `"lowpass"` removes high frequencies (blurs). `"highpass"` removes low frequencies (enhances edges). |
| `cutoff` | `float` | Normalised cutoff in `(0, 1]`. `0.1` = very blurry / only coarse edges. `0.9` = near-original. |
| `order` | `int` | Butterworth order. `1–2` = gentle roll-off. `8–10` = near brick-wall. |

**Raises** — `ValueError` if `kind` is not `"lowpass"` or `"highpass"`.

**Examples**

```python
from medipixel.core.contrast import frequency_filter

# Gentle noise removal
smooth = frequency_filter(image, kind="lowpass", cutoff=0.3, order=2)

# Aggressive noise removal — loses fine detail
very_smooth = frequency_filter(image, kind="lowpass", cutoff=0.1, order=4)

# Edge enhancement
edges = frequency_filter(image, kind="highpass", cutoff=0.2, order=2)

# Near brick-wall lowpass — may show ringing at edges
sharp_cutoff = frequency_filter(image, kind="lowpass", cutoff=0.3, order=8)
```

**Choosing cutoff and order**

| Goal | cutoff | order |
|---|---|---|
| Light denoising, preserve detail | 0.4–0.6 | 2 |
| Moderate denoising | 0.2–0.4 | 2–4 |
| Heavy denoising, accept blur | 0.1–0.2 | 2 |
| Sharp edge enhancement | 0.1–0.3 | 2–4 |

!!! info "Why Butterworth and not ideal?"
    An ideal lowpass filter (cutoff at exactly fc, zero everywhere else)
    produces Gibbs ringing — bright and dark fringes along edges in the
    spatial domain. The Butterworth smooth transition eliminates this.
    Higher order approaches ideal but ringing returns. Order 2 is the
    standard clinical choice.

!!! warning "High-order filters and uint8"
    At order 8–10, the filter approaches a brick wall. The resulting image
    may have float values outside [0, 255] before clipping. MediPixel
    normalises the output back to uint8 automatically, but very aggressive
    settings can produce artefacts that look like contrast inversion.