# core/denoiser.py

Spatial denoising filters. All functions accept and return numpy `uint8` arrays.
No UI dependencies.

---

## Choosing a filter

| Filter | Speed | Edge preservation | Best against | Avoid when |
|---|---|---|---|---|
| Median | Fast | Good | Salt & Pepper | Fine texture matters |
| Bilateral | Medium | Excellent | Gaussian | Real-time required |
| Non-Local Means | Slow | Best | Gaussian | Image is large + time-critical |

**Rule of thumb:** If you added salt & pepper noise, use median. If you added
Gaussian or Poisson noise and care about preserving thin structures, use NLM.
Bilateral is the middle ground — better than median for Gaussian, faster than NLM.

---

## `median`

```python
def median(image: np.ndarray, kernel_size: int = 3) -> np.ndarray
```

Replaces each pixel with the **median** of its k×k neighbourhood.
Because it uses the median rather than the mean, a small number of extreme
values (salt & pepper) have no effect — they need to be the majority to
influence the result.

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `image` | `np.ndarray` | — | 2-D uint8 array |
| `kernel_size` | `int` | `3` | Must be odd: 3, 5, 7… Larger = more smoothing, more blur. |

**Example**

```python
from medipixel.core.denoiser import median

denoised = median(image, kernel_size=3)   # gentle
stronger = median(image, kernel_size=7)   # removes more noise, blurs more
```

!!! warning "Kernel size trade-off"
    `kernel_size=3` removes isolated noise pixels cleanly.
    `kernel_size=7` starts to blur edges and thin structures.
    For salt & pepper, `kernel_size=3` is almost always optimal.

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

Edge-preserving Gaussian blur. Weights each neighbour by **both** spatial
distance and intensity similarity. Pixels across an edge have very different
intensities, so they get near-zero weight — the edge is preserved while
uniform regions are smoothed.

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `image` | `np.ndarray` | — | 2-D uint8 array |
| `d` | `int` | `9` | Diameter of the pixel neighbourhood. Larger = slower. |
| `sigma_color` | `float` | `75.0` | Intensity range. Large = blur across bigger intensity differences (softer edge preservation). |
| `sigma_space` | `float` | `75.0` | Spatial range. Large = consider farther pixels. |

**Example**

```python
from medipixel.core.denoiser import bilateral

# Gentle — preserve fine edges
gentle = bilateral(image, d=5, sigma_color=50, sigma_space=50)

# Stronger — heavier smoothing, still edge-aware
strong = bilateral(image, d=9, sigma_color=100, sigma_space=100)
```

!!! tip "sigma_color is the key parameter"
    If `sigma_color` is too small, the filter barely blurs anything
    (every intensity difference looks like an edge). If too large, it
    degrades to a regular Gaussian blur. `75` works well for uint8 images.

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

Finds similar patches **anywhere in the image** and averages them together.
Unlike local filters, NLM uses global information — repeated structures
(blood vessels, tissue texture) are averaged with their matches across the
whole image, recovering detail that local filters destroy.

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `image` | `np.ndarray` | — | 2-D uint8 array |
| `h` | `float` | `10.0` | Filter strength. Higher = smoother but blurrier. Match to noise sigma. |
| `template_window` | `int` | `7` | Patch size for comparison (odd, pixels). Larger = more context, slower. |
| `search_window` | `int` | `21` | Search area (odd, pixels). Larger = more matches found, much slower. |

**Example**

```python
from medipixel.core.denoiser import non_local_means

# h should roughly match the noise sigma
light_denoise = non_local_means(image, h=7)
heavy_denoise = non_local_means(image, h=20)
```

!!! info "Complexity"
    NLM is O(N² · w²) where N is the number of pixels and w is the search
    window size. On a 512×512 image with default settings, expect 2–5 seconds.
    This is why MediPixel uses a 120ms debounce on this slider.

!!! tip "Setting h"
    A good starting point: set `h` equal to the noise sigma you added.
    If you added Gaussian noise with `sigma=25`, start with `h=25`.
    Reduce if the result looks over-smoothed.