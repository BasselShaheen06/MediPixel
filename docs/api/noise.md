# core/noise.py

Pure noise generators. All functions accept and return numpy `uint8` arrays.
No UI dependencies — use them independently in scripts or notebooks.

---

## When to use which noise model

| You want to simulate… | Use |
|---|---|
| Detector electronics, amplifier noise | `add_gaussian` |
| Dead pixels, transmission errors | `add_salt_and_pepper` |
| Low-dose X-ray, nuclear medicine | `add_poisson` |

The choice matters because optimal denoising depends on the noise model.
Applying a median filter to Gaussian noise works poorly; applying NLM to
salt & pepper is overkill. Match the noise to the physics.

---

## `add_gaussian`

```python
def add_gaussian(image: np.ndarray, sigma: float = 25.0) -> np.ndarray
```

Add zero-mean Gaussian noise. The noise is **additive and signal-independent** —
bright and dark regions are corrupted equally.

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `image` | `np.ndarray` | — | 2-D uint8 array |
| `sigma` | `float` | `25.0` | Standard deviation on the 0–255 scale. `10` = light, `50` = heavy. |

**Returns** — uint8 array, same shape as input, values clipped to `[0, 255]`.

**Example**

```python
from medipixel.core.noise import add_gaussian
import numpy as np

image  = np.array(...)
light  = add_gaussian(image, sigma=10)
heavy  = add_gaussian(image, sigma=50)
```

!!! tip "Typical values"
    `sigma=15–25` — visible but mild, realistic for moderate dose reduction.
    `sigma=40–60` — heavy degradation, similar to very low dose CT.

---

## `add_salt_and_pepper`

```python
def add_salt_and_pepper(image: np.ndarray, prob: float = 0.05) -> np.ndarray
```

Add impulse noise. Each pixel is independently corrupted with probability `prob` —
half become 255 (salt), half become 0 (pepper).

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `image` | `np.ndarray` | — | 2-D uint8 array |
| `prob` | `float` | `0.05` | Total corrupted-pixel fraction. `0.01` = 1%, `0.1` = 10%. |

**Example**

```python
from medipixel.core.noise import add_salt_and_pepper

mild  = add_salt_and_pepper(image, prob=0.01)
heavy = add_salt_and_pepper(image, prob=0.08)
```

!!! warning "Common mistake"
    `prob=0.05` means 5 in every 100 pixels are corrupted — already very visible.
    Stay below `0.05` for realistic simulation.

!!! tip "Best denoising pair"
    Always use **Median filter** with this noise type. Bilateral and NLM
    are designed for continuous noise distributions and handle impulse noise poorly.

---

## `add_poisson`

```python
def add_poisson(image: np.ndarray, scale: float = 1.0) -> np.ndarray
```

Add signal-dependent Poisson (shot) noise. **Brighter regions get more noise**
because they correspond to higher photon counts with higher variance.

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `image` | `np.ndarray` | — | 2-D uint8 array |
| `scale` | `float` | `1.0` | Higher scale = fewer photons = more noise. Dose reduction factor. |

**Example**

```python
from medipixel.core.noise import add_poisson

normal_dose = add_poisson(image, scale=1.0)   # baseline
low_dose    = add_poisson(image, scale=4.0)   # 4x less dose
ultra_low   = add_poisson(image, scale=10.0)  # extreme low-dose simulation
```

!!! info "Physics note"
    The variance of a Poisson distribution equals its mean. If a pixel has
    true intensity λ, the measured value is ~Poisson(λ) with std = √λ.
    This is why Poisson noise is most visible at tissue boundaries —
    the noise jumps where intensity jumps.