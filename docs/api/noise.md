# core/noise.py

Pure noise generators. All functions accept and return numpy `uint8` arrays. No UI dependencies.

---

## `add_gaussian`

```python
def add_gaussian(image: np.ndarray, sigma: float = 25.0) -> np.ndarray
```

Add zero-mean Gaussian noise.

**Parameters:**

| Name | Type | Default | Description |
|---|---|---|---|
| `image` | `np.ndarray` | — | 2-D uint8 array |
| `sigma` | `float` | `25.0` | Standard deviation on the 0–255 scale |

**Returns:** Noisy image clipped to uint8 `[0, 255]`.

**Example:**

```python
from medipixel.core.noise import add_gaussian
noisy = add_gaussian(image, sigma=40)
```

---

## `add_salt_and_pepper`

```python
def add_salt_and_pepper(image: np.ndarray, prob: float = 0.05) -> np.ndarray
```

Add impulse noise. Half the corrupted pixels become 255 (salt), half become 0 (pepper).

**Parameters:**

| Name | Type | Default | Description |
|---|---|---|---|
| `image` | `np.ndarray` | — | 2-D uint8 array |
| `prob` | `float` | `0.05` | Total fraction of corrupted pixels (0.0–1.0) |

---

## `add_poisson`

```python
def add_poisson(image: np.ndarray, scale: float = 1.0) -> np.ndarray
```

Add signal-dependent Poisson (shot) noise. Higher `scale` means less noise.

**Parameters:**

| Name | Type | Default | Description |
|---|---|---|---|
| `image` | `np.ndarray` | — | 2-D uint8 array |
| `scale` | `float` | `1.0` | Scales the Poisson lambda. Higher = less noise. |