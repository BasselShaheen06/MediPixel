# The Science

This page explains the signal processing and imaging physics behind each operation in MediPixel. Understanding *why* each method works — not just that it does — is what separates a tool you built from a tool you understand.

---

## Noise models

### Why three different models?

Noise in medical imaging is not a single phenomenon. Different physical processes produce fundamentally different statistical distributions, and the correct denoising strategy depends on the noise model.

### Gaussian noise

**Physical origin:** Thermal noise in the detector electronics, amplifier noise, and quantisation noise. These noise sources are independent and additive — they add on top of the signal regardless of its intensity.

**Model:** Each pixel $p_{noisy} = p_{clean} + \mathcal{N}(0, \sigma^2)$

The noise is zero-mean and signal-independent. This means bright regions and dark regions are corrupted equally.

**In MediPixel:** `noise.add_gaussian(image, sigma)` adds $\mathcal{N}(0, \sigma^2)$ drawn independently per pixel.

### Salt & Pepper noise

**Physical origin:** Dead pixels in the detector array, bit-flip errors in data transmission, read errors in storage. Each corrupted pixel is independently driven to the maximum (salt, 255) or minimum (pepper, 0) value.

**Model:** Each pixel is corrupted with probability $p$: half become 255, half become 0.

This is an *impulse* noise model — the corruption is extreme and sparse. A median filter is optimal against this type because it replaces each pixel with the neighbourhood median, which is unaffected by a small number of extreme values.

### Poisson (shot) noise

**Physical origin:** The quantum nature of light. X-ray photons and gamma rays arrive at the detector as discrete particles. The number of photons detected per pixel follows a Poisson distribution, with variance equal to the mean count. This means **brighter regions have more noise** — the noise is signal-dependent.

**Model:** $p_{noisy} \sim \text{Poisson}(\lambda \cdot p_{clean}) / \lambda$

This is the dominant noise source in low-dose CT, X-ray fluoroscopy, and nuclear medicine (PET, SPECT). It explains why low-dose scans look noisier in the brighter tissue regions.

---

## Denoising filters

### Median filter

Replaces each pixel with the median of its $k \times k$ neighbourhood. Because it uses the *median* rather than the mean, a small number of extreme values (salt & pepper) have no effect on the result — they need to be the majority to influence the median.

**Trade-off:** Non-linear, so it can preserve edges well. But it removes fine texture and thin structures when the kernel is large.

### Bilateral filter

A Gaussian blur that is weighted by **both** spatial distance and intensity similarity:

$$p_{out}(x) = \frac{1}{W} \sum_{y} G_s(\|x-y\|) \cdot G_r(|I(x) - I(y)|) \cdot I(y)$$

Where $G_s$ is the spatial Gaussian and $G_r$ is the range (intensity) Gaussian.

The key insight: pixels on the other side of an edge have very different intensity values, so $G_r$ assigns them near-zero weight. The edge is therefore preserved while noise within a uniform region is averaged away.

**Parameters:**
- `sigma_color` — how much intensity difference is tolerated. Large value = blur across bigger intensity differences (softer edge preservation).
- `sigma_space` — spatial range. Large value = consider farther pixels.

### Non-Local Means (NLM)

Both median and bilateral only use pixels in a local neighbourhood. NLM uses the entire image:

$$p_{out}(x) = \frac{1}{Z(x)} \sum_{y} w(x, y) \cdot I(y)$$

The weight $w(x, y)$ is based on the **similarity of a patch** centred at $x$ to a patch centred at $y$. If two patches look alike anywhere in the image, their central pixels are likely to have the same true value, so they are averaged together.

This is why NLM preserves fine structure that local filters destroy — it finds similar patches across the whole image and uses them all to estimate the clean value. The cost is computation: it is $O(n^2)$ in the number of pixels compared to $O(n \cdot k^2)$ for local filters.

---

## Butterworth frequency filter

### Why frequency domain?

Noise and signal often occupy different parts of the frequency spectrum. Gaussian noise is broadband — it has energy at all frequencies. The structural content of a medical image (anatomy, organ boundaries) is concentrated in the low and mid frequencies. A lowpass filter that removes high-frequency energy will remove noise while preserving structure.

### The Butterworth transfer function

An ideal lowpass filter (brick wall) has the transfer function $H(f) = 1$ for $f < f_c$ and $H(f) = 0$ for $f \geq f_c$. In the spatial domain, this corresponds to convolution with a sinc function, which causes severe ringing artefacts (Gibbs phenomenon) at edges.

The Butterworth filter provides a smooth transition:

$$H(f) = \frac{1}{1 + (f / f_c)^{2n}}$$

where $f_c$ is the cutoff frequency and $n$ is the filter order. Higher order = steeper roll-off, approaching a brick wall as $n \to \infty$.

**In MediPixel:** The filter is applied by computing the 2D FFT of the image, multiplying by $H(f)$, and taking the inverse FFT.

---

## Contrast enhancement

### Histogram equalization

Maps pixel intensities so the output histogram is approximately uniform. Mathematically, this is achieved by using the CDF of the input as the mapping function:

$$T(r) = \lfloor (L-1) \cdot \text{CDF}(r) \rfloor$$

Where $r$ is the input intensity, $L=256$ is the number of levels, and CDF is the cumulative distribution function.

**Problem:** A global CDF treats the whole image identically. In medical images with large uniform regions (background, bone), the CDF is dominated by those regions and the contrast in soft tissue — where it matters — is not improved.

### CLAHE

**Contrast Limited Adaptive Histogram Equalization** divides the image into tiles (default $8 \times 8$) and equalises each tile independently. This provides local contrast enhancement that adapts to regional intensity distributions.

The critical addition over AHE (Adaptive HE): a **clip limit**. Before computing the CDF, the histogram of each tile is clipped at the clip limit value. Excess counts are redistributed uniformly across all bins. This prevents the noise amplification that AHE causes in uniform regions — a uniform region has a spike in its histogram, and without the clip, that spike gets stretched across the full range, amplifying noise.

CLAHE is the standard preprocessing step in medical image analysis pipelines. It is used in retinal vessel segmentation, lung nodule detection, and mammography CAD.

### Adaptive gamma correction

The gamma function $I_{out} = I_{in}^{\gamma}$ is a nonlinear intensity mapping:

- $\gamma < 1$ brightens a dark image (expands the shadow tones)
- $\gamma > 1$ darkens a bright image (compresses the highlight tones)

MediPixel computes gamma adaptively from the image's mean brightness rather than requiring manual input:

- If mean brightness $< 0.5$: $\gamma = 0.5 + \text{mean}$ → range $[0.5, 1.0)$, brightening
- If mean brightness $\geq 0.5$: $\gamma = 1.0 + (\text{mean} - 0.5)$ → range $[1.0, 1.5]$, darkening

---

## SNR and CNR

### Signal-to-Noise Ratio

$$\text{SNR} = \frac{\mu_{\text{signal}}}{\sigma_{\text{noise}}}$$

SNR measures how much the signal stands above the noise floor. A higher SNR means the image can be acquired at lower dose (in CT/X-ray) or shorter scan time (in MRI) while still being diagnostically useful.

The noise ROI should be placed in a region with no signal — background air in CT, or a uniform phantom region.

### Contrast-to-Noise Ratio

$$\text{CNR} = \frac{|\mu_{\text{A}} - \mu_{\text{B}}|}{\sigma_{\text{noise}}}$$

CNR measures whether two tissue types are distinguishable. A lesion might be clearly visible in the raw image (high CNR) but after aggressive denoising it might become indistinct (lower CNR) because the denoising blurred the boundary.

This is why MediPixel calculates both metrics for the Original, Viewport 1, and Viewport 2 simultaneously — you can directly measure whether your processing improved or degraded diagnostic quality.

### NEMA and ACR standards

SNR and CNR as computed here are consistent with the NEMA MS 1 standard for MRI and the ACR CT accreditation phantom measurements. The formulas above use the mean of the signal ROI and the standard deviation of the noise ROI, which is the standard definition in both protocols.