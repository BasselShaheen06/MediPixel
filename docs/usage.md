# Usage

## Loading an image

Click **Load image** in the toolbar. MediPixel accepts:

- DICOM (`.dcm`) — single frame and multi-frame
- PNG, JPEG, BMP, TIFF — loaded with a choice between color and grayscale

The image appears on the **Original** tab. All pixel values are normalised to uint8 `[0, 255]` on load.

---

## Onboarding and UI preferences

- **Startup cover page**: a short splash screen appears before the main window opens.
- **Guided tour**: use the guide action to walk through the main controls and workflow.
- **Font family selector**: choose between bundled UI fonts from the top controls.
- **Font scale (%)**: adjust global text size with the numeric percentage control.

Both font family and scale are persisted with `QSettings` and restored on next launch.

---

## The processing pipeline

Controls are in the left sidebar, grouped into collapsible sections. Click any section title to hide or show its controls.

### Pipeline order

Operations are applied in this fixed order every time any control changes:

```
subsample → add noise → denoise → frequency filter → contrast enhance → display
```

**Target viewport** — choose whether your settings send the result to Viewport 1 or Viewport 2. The original image is never modified.

### Noise

| Type | Model | When to use |
|---|---|---|
| Gaussian | Additive zero-mean normal | Thermal / electronic detector noise |
| Salt & Pepper | Random pixel impulses to 0 or 255 | Dead pixels, bit-flip errors |
| Poisson | Signal-dependent shot noise | Low-dose X-ray, nuclear medicine |

**Strength** controls the standard deviation (Gaussian), corrupted-pixel fraction (Salt & Pepper), or Poisson scale factor.

### Denoising

| Filter | Edge preservation | Best against |
|---|---|---|
| Median | Good | Salt & Pepper |
| Bilateral | Excellent | Gaussian |
| Non-Local Means | Best | Gaussian — preserves fine structure |

Non-Local Means is significantly slower than the others. Use the 120ms debounce — the result appears shortly after you stop dragging the slider.

### Frequency filter

A **Butterworth** filter applied in the Fourier domain. The Butterworth shape avoids the ringing artefacts of an ideal brick-wall filter while still providing a controllable roll-off.

- **Cutoff %** — normalised cutoff frequency. Low values pass only coarse structure; high values pass almost everything.
- **Order** — steepness of the roll-off. Order 2 is gentle; order 8 is near-brick-wall.
- **Lowpass** — removes high-frequency detail (noise, edges). Blurs the image.
- **Highpass** — removes low-frequency background. Enhances edges and texture.

### Contrast enhancement

| Method | What it does |
|---|---|
| Histogram equalization | Redistributes intensities globally. Maximises contrast but amplifies noise. |
| CLAHE | Equalises each tile independently. The clip limit prevents noise amplification. |
| Adaptive gamma | Adjusts gamma based on mean brightness. Brightens dark images, darkens bright ones. |

---

## Canvas interaction

The main canvas uses **pyqtgraph**:

- **Scroll wheel** — zoom in / zoom out centred on cursor
- **Right-click drag** — pan the image
- **Double-click** — reset zoom and pan to fit

Switch between the original and processed versions using the tab bar at the top of the canvas.

---

## ROI selection and metrics

### Workflow

1. Click **Start ROI selection** in the sidebar
2. A red box (Signal A) appears centred on the image
3. Drag the box to your region of interest — position it over bright tissue
4. Drag any corner handle to resize
5. Click **✓ Confirm position** — Signal A is locked, Signal B box appears
6. Repeat for Signal B (a second tissue region) and Noise (background / air)
7. Click **SNR** or **CNR** — results appear for Original, Viewport 1, and Viewport 2

After finishing you can still drag all three boxes to adjust positions. Click **Redo ROI selection** to start over.

### Formulas

$\text{SNR} = \frac{\mu_{\text{signal}}}{\sigma_{\text{noise}}}$

$\text{CNR} = \frac{|\mu_{\text{signal A}} - \mu_{\text{signal B}}|}{\sigma_{\text{noise}}}$

Where $\mu$ is the mean pixel value inside the ROI and $\sigma$ is the standard deviation.

!!! tip "Choosing ROI positions"
    For SNR: place Signal A over a uniform bright region (white matter in MRI, bone in CT). Place Noise in the background outside the body.

    For CNR: place Signal A and Signal B over two different tissue types you want to distinguish. Place Noise in the background.

---

## Histogram tab

Switch to the **Histogram** tab using the tab bar.

| Control | Function |
|---|---|
| Source dropdown | Select which image to analyse — Original, Viewport 1, Viewport 2, or any of the three ROI regions |
| Overlay all | Plot Original, VP1, VP2 on one chart for direct comparison |
| Show CDF | Toggle the cumulative distribution function on a right Y-axis (0–100%) |
| Export PNG | Save the current histogram figure at 150 DPI |

## Color images

When loading a PNG or JPEG, a dialog asks whether to keep color or convert
to grayscale. For DICOM, grayscale is always used.

In color mode, contrast operations (CLAHE, histogram equalization, gamma)
work on the luminance channel only — hue and saturation are preserved.
SNR and CNR are measured on the ITU-R BT.601 luminance: Y = 0.299R + 0.587G + 0.114B.
The histogram tab shows R, G, B as separate lines.

### Reading the CDF

The CDF shows what percentage of pixels fall below each intensity level. A steep CDF means high contrast — intensity is spread across the full range. A flat CDF means low contrast — most pixels are clustered in a narrow band.

---

## Saving results

Click **Save result** in the toolbar. The currently selected target viewport is saved. Supports PNG, JPEG, and BMP.