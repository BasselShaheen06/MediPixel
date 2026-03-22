# Installation

## Requirements

- Python 3.10 or later
- Windows, macOS, or Linux

## Step 1 — Clone the repository

```bash
git clone https://github.com/BasselShaheen06/MediPixel.git
cd MediPixel
```

## Step 2 — Create a virtual environment

=== "Windows (PowerShell)"

    ```powershell
    python -m venv venv
    venv\Scripts\activate
    ```

=== "macOS / Linux"

    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

You will see `(venv)` at the start of your prompt when the environment is active.

## Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

This installs:

| Package | Purpose |
|---|---|
| `PyQt5` | GUI framework |
| `pyqtgraph` | Main canvas — zoom, pan, interactive ROI |
| `numpy` | Array operations throughout |
| `opencv-python` | Denoising filters (median, bilateral, NLM) |
| `scikit-image` | Adaptive gamma correction |
| `matplotlib` | Thumbnail viewports and histogram |
| `Pillow` | Image loading and interpolation |
| `pydicom` | DICOM file reading |
| `scipy` | Unused directly — available for extensions |

## Step 4 — Run

```bash
python main.py
```

---

## DICOM support

MediPixel reads DICOM files via `pydicom`. Compressed DICOM (JPEG, JPEG 2000, RLE) requires additional backends:

```bash
pip install pylibjpeg pylibjpeg-libjpeg   # JPEG / JPEG-LS
pip install python-gdcm                    # JPEG 2000 / RLE (Windows-friendly)
```

If you open a compressed DICOM and see a pixel decode error, install one of the above and try again.

---

## Building the documentation locally

```bash
pip install mkdocs-material
mkdocs serve
```

Then open `http://127.0.0.1:8000` in your browser. The site live-reloads as you edit the markdown files.