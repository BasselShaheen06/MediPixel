# Changelog

All notable changes are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [1.1.0] — March 2026

### Added
- **pyqtgraph main canvas** — scroll-to-zoom, right-click-drag pan, native ROI handles
- **Guided ROI workflow** — step-by-step Signal A → Signal B → Noise placement with confirm button
- **Rubber-band ROI drawing** — drag to draw boxes, drag corners to resize
- **Histogram tab enhancements** — ROI histogram, CDF overlay, multi-source comparison, PNG export
- **Split debounce** — view sliders update at 30ms, compute sliders at 120ms
- **Collapsible sidebar sections** — click section title to hide/show
- **Resizable panels** — drag QSplitter handles between sidebar, canvas, and thumbnails
- **Sidebar toggle** and **thumbnail toggle** buttons in toolbar
- **Mode indicator pill** in toolbar showing current canvas interaction mode
- **MkDocs Material documentation site** — deployed at basselshaheen06.github.io/MediPixel
- **GitHub Actions** — automatic docs deployment on every push to main

### Changed
- Main canvas migrated from matplotlib to pyqtgraph
- ROI interaction rewritten — no longer requires clicking separate buttons for each step
- Histogram source dropdown expanded to include ROI regions
- Processing pipeline has single code path — no duplicate contrast logic

### Fixed
- SNR/CNR returning `inf` — now uses `roi.getArrayRegion()` for correct coordinate mapping
- Tab text clipping — tabs now use `setFixedWidth` instead of `setMinimumWidth`
- Slider lag — debounce prevents mid-drag pipeline reruns

---

## [1.0.0] — January 2026

### Added
- Initial modular architecture: `core/noise.py`, `core/denoiser.py`, `core/contrast.py`
- `ui/canvas.py` — DraggableCanvas for matplotlib viewports
- `ui/main_window.py` — Qt layout with sidebar, main canvas, right thumbnails
- Gaussian, Salt & Pepper, Poisson noise generators
- Median, Bilateral, Non-Local Means denoising filters
- Butterworth frequency filter (lowpass + highpass)
- Histogram equalization, CLAHE, adaptive gamma correction
- SNR and CNR calculation from drawn ROIs
- DICOM and standard image format loading
- `pyproject.toml` — installable package with `pip install .`
- MIT License

### Architecture
- `core/` modules have no UI dependencies — pure numpy/cv2 functions
- Hard boundary: `core/` never imports PyQt5, pyqtgraph, or matplotlib
- Single processing pipeline in `_process()` — no duplicated logic

---

[1.1.0]: https://github.com/BasselShaheen06/MediPixel/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/BasselShaheen06/MediPixel/releases/tag/v1.0.0