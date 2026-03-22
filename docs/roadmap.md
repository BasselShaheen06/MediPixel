# Roadmap

Features that are complete, in progress, or planned. If you want to work on
any of these, check the [contributing guide](contributing.md) and open an issue first.

---

## Completed

- [x] Gaussian, Salt & Pepper, and Poisson noise generation
- [x] Median, Bilateral, and Non-Local Means denoising
- [x] Butterworth frequency filter (lowpass + highpass)
- [x] Histogram equalization, CLAHE, adaptive gamma correction
- [x] SNR and CNR from interactive rubber-band ROI boxes
- [x] Three guided ROI workflow (Signal A → Signal B → Noise → calculate)
- [x] pyqtgraph main canvas with scroll-to-zoom and drag-to-pan
- [x] Full histogram tab: ROI histogram, CDF overlay, multi-source comparison, PNG export
- [x] Collapsible sidebar sections
- [x] Resizable panels via QSplitter
- [x] Dual viewport thumbnails
- [x] DICOM file loading (single frame)
- [x] Save processed image (PNG, JPEG, BMP)
- [x] Split debounce: view operations 30ms, compute operations 120ms

---

## In progress

- [ ] Fix SNR/CNR coordinate mapping edge cases with pyqtgraph ROI
- [ ] Window/level control via left-click drag (radiologist standard interaction)
- [ ] Demo GIF and screenshots in README and docs

---

## Planned — short term

- [ ] DICOM metadata inspector (patient info, acquisition parameters)
- [ ] Brightness and contrast sliders directly on the main canvas
- [ ] Keyboard shortcuts for common operations (load, reset ROI, calculate)
- [ ] Pixel value readout on mouse hover
- [ ] Export ROI statistics as CSV

---

## Planned — longer term

- [ ] Multi-frame DICOM (cine playback)
- [ ] NIfTI file support (`.nii`, `.nii.gz`)
- [ ] 3D volume rendering (MPR — axial, sagittal, coronal views)
- [ ] Measurement tools (distance, angle, area in physical units)
- [ ] Batch processing mode (apply pipeline to a folder of images)
- [ ] Plugin architecture for custom processing steps

---

## Will not implement

- **Clinical annotation tools** — MediPixel is a teaching and research tool,
  not a clinical workstation. Arrow annotations, text labels, and diagnostic
  reporting are out of scope. Use OsiriX, 3D Slicer, or Horos for clinical work.

- **Cloud or network DICOM (PACS integration)** — out of scope for a local
  desktop application.

---

## Suggest a feature

Open a [Feature Request issue](https://github.com/BasselShaheen06/MediPixel/issues/new?template=feature_request.md)
on GitHub. Describe what you want to do and why the current tool doesn't support it.