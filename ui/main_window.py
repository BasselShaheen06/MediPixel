"""
MedicalImageApp — the main application window for MediPixel.

Responsibilities:
  - Build and wire the Qt UI
  - Delegate all image-processing work to core/
  - Keep UI state in sync with processed images

What this file must NOT do:
  - Implement any signal/noise/contrast algorithm directly
  - Import cv2, skimage, or scipy
"""

import numpy as np
import pydicom
from PIL import Image

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QGridLayout, QVBoxLayout, QHBoxLayout,
    QFrame, QPushButton, QLabel, QSlider, QComboBox, QSpinBox,
    QScrollArea, QFileDialog, QApplication,
)
from PyQt5.QtCore import Qt
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from medipixel.core import noise, denoiser, contrast
from medipixel.ui.canvas import DraggableCanvas
from medipixel.ui.histogram import HistogramWindow

# ── Layout constants ──────────────────────────────────────────────────────────
_WINDOW_W = 1400
_WINDOW_H = 900
_CONTROL_W = 210
_CANVAS_MIN = 400
_ROI_SIZE = 20          # Fixed ROI side length in pixels


class MedicalImageApp(QMainWindow):
    """Main window: loads an image, applies processing, shows three viewports."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("MediPixel")
        self.setGeometry(100, 100, _WINDOW_W, _WINDOW_H)

        # ── Image state ───────────────────────────────────────────────────────
        self.original_image: np.ndarray | None = None
        # Per-viewport processed images (index 1 and 2)
        self.viewport_images: dict[int, np.ndarray | None] = {1: None, 2: None}

        # ── ROI state ─────────────────────────────────────────────────────────
        self._roi_state = "idle"      # idle → signal → signal2 → noise → done
        self._signal_coords = None
        self._signal2_coords = None
        self._noise_coords = None
        self._roi_patches: list = []  # all Rectangle patches added to axes

        # ── Build UI ──────────────────────────────────────────────────────────
        self._build_ui()
        self._connect_signals()

    # =========================================================================
    # UI construction
    # =========================================================================

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        grid = QGridLayout(root)
        grid.setSpacing(5)

        # Control panel
        self._ctrl = self._build_control_panel()
        grid.addWidget(self._ctrl, 0, 0, 2, 1)

        # Three canvases (main viewport + 2 side viewports)
        self._figures = [Figure() for _ in range(3)]
        self._canvases = [DraggableCanvas(fig) for fig in self._figures]
        self._axes = [fig.add_subplot(111) for fig in self._figures]

        for canvas in self._canvases:
            canvas.setMinimumSize(_CANVAS_MIN, _CANVAS_MIN)

        def _scroll(canvas):
            sa = QScrollArea()
            sa.setWidget(canvas)
            sa.setWidgetResizable(True)
            return sa

        grid.addWidget(_scroll(self._canvases[0]), 0, 1, 2, 1)
        grid.addWidget(_scroll(self._canvases[1]), 0, 2)
        grid.addWidget(_scroll(self._canvases[2]), 1, 2)

        grid.setColumnStretch(0, 0)
        grid.setColumnStretch(1, 2)
        grid.setColumnStretch(2, 1)

        # Histogram window (hidden by default)
        self._histogram = HistogramWindow()

    def _build_control_panel(self) -> QFrame:
        panel = QFrame()
        panel.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        panel.setFixedWidth(_CONTROL_W)
        layout = QVBoxLayout(panel)
        layout.setSpacing(6)
        layout.setContentsMargins(8, 8, 8, 8)

        _btn = lambda text: self._styled_button(text)

        # File
        self._load_btn = _btn("Load image")
        layout.addWidget(self._load_btn)

        # ROI
        self._status_lbl = QLabel("Load an image to start")
        self._status_lbl.setWordWrap(True)
        self._status_lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._status_lbl)

        self._roi_signal_btn  = _btn("Select signal ROI")
        self._roi_signal2_btn = _btn("Select signal ROI 2")
        self._roi_noise_btn   = _btn("Select noise ROI")
        self._calc_snr_btn    = _btn("Calculate SNR")
        self._calc_cnr_btn    = _btn("Calculate CNR")
        self._reset_btn       = _btn("Reset ROIs")

        for btn in [self._roi_signal_btn, self._roi_signal2_btn,
                    self._roi_noise_btn, self._calc_snr_btn,
                    self._calc_cnr_btn, self._reset_btn]:
            layout.addWidget(btn)

        layout.addWidget(self._hsep())

        # Viewport selector
        layout.addWidget(QLabel("Apply processing to:"))
        self._viewport_sel = QComboBox()
        self._viewport_sel.addItems(["Viewport 1", "Viewport 2"])
        layout.addWidget(self._viewport_sel)

        layout.addWidget(self._hsep())

        # Resolution / FOV
        layout.addWidget(QLabel("Resolution (subsampling):"))
        self._resolution = QComboBox()
        self._resolution.addItems([str(i) for i in range(1, 11)])
        layout.addWidget(self._resolution)

        layout.addWidget(QLabel("FOV window (pixels):"))
        self._fov_spin = QSpinBox()
        self._fov_spin.setRange(10, 5000)
        self._fov_spin.setValue(400)
        layout.addWidget(self._fov_spin)

        layout.addWidget(self._hsep())

        # Zoom + interpolation
        layout.addWidget(QLabel("Zoom:"))
        self._zoom_slider = QSlider(Qt.Horizontal)
        self._zoom_slider.setRange(1, 10)
        self._zoom_slider.setValue(1)
        self._zoom_slider.setTickPosition(QSlider.TicksBelow)
        self._zoom_slider.setTickInterval(1)
        layout.addWidget(self._zoom_slider)

        layout.addWidget(QLabel("Interpolation:"))
        self._interp = QComboBox()
        self._interp.addItems(["Nearest", "Bilinear", "Cubic"])
        self._interp.setCurrentIndex(1)
        layout.addWidget(self._interp)

        layout.addWidget(self._hsep())

        # Noise
        layout.addWidget(QLabel("Noise type:"))
        self._noise_type = QComboBox()
        self._noise_type.addItems(["None", "Gaussian", "Salt & Pepper", "Poisson"])
        layout.addWidget(self._noise_type)

        layout.addWidget(QLabel("Noise strength:"))
        self._noise_strength = QSlider(Qt.Horizontal)
        self._noise_strength.setRange(1, 100)
        self._noise_strength.setValue(25)
        layout.addWidget(self._noise_strength)

        layout.addWidget(QLabel("Denoising:"))
        self._denoise_method = QComboBox()
        self._denoise_method.addItems(["None", "Median", "Bilateral", "Non-local Means"])
        layout.addWidget(self._denoise_method)

        layout.addWidget(self._hsep())

        # Frequency filter
        layout.addWidget(QLabel("Frequency filter:"))
        self._filter_type = QComboBox()
        self._filter_type.addItems(["None", "Lowpass", "Highpass"])
        layout.addWidget(self._filter_type)

        layout.addWidget(QLabel("Cutoff frequency (%):"))
        self._cutoff = QSlider(Qt.Horizontal)
        self._cutoff.setRange(1, 100)
        self._cutoff.setValue(50)
        layout.addWidget(self._cutoff)

        layout.addWidget(QLabel("Filter order:"))
        self._filter_order = QSpinBox()
        self._filter_order.setRange(1, 10)
        self._filter_order.setValue(2)
        layout.addWidget(self._filter_order)

        layout.addWidget(self._hsep())

        # Contrast
        layout.addWidget(QLabel("Contrast enhancement:"))
        self._contrast_method = QComboBox()
        self._contrast_method.addItems(
            ["None", "Histogram equalization", "CLAHE", "Adaptive gamma"]
        )
        layout.addWidget(self._contrast_method)

        layout.addWidget(QLabel("CLAHE clip limit:"))
        self._clahe_clip = QSlider(Qt.Horizontal)
        self._clahe_clip.setRange(1, 50)
        self._clahe_clip.setValue(20)
        layout.addWidget(self._clahe_clip)

        layout.addWidget(QLabel("CLAHE grid size:"))
        self._clahe_grid = QSlider(Qt.Horizontal)
        self._clahe_grid.setRange(2, 16)
        self._clahe_grid.setValue(8)
        layout.addWidget(self._clahe_grid)

        layout.addWidget(self._hsep())

        # Histogram source + button
        layout.addWidget(QLabel("Histogram source:"))
        self._hist_source = QComboBox()
        self._hist_source.addItems(["Main viewport", "Viewport 1", "Viewport 2"])
        layout.addWidget(self._hist_source)

        self._show_hist_btn = self._styled_button("Show histogram")
        layout.addWidget(self._show_hist_btn)

        layout.addStretch()
        return panel

    # ── Small helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _styled_button(text: str) -> QPushButton:
        btn = QPushButton(text)
        btn.setStyleSheet(
            "QPushButton { min-height:22px; padding:2px 6px; font-size:12px;"
            " background:#f0f0f0; border:1px solid #ccc; border-radius:3px; }"
            "QPushButton:hover { background:#e0e0e0; }"
            "QPushButton:disabled { color:#aaa; }"
        )
        return btn

    @staticmethod
    def _hsep() -> QFrame:
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        return sep

    # =========================================================================
    # Signal wiring
    # =========================================================================

    def _connect_signals(self):
        self._load_btn.clicked.connect(self._load_image)

        self._roi_signal_btn.clicked.connect(lambda: self._start_roi("signal"))
        self._roi_signal2_btn.clicked.connect(lambda: self._start_roi("signal2"))
        self._roi_noise_btn.clicked.connect(lambda: self._start_roi("noise"))
        self._calc_snr_btn.clicked.connect(self._calculate_snr)
        self._calc_cnr_btn.clicked.connect(self._calculate_cnr)
        self._reset_btn.clicked.connect(self._reset_rois)

        self._show_hist_btn.clicked.connect(self._show_histogram)
        self._hist_source.currentIndexChanged.connect(self._refresh_histogram)

        # Every processing parameter triggers a reprocess
        for widget in [
            self._resolution, self._viewport_sel, self._noise_type,
            self._denoise_method, self._filter_type, self._contrast_method,
            self._interp,
        ]:
            widget.currentIndexChanged.connect(self._process)

        for slider in [
            self._zoom_slider, self._noise_strength, self._cutoff,
            self._clahe_clip, self._clahe_grid,
        ]:
            slider.valueChanged.connect(self._process)

        self._fov_spin.valueChanged.connect(self._process)
        self._filter_order.valueChanged.connect(self._process)

    # =========================================================================
    # Image processing pipeline  ← single code path, no duplication
    # =========================================================================

    def _process(self):
        """
        Run the full processing pipeline and update the target viewport.

        Pipeline order:
          1. Subsample (resolution)
          2. Add noise
          3. Denoise
          4. Frequency filter
          5. Contrast enhancement
          6. Zoom + interpolation
          7. Display + FOV crop
        """
        if self.original_image is None:
            return

        scale = int(self._resolution.currentText())
        img = self.original_image[::scale, ::scale].copy()

        # 1. Noise
        noise_choice = self._noise_type.currentText()
        strength = self._noise_strength.value()
        if noise_choice == "Gaussian":
            img = noise.add_gaussian(img, sigma=float(strength))
        elif noise_choice == "Salt & Pepper":
            img = noise.add_salt_and_pepper(img, prob=strength / 500.0)
        elif noise_choice == "Poisson":
            img = noise.add_poisson(img, scale=strength / 25.0)

        # 2. Denoise
        denoise_choice = self._denoise_method.currentText()
        if denoise_choice == "Median":
            img = denoiser.median(img)
        elif denoise_choice == "Bilateral":
            img = denoiser.bilateral(img)
        elif denoise_choice == "Non-local Means":
            img = denoiser.non_local_means(img)

        # 3. Frequency filter
        filter_choice = self._filter_type.currentText()
        if filter_choice != "None":
            img = contrast.frequency_filter(
                img,
                kind=filter_choice.lower(),
                cutoff=self._cutoff.value() / 100.0,
                order=self._filter_order.value(),
            )

        # 4. Contrast enhancement
        contrast_choice = self._contrast_method.currentText()
        if contrast_choice == "Histogram equalization":
            img = contrast.histogram_equalization(img)
        elif contrast_choice == "CLAHE":
            img = contrast.clahe(
                img,
                clip_limit=self._clahe_clip.value() / 10.0,
                tile_grid_size=(self._clahe_grid.value(), self._clahe_grid.value()),
            )
        elif contrast_choice == "Adaptive gamma":
            img = contrast.adaptive_gamma(img)

        # 5. Zoom + interpolation
        zoom = self._zoom_slider.value()
        if zoom > 1:
            interp_map = {
                "Nearest": Image.Resampling.NEAREST,
                "Bilinear": Image.Resampling.BILINEAR,
                "Cubic": Image.Resampling.BICUBIC,
            }
            pil_img = Image.fromarray(img)
            pil_img = pil_img.resize(
                (img.shape[1] * zoom, img.shape[0] * zoom),
                interp_map.get(self._interp.currentText(), Image.Resampling.BILINEAR),
            )
            img = np.array(pil_img)

        # 6. Store + display
        target = self._viewport_sel.currentIndex() + 1
        self.viewport_images[target] = img
        ax = self._axes[target]
        canvas = self._canvases[target]

        self._display(img, ax, canvas)

        # 7. FOV crop via axis limits
        h, w = img.shape
        fov = self._fov_spin.value()
        cx, cy = w // 2, h // 2
        half = fov // 2
        ax.set_xlim(max(0, cx - half), min(w, cx + half))
        ax.set_ylim(min(h, cy + half), max(0, cy - half))
        canvas.draw()

        if self._histogram.isVisible():
            self._refresh_histogram()

    # =========================================================================
    # Image loading + display
    # =========================================================================

    def _load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load image", "",
            "Medical images (*.dcm *.png *.jpg *.jpeg *.bmp *.tiff);;All files (*)",
        )
        if not path:
            return

        if path.lower().endswith(".dcm"):
            self.original_image = pydicom.dcmread(path).pixel_array.astype(np.uint8)
        else:
            self.original_image = np.array(Image.open(path).convert("L"))

        self._display(self.original_image, self._axes[0], self._canvases[0])
        self._status_lbl.setText("Image loaded — select signal ROI to begin")

    def _display(self, image: np.ndarray, ax, canvas):
        ax.clear()
        ax.imshow(image, cmap="gray")
        ax.axis("on")
        ax.set_xlim(0, image.shape[1])
        ax.set_ylim(image.shape[0], 0)
        canvas.draw()

    # =========================================================================
    # ROI selection
    # =========================================================================

    def _start_roi(self, which: str):
        """Arm the canvas click handler for a specific ROI type."""
        self._roi_state = which
        self._status_lbl.setText(f"Click image to place {which.replace('2', ' 2')} ROI")
        self._cid = self._figures[0].canvas.mpl_connect(
            "button_press_event", self._on_roi_click
        )

    def _on_roi_click(self, event):
        if event.inaxes != self._axes[0] or event.xdata is None:
            return

        x, y = int(event.xdata), int(event.ydata)
        s = _ROI_SIZE
        colors = {"signal": "red", "signal2": "blue", "noise": "green"}
        color = colors.get(self._roi_state, "white")

        for ax in self._axes:
            patch = Rectangle((x, y), s, s, linewidth=1,
                               edgecolor=color, facecolor="none")
            ax.add_patch(patch)
            self._roi_patches.append(patch)

        for canvas in self._canvases:
            canvas.draw()

        if self._roi_state == "signal":
            self._signal_coords = (x, y, s, s)
            self._roi_state = "idle"
            self._status_lbl.setText("Signal ROI set — select signal ROI 2")
        elif self._roi_state == "signal2":
            self._signal2_coords = (x, y, s, s)
            self._roi_state = "idle"
            self._status_lbl.setText("Signal ROI 2 set — select noise ROI")
        elif self._roi_state == "noise":
            self._noise_coords = (x, y, s, s)
            self._roi_state = "done"
            self._status_lbl.setText("All ROIs set — ready to calculate SNR / CNR")

        self._figures[0].canvas.mpl_disconnect(self._cid)

    def _reset_rois(self):
        for patch in self._roi_patches:
            patch.remove()
        self._roi_patches.clear()
        self._signal_coords = self._signal2_coords = self._noise_coords = None
        self._roi_state = "idle"
        self._status_lbl.setText("ROIs cleared")
        for canvas in self._canvases:
            canvas.draw()

    # =========================================================================
    # SNR / CNR calculations
    # =========================================================================

    def _roi_pixels(self, image: np.ndarray, coords) -> np.ndarray:
        x, y, w, h = coords
        return image[y : y + h, x : x + w]

    def _calculate_snr(self):
        if self._signal_coords is None or self._noise_coords is None:
            self._status_lbl.setText("Place signal and noise ROIs first")
            return

        lines = ["SNR results:"]
        for label, img in [
            ("Main", self.original_image),
            ("Viewport 1", self.viewport_images[1]),
            ("Viewport 2", self.viewport_images[2]),
        ]:
            if img is None:
                continue
            sig = self._roi_pixels(img, self._signal_coords).mean()
            nse = self._roi_pixels(img, self._noise_coords).std()
            snr = sig / nse if nse > 0 else float("inf")
            lines.append(f"  {label}: {snr:.2f}")

        self._status_lbl.setText("\n".join(lines))

    def _calculate_cnr(self):
        if any(c is None for c in [
            self._signal_coords, self._signal2_coords, self._noise_coords
        ]):
            self._status_lbl.setText("Place all three ROIs first")
            return

        lines = ["CNR results:"]
        for label, img in [
            ("Main", self.original_image),
            ("Viewport 1", self.viewport_images[1]),
            ("Viewport 2", self.viewport_images[2]),
        ]:
            if img is None:
                continue
            s1 = self._roi_pixels(img, self._signal_coords).mean()
            s2 = self._roi_pixels(img, self._signal2_coords).mean()
            nse = self._roi_pixels(img, self._noise_coords).std()
            cnr = abs(s1 - s2) / nse if nse > 0 else float("inf")
            lines.append(f"  {label}: {cnr:.2f}")

        self._status_lbl.setText("\n".join(lines))

    # =========================================================================
    # Histogram
    # =========================================================================

    def _show_histogram(self):
        self._refresh_histogram()
        self._histogram.show()
        self._histogram.raise_()

    def _refresh_histogram(self):
        source = self._hist_source.currentText()
        img_map = {
            "Main viewport": self.original_image,
            "Viewport 1": self.viewport_images[1],
            "Viewport 2": self.viewport_images[2],
        }
        img = img_map.get(source)
        if img is not None:
            self._histogram.update(img, source)
