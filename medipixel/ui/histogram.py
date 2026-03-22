"""
HistogramWindow — a separate QMainWindow that plots pixel intensity
histograms with basic statistics.

Kept separate from main_window.py because it is an independent
display panel with its own lifecycle (show/hide without affecting
the main viewer).
"""

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QWidget, QLabel


class HistogramWindow(QMainWindow):
    """Displays a pixel-intensity histogram and basic statistics."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pixel Intensity Histogram")
        self.setGeometry(800, 100, 600, 420)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        self._figure = Figure(figsize=(6, 4), dpi=100)
        self._canvas = FigureCanvas(self._figure)
        self._ax = self._figure.add_subplot(111)
        layout.addWidget(self._canvas)

        self._stats_label = QLabel()
        layout.addWidget(self._stats_label)

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, image: np.ndarray, title: str = "") -> None:
        """
        Replot the histogram for *image*.

        Args:
            image: 2-D uint8 numpy array.
            title: Label shown above the plot.
        """
        self._ax.clear()

        hist, bins = np.histogram(image.flatten(), bins=256, range=(0, 255))
        self._ax.bar(bins[:-1], hist, width=1, color="steelblue", alpha=0.8)
        self._ax.set_title(f"Histogram — {title}" if title else "Histogram")
        self._ax.set_xlabel("Pixel intensity")
        self._ax.set_ylabel("Frequency")
        self._ax.set_ylim(0, hist.max() * 1.1)
        self._ax.grid(True, alpha=0.3)

        self._stats_label.setText(
            f"Mean: {image.mean():.1f}   "
            f"Median: {float(np.median(image)):.1f}   "
            f"Std: {image.std():.1f}   "
            f"Min: {int(image.min())}   "
            f"Max: {int(image.max())}"
        )
        self._canvas.draw()
