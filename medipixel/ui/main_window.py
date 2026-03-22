"""
MedicalImageApp — MediPixel main window.
Color scheme: Clean Teal (D)

Main canvas: pyqtgraph ImageView
  - scroll to zoom, right-click drag to pan, native
  - RectROI for interactive ROI boxes (movable + resizable with handles)
  - Guided ROI workflow: Signal A → confirm → Signal B → confirm → Noise → confirm

Thumbnails + histogram: matplotlib (unchanged)

Debounce:
  view (zoom/FOV/resolution) → 30ms
  compute (noise/filter/contrast) → 120ms
"""

import numpy as np
import pydicom
from PIL import Image

import pyqtgraph as pg
from pyqtgraph import ImageView, ROI, RectROI, ImageItem

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QFrame, QPushButton, QLabel, QSlider, QComboBox, QSpinBox,
    QFileDialog, QSplitter, QScrollArea, QSizePolicy,
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.patches import FancyBboxPatch

from medipixel.core import noise, denoiser, contrast
from medipixel.ui.canvas import DraggableCanvas

# ── pyqtgraph global config ───────────────────────────────────────────────────
pg.setConfigOptions(imageAxisOrder="row-major", antialias=True)
pg.setConfigOption("background", "#FFFFFF")
pg.setConfigOption("foreground", "#0D1B1A")

# ── Palette ───────────────────────────────────────────────────────────────────
C_BG         = "#F4F7F7"
C_SURFACE    = "#FFFFFF"
C_SIDEBAR    = "#F4F7F7"
C_ACCENT     = "#00695C"
C_ACCENT_H   = "#00897B"
C_ACCENT_L   = "#E0F2F1"
C_TEXT       = "#0D1B1A"
C_TEXT_SEC   = "#5A7A78"
C_TEXT_TER   = "#9ABCB8"
C_BORDER     = "#E0ECEB"
C_BORDER_MED = "#C4D8D6"
C_CARD_BG    = "#F0F7F6"

# ROI pen colors
ROI_CFG = {
    "signal_a": {"pen": pg.mkPen("#B71C1C", width=2),
                 "hpen": pg.mkPen("#EF5350", width=2),
                 "label": "Signal A"},
    "signal_b": {"pen": pg.mkPen("#1565C0", width=2),
                 "hpen": pg.mkPen("#42A5F5", width=2),
                 "label": "Signal B"},
    "noise":    {"pen": pg.mkPen("#1B5E20", width=2),
                 "hpen": pg.mkPen("#66BB6A", width=2),
                 "label": "Noise"},
}
ROI_ORDER = ["signal_a", "signal_b", "noise"]

FONT         = 13
FONT_S       = 12
FONT_XS      = 11
SIDEBAR_W    = 272
RIGHT_W      = 286

DEBOUNCE_VIEW    = 30
DEBOUNCE_COMPUTE = 120


# =============================================================================
# Style helpers (identical to previous version)
# =============================================================================

def _combo(items, w=None):
    cb = QComboBox()
    cb.addItems(items)
    cb.setFixedHeight(28)
    if w: cb.setFixedWidth(w)
    cb.setStyleSheet(f"""
        QComboBox {{
            background:{C_SURFACE}; color:{C_TEXT};
            border:1px solid {C_BORDER_MED}; border-radius:5px;
            padding:0 8px; font-size:{FONT_S}px;
        }}
        QComboBox:hover {{ border-color:{C_ACCENT}; }}
        QComboBox::drop-down {{ border:none; width:16px; }}
        QComboBox QAbstractItemView {{
            background:{C_SURFACE}; color:{C_TEXT};
            border:1px solid {C_BORDER_MED};
            selection-background-color:{C_ACCENT_L};
            selection-color:{C_ACCENT};
            font-size:{FONT_S}px;
        }}
    """)
    return cb


def _spinbox(lo, hi, val, w=72):
    sp = QSpinBox()
    sp.setRange(lo, hi)
    sp.setValue(val)
    sp.setFixedHeight(28)
    sp.setFixedWidth(w)
    sp.setStyleSheet(f"""
        QSpinBox {{
            background:{C_SURFACE}; color:{C_TEXT};
            border:1px solid {C_BORDER_MED}; border-radius:5px;
            padding:0 6px; font-size:{FONT_S}px;
        }}
        QSpinBox:hover {{ border-color:{C_ACCENT}; }}
    """)
    return sp


def _slider(lo, hi, val):
    sl = QSlider(Qt.Horizontal)
    sl.setRange(lo, hi)
    sl.setValue(val)
    sl.setFixedHeight(20)
    sl.setStyleSheet(f"""
        QSlider::groove:horizontal {{
            height:3px; background:{C_BORDER}; border-radius:2px;
        }}
        QSlider::sub-page:horizontal {{
            background:{C_ACCENT}; border-radius:2px;
        }}
        QSlider::handle:horizontal {{
            width:14px; height:14px; margin:-6px 0;
            background:{C_SURFACE}; border-radius:7px;
            border:1.5px solid {C_ACCENT};
        }}
        QSlider::handle:horizontal:hover {{ background:{C_ACCENT_L}; }}
        QSlider::handle:horizontal:pressed {{ background:{C_ACCENT}; }}
    """)
    return sl


def _btn_primary(text, h=30):
    b = QPushButton(text)
    b.setFixedHeight(h)
    b.setStyleSheet(f"""
        QPushButton {{
            background:{C_ACCENT}; color:#FFF;
            border:none; border-radius:6px;
            font-size:{FONT_S}px; font-weight:600; padding:0 14px;
        }}
        QPushButton:hover  {{ background:{C_ACCENT_H}; }}
        QPushButton:pressed{{ background:#004D40; }}
        QPushButton:disabled{{ background:{C_BORDER}; color:{C_TEXT_TER}; }}
    """)
    return b


def _btn_secondary(text, h=30):
    b = QPushButton(text)
    b.setFixedHeight(h)
    b.setStyleSheet(f"""
        QPushButton {{
            background:{C_SURFACE}; color:{C_TEXT};
            border:1px solid {C_BORDER_MED}; border-radius:6px;
            font-size:{FONT_S}px; padding:0 14px;
        }}
        QPushButton:hover  {{ border-color:{C_ACCENT}; color:{C_ACCENT}; }}
        QPushButton:pressed{{ background:{C_BORDER}; }}
        QPushButton:disabled{{ color:{C_TEXT_TER}; }}
    """)
    return b


def _lbl(text, size=FONT_S, color=None, bold=False):
    l = QLabel(text)
    l.setStyleSheet(
        f"color:{color or C_TEXT}; font-size:{size}px; "
        f"font-weight:{'600' if bold else '400'}; background:transparent;"
    )
    return l


def _hsep():
    f = QFrame()
    f.setFrameShape(QFrame.HLine)
    f.setFixedHeight(1)
    f.setStyleSheet(f"background:{C_BORDER}; border:none;")
    return f


def _slider_row(label, lo, hi, val):
    w = QWidget()
    w.setStyleSheet("background:transparent;")
    lay = QVBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(2)
    top = QHBoxLayout()
    top.setContentsMargins(0, 0, 0, 0)
    top.addWidget(_lbl(label, size=FONT_XS, color=C_TEXT_SEC))
    top.addStretch()
    val_lbl = _lbl(str(val), size=FONT_XS, bold=True)
    val_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
    top.addWidget(val_lbl)
    sl = _slider(lo, hi, val)
    sl.valueChanged.connect(lambda v, vl=val_lbl: vl.setText(str(v)))
    lay.addLayout(top)
    lay.addWidget(sl)
    return w, sl


def _inline(label, widget):
    w = QWidget()
    w.setStyleSheet("background:transparent;")
    lay = QHBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(8)
    lay.addWidget(_lbl(label, size=FONT_XS, color=C_TEXT_SEC))
    lay.addStretch()
    lay.addWidget(widget)
    return w


# =============================================================================
# Card + Section (same as before)
# =============================================================================

class Card(QWidget):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            Card {{
                background:{C_SURFACE};
                border:1px solid {C_BORDER};
                border-left:3px solid {C_ACCENT};
                border-radius:6px;
            }}
        """)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 8, 12, 10)
        lay.setSpacing(6)
        lbl = QLabel(title)
        lbl.setStyleSheet(f"""
            color:{C_ACCENT}; font-size:{FONT_XS}px;
            font-weight:700; letter-spacing:0.5px;
            background:transparent; margin-bottom:1px;
        """)
        lay.addWidget(lbl)
        self._lay = lay

    def add(self, w):        self._lay.addWidget(w)
    def add_layout(self, l): self._lay.addLayout(l)


class Section(QWidget):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background:transparent;")
        self._collapsed = False
        self._title = title
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        self._btn = QPushButton(f"▾  {title}")
        self._btn.setFixedHeight(30)
        self._btn.setStyleSheet(f"""
            QPushButton {{
                background:transparent; color:{C_TEXT};
                border:none; text-align:left;
                font-size:{FONT}px; font-weight:600; padding:0 2px;
            }}
            QPushButton:hover {{ color:{C_ACCENT}; }}
        """)
        self._btn.clicked.connect(self._toggle)
        lay.addWidget(self._btn)
        self._body = QWidget()
        self._body.setStyleSheet("background:transparent;")
        self._body_lay = QVBoxLayout(self._body)
        self._body_lay.setContentsMargins(0, 0, 0, 0)
        self._body_lay.setSpacing(6)
        lay.addWidget(self._body)

    def add_card(self, c): self._body_lay.addWidget(c)

    def _toggle(self):
        self._collapsed = not self._collapsed
        self._body.setVisible(not self._collapsed)
        self._btn.setText(
            f"{'▸' if self._collapsed else '▾'}  {self._title}"
        )


# =============================================================================
# Main window
# =============================================================================

class MedicalImageApp(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("MediPixel")
        self.setGeometry(60, 60, 1580, 940)
        self.setStyleSheet(f"QMainWindow {{ background:{C_BG}; }}")

        # Image state
        self.original_image: np.ndarray | None = None
        self.viewport_images: dict[int, np.ndarray | None] = {1: None, 2: None}

        # Debounce timers
        self._timer_view = QTimer()
        self._timer_view.setSingleShot(True)
        self._timer_view.setInterval(DEBOUNCE_VIEW)
        self._timer_view.timeout.connect(self._process)

        self._timer_compute = QTimer()
        self._timer_compute.setSingleShot(True)
        self._timer_compute.setInterval(DEBOUNCE_COMPUTE)
        self._timer_compute.timeout.connect(self._process)

        # ROI state — guided workflow
        # _roi_step: index into ROI_ORDER (0=signal_a, 1=signal_b, 2=noise, 3=done)
        self._roi_step: int = -1
        self._pg_rois: dict[str, RectROI] = {}

        self._current_tab = 0

        self._build_ui()
        self._connect_signals()

    # =========================================================================
    # Build UI
    # =========================================================================

    def _build_ui(self):
        root = QWidget()
        root.setStyleSheet(f"background:{C_BG};")
        self.setCentralWidget(root)
        lay = QVBoxLayout(root)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self._build_toolbar())

        self._splitter = QSplitter(Qt.Horizontal)
        self._splitter.setHandleWidth(3)
        self._splitter.setStyleSheet(f"""
            QSplitter::handle {{ background:{C_BORDER}; }}
            QSplitter::handle:hover {{ background:{C_ACCENT}; }}
        """)
        self._splitter.addWidget(self._build_sidebar())
        self._splitter.addWidget(self._build_centre())
        self._splitter.addWidget(self._build_right())
        self._splitter.setSizes([SIDEBAR_W, 1010, RIGHT_W])
        self._splitter.setCollapsible(0, True)
        self._splitter.setCollapsible(2, True)
        lay.addWidget(self._splitter, stretch=1)

    # ── Toolbar ───────────────────────────────────────────────────────────────

    def _build_toolbar(self):
        bar = QWidget()
        bar.setFixedHeight(48)
        bar.setStyleSheet(
            f"background:{C_SURFACE}; border-bottom:1px solid {C_BORDER};"
        )
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(16, 0, 16, 0)
        lay.setSpacing(10)

        lay.addWidget(_lbl("MediPixel", size=15, bold=True))
        lay.addWidget(_lbl("·", size=15, color=C_TEXT_TER))
        lay.addWidget(_lbl("Medical Image Processing Workstation",
                           size=FONT_S, color=C_TEXT_SEC))
        lay.addStretch()

        # ROI step indicator
        self._step_pill = QLabel("")
        self._step_pill.setVisible(False)
        self._step_pill.setStyleSheet(f"""
            color:{C_ACCENT}; background:{C_ACCENT_L};
            border:1px solid {C_BORDER_MED};
            border-radius:10px; padding:2px 12px;
            font-size:{FONT_XS}px; font-weight:600;
        """)
        lay.addWidget(self._step_pill)

        self._sidebar_toggle = _btn_secondary("⇤  Hide sidebar", h=32)
        self._sidebar_toggle.setFixedWidth(134)
        self._sidebar_toggle.clicked.connect(self._toggle_sidebar)
        lay.addWidget(self._sidebar_toggle)

        self._right_toggle = _btn_secondary("Thumbnails  ⇥", h=32)
        self._right_toggle.setFixedWidth(124)
        self._right_toggle.clicked.connect(self._toggle_right)
        lay.addWidget(self._right_toggle)

        self._load_btn = _btn_primary("Load image", h=32)
        self._save_btn = _btn_secondary("Save result", h=32)
        lay.addWidget(self._load_btn)
        lay.addWidget(self._save_btn)
        return bar

    # ── Sidebar ───────────────────────────────────────────────────────────────

    def _build_sidebar(self):
        outer = QWidget()
        outer.setMinimumWidth(180)
        outer.setStyleSheet(f"background:{C_SIDEBAR};")
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet(f"""
            QScrollArea {{ border:none; background:{C_SIDEBAR}; }}
            QScrollBar:vertical {{ background:transparent; width:4px; }}
            QScrollBar::handle:vertical {{
                background:{C_BORDER_MED}; border-radius:2px; min-height:20px;
            }}
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {{ height:0; background:none; }}
        """)
        inner = QWidget()
        inner.setStyleSheet(f"background:{C_SIDEBAR};")
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(12, 14, 12, 14)
        lay.setSpacing(10)

        # Pipeline
        pipeline = Section("Processing pipeline")

        c = Card("Target viewport")
        self._viewport_sel = _combo(["Viewport 1", "Viewport 2"])
        c.add(self._viewport_sel)
        pipeline.add_card(c)

        c = Card("Noise")
        self._noise_type = _combo(
            ["None", "Gaussian", "Salt & Pepper", "Poisson"])
        c.add(self._noise_type)
        w, self._noise_strength = _slider_row("Strength", 1, 100, 25)
        c.add(w)
        pipeline.add_card(c)

        c = Card("Denoising")
        self._denoise_method = _combo(
            ["None", "Median", "Bilateral", "Non-local Means"])
        c.add(self._denoise_method)
        pipeline.add_card(c)

        c = Card("Frequency filter")
        self._filter_type = _combo(["None", "Lowpass", "Highpass"])
        c.add(self._filter_type)
        w, self._cutoff = _slider_row("Cutoff %", 1, 100, 50)
        c.add(w)
        self._filter_order = _spinbox(1, 10, 2)
        c.add(_inline("Order", self._filter_order))
        pipeline.add_card(c)

        c = Card("Contrast enhancement")
        self._contrast_method = _combo(
            ["None", "Histogram equalization", "CLAHE", "Adaptive gamma"])
        c.add(self._contrast_method)
        w, self._clahe_clip = _slider_row("CLAHE clip", 1, 50, 20)
        c.add(w)
        w, self._clahe_grid = _slider_row("CLAHE grid", 2, 16, 8)
        c.add(w)
        pipeline.add_card(c)
        lay.addWidget(pipeline)
        lay.addWidget(_hsep())

        # View
        view = Section("View")
        c = Card("Display settings")
        self._resolution = _combo([str(i) for i in range(1, 11)], w=60)
        c.add(_inline("Resolution", self._resolution))
        self._interp = _combo(["Nearest", "Bilinear", "Cubic"], w=90)
        self._interp.setCurrentIndex(1)
        c.add(_inline("Interpolation", self._interp))
        view.add_card(c)
        lay.addWidget(view)
        lay.addWidget(_hsep())

        # ROI & Metrics
        roi_sec = Section("ROI & Metrics")

        c = Card("Guided ROI selection")
        self._roi_status = QLabel("Load an image to start")
        self._roi_status.setWordWrap(True)
        self._roi_status.setStyleSheet(
            f"color:{C_TEXT_SEC}; font-size:{FONT_XS}px; "
            f"font-style:italic; background:transparent;"
        )
        c.add(self._roi_status)

        # Start button + confirm button
        self._start_roi_btn   = _btn_primary("Start ROI selection", h=28)
        self._confirm_roi_btn = _btn_primary("✓  Confirm position",  h=28)
        self._confirm_roi_btn.setVisible(False)
        self._reset_roi_btn   = _btn_secondary("Reset all ROIs",     h=28)
        c.add(self._start_roi_btn)
        c.add(self._confirm_roi_btn)
        c.add(self._reset_roi_btn)
        roi_sec.add_card(c)

        # ROI indicators — show current state of each ROI
        c = Card("ROI status")
        self._roi_indicators: dict[str, QLabel] = {}
        for key in ROI_ORDER:
            lbl = QLabel(f"{ROI_CFG[key]['label']}:  not set")
            color = ROI_CFG[key]["pen"].color().name()
            lbl.setStyleSheet(
                f"color:{color}; font-size:{FONT_XS}px; background:transparent;"
            )
            c.add(lbl)
            self._roi_indicators[key] = lbl
        roi_sec.add_card(c)

        c = Card("Metrics")
        mr = QHBoxLayout()
        mr.setSpacing(6)
        self._calc_snr_btn = _btn_primary("SNR", h=26)
        self._calc_cnr_btn = _btn_primary("CNR", h=26)
        self._calc_snr_btn.setEnabled(False)
        self._calc_cnr_btn.setEnabled(False)
        mr.addWidget(self._calc_snr_btn)
        mr.addWidget(self._calc_cnr_btn)
        mw = QWidget()
        mw.setStyleSheet("background:transparent;")
        mw.setLayout(mr)
        c.add(mw)

        self._metric_display = QLabel("")
        self._metric_display.setWordWrap(True)
        self._metric_display.setStyleSheet(f"""
            color:{C_TEXT}; font-size:{FONT_S}px;
            background:{C_CARD_BG};
            border:1px solid {C_BORDER};
            border-radius:5px; padding:8px 10px; min-height:44px;
        """)
        c.add(self._metric_display)
        roi_sec.add_card(c)
        lay.addWidget(roi_sec)
        lay.addStretch()

        scroll.setWidget(inner)
        ol = QVBoxLayout(outer)
        ol.setContentsMargins(0, 0, 0, 0)
        ol.addWidget(scroll)
        return outer

    # ── Centre (tab bar + pyqtgraph canvas + histogram) ───────────────────────

    def _build_centre(self):
        w = QWidget()
        w.setStyleSheet(f"background:{C_BG};")
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self._build_tab_bar())

        # ── pyqtgraph ImageView ────────────────────────────────────────────────
        self._pg_view = ImageView()
        # Strip default pyqtgraph UI elements we don't need
        self._pg_view.ui.histogram.hide()
        self._pg_view.ui.roiBtn.hide()
        self._pg_view.ui.menuBtn.hide()
        self._pg_view.ui.roiPlot.hide()

        # Style the pyqtgraph canvas to match our palette
        self._pg_view.setStyleSheet(f"background:{C_SURFACE};")
        self._pg_view.getView().setBackgroundColor(C_SURFACE)
        self._pg_view.getView().setMenuEnabled(False)

        self._canvas_wrap = self._pg_view
        lay.addWidget(self._canvas_wrap, stretch=1)

        # Draw placeholder text using a pyqtgraph TextItem
        self._placeholder = pg.TextItem(
            "Load an image to begin",
            color=C_TEXT_SEC, anchor=(0.5, 0.5),
        )
        self._placeholder.setFont(
            __import__("PyQt5.QtGui", fromlist=["QFont"]).QFont(
                "sans-serif", 13
            )
        )
        self._pg_view.addItem(self._placeholder)
        self._placeholder.setPos(0.5, 0.5)

        # ── Histogram wrap (matplotlib, hidden by default) ────────────────────
        self._hist_wrap = QWidget()
        self._hist_wrap.setStyleSheet(f"background:{C_BG};")
        self._hist_wrap.setVisible(False)
        hw = QVBoxLayout(self._hist_wrap)
        hw.setContentsMargins(16, 16, 16, 12)
        hw.setSpacing(10)
        self._hist_fig = Figure(facecolor=C_SURFACE)
        self._hist_canvas = FigureCanvas(self._hist_fig)
        hw.addWidget(self._hist_canvas, stretch=1)

        hc = QHBoxLayout()
        hc.addWidget(_lbl("Source:", color=C_TEXT_SEC, size=FONT_S))
        self._hist_source = _combo(
            ["Original", "Viewport 1", "Viewport 2",
             "ROI — Signal A", "ROI — Signal B", "ROI — Noise"],
            w=160,
        )
        hc.addWidget(self._hist_source)

        self._compare_btn = _btn_secondary("Overlay all", h=28)
        self._compare_btn.setFixedWidth(96)
        hc.addWidget(self._compare_btn)

        self._cdf_btn = _btn_secondary("Show CDF", h=28)
        self._cdf_btn.setCheckable(True)
        self._cdf_btn.setFixedWidth(88)
        hc.addWidget(self._cdf_btn)

        self._export_hist_btn = _btn_secondary("Export PNG", h=28)
        self._export_hist_btn.setFixedWidth(96)
        hc.addWidget(self._export_hist_btn)

        hc.addStretch()
        hw.addLayout(hc)
        lay.addWidget(self._hist_wrap, stretch=1)
        return w

    def _build_tab_bar(self):
        bar = QWidget()
        bar.setFixedHeight(40)
        bar.setStyleSheet(
            f"background:{C_SURFACE}; border-bottom:1px solid {C_BORDER};"
        )
        # Use a plain QHBoxLayout with no stretch between buttons
        # and explicit minimum widths — this is the tab clip fix
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        self._tab_btns = []
        # Explicit widths: wider than text + 32px padding each side
        tab_defs = [
            ("Original",   110),
            ("Viewport 1", 110),
            ("Viewport 2", 110),
            ("Histogram",  110),
        ]
        tab_style = f"""
            QPushButton {{
                background:transparent; color:{C_TEXT_SEC};
                border:none;
                border-bottom:2px solid transparent;
                border-right:1px solid {C_BORDER};
                border-radius:0;
                font-size:{FONT}px;
            }}
            QPushButton:hover {{ color:{C_TEXT}; background:{C_CARD_BG}; }}
            QPushButton:checked {{
                color:{C_ACCENT};
                border-bottom:2px solid {C_ACCENT};
                font-weight:600;
                background:{C_SURFACE};
            }}
        """
        for i, (name, width) in enumerate(tab_defs):
            btn = QPushButton(name)
            btn.setCheckable(True)
            btn.setChecked(i == 0)
            btn.setFixedHeight(40)
            btn.setFixedWidth(width)   # fixed, not minimum — cannot be squeezed
            btn.setStyleSheet(tab_style)
            btn.clicked.connect(lambda _, idx=i: self._switch_tab(idx))
            lay.addWidget(btn)
            self._tab_btns.append(btn)

        lay.addStretch()
        return bar

    # ── Right panel ───────────────────────────────────────────────────────────

    def _build_right(self):
        w = QWidget()
        w.setMinimumWidth(160)
        w.setStyleSheet(
            f"background:{C_SURFACE}; border-left:1px solid {C_BORDER};"
        )
        lay = QVBoxLayout(w)
        lay.setContentsMargins(10, 14, 10, 14)
        lay.setSpacing(10)
        lay.addWidget(_lbl("Viewports", bold=True))
        lay.addWidget(_hsep())

        self._vp_figs     = {}
        self._vp_canvases = {}
        self._vp_axes     = {}

        for i in [1, 2]:
            card = QWidget()
            card.setStyleSheet(
                f"background:{C_SURFACE}; border:1px solid {C_BORDER}; border-radius:6px;"
            )
            cl = QVBoxLayout(card)
            cl.setContentsMargins(0, 0, 0, 0)
            cl.setSpacing(0)

            lbar = QWidget()
            lbar.setFixedHeight(22)
            lbar.setStyleSheet(
                f"background:{C_CARD_BG}; border-bottom:1px solid {C_BORDER};"
                f"border-radius:6px 6px 0 0;"
            )
            ll = QHBoxLayout(lbar)
            ll.setContentsMargins(8, 0, 8, 0)
            vl = QLabel(f"Viewport {i}")
            vl.setStyleSheet(
                f"color:{C_ACCENT}; font-size:10px; font-weight:700; background:transparent;"
            )
            ll.addWidget(vl)
            cl.addWidget(lbar)

            fig = Figure(facecolor="#0D1B1A")
            canvas = DraggableCanvas(fig)
            canvas.setMinimumHeight(165)
            ax = fig.add_subplot(111)
            ax.set_facecolor("#0D1B1A")
            ax.axis("off")
            fig.tight_layout(pad=0.1)
            self._vp_figs[i]     = fig
            self._vp_canvases[i] = canvas
            self._vp_axes[i]     = ax
            cl.addWidget(canvas)
            lay.addWidget(card)

        lay.addStretch()
        return w

    # =========================================================================
    # Signal wiring
    # =========================================================================

    def _connect_signals(self):
        self._load_btn.clicked.connect(self._load_image)
        self._save_btn.clicked.connect(self._save_result)

        self._start_roi_btn.clicked.connect(self._start_roi_workflow)
        self._confirm_roi_btn.clicked.connect(self._confirm_roi_step)
        self._reset_roi_btn.clicked.connect(self._reset_rois)
        self._calc_snr_btn.clicked.connect(self._calculate_snr)
        self._calc_cnr_btn.clicked.connect(self._calculate_cnr)

        self._hist_source.currentIndexChanged.connect(self._refresh_histogram)
        self._compare_btn.clicked.connect(self._compare_histograms)
        self._cdf_btn.toggled.connect(self._refresh_histogram)
        self._export_hist_btn.clicked.connect(self._export_histogram)

        # View — fast debounce
        for w in [self._resolution, self._interp, self._viewport_sel]:
            w.currentIndexChanged.connect(lambda: self._timer_view.start())

        # Compute — slower debounce
        for w in [self._noise_type, self._denoise_method,
                  self._filter_type, self._contrast_method]:
            w.currentIndexChanged.connect(lambda: self._timer_compute.start())
        for sl in [self._noise_strength, self._cutoff,
                   self._clahe_clip, self._clahe_grid]:
            sl.valueChanged.connect(lambda: self._timer_compute.start())
        self._filter_order.valueChanged.connect(
            lambda: self._timer_compute.start()
        )

    # =========================================================================
    # Panel toggles
    # =========================================================================

    def _toggle_sidebar(self):
        sizes = self._splitter.sizes()
        if sizes[0] > 0:
            self._splitter.setSizes([0, sizes[1]+sizes[0], sizes[2]])
            self._sidebar_toggle.setText("⇥  Show sidebar")
        else:
            self._splitter.setSizes([SIDEBAR_W,
                                     max(sizes[1]-SIDEBAR_W, 400), sizes[2]])
            self._sidebar_toggle.setText("⇤  Hide sidebar")

    def _toggle_right(self):
        sizes = self._splitter.sizes()
        if sizes[2] > 0:
            self._splitter.setSizes([sizes[0], sizes[1]+sizes[2], 0])
            self._right_toggle.setText("⇤  Show thumbs")
        else:
            self._splitter.setSizes([sizes[0],
                                     max(sizes[1]-RIGHT_W, 400), RIGHT_W])
            self._right_toggle.setText("Thumbnails  ⇥")

    # =========================================================================
    # Tab switching
    # =========================================================================

    def _switch_tab(self, idx):
        self._current_tab = idx
        for i, btn in enumerate(self._tab_btns):
            btn.setChecked(i == idx)

        is_hist = (idx == 3)
        self._canvas_wrap.setVisible(not is_hist)
        self._hist_wrap.setVisible(is_hist)

        if is_hist:
            self._refresh_histogram()
            return

        img_map   = {0: self.original_image,
                     1: self.viewport_images[1],
                     2: self.viewport_images[2]}
        title_map = {0: "Original", 1: "Viewport 1", 2: "Viewport 2"}
        img = img_map.get(idx)
        if img is not None:
            self._show_image(img, title_map[idx])
        elif idx == 0 and self.original_image is None:
            self._pg_view.clear()
            self._placeholder.setVisible(True)

    # =========================================================================
    # pyqtgraph image display
    # =========================================================================

    def _show_image(self, image: np.ndarray, title: str = ""):
        """Display a numpy uint8 array in the pyqtgraph ImageView."""
        self._placeholder.setVisible(False)
        # pyqtgraph expects (rows, cols) for grayscale
        self._pg_view.setImage(
            image,
            autoRange=True,
            autoLevels=False,
            levels=(0, 255),
        )
        # Remove default colourmap so it stays grayscale
        self._pg_view.setColorMap(pg.colormap.get("grey", source="matplotlib"))

    # =========================================================================
    # Processing pipeline
    # =========================================================================

    def _process(self):
        if self.original_image is None:
            return

        scale = int(self._resolution.currentText())
        img   = self.original_image[::scale, ::scale].copy()

        nc = self._noise_type.currentText()
        st = self._noise_strength.value()
        if nc == "Gaussian":
            img = noise.add_gaussian(img, sigma=float(st))
        elif nc == "Salt & Pepper":
            img = noise.add_salt_and_pepper(img, prob=st / 500.0)
        elif nc == "Poisson":
            img = noise.add_poisson(img, scale=st / 25.0)

        dc = self._denoise_method.currentText()
        if dc == "Median":
            img = denoiser.median(img)
        elif dc == "Bilateral":
            img = denoiser.bilateral(img)
        elif dc == "Non-local Means":
            img = denoiser.non_local_means(img)

        fc = self._filter_type.currentText()
        if fc != "None":
            img = contrast.frequency_filter(
                img, kind=fc.lower(),
                cutoff=self._cutoff.value() / 100.0,
                order=self._filter_order.value(),
            )

        cc = self._contrast_method.currentText()
        if cc == "Histogram equalization":
            img = contrast.histogram_equalization(img)
        elif cc == "CLAHE":
            img = contrast.clahe(
                img,
                clip_limit=self._clahe_clip.value() / 10.0,
                tile_grid_size=(self._clahe_grid.value(),
                                self._clahe_grid.value()),
            )
        elif cc == "Adaptive gamma":
            img = contrast.adaptive_gamma(img)

        zoom = 1  # zoom handled natively by pyqtgraph scroll
        imap = {"Nearest":  Image.Resampling.NEAREST,
                "Bilinear": Image.Resampling.BILINEAR,
                "Cubic":    Image.Resampling.BICUBIC}
        # Interpolation applied only if resolution > 1 (upscale back)
        if scale > 1:
            img = np.array(
                Image.fromarray(img).resize(
                    (self.original_image.shape[1],
                     self.original_image.shape[0]),
                    imap.get(self._interp.currentText(),
                             Image.Resampling.BILINEAR),
                )
            )

        target = self._viewport_sel.currentIndex() + 1
        self.viewport_images[target] = img

        # Always update thumbnail
        self._display_thumb(img, self._vp_axes[target],
                            self._vp_canvases[target])

        # Update main canvas for whichever tab is currently visible
        if self._current_tab == 0:
            # On Original tab — don't overwrite, but show a hint
            pass
        elif self._current_tab == target:
            self._show_image(img, f"Viewport {target}")
        # If user is on a different viewport tab, don't disturb it —
        # it will refresh when they switch to it via _switch_tab

        if self._current_tab == 3:
            self._refresh_histogram()

    # =========================================================================
    # Load / Save
    # =========================================================================

    def _load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load image", "",
            "Images (*.dcm *.png *.jpg *.jpeg *.bmp *.tiff);;All files (*)",
        )
        if not path:
            return
        if path.lower().endswith(".dcm"):
            arr = pydicom.dcmread(path).pixel_array
            if arr.dtype != np.uint8:
                mn, mx = arr.min(), arr.max()
                arr = ((arr-mn)*255.0/max(mx-mn, 1)).astype(np.uint8)
            self.original_image = arr
        else:
            self.original_image = np.array(Image.open(path).convert("L"))

        self._show_image(self.original_image, "Original")
        self._switch_tab(0)
        self._roi_status.setText(
            "Click \"Start ROI selection\" to begin placing ROIs"
        )
        self._start_roi_btn.setEnabled(True)

    def _save_result(self):
        target = self._viewport_sel.currentIndex() + 1
        img = self.viewport_images.get(target)
        if img is None:
            self._roi_status.setText("Process an image first")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save image", "",
            "PNG (*.png);;JPEG (*.jpg);;BMP (*.bmp)",
        )
        if path:
            Image.fromarray(img).save(path)
            self._roi_status.setText(f"Saved: {path.split('/')[-1]}")

    # =========================================================================
    # Thumbnail display (matplotlib, right panel)
    # =========================================================================

    def _display_thumb(self, image, ax, canvas,
                       facecolor="#0D1B1A"):
        ax.clear()
        ax.set_facecolor(facecolor)
        ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        ax.axis("off")
        canvas.figure.tight_layout(pad=0.1)
        canvas.draw()

    # =========================================================================
    # Guided ROI workflow
    # =========================================================================

    def _start_roi_workflow(self):
        """Begin guided workflow from step 0 (Signal A)."""
        if self.original_image is None:
            return
        self._reset_rois()
        self._roi_step = 0
        self._place_roi_for_step()

    def _place_roi_for_step(self):
        """Place a RectROI for the current step and update UI."""
        if self._roi_step >= len(ROI_ORDER):
            self._finish_roi_workflow()
            return

        key   = ROI_ORDER[self._roi_step]
        cfg   = ROI_CFG[key]
        h, w  = self.original_image.shape

        # Place ROI in centre of image, sized ~15% of image dimensions
        rw = max(int(w * 0.15), 20)
        rh = max(int(h * 0.15), 20)
        x0 = (w - rw) // 2
        y0 = (h - rh) // 2

        roi = RectROI(
            [x0, y0], [rw, rh],
            pen=cfg["pen"],
            hoverPen=cfg["hpen"],
            handlePen=cfg["pen"],
            handleHoverPen=cfg["hpen"],
            removable=False,
            rotatable=False,
        )
        self._pg_view.addItem(roi)
        self._pg_rois[key] = roi

        # Update UI
        label = cfg["label"]
        self._roi_status.setText(
            f"Step {self._roi_step + 1}/3:  Position the {label} box, "
            f"then click ✓ Confirm"
        )
        self._step_pill.setText(f"Placing: {label}")
        self._step_pill.setVisible(True)
        self._confirm_roi_btn.setVisible(True)
        self._start_roi_btn.setVisible(False)

    def _confirm_roi_step(self):
        """User confirmed current ROI — advance to next step."""
        if self._roi_step < len(ROI_ORDER):
            key = ROI_ORDER[self._roi_step]
            # Mark indicator as set
            color = ROI_CFG[key]["pen"].color().name()
            x, y, w, h = self._get_roi_coords(key)
            self._roi_indicators[key].setText(
                f"{ROI_CFG[key]['label']}:  ({x}, {y})  {w}×{h}px  ✓"
            )
            self._roi_step += 1
            self._place_roi_for_step()

    def _finish_roi_workflow(self):
        """All three ROIs confirmed — enable metrics."""
        self._roi_status.setText(
            "All ROIs set. Adjust positions by dragging boxes or handles, "
            "then calculate SNR / CNR."
        )
        self._step_pill.setVisible(False)
        self._confirm_roi_btn.setVisible(False)
        self._start_roi_btn.setVisible(True)
        self._start_roi_btn.setText("Redo ROI selection")
        self._calc_snr_btn.setEnabled(True)
        self._calc_cnr_btn.setEnabled(True)

    def _reset_rois(self):
        """Remove all ROIs from the pyqtgraph view and reset state."""
        for roi in self._pg_rois.values():
            try:
                self._pg_view.removeItem(roi)
            except Exception:
                pass
        self._pg_rois.clear()
        self._roi_step = -1
        self._confirm_roi_btn.setVisible(False)
        self._start_roi_btn.setVisible(True)
        self._start_roi_btn.setText("Start ROI selection")
        self._step_pill.setVisible(False)
        self._calc_snr_btn.setEnabled(False)
        self._calc_cnr_btn.setEnabled(False)
        self._metric_display.setText("")
        for key in ROI_ORDER:
            self._roi_indicators[key].setText(
                f"{ROI_CFG[key]['label']}:  not set"
            )
        self._roi_status.setText("ROIs cleared")

    def _get_roi_coords(self, key: str):
        """
        Returns (x, y, w, h) in image pixel coordinates for the named ROI.
        pyqtgraph RectROI pos() is (col, row) = (x, y).
        """
        roi = self._pg_rois.get(key)
        if roi is None:
            return 0, 0, 1, 1
        pos  = roi.pos()
        size = roi.size()
        x = max(0, int(pos.x()))
        y = max(0, int(pos.y()))
        w = max(1, int(size.x()))
        h = max(1, int(size.y()))
        return x, y, w, h

    # =========================================================================
    # SNR / CNR
    # =========================================================================

    def _roi_pixels(self, image: np.ndarray, key: str) -> np.ndarray:
        """
        Extract pixels under a ROI using pyqtgraph's own coordinate mapping.
        getArrayRegion handles axis order, zoom, and pan correctly.
        Falls back to manual slice if something goes wrong.
        """
        roi = self._pg_rois.get(key)
        if roi is None:
            return np.array([0], dtype=np.uint8)
        try:
            img_item = self._pg_view.getImageItem()
            region = roi.getArrayRegion(image, img_item)
            if region is not None and region.size > 0:
                return region.astype(np.float32)
        except Exception:
            pass
        # Fallback: manual slice with clamped coords
        x, y, w, h = self._get_roi_coords(key)
        H, W = image.shape
        x2 = min(x + w, W)
        y2 = min(y + h, H)
        region = image[y:y2, x:x2]
        return region.astype(np.float32) if region.size > 0 else np.array([0.0])

    def _calculate_snr(self):
        if "signal_a" not in self._pg_rois or "noise" not in self._pg_rois:
            self._metric_display.setText("Complete ROI selection first")
            return
        lines = ["SNR"]
        for label, img in [("Original",   self.original_image),
                            ("Viewport 1", self.viewport_images[1]),
                            ("Viewport 2", self.viewport_images[2])]:
            if img is None: continue
            sig = self._roi_pixels(img, "signal_a").mean()
            nse = self._roi_pixels(img, "noise").std()
            snr = sig / nse if nse > 0 else float("inf")
            lines.append(f"{label}:  {snr:.2f}")
        self._metric_display.setText("\n".join(lines))

    def _calculate_cnr(self):
        if not all(k in self._pg_rois for k in ROI_ORDER):
            self._metric_display.setText("Complete ROI selection first")
            return
        lines = ["CNR"]
        for label, img in [("Original",   self.original_image),
                            ("Viewport 1", self.viewport_images[1]),
                            ("Viewport 2", self.viewport_images[2])]:
            if img is None: continue
            s1  = self._roi_pixels(img, "signal_a").mean()
            s2  = self._roi_pixels(img, "signal_b").mean()
            nse = self._roi_pixels(img, "noise").std()
            cnr = abs(s1-s2) / nse if nse > 0 else float("inf")
            lines.append(f"{label}:  {cnr:.2f}")
        self._metric_display.setText("\n".join(lines))

    # =========================================================================
    # Histogram (matplotlib) — full image, ROI, CDF, overlay, export
    # =========================================================================

    def _get_hist_pixels(self, source: str) -> tuple[np.ndarray | None, str]:
        """
        Returns (pixel_array, label) for the given source string.
        Handles both whole-image sources and ROI sources.
        """
        img_map = {
            "Original":   self.original_image,
            "Viewport 1": self.viewport_images[1],
            "Viewport 2": self.viewport_images[2],
        }
        roi_map = {
            "ROI — Signal A": "signal_a",
            "ROI — Signal B": "signal_b",
            "ROI — Noise":    "noise",
        }

        if source in img_map:
            return img_map[source], source

        if source in roi_map:
            key = roi_map[source]
            if key not in self._pg_rois:
                return None, source
            # Use the current tab's image as the base
            tab_img_map = {
                0: self.original_image,
                1: self.viewport_images[1],
                2: self.viewport_images[2],
            }
            base = tab_img_map.get(self._current_tab, self.original_image)
            if base is None:
                base = self.original_image
            if base is None:
                return None, source
            region = self._roi_pixels(base, key)
            return region.astype(np.uint8) if region is not None else None, source

        return None, source

    def _plot_histogram_on_ax(
        self,
        ax,
        pixels: np.ndarray,
        label: str,
        color: str,
        show_cdf: bool,
        alpha: float = 0.72,
        lw: float = 0,
    ):
        """
        Plot a histogram (and optionally its CDF) on the given axes.
        Uses a twin axis for the CDF so scales don't collide.
        """
        flat = pixels.flatten().astype(np.float32)
        hist, bins = np.histogram(flat, bins=256, range=(0, 255))

        if show_cdf:
            # Main axis: histogram (bar or line)
            ax.bar(bins[:-1], hist, width=1, color=color,
                   alpha=alpha, linewidth=lw, label=f"{label} histogram")

            # Twin axis: CDF
            ax2 = ax.twinx()
            cdf = np.cumsum(hist).astype(np.float64)
            cdf = cdf / cdf[-1] * 100.0   # normalise to %
            ax2.plot(bins[:-1], cdf, color=color,
                     linewidth=1.5, linestyle="--",
                     alpha=0.85, label=f"{label} CDF")
            ax2.set_ylabel("Cumulative %", color=C_TEXT_SEC, fontsize=10)
            ax2.tick_params(colors=C_TEXT_SEC, labelsize=9)
            ax2.set_ylim(0, 105)
            ax2.spines["right"].set_color(C_BORDER_MED)
            for sp in ["top", "left", "bottom"]:
                ax2.spines[sp].set_visible(False)
            return ax2
        else:
            ax.bar(bins[:-1], hist, width=1, color=color,
                   alpha=alpha, linewidth=lw, label=label)
            return None

    def _style_hist_ax(self, ax, title: str):
        ax.set_xlabel("Pixel intensity", color=C_TEXT_SEC, fontsize=11)
        ax.set_ylabel("Frequency",       color=C_TEXT_SEC, fontsize=11)
        ax.set_title(title,              color=C_TEXT,     fontsize=11)
        ax.tick_params(colors=C_TEXT_SEC, labelsize=10)
        for sp in ax.spines.values():
            sp.set_color(C_BORDER)
        ax.grid(True, alpha=0.3, color=C_BORDER)

    def _refresh_histogram(self):
        src      = self._hist_source.currentText()
        show_cdf = self._cdf_btn.isChecked()

        pixels, label = self._get_hist_pixels(src)
        if pixels is None:
            return

        self._hist_fig.clear()
        self._hist_fig.patch.set_facecolor(C_SURFACE)
        ax = self._hist_fig.add_subplot(111)
        ax.set_facecolor(C_SURFACE)

        flat = pixels.flatten()
        mean_v = flat.mean()
        std_v  = flat.std()
        min_v  = int(flat.min())
        max_v  = int(flat.max())

        self._plot_histogram_on_ax(
            ax, pixels, label, C_ACCENT, show_cdf
        )

        cdf_note = "  +CDF" if show_cdf else ""
        is_roi   = src.startswith("ROI")
        roi_note = "  [ROI region]" if is_roi else ""
        title    = (
            f"{label}{roi_note}{cdf_note}   "
            f"mean {mean_v:.1f}   std {std_v:.1f}   "
            f"[{min_v}, {max_v}]"
        )
        self._style_hist_ax(ax, title)
        self._hist_fig.tight_layout(pad=1.2)
        self._hist_canvas.draw()

    def _compare_histograms(self):
        """Overlay whole-image histograms for Original, VP1, VP2."""
        show_cdf = self._cdf_btn.isChecked()

        self._hist_fig.clear()
        self._hist_fig.patch.set_facecolor(C_SURFACE)
        ax = self._hist_fig.add_subplot(111)
        ax.set_facecolor(C_SURFACE)

        cfg = {
            "Original":   (C_TEXT,    1.8),
            "Viewport 1": (C_ACCENT,  1.5),
            "Viewport 2": ("#1565C0", 1.5),
        }
        img_map = {
            "Original":   self.original_image,
            "Viewport 1": self.viewport_images[1],
            "Viewport 2": self.viewport_images[2],
        }
        plotted = False
        for label, img in img_map.items():
            if img is None:
                continue
            color, lw = cfg[label]
            hist, bins = np.histogram(
                img.flatten(), bins=256, range=(0, 255))
            if show_cdf:
                ax.plot(bins[:-1], hist, color=color,
                        label=f"{label} hist", linewidth=lw, alpha=0.6)
                cdf = np.cumsum(hist).astype(np.float64)
                cdf = cdf / cdf[-1] * 100.0
                ax2 = ax.twinx()
                ax2.plot(bins[:-1], cdf, color=color,
                         linewidth=1.5, linestyle="--",
                         alpha=0.85, label=f"{label} CDF")
                ax2.set_ylabel("Cumulative %",
                                color=C_TEXT_SEC, fontsize=10)
                ax2.tick_params(colors=C_TEXT_SEC, labelsize=9)
                ax2.set_ylim(0, 105)
            else:
                ax.plot(bins[:-1], hist, color=color,
                        label=label, linewidth=lw, alpha=0.85)
            plotted = True

        if plotted:
            ax.legend(fontsize=10, framealpha=0.9)
            cdf_note = " + CDF" if show_cdf else ""
            self._style_hist_ax(ax, f"Histogram overlay{cdf_note}")
            self._hist_fig.tight_layout(pad=1.2)
            self._hist_canvas.draw()

    def _export_histogram(self):
        """Save the current histogram figure as a PNG."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Export histogram", "histogram.png",
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)",
        )
        if path:
            self._hist_fig.savefig(
                path, dpi=150, bbox_inches="tight",
                facecolor=C_SURFACE,
            )
            self._roi_status.setText(
                f"Histogram exported: {path.split('/')[-1]}"
            )