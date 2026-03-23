"""
MedicalImageApp — MediPixel main window.
Color scheme: Clean Teal (D)

New in this version:
  - Color/grayscale choice dialog on image load
  - Aspect-ratio locked pyqtgraph canvas
  - ROI overlays drawn on sidebar thumbnails
  - Per-series histogram color pickers (QColorDialog)
  - Tooltips on every control
  - Guided tour (GuidedTour class)
"""

import numpy as np
import pydicom
import re
from pathlib import Path
from PIL import Image

import pyqtgraph as pg
from pyqtgraph import ImageView, RectROI

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QFrame, QPushButton, QLabel, QSlider, QComboBox, QSpinBox,
    QFileDialog, QSplitter, QScrollArea, QSizePolicy, QDialog,
    QColorDialog, QApplication,
)
from PyQt5.QtCore import Qt, QTimer, QSettings, QRect, QPoint
from PyQt5.QtGui import QColor, QPainter, QPen, QFont, QFontDatabase

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.patches import FancyBboxPatch, Rectangle as MplRect

from medipixel.core import noise, denoiser, contrast
from medipixel.core.contrast import to_luminance
from medipixel.ui.canvas import DraggableCanvas

# ── pyqtgraph config ──────────────────────────────────────────────────────────
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

ROI_CFG = {
    "signal_a": {"pen": pg.mkPen("#B71C1C", width=2),
                 "hpen": pg.mkPen("#EF5350", width=2),
                 "label": "Signal A", "mpl_color": "#B71C1C"},
    "signal_b": {"pen": pg.mkPen("#1565C0", width=2),
                 "hpen": pg.mkPen("#42A5F5", width=2),
                 "label": "Signal B", "mpl_color": "#1565C0"},
    "noise":    {"pen": pg.mkPen("#1B5E20", width=2),
                 "hpen": pg.mkPen("#66BB6A", width=2),
                 "label": "Noise", "mpl_color": "#1B5E20"},
}
ROI_ORDER = ["signal_a", "signal_b", "noise"]

# Default histogram series colors
HIST_COLORS_DEFAULT = {
    "Original":   "#0D1B1A",
    "Viewport 1": "#00695C",
    "Viewport 2": "#1565C0",
}

FONT         = 13
FONT_S       = 12
FONT_XS      = 11
SIDEBAR_W    = 272
RIGHT_W      = 286
DEBOUNCE_VIEW    = 30
DEBOUNCE_COMPUTE = 120

APP_FONT_CHOICES = [
    "IBM Plex Serif",
    "Passero One",
    "Silkscreen",
]
FONT_SCALE_MIN = 70
FONT_SCALE_MAX = 170
FONT_SCALE_DEFAULT = 100


# =============================================================================
# Style helpers
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


def _slider_row(label, lo, hi, val, tooltip=""):
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
    if tooltip:
        sl.setToolTip(tooltip)
        w.setToolTip(tooltip)
    lay.addLayout(top)
    lay.addWidget(sl)
    return w, sl


def _inline(label, widget, tooltip=""):
    w = QWidget()
    w.setStyleSheet("background:transparent;")
    lay = QHBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(8)
    lay.addWidget(_lbl(label, size=FONT_XS, color=C_TEXT_SEC))
    lay.addStretch()
    lay.addWidget(widget)
    if tooltip:
        w.setToolTip(tooltip)
        widget.setToolTip(tooltip)
    return w


# =============================================================================
# Color swatch button for histogram color picking
# =============================================================================

class ColorSwatchBtn(QPushButton):
    """Small square button showing a color — click to open QColorDialog."""

    def __init__(self, color: str, parent=None):
        super().__init__(parent)
        self._color = color
        self.setFixedSize(22, 22)
        self._update_style()

    def _update_style(self):
        self.setStyleSheet(f"""
            QPushButton {{
                background:{self._color};
                border:1.5px solid {C_BORDER_MED};
                border-radius:4px;
            }}
            QPushButton:hover {{ border-color:{C_ACCENT}; }}
        """)

    def color(self) -> str:
        return self._color

    def mousePressEvent(self, event):
        qc = QColorDialog.getColor(
            QColor(self._color), self, "Choose color"
        )
        if qc.isValid():
            self._color = qc.name()
            self._update_style()
        super().mousePressEvent(event)


# =============================================================================
# Card + Section
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
# Color/Grayscale load dialog
# =============================================================================

class ColorModeDialog(QDialog):
    """
    Shown when loading a color image (PNG/JPEG with 3 channels).
    Lets the user choose between keeping color or converting to grayscale.
    """

    def __init__(self, thumbnail: np.ndarray, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Image color mode")
        self.setFixedWidth(480)
        self.setStyleSheet(f"background:{C_SURFACE};")
        self.choice = "grayscale"   # default

        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 20, 24, 20)
        lay.setSpacing(16)

        title = QLabel("How would you like to load this image?")
        title.setStyleSheet(
            f"color:{C_TEXT}; font-size:14px; font-weight:600;"
        )
        lay.addWidget(title)

        # Thumbnail preview
        if thumbnail is not None:
            try:
                from PIL import Image as PILImage
                from PyQt5.QtGui import QPixmap, QImage
                rgb = thumbnail if thumbnail.ndim == 3 else np.stack([thumbnail]*3, axis=2)
                h, w = rgb.shape[:2]
                qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
                pix  = QPixmap.fromImage(qimg).scaledToWidth(
                    432, Qt.SmoothTransformation
                )
                thumb_lbl = QLabel()
                thumb_lbl.setPixmap(pix)
                thumb_lbl.setAlignment(Qt.AlignCenter)
                lay.addWidget(thumb_lbl)
            except Exception:
                pass

        # Option cards
        btn_row = QHBoxLayout()
        btn_row.setSpacing(12)

        def _option_card(title_text, desc, value):
            card = QWidget()
            card.setStyleSheet(f"""
                QWidget {{
                    background:{C_CARD_BG};
                    border:2px solid {C_BORDER_MED};
                    border-radius:8px;
                }}
                QWidget:hover {{
                    border-color:{C_ACCENT};
                    background:{C_ACCENT_L};
                }}
            """)
            cl = QVBoxLayout(card)
            cl.setContentsMargins(14, 12, 14, 12)
            cl.setSpacing(4)

            t = QLabel(title_text)
            t.setStyleSheet(
                f"color:{C_TEXT}; font-size:13px; font-weight:600; background:transparent;"
            )
            d = QLabel(desc)
            d.setWordWrap(True)
            d.setStyleSheet(
                f"color:{C_TEXT_SEC}; font-size:11px; background:transparent;"
            )
            cl.addWidget(t)
            cl.addWidget(d)

            btn = QPushButton(f"Use {title_text.lower()}")
            btn.setStyleSheet(f"""
                QPushButton {{
                    background:{C_ACCENT}; color:#FFF;
                    border:none; border-radius:5px;
                    font-size:12px; font-weight:600;
                    padding:5px 0; margin-top:4px;
                }}
                QPushButton:hover {{ background:{C_ACCENT_H}; }}
            """)
            btn.clicked.connect(lambda _, v=value: self._choose(v))
            cl.addWidget(btn)
            return card

        btn_row.addWidget(_option_card(
            "Color",
            "Keep the original RGB channels. Processing operates on "
            "luminance for metrics — hue is preserved.",
            "color",
        ))
        btn_row.addWidget(_option_card(
            "Grayscale",
            "Convert to single-channel luminance. Standard for CT, MRI, "
            "and X-ray. Required if the image represents a single physical quantity.",
            "grayscale",
        ))
        lay.addLayout(btn_row)

    def _choose(self, value: str):
        self.choice = value
        self.accept()


# =============================================================================
# Guided tour
# =============================================================================

class TourStep:
    def __init__(self, target: QWidget, title: str, text: str):
        self.target = target
        self.title  = title
        self.text   = text


class GuidedTour(QWidget):
    """
    Semi-transparent overlay with a spotlight cutout and tooltip popup.
    Walks through a list of TourStep objects one by one.
    """

    def __init__(self, parent: QMainWindow):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.setStyleSheet("background:transparent;")
        self._steps: list[TourStep] = []
        self._idx   = 0
        self._active = False
        self._overlay_alpha = 112

        # Popup card
        self._popup = QWidget(self)
        self._popup.setStyleSheet(f"""
            QWidget {{
                background:{C_SURFACE};
                border:1px solid {C_BORDER_MED};
                border-radius:12px;
            }}
        """)
        self._popup.setFixedWidth(350)
        pl = QVBoxLayout(self._popup)
        pl.setContentsMargins(20, 18, 20, 18)
        pl.setSpacing(12)

        self._tour_title = QLabel()
        self._tour_title.setStyleSheet(
            f"color:{C_ACCENT}; font-size:17px; font-weight:800; background:transparent;"
        )
        self._tour_title.setWordWrap(True)

        self._msg_frame = QFrame()
        self._msg_frame.setStyleSheet(f"""
            QFrame {{
                background:{C_CARD_BG};
                border:1px solid {C_BORDER_MED};
                border-radius:10px;
            }}
        """)
        msg_lay = QVBoxLayout(self._msg_frame)
        msg_lay.setContentsMargins(14, 10, 14, 10)
        msg_lay.setSpacing(0)

        self._tour_text = QLabel()
        self._tour_text.setWordWrap(True)
        self._tour_text.setStyleSheet(
            f"color:{C_TEXT}; font-size:12px; background:transparent; line-height:1.5;"
        )
        msg_lay.addWidget(self._tour_text)

        nav = QHBoxLayout()
        self._skip_btn = QPushButton("Skip tour")
        self._skip_btn.setStyleSheet(f"""
            QPushButton {{
                background:transparent; color:{C_TEXT_SEC};
                border:1px solid {C_BORDER_MED}; border-radius:5px;
                font-size:11px; padding:4px 10px;
            }}
            QPushButton:hover {{ color:{C_ACCENT}; border-color:{C_ACCENT}; }}
        """)
        self._next_btn = QPushButton("Next  →")
        self._next_btn.setStyleSheet(f"""
            QPushButton {{
                background:{C_ACCENT}; color:#FFF;
                border:none; border-radius:5px;
                font-size:11px; font-weight:600; padding:4px 14px;
            }}
            QPushButton:hover {{ background:{C_ACCENT_H}; }}
        """)
        self._counter = QLabel()
        self._counter.setStyleSheet(
            f"color:{C_TEXT_TER}; font-size:10px; background:transparent;"
        )
        nav.addWidget(self._skip_btn)
        nav.addStretch()
        nav.addWidget(self._counter)
        nav.addWidget(self._next_btn)

        pl.addWidget(self._tour_title)
        pl.addWidget(self._msg_frame)
        pl.addLayout(nav)

        self._skip_btn.clicked.connect(self.end)
        self._next_btn.clicked.connect(self._advance)

        self.hide()

    def set_steps(self, steps: list[TourStep]):
        self._steps = steps

    def start(self):
        if not self._steps:
            return
        self._idx    = 0
        self._active = True
        self.show()
        self.raise_()
        self.resize(self.parent().size())
        self._show_step()

    def end(self):
        self._active = False
        self.hide()
        # Remember that tour was shown
        QSettings("MediPixel", "MediPixel").setValue("tour_shown", True)

    def _advance(self):
        self._idx += 1
        if self._idx >= len(self._steps):
            self.end()
        else:
            self._show_step()

    def _show_step(self):
        step = self._steps[self._idx]
        self._tour_title.setText(step.title)
        self._tour_text.setText(step.text)
        self._counter.setText(f"{self._idx + 1} / {len(self._steps)}")
        self._next_btn.setText(
            "Finish  ✓" if self._idx == len(self._steps) - 1 else "Next  →"
        )
        self.update()
        self._position_popup(step.target)

    def _position_popup(self, target: QWidget):
        self._popup.adjustSize()
        # Map target coordinates via global space (safe for non-ancestor widgets).
        pos  = self.mapFromGlobal(target.mapToGlobal(QPoint(0, 0)))
        rect = QRect(pos, target.size())

        pw = self._popup.width()
        ph = self._popup.height()
        ow = self.width()
        oh = self.height()

        # Prefer placing popup to the right of target, then below, then left
        x = rect.right() + 12
        y = rect.top()
        if x + pw > ow - 10:
            x = rect.left() - pw - 12
        if x < 10:
            x = rect.left()
            y = rect.bottom() + 12
        y = max(10, min(y, oh - ph - 10))

        self._popup.move(x, y)
        self._popup.show()
        self._popup.raise_()

    def paintEvent(self, event):
        if not self._active or not self._steps:
            super().paintEvent(event)
            return

        step = self._steps[self._idx]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        pos = self.mapFromGlobal(step.target.mapToGlobal(QPoint(0, 0)))
        rect = QRect(pos.x() - 6, pos.y() - 6,
                     step.target.width() + 12, step.target.height() + 12)
        rect = rect.intersected(self.rect())

        dim = QColor(0, 0, 0, self._overlay_alpha)
        if rect.isValid() and not rect.isNull():
            top_h = max(0, rect.top())
            left_w = max(0, rect.left())
            right_x = rect.right() + 1
            right_w = max(0, self.width() - right_x)
            bottom_y = rect.bottom() + 1
            bottom_h = max(0, self.height() - bottom_y)

            painter.fillRect(0, 0, self.width(), top_h, dim)
            painter.fillRect(0, rect.top(), left_w, rect.height(), dim)
            painter.fillRect(right_x, rect.top(), right_w, rect.height(), dim)
            painter.fillRect(0, bottom_y, self.width(), bottom_h, dim)
        else:
            painter.fillRect(self.rect(), dim)

        pen = QPen(QColor(C_ACCENT_L), 2)
        painter.setPen(pen)
        painter.drawRoundedRect(rect, 8, 8)

    def resizeEvent(self, event):
        parent = self.parentWidget()
        if parent is not None and self.size() != parent.size():
            self.resize(parent.size())
        super().resizeEvent(event)



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
        self._color_mode = False   # True = color, False = grayscale

        # Debounce timers
        self._timer_view = QTimer()
        self._timer_view.setSingleShot(True)
        self._timer_view.setInterval(DEBOUNCE_VIEW)
        self._timer_view.timeout.connect(self._process)

        self._timer_compute = QTimer()
        self._timer_compute.setSingleShot(True)
        self._timer_compute.setInterval(DEBOUNCE_COMPUTE)
        self._timer_compute.timeout.connect(self._process)

        # ROI state
        self._roi_step: int = -1
        self._pg_rois: dict[str, RectROI] = {}

        # Histogram series colors (mutable)
        self._hist_colors = dict(HIST_COLORS_DEFAULT)

        self._current_tab = 0

        self._build_ui()
        self._connect_signals()
        self._setup_tour()
        self._init_font_preference()

        # Auto-show tour on first launch
        settings = QSettings("MediPixel", "MediPixel")
        if not settings.value("tour_shown", False):
            QTimer.singleShot(800, self._tour.start)

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

        self._font_sel = _combo(APP_FONT_CHOICES, w=170)
        self._font_sel.setToolTip("Choose app font family")
        lay.addWidget(self._font_sel)

        self._font_scale = _spinbox(
            FONT_SCALE_MIN, FONT_SCALE_MAX, FONT_SCALE_DEFAULT, w=64
        )
        self._font_scale.setSuffix("%")
        self._font_scale.setToolTip(
            "Scale all UI font sizes proportionally"
        )
        lay.addWidget(_lbl("Font", size=FONT_XS, color=C_TEXT_SEC))
        lay.addWidget(self._font_scale)
        lay.addStretch()

        # Color mode badge
        self._mode_badge = QLabel("Grayscale")
        self._mode_badge.setStyleSheet(f"""
            color:{C_TEXT_SEC}; background:{C_CARD_BG};
            border:1px solid {C_BORDER_MED};
            border-radius:10px; padding:2px 10px;
            font-size:{FONT_XS}px;
        """)
        lay.addWidget(self._mode_badge)

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

        # Tour button
        self._tour_btn = _btn_secondary("?  Guide", h=32)
        self._tour_btn.setFixedWidth(80)
        self._tour_btn.setToolTip("Start the guided tour")
        lay.addWidget(self._tour_btn)

        self._sidebar_toggle = _btn_secondary("⇤  Hide sidebar", h=32)
        self._sidebar_toggle.setFixedWidth(134)
        self._sidebar_toggle.setToolTip("Hide or show the control sidebar")
        self._sidebar_toggle.clicked.connect(self._toggle_sidebar)
        lay.addWidget(self._sidebar_toggle)

        self._right_toggle = _btn_secondary("Thumbnails  ⇥", h=32)
        self._right_toggle.setFixedWidth(124)
        self._right_toggle.setToolTip("Hide or show the viewport thumbnails")
        self._right_toggle.clicked.connect(self._toggle_right)
        lay.addWidget(self._right_toggle)

        self._load_btn = _btn_primary("Load image", h=32)
        self._load_btn.setToolTip(
            "Load a DICOM, PNG, JPEG, BMP, or TIFF image"
        )
        self._save_btn = _btn_secondary("Save result", h=32)
        self._save_btn.setToolTip("Save the current processed viewport")
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
        self._viewport_sel.setToolTip(
            "Which viewport receives the processed result"
        )
        c.add(self._viewport_sel)
        pipeline.add_card(c)

        c = Card("Noise")
        self._noise_type = _combo(
            ["None", "Gaussian", "Salt & Pepper", "Poisson"])
        self._noise_type.setToolTip(
            "Gaussian: electronic/thermal noise\n"
            "Salt & Pepper: dead pixels / impulse noise\n"
            "Poisson: photon shot noise (signal-dependent)"
        )
        c.add(self._noise_type)
        w, self._noise_strength = _slider_row(
            "Strength", 1, 100, 25,
            tooltip="Noise intensity — sigma for Gaussian, fraction for S&P, dose for Poisson"
        )
        c.add(w)
        pipeline.add_card(c)

        c = Card("Denoising")
        self._denoise_method = _combo(
            ["None", "Median", "Bilateral", "Non-local Means"])
        self._denoise_method.setToolTip(
            "Median: best for salt & pepper\n"
            "Bilateral: edge-preserving, good for Gaussian\n"
            "Non-local Means: best detail preservation, slowest"
        )
        c.add(self._denoise_method)
        pipeline.add_card(c)

        c = Card("Frequency filter")
        self._filter_type = _combo(["None", "Lowpass", "Highpass"])
        self._filter_type.setToolTip(
            "Butterworth filter in Fourier domain\n"
            "Lowpass: removes high-frequency noise (blurs)\n"
            "Highpass: enhances edges and fine detail"
        )
        c.add(self._filter_type)
        w, self._cutoff = _slider_row(
            "Cutoff %", 1, 100, 50,
            tooltip="Normalised cutoff frequency. Low = more aggressive filtering."
        )
        c.add(w)
        self._filter_order = _spinbox(1, 10, 2)
        self._filter_order.setToolTip(
            "Butterworth order — higher = steeper roll-off, more ringing risk"
        )
        c.add(_inline("Order", self._filter_order))
        pipeline.add_card(c)

        c = Card("Contrast enhancement")
        self._contrast_method = _combo(
            ["None", "Histogram equalization", "CLAHE", "Adaptive gamma"])
        self._contrast_method.setToolTip(
            "Histogram EQ: global, maximises contrast, amplifies noise\n"
            "CLAHE: local per-tile, clip limit prevents noise amplification (recommended)\n"
            "Adaptive gamma: auto brightness correction"
        )
        c.add(self._contrast_method)
        w, self._clahe_clip = _slider_row(
            "CLAHE clip", 1, 50, 20,
            tooltip="Clip limit per tile. Low = less noise amplification. 20 = clip_limit 2.0"
        )
        c.add(w)
        w, self._clahe_grid = _slider_row(
            "CLAHE grid", 2, 16, 8,
            tooltip="Tile grid size. Larger = more localised enhancement"
        )
        c.add(w)
        pipeline.add_card(c)
        lay.addWidget(pipeline)
        lay.addWidget(_hsep())

        # View
        view = Section("View")
        c = Card("Display settings")
        self._resolution = _combo([str(i) for i in range(1, 11)], w=60)
        self._resolution.setToolTip(
            "Subsampling factor. 2 = display every 2nd pixel (faster processing)"
        )
        c.add(_inline("Resolution", self._resolution))
        self._interp = _combo(["Nearest", "Bilinear", "Cubic"], w=90)
        self._interp.setCurrentIndex(1)
        self._interp.setToolTip(
            "Interpolation when upscaling.\n"
            "Nearest: crisp pixels\n"
            "Bilinear: smooth\n"
            "Cubic: sharpest edges"
        )
        c.add(_inline("Interpolation", self._interp))
        view.add_card(c)
        lay.addWidget(view)
        lay.addWidget(_hsep())

        # ROI & Metrics
        roi_sec = Section("ROI & Metrics")

        c = Card("Region selection")
        self._roi_status = QLabel("Load an image to start")
        self._roi_status.setWordWrap(True)
        self._roi_status.setStyleSheet(
            f"color:{C_TEXT_SEC}; font-size:{FONT_XS}px; "
            f"font-style:italic; background:transparent;"
        )
        c.add(self._roi_status)

        hint = _lbl(
            "Select button → drag on canvas → confirm → next ROI",
            size=FONT_XS, color=C_TEXT_TER,
        )
        hint.setWordWrap(True)
        c.add(hint)

        grid = QGridLayout()
        grid.setSpacing(5)
        self._start_roi_btn   = _btn_primary("Start ROI selection", h=28)
        self._confirm_roi_btn = _btn_primary("✓  Confirm position",  h=28)
        self._confirm_roi_btn.setVisible(False)
        self._reset_roi_btn   = _btn_secondary("Reset all ROIs",     h=28)
        self._start_roi_btn.setToolTip(
            "Begin guided 3-step ROI placement:\n"
            "1. Signal A  2. Signal B  3. Noise region"
        )
        self._confirm_roi_btn.setToolTip(
            "Confirm current ROI position and move to next step"
        )
        self._reset_roi_btn.setToolTip("Remove all ROI boxes")
        c.add(self._start_roi_btn)
        c.add(self._confirm_roi_btn)
        c.add(self._reset_roi_btn)
        roi_sec.add_card(c)

        # ROI indicators
        c = Card("ROI status")
        self._roi_indicators: dict[str, QLabel] = {}
        for key in ROI_ORDER:
            lbl = QLabel(f"{ROI_CFG[key]['label']}:  not set")
            color = ROI_CFG[key]["mpl_color"]
            lbl.setStyleSheet(
                f"color:{color}; font-size:{FONT_XS}px; background:transparent;"
            )
            c.add(lbl)
            self._roi_indicators[key] = lbl
        roi_sec.add_card(c)

        # Metrics
        c = Card("Metrics")
        mr = QHBoxLayout()
        mr.setSpacing(6)
        self._calc_snr_btn = _btn_primary("SNR", h=26)
        self._calc_cnr_btn = _btn_primary("CNR", h=26)
        self._calc_snr_btn.setEnabled(False)
        self._calc_cnr_btn.setEnabled(False)
        self._calc_snr_btn.setToolTip(
            "SNR = mean(Signal A) / std(Noise)\n"
            "Requires Signal A and Noise ROIs"
        )
        self._calc_cnr_btn.setToolTip(
            "CNR = |mean(Signal A) - mean(Signal B)| / std(Noise)\n"
            "Requires all three ROIs"
        )
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

    # ── Centre ────────────────────────────────────────────────────────────────

    def _build_centre(self):
        w = QWidget()
        w.setStyleSheet(f"background:{C_BG};")
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self._build_tab_bar())

        self._canvas_wrap = QWidget()
        self._canvas_wrap.setStyleSheet(f"background:{C_SURFACE};")
        cw = QVBoxLayout(self._canvas_wrap)
        cw.setContentsMargins(0, 0, 0, 0)

        self._pg_view = ImageView()
        self._pg_view.ui.histogram.hide()
        self._pg_view.ui.roiBtn.hide()
        self._pg_view.ui.menuBtn.hide()
        self._pg_view.ui.roiPlot.hide()
        self._pg_view.setStyleSheet(f"background:{C_SURFACE};")
        self._pg_view.getView().setBackgroundColor(C_SURFACE)
        self._pg_view.getView().setMenuEnabled(False)

        # Lock aspect ratio so image proportions are always preserved
        self._pg_view.getView().setAspectLocked(True)

        self._placeholder = pg.TextItem(
            "Load an image to begin",
            color=C_TEXT_SEC, anchor=(0.5, 0.5),
        )
        self._pg_view.addItem(self._placeholder)
        self._placeholder.setPos(0.5, 0.5)

        cw.addWidget(self._pg_view)
        lay.addWidget(self._canvas_wrap, stretch=1)

        # Histogram panel
        self._hist_wrap = QWidget()
        self._hist_wrap.setStyleSheet(f"background:{C_BG};")
        self._hist_wrap.setVisible(False)
        hw = QVBoxLayout(self._hist_wrap)
        hw.setContentsMargins(16, 16, 16, 12)
        hw.setSpacing(10)
        self._hist_fig = Figure(facecolor=C_SURFACE)
        self._hist_canvas = FigureCanvas(self._hist_fig)
        self._hist_ax = self._hist_fig.add_subplot(111)
        hw.addWidget(self._hist_canvas, stretch=1)

        # Histogram controls — source, overlay, CDF, color pickers, export
        hc = QHBoxLayout()
        hc.setSpacing(8)
        hc.addWidget(_lbl("Source:", color=C_TEXT_SEC, size=FONT_S))

        self._hist_source = _combo(
            ["Original", "Viewport 1", "Viewport 2",
             "ROI — Signal A", "ROI — Signal B", "ROI — Noise"],
            w=160,
        )
        self._hist_source.setToolTip(
            "Choose which image or ROI region to plot"
        )
        hc.addWidget(self._hist_source)

        checkbox_style = f"""
            QCheckBox {{
                color:{C_TEXT}; font-size:{FONT_S}px; spacing:5px;
                background:transparent;
            }}
            QCheckBox::indicator {{
                width:16px; height:16px;
                border:1.5px solid {C_BORDER_MED};
                border-radius:4px; background:{C_SURFACE};
            }}
            QCheckBox::indicator:checked {{
                background:{C_ACCENT}; border-color:{C_ACCENT};
                image:none;
            }}
            QCheckBox::indicator:hover {{
                border-color:{C_ACCENT};
            }}
        """
        from PyQt5.QtWidgets import QCheckBox

        self._compare_cb = QCheckBox("Overlay all")
        self._compare_cb.setStyleSheet(checkbox_style)
        self._compare_cb.setToolTip(
            "Overlay Original, VP1, and VP2 on one chart"
        )
        hc.addWidget(self._compare_cb)

        self._cdf_btn = QCheckBox("Show CDF")
        self._cdf_btn.setStyleSheet(checkbox_style)
        self._cdf_btn.setToolTip(
            "Add cumulative distribution function on right axis"
        )
        hc.addWidget(self._cdf_btn)

        # Per-series color swatches
        hc.addWidget(_lbl("Colors:", color=C_TEXT_SEC, size=FONT_XS))
        self._hist_swatches: dict[str, ColorSwatchBtn] = {}
        for key, default_color in HIST_COLORS_DEFAULT.items():
            swatch = ColorSwatchBtn(default_color)
            swatch.setToolTip(f"Click to change {key} histogram color")
            swatch.clicked.connect(self._on_swatch_changed)
            hc.addWidget(swatch)
            label = QLabel(key.replace("Viewport ", "VP"))
            label.setStyleSheet(
                f"color:{C_TEXT_SEC}; font-size:10px; background:transparent;"
            )
            hc.addWidget(label)
            self._hist_swatches[key] = swatch

        self._export_hist_btn = _btn_secondary("Export PNG", h=28)
        self._export_hist_btn.setFixedWidth(96)
        self._export_hist_btn.setToolTip("Save histogram as PNG/PDF/SVG")
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
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        self._tab_btns = []
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
                border-radius:0; font-size:{FONT}px;
            }}
            QPushButton:hover {{ color:{C_TEXT}; background:{C_CARD_BG}; }}
            QPushButton:checked {{
                color:{C_ACCENT};
                border-bottom:2px solid {C_ACCENT};
                font-weight:600; background:{C_SURFACE};
            }}
        """
        for i, (name, width) in enumerate(tab_defs):
            btn = QPushButton(name)
            btn.setCheckable(True)
            btn.setChecked(i == 0)
            btn.setFixedHeight(40)
            btn.setFixedWidth(width)
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
    # Guided tour setup
    # =========================================================================

    def _setup_tour(self):
        self._tour = GuidedTour(self)
        # Steps are set after build_ui so widgets exist
        # We schedule step setup after show() so geometry is available
        QTimer.singleShot(200, self._init_tour_steps)
        self._tour_btn.clicked.connect(self._tour.start)

    def _init_tour_steps(self):
        # Find sidebar widget for targeting
        sidebar = self._splitter.widget(0)
        centre  = self._splitter.widget(1)
        right   = self._splitter.widget(2)

        steps = [
            TourStep(
                self._load_btn,
                "Welcome! Start By Loading An Image",
                "Click here to open a DICOM, PNG, JPEG, BMP, or TIFF image. "
                "If your image is color, MediPixel will help you choose color or grayscale in one step.",
            ),
            TourStep(
                sidebar,
                "Processing Pipeline",
                "This left panel is your workflow hub. "
                "Controls are grouped in sections, and you can collapse any section to keep things clean.",
            ),
            TourStep(
                centre,
                "Main canvas",
                "This is your main viewing area. "
                "Use the mouse wheel to zoom and right-click drag to pan. "
                "Tabs above let you move between Original, processed viewports, and Histogram.",
            ),
            TourStep(
                right,
                "Viewport Thumbnails",
                "Here you can quickly compare Viewport 1 and Viewport 2 side by side. "
                "ROI overlays also appear here for a fast visual check.",
            ),
            TourStep(
                self._start_roi_btn,
                "ROI Selection",
                "Click Start ROI selection and follow the guided steps: Signal A, Signal B, then Noise. "
                "You can drag each ROI box or its corners anytime to fine-tune placement.",
            ),
            TourStep(
                self._calc_snr_btn,
                "Metrics",
                "After ROI placement, compute SNR and CNR from here. "
                "Results are shown together for the Original image and both processed viewports.",
            ),
            TourStep(
                self._tab_btns[3],
                "Histogram Tab",
                "Use this tab to explore intensity distributions. "
                "You can inspect full images or ROIs, enable CDF, and overlay all series for comparison.",
            ),
        ]
        self._tour.set_steps(steps)

    # =========================================================================
    # Signal wiring
    # =========================================================================

    def _connect_signals(self):
        self._load_btn.clicked.connect(self._load_image)
        self._save_btn.clicked.connect(self._save_result)
        self._font_sel.currentTextChanged.connect(self._on_font_family_changed)
        self._font_scale.valueChanged.connect(self._on_font_scale_changed)

        self._start_roi_btn.clicked.connect(self._start_roi_workflow)
        self._confirm_roi_btn.clicked.connect(self._confirm_roi_step)
        self._reset_roi_btn.clicked.connect(self._reset_rois)
        self._calc_snr_btn.clicked.connect(self._calculate_snr)
        self._calc_cnr_btn.clicked.connect(self._calculate_cnr)

        self._hist_source.currentIndexChanged.connect(self._refresh_histogram)
        self._compare_cb.stateChanged.connect(self._on_hist_mode_changed)
        self._cdf_btn.stateChanged.connect(self._on_hist_mode_changed)
        self._export_hist_btn.clicked.connect(self._export_histogram)

        for w in [self._resolution, self._interp, self._viewport_sel]:
            w.currentIndexChanged.connect(lambda: self._timer_view.start())

        for w in [self._noise_type, self._denoise_method,
                  self._filter_type, self._contrast_method]:
            w.currentIndexChanged.connect(lambda: self._timer_compute.start())
        for sl in [self._noise_strength, self._cutoff,
                   self._clahe_clip, self._clahe_grid]:
            sl.valueChanged.connect(lambda: self._timer_compute.start())
        self._filter_order.valueChanged.connect(
            lambda: self._timer_compute.start()
        )

    def _on_swatch_changed(self):
        """Any color swatch changed — sync to _hist_colors and refresh."""
        for key, swatch in self._hist_swatches.items():
            self._hist_colors[key] = swatch.color()
        self._refresh_histogram()

    def _norm_font_name(self, name: str) -> str:
        return re.sub(r"[^a-z0-9]", "", name.lower())

    def _ensure_local_fonts_loaded(self):
        if getattr(self, "_local_fonts_loaded", False):
            return

        root = Path(__file__).resolve().parents[2]
        fonts_dir = root / "assets" / "fonts"
        if fonts_dir.exists():
            for p in fonts_dir.rglob("*"):
                if p.is_file() and p.suffix.lower() in {".ttf", ".otf"}:
                    QFontDatabase.addApplicationFont(str(p))

        self._local_fonts_loaded = True

    def _resolve_font_family(self, requested: str) -> str | None:
        self._ensure_local_fonts_loaded()
        families = list(QFontDatabase().families())
        if requested in families:
            return requested

        req_n = self._norm_font_name(requested)
        # First try direct fuzzy contains match.
        for fam in families:
            fam_n = self._norm_font_name(fam)
            if req_n and (req_n in fam_n or fam_n in req_n):
                return fam
        return None

    def _scale_stylesheet_fonts(self, stylesheet: str, scale: float) -> str:
        if not stylesheet:
            return stylesheet

        def _repl(match):
            v = float(match.group(1))
            sv = max(6.0, v * scale)
            sval = f"{sv:.1f}".rstrip("0").rstrip(".")
            return f"font-size:{sval}px"

        return re.sub(
            r"font-size\s*:\s*(\d+(?:\.\d+)?)px",
            _repl,
            stylesheet,
            flags=re.IGNORECASE,
        )

    def _apply_font_scale(self, scale_pct: int, save: bool = True):
        app = QApplication.instance()
        if app is None:
            return

        scale_pct = max(FONT_SCALE_MIN, min(FONT_SCALE_MAX, int(scale_pct)))
        scale = scale_pct / 100.0

        widgets = [self] + self.findChildren(QWidget)
        for w in widgets:
            base_ss = w.property("_base_stylesheet")
            if base_ss is None:
                base_ss = w.styleSheet()
                w.setProperty("_base_stylesheet", base_ss)
            w.setStyleSheet(self._scale_stylesheet_fonts(base_ss, scale))

            base_pt = w.property("_base_point_size")
            if base_pt is None:
                pt = w.font().pointSizeF()
                base_pt = pt if pt > 0 else 10.0
                w.setProperty("_base_point_size", float(base_pt))

            f = QFont(w.font())
            f.setPointSizeF(float(base_pt) * scale)
            w.setFont(f)

        if save:
            QSettings("MediPixel", "MediPixel").setValue(
                "font_scale_pct", scale_pct
            )

    def _apply_font_family(self, family: str, save: bool = True):
        app = QApplication.instance()
        if app is None:
            return
        resolved = self._resolve_font_family(family)
        if resolved is None:
            return

        base_size = app.font().pointSize() if app.font().pointSize() > 0 else 10
        font = QFont(resolved, base_size)
        app.setStyleSheet(
            f"QWidget {{ font-family: '{resolved}', 'Segoe UI', sans-serif; }}"
        )
        app.setFont(font)

        self.setFont(font)
        for child in self.findChildren(QWidget):
            child.setFont(font)

        # Reapply size scaling after family switch.
        self._apply_font_scale(self._font_scale.value(), save=False)

        if save:
            QSettings("MediPixel", "MediPixel").setValue("font_family", resolved)

    def _on_font_family_changed(self, family: str):
        self._apply_font_family(family, save=True)

    def _on_font_scale_changed(self, value: int):
        self._apply_font_scale(value, save=True)

    def _init_font_preference(self):
        settings = QSettings("MediPixel", "MediPixel")
        saved_family = settings.value("font_family", "IBM Plex Serif")
        saved_scale = settings.value("font_scale_pct", FONT_SCALE_DEFAULT)

        self._font_sel.blockSignals(True)
        self._font_scale.blockSignals(True)

        if isinstance(saved_family, str):
            resolved = self._resolve_font_family(saved_family)
            if resolved:
                for i in range(self._font_sel.count()):
                    txt = self._font_sel.itemText(i)
                    if self._resolve_font_family(txt) == resolved:
                        self._font_sel.setCurrentIndex(i)
                        break

        try:
            self._font_scale.setValue(int(saved_scale))
        except Exception:
            self._font_scale.setValue(FONT_SCALE_DEFAULT)

        self._font_sel.blockSignals(False)
        self._font_scale.blockSignals(False)

        self._apply_font_family(self._font_sel.currentText(), save=False)
        self._apply_font_scale(self._font_scale.value(), save=False)

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
    # Image display
    # =========================================================================

    def _show_image(self, image: np.ndarray, title: str = ""):
        """Display in pyqtgraph — handles both grayscale and color."""
        self._placeholder.setVisible(False)
        if image.ndim == 3:
            # pyqtgraph color: expects (H, W, 3) with row-major
            self._pg_view.setImage(
                image, autoRange=True, autoLevels=True,
            )
        else:
            self._pg_view.setImage(
                image, autoRange=True, autoLevels=False, levels=(0, 255),
            )
            self._pg_view.setColorMap(
                pg.colormap.get("grey", source="matplotlib")
            )

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

        # Upscale back if subsampled
        imap = {"Nearest":  Image.Resampling.NEAREST,
                "Bilinear": Image.Resampling.BILINEAR,
                "Cubic":    Image.Resampling.BICUBIC}
        if scale > 1:
            target_size = (self.original_image.shape[1],
                           self.original_image.shape[0])
            if img.ndim == 3:
                pil = Image.fromarray(img).resize(
                    target_size,
                    imap.get(self._interp.currentText(),
                             Image.Resampling.BILINEAR)
                )
                img = np.array(pil)
            else:
                pil = Image.fromarray(img).resize(
                    target_size,
                    imap.get(self._interp.currentText(),
                             Image.Resampling.BILINEAR)
                )
                img = np.array(pil)

        target = self._viewport_sel.currentIndex() + 1
        self.viewport_images[target] = img

        # Update thumbnail with ROI overlays
        self._display_thumb(img, self._vp_axes[target],
                            self._vp_canvases[target],
                            draw_rois=True)

        if self._current_tab == target:
            self._show_image(img, f"Viewport {target}")

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
            # DICOM is always grayscale
            self.original_image = arr
            self._color_mode = False
            self._mode_badge.setText("Grayscale  (DICOM)")
        else:
            raw = np.array(Image.open(path))
            if raw.ndim == 3 and raw.shape[2] >= 3:
                # Color image — ask user
                rgb = raw[:, :, :3]
                dlg = ColorModeDialog(rgb, self)
                dlg.exec_()
                if dlg.choice == "color":
                    self.original_image = rgb.astype(np.uint8)
                    self._color_mode = True
                    self._mode_badge.setText("Color  (RGB)")
                else:
                    gray = to_luminance(rgb.astype(np.uint8))
                    self.original_image = gray
                    self._color_mode = False
                    self._mode_badge.setText("Grayscale")
            else:
                # Already grayscale
                self.original_image = raw.astype(np.uint8)
                self._color_mode = False
                self._mode_badge.setText("Grayscale")

        self._show_image(self.original_image, "Original")
        self._switch_tab(0)
        self._roi_status.setText(
            "Click 'Start ROI selection' to begin placing ROIs"
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
    # Thumbnail display with ROI overlays
    # =========================================================================

    def _display_thumb(self, image, ax, canvas,
                       facecolor="#0D1B1A", draw_rois=False):
        ax.clear()
        ax.set_facecolor(facecolor)

        if image.ndim == 3:
            ax.imshow(image)
        else:
            ax.imshow(image, cmap="gray", vmin=0, vmax=255)

        ax.axis("off")

        # Overlay ROI boxes if requested and any are placed
        if draw_rois:
            h_img, w_img = image.shape[:2]
            for key, roi in self._pg_rois.items():
                if not roi.placed:
                    continue
                try:
                    x, y, w, ht = self._get_roi_coords(key)
                    # Clamp to image bounds
                    x  = max(0, min(x, w_img - 1))
                    y  = max(0, min(y, h_img - 1))
                    w  = max(1, min(w,  w_img - x))
                    ht = max(1, min(ht, h_img - y))
                    rect = MplRect(
                        (x, y), w, ht,
                        linewidth=1.5,
                        edgecolor=ROI_CFG[key]["mpl_color"],
                        facecolor="none",
                        linestyle="--",
                    )
                    ax.add_patch(rect)
                except Exception:
                    pass

        canvas.figure.tight_layout(pad=0.1)
        canvas.draw()

    # =========================================================================
    # ROI workflow
    # =========================================================================

    def _start_roi_workflow(self):
        if self.original_image is None:
            return
        self._reset_rois()
        self._roi_step = 0
        self._place_roi_for_step()

    def _place_roi_for_step(self):
        if self._roi_step >= len(ROI_ORDER):
            self._finish_roi_workflow()
            return

        key  = ROI_ORDER[self._roi_step]
        cfg  = ROI_CFG[key]
        h, w = self.original_image.shape[:2]

        rw = max(int(w * 0.15), 20)
        rh = max(int(h * 0.15), 20)
        x0 = (w - rw) // 2
        y0 = (h - rh) // 2

        roi = RectROI(
            [x0, y0], [rw, rh],
            pen=cfg["pen"], hoverPen=cfg["hpen"],
            handlePen=cfg["pen"], handleHoverPen=cfg["hpen"],
            removable=False, rotatable=False,
        )
        # Mark as placed immediately so thumbnails draw it
        roi.placed = True
        self._pg_view.addItem(roi)
        self._pg_rois[key] = roi

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
        if self._roi_step < len(ROI_ORDER):
            key = ROI_ORDER[self._roi_step]
            x, y, w, h = self._get_roi_coords(key)
            self._roi_indicators[key].setText(
                f"{ROI_CFG[key]['label']}:  ({x}, {y})  {w}×{h}px  ✓"
            )
            self._roi_step += 1
            self._place_roi_for_step()

    def _finish_roi_workflow(self):
        self._roi_status.setText(
            "All ROIs set. Drag boxes or handles to adjust, "
            "then calculate SNR / CNR."
        )
        self._step_pill.setVisible(False)
        self._confirm_roi_btn.setVisible(False)
        self._start_roi_btn.setVisible(True)
        self._start_roi_btn.setText("Redo ROI selection")
        self._calc_snr_btn.setEnabled(True)
        self._calc_cnr_btn.setEnabled(True)

        # Refresh thumbnails to show all ROI boxes
        for i in [1, 2]:
            if self.viewport_images[i] is not None:
                self._display_thumb(
                    self.viewport_images[i],
                    self._vp_axes[i], self._vp_canvases[i],
                    draw_rois=True,
                )

    def _reset_rois(self):
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

        # Refresh thumbnails to remove ROI overlays
        for i in [1, 2]:
            if self.viewport_images[i] is not None:
                self._display_thumb(
                    self.viewport_images[i],
                    self._vp_axes[i], self._vp_canvases[i],
                    draw_rois=False,
                )

    def _get_roi_coords(self, key: str):
        roi = self._pg_rois.get(key)
        if roi is None:
            return 0, 0, 1, 1
        pos  = roi.pos()
        size = roi.size()
        return (max(0, int(pos.x())),  max(0, int(pos.y())),
                max(1, int(size.x())), max(1, int(size.y())))

    # =========================================================================
    # SNR / CNR  (uses luminance channel for color images)
    # =========================================================================

    def _roi_pixels(self, image: np.ndarray, key: str) -> np.ndarray:
        roi = self._pg_rois.get(key)
        if roi is None:
            return np.array([0.0])

        # For color images, extract luminance before measuring
        if image.ndim == 3:
            meas_image = to_luminance(image)
        else:
            meas_image = image

        try:
            img_item = self._pg_view.getImageItem()
            region   = roi.getArrayRegion(meas_image, img_item)
            if region is not None and region.size > 0:
                return region.astype(np.float32)
        except Exception:
            pass

        # Fallback
        x, y, w, h = self._get_roi_coords(key)
        H, W = meas_image.shape[:2]
        x2 = min(x + w, W)
        y2 = min(y + h, H)
        region = meas_image[y:y2, x:x2]
        return region.astype(np.float32) if region.size > 0 else np.array([0.0])

    def _calculate_snr(self):
        if "signal_a" not in self._pg_rois or "noise" not in self._pg_rois:
            self._metric_display.setText("Complete ROI selection first")
            return
        note = "  (luminance channel)" if self._color_mode else ""
        lines = [f"SNR{note}"]
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
        note = "  (luminance channel)" if self._color_mode else ""
        lines = [f"CNR{note}"]
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
    # Histogram
    # =========================================================================

    def _get_hist_pixels(self, source: str):
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
            base = self.original_image
            if base is None:
                return None, source
            region = self._roi_pixels(base, key)
            return (region.astype(np.uint8)
                    if region is not None else None), source
        return None, source

    def _on_hist_mode_changed(self):
        """Route to overlay or single histogram based on checkbox states."""
        if self._compare_cb.isChecked():
            self._compare_histograms()
        else:
            self._refresh_histogram()

    def _prepare_hist_axes(self):
        """Reset histogram axes, including any previous twinx CDF axis."""
        for ax in list(self._hist_fig.axes):
            if ax is not self._hist_ax:
                self._hist_fig.delaxes(ax)
        self._hist_ax.clear()
        self._hist_fig.patch.set_facecolor(C_SURFACE)
        self._hist_ax.set_facecolor(C_SURFACE)

    def _refresh_histogram(self):
        src      = self._hist_source.currentText()
        show_cdf = self._cdf_btn.isChecked()

        pixels, label = self._get_hist_pixels(src)
        if pixels is None:
            return

        self._prepare_hist_axes()

        color = self._hist_colors.get(src, C_ACCENT)
        flat  = pixels.flatten().astype(np.float32)

        # For color images plot per-channel R/G/B lines
        if self._color_mode and pixels.ndim == 3 and "ROI" not in src:
            ch_colors = ["#C62828", "#2E7D32", "#1565C0"]
            ch_names  = ["R", "G", "B"]
            ax2 = None
            for c, (col, name) in enumerate(zip(ch_colors, ch_names)):
                ch   = pixels[:, :, c].flatten().astype(np.float32)
                hist, bins = np.histogram(ch, bins=256, range=(0, 255))
                self._hist_ax.plot(bins[:-1], hist, color=col,
                                   label=name, linewidth=1.5, alpha=0.85)
                if show_cdf:
                    if ax2 is None:
                        ax2 = self._hist_ax.twinx()
                        ax2.set_ylabel("Cumulative %",
                                       color=C_TEXT_SEC, fontsize=10)
                        ax2.tick_params(colors=C_TEXT_SEC, labelsize=9)
                        ax2.set_ylim(0, 105)
                    cdf = np.cumsum(hist).astype(np.float64)
                    cdf = cdf / cdf[-1] * 100.0
                    ax2.plot(bins[:-1], cdf, color=col,
                             linewidth=1, linestyle="--", alpha=0.6)
            self._hist_ax.legend(fontsize=10)
            title = f"{label}  (R/G/B channels)"
        else:
            hist, bins = np.histogram(flat, bins=256, range=(0, 255))
            self._hist_ax.bar(bins[:-1], hist, width=1,
                              color=color, alpha=0.72, linewidth=0)
            if show_cdf:
                ax2 = self._hist_ax.twinx()
                cdf = np.cumsum(hist).astype(np.float64)
                cdf = cdf / cdf[-1] * 100.0
                ax2.plot(bins[:-1], cdf, color=color,
                         linewidth=1.5, linestyle="--", alpha=0.85)
                ax2.set_ylabel("Cumulative %", color=C_TEXT_SEC, fontsize=10)
                ax2.tick_params(colors=C_TEXT_SEC, labelsize=9)
                ax2.set_ylim(0, 105)
            title = (f"{label}   mean {flat.mean():.1f}   "
                     f"std {flat.std():.1f}   "
                     f"[{int(flat.min())}, {int(flat.max())}]")

        self._hist_ax.set_xlabel("Pixel intensity",
                                  color=C_TEXT_SEC, fontsize=11)
        self._hist_ax.set_ylabel("Frequency", color=C_TEXT_SEC, fontsize=11)
        self._hist_ax.set_title(title, color=C_TEXT, fontsize=11)
        self._hist_ax.tick_params(colors=C_TEXT_SEC, labelsize=10)
        for sp in self._hist_ax.spines.values():
            sp.set_color(C_BORDER)
        self._hist_ax.grid(True, alpha=0.35, color=C_BORDER)
        self._hist_fig.tight_layout(pad=1.2)
        self._hist_canvas.draw()

    def _compare_histograms(self):
        show_cdf = self._cdf_btn.isChecked()
        self._prepare_hist_axes()

        img_map = {
            "Original":   self.original_image,
            "Viewport 1": self.viewport_images[1],
            "Viewport 2": self.viewport_images[2],
        }
        plotted = False
        ax2 = None
        for label, img in img_map.items():
            if img is None: continue
            color = self._hist_colors.get(label, C_ACCENT)
            # For color images use luminance for overlay comparison
            arr = to_luminance(img) if img.ndim == 3 else img
            hist, bins = np.histogram(arr.flatten(), bins=256, range=(0, 255))
            self._hist_ax.plot(bins[:-1], hist, color=color,
                               label=label, linewidth=1.5, alpha=0.85)
            if show_cdf:
                if ax2 is None:
                    ax2 = self._hist_ax.twinx()
                    ax2.set_ylabel("Cumulative %",
                                   color=C_TEXT_SEC, fontsize=10)
                    ax2.tick_params(colors=C_TEXT_SEC, labelsize=9)
                    ax2.set_ylim(0, 105)
                cdf = np.cumsum(hist).astype(np.float64)
                cdf = cdf / cdf[-1] * 100.0
                ax2.plot(bins[:-1], cdf, color=color,
                         linewidth=1, linestyle="--", alpha=0.7)
            plotted = True

        if plotted:
            self._hist_ax.legend(fontsize=10, framealpha=0.9)
            note = " + CDF" if show_cdf else ""
            self._hist_ax.set_xlabel("Pixel intensity",
                                      color=C_TEXT_SEC, fontsize=11)
            self._hist_ax.set_ylabel("Frequency",
                                      color=C_TEXT_SEC, fontsize=11)
            self._hist_ax.set_title(f"Histogram overlay{note}",
                                     color=C_TEXT, fontsize=11)
            self._hist_ax.tick_params(colors=C_TEXT_SEC, labelsize=10)
            for sp in self._hist_ax.spines.values():
                sp.set_color(C_BORDER)
            self._hist_ax.grid(True, alpha=0.35, color=C_BORDER)
            self._hist_fig.tight_layout(pad=1.2)
            self._hist_canvas.draw()

    def _export_histogram(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export histogram", "histogram.png",
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)",
        )
        if path:
            self._hist_fig.savefig(
                path, dpi=150, bbox_inches="tight", facecolor=C_SURFACE,
            )
            self._roi_status.setText(
                f"Histogram exported: {path.split('/')[-1]}"
            )