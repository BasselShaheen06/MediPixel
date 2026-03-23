"""
MediPixel — entry point.

Run with:
    python main.py
or after `pip install .`:
    medipixel
"""

import sys
import re
from pathlib import Path
import urllib.request

from PyQt5.QtCore import QByteArray, Qt, QTimer, QSettings
from PyQt5.QtGui import QColor, QFont, QFontDatabase, QLinearGradient, QPainter, QRadialGradient
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
from medipixel.ui.main_window import MedicalImageApp


class CoverPage(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setFixedSize(760, 420)
        self.setAttribute(Qt.WA_TranslucentBackground, False)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(40, 40, 40, 40)
        lay.setSpacing(14)

        title = QLabel("MediPixel")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(
            "color:#004D40; font-size:56px; font-weight:800; letter-spacing:2px;"
        )

        subtitle = QLabel("Medical Image Processing Workstation")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet(
            "color:#1D5F57; font-size:18px; font-weight:600;"
        )

        hint = QLabel("Preparing your workspace...")
        hint.setAlignment(Qt.AlignCenter)
        hint.setStyleSheet("color:#3D7A73; font-size:13px; font-weight:500;")

        lay.addStretch()
        lay.addWidget(title)
        lay.addWidget(subtitle)
        lay.addWidget(hint)
        lay.addStretch()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Base soft blend.
        bg = QLinearGradient(0, 0, self.width(), self.height())
        bg.setColorAt(0.0, QColor("#D2F0EA"))
        bg.setColorAt(1.0, QColor("#BFD9FF"))
        painter.fillRect(self.rect(), bg)

        # Blurry color clouds.
        c1 = QRadialGradient(self.width() * 0.28, self.height() * 0.35, 260)
        c1.setColorAt(0.0, QColor(160, 220, 210, 140))
        c1.setColorAt(1.0, QColor(160, 220, 210, 0))
        painter.fillRect(self.rect(), c1)

        c2 = QRadialGradient(self.width() * 0.72, self.height() * 0.62, 280)
        c2.setColorAt(0.0, QColor(165, 190, 245, 135))
        c2.setColorAt(1.0, QColor(165, 190, 245, 0))
        painter.fillRect(self.rect(), c2)


def _try_load_google_font(css_family_query: str) -> str | None:
    """Load a Google Font family at runtime and return loaded family name."""
    css_url = f"https://fonts.googleapis.com/css2?family={css_family_query}&display=swap"
    try:
        req = urllib.request.Request(
            css_url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/122.0.0.0 Safari/537.36"
                )
            },
        )
        with urllib.request.urlopen(req, timeout=4) as r:
            css = r.read().decode("utf-8", errors="ignore")

        m = re.search(r"url\((https://[^)]+)\)", css)
        if m:
            with urllib.request.urlopen(m.group(1), timeout=4) as fr:
                font_data = fr.read()
            fid = QFontDatabase.addApplicationFontFromData(
                QByteArray(font_data)
            )
            if fid != -1:
                families = QFontDatabase.applicationFontFamilies(fid)
                if families:
                    return families[0]
    except Exception:
        pass

    return None


def _try_load_local_font() -> str | None:
    """Load a local app font file from assets/fonts if present."""
    root = Path(__file__).resolve().parent
    fonts_dir = root / "assets" / "fonts"
    if not fonts_dir.exists():
        return None

    # Explicit user-provided paths first.
    preferred_paths = [
        fonts_dir / "IBM_Plex_Serif,Passero_One,Silkscreen" / "IBM_Plex_Serif" / "IBMPlexSerif-SemiBold.ttf",
        fonts_dir / "IBM_Plex_Serif,Passero_One,Silkscreen" / "Passero_One" / "PasseroOne-Regular.ttf",
        fonts_dir / "IBM_Plex_Serif,Passero_One,Silkscreen" / "Silkscreen" / "Silkscreen-Regular.ttf",
        fonts_dir / "IBM_Plex_Serif,Passero_One,Silkscreen" / "Silkscreen" / "Silkscreen-Bold.ttf",
    ]

    discovered = [
        p for p in fonts_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in {".ttf", ".otf"}
    ]

    # Preserve order: explicit paths first, then the rest.
    ordered_paths: list[Path] = []
    seen: set[str] = set()
    for p in preferred_paths + discovered:
        k = str(p.resolve()) if p.exists() else str(p)
        if k in seen:
            continue
        seen.add(k)
        ordered_paths.append(p)

    loaded_families: list[str] = []
    for font_path in ordered_paths:
        if not font_path.exists():
            continue
        fid = QFontDatabase.addApplicationFont(str(font_path))
        if fid == -1:
            continue
        families = QFontDatabase.applicationFontFamilies(fid)
        loaded_families.extend(families)

    preferred_families = [
        "IBM Plex Serif",
        "Passero One",
        "Silkscreen",
        "Press Start 2P",
        "VT323",
        "Pixeled",
        "Pixel Operator",
    ]
    for fam in preferred_families:
        if fam in loaded_families:
            return fam

    if loaded_families:
        return loaded_families[0]

    return None


def _choose_app_font_family() -> str:
    """Prefer bundled UI fonts with safe fallbacks for readability."""
    settings = QSettings("MediPixel", "MediPixel")
    preferred_saved = settings.value("font_family", "IBM Plex Serif")

    loaded_local = _try_load_local_font()
    installed = set(QFontDatabase().families())

    if isinstance(preferred_saved, str) and preferred_saved in installed:
        return preferred_saved

    if loaded_local and loaded_local in installed:
        return loaded_local

    # Local fallback order.
    for name in [
        "IBM Plex Serif",
        "Passero One",
        "Silkscreen",
        "Press Start 2P",
        "VT323",
        "Pixel Operator",
        "Pixeled",
    ]:
        if name in installed:
            return name

    # Try loading from Google Fonts at runtime (network optional).
    for query in [
        "Press+Start+2P",
        "VT323",
        "Silkscreen:wght@400;700",
    ]:
        loaded = _try_load_google_font(query)
        if loaded:
            return loaded

    return "IBM Plex Serif" if "IBM Plex Serif" in installed else "Segoe UI"


def main():
    app = QApplication(sys.argv)

    def _force_font_tree(root: QWidget, font: QFont):
        root.setFont(font)
        for child in root.findChildren(QWidget):
            child.setFont(font)

    # Use configured app-wide family, with graceful fallback.
    app_family = _choose_app_font_family()
    app.setStyleSheet(
        f"QWidget {{ font-family: '{app_family}', 'Segoe UI', sans-serif; }}"
    )
    app_font = QFont(app_family, 10)
    app.setFont(app_font)

    cover = CoverPage()
    _force_font_tree(cover, app_font)
    cover.show()

    window = MedicalImageApp()
    _force_font_tree(window, app_font)

    def _open_main_window():
        cover.close()
        window.show()

    QTimer.singleShot(5000, _open_main_window)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
