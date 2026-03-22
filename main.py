"""
MediPixel — entry point.

Run with:
    python main.py
or after `pip install .`:
    medipixel
"""

import sys
from PyQt5.QtWidgets import QApplication
from medipixel.ui.main_window import MedicalImageApp


def main():
    app = QApplication(sys.argv)
    window = MedicalImageApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
