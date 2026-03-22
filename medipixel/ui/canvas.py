"""
DraggableCanvas — a Matplotlib FigureCanvas with click-drag panning.

Kept in its own file because it is a reusable Qt widget that has
nothing to do with medical image logic.
"""

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtCore import Qt


class DraggableCanvas(FigureCanvas):
    """Matplotlib canvas that pans the axes on left-button drag."""

    def __init__(self, figure):
        super().__init__(figure)
        self._dragging = False
        self._last_pos = None

        self.mpl_connect("button_press_event", self._on_press)
        self.mpl_connect("button_release_event", self._on_release)
        self.mpl_connect("motion_notify_event", self._on_move)

    # ── Mouse handlers ────────────────────────────────────────────────────────

    def _on_press(self, event):
        if event.inaxes and event.button == 1:
            self._dragging = True
            self._last_pos = (event.x, event.y)
            self.setCursor(Qt.ClosedHandCursor)

    def _on_release(self, event):
        self._dragging = False
        self._last_pos = None
        self.setCursor(Qt.ArrowCursor)

    def _on_move(self, event):
        if not (self._dragging and event.inaxes and self._last_pos):
            return

        dx_px = event.x - self._last_pos[0]
        dy_px = event.y - self._last_pos[1]
        self._last_pos = (event.x, event.y)

        ax = self.figure.axes[0]
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()

        # Convert pixel displacement → data-coordinate displacement
        dx = -dx_px * (x1 - x0) / self.width()
        dy = -dy_px * (y1 - y0) / self.height()

        ax.set_xlim(x0 + dx, x1 + dx)
        ax.set_ylim(y0 + dy, y1 + dy)
        self.draw_idle()
