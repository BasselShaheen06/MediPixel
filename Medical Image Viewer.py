import sys
import cv2
from skimage import exposure
import numpy as np
from PIL import Image
import pydicom
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QFileDialog,
    QLabel, QSlider, QGridLayout, QComboBox, QSpinBox, QFrame, QScrollArea
)
from PyQt5.QtCore import Qt, QPoint
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from scipy import ndimage


class NoiseGenerator:
    @staticmethod
    def add_gaussian_noise(image, mean=0, sigma=25):
        """Add Gaussian noise to image"""
        noise = np.random.normal(mean, sigma, image.shape)
        noisy_image = image + noise
        return np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    @staticmethod
    def add_salt_and_pepper(image, prob=0.05):
        """Add salt and pepper noise"""
        noisy_image = np.copy(image)
        # Salt
        salt_mask = np.random.random(image.shape) < prob/2
        noisy_image[salt_mask] = 255
        # Pepper
        pepper_mask = np.random.random(image.shape) < prob/2
        noisy_image[pepper_mask] = 0
        return noisy_image
        
    @staticmethod
    def add_poisson_noise(image, scale=1.0):
        """Add Poisson noise"""
        noisy_image = np.random.poisson(image * scale) / scale
        return np.clip(noisy_image, 0, 255).astype(np.uint8)

class Denoiser:
    @staticmethod
    def median_filter(image, kernel_size=3):
        """Apply median filter"""
        return cv2.medianBlur(image, kernel_size)
    
    @staticmethod
    def bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):
        """Apply bilateral filter"""
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    
    @staticmethod
    def nlm_filter(image, h=10, window_size=7, search_size=21):
        """Apply Non-local Means denoising"""
        return cv2.fastNlMeansDenoising(image, None, h, window_size, search_size)

class ContrastEnhancement:
    @staticmethod
    def apply_histogram_equalization(image):
        """Apply standard histogram equalization"""
        if image is None or not isinstance(image, np.ndarray):
            return None
        
        # Ensure image is in uint8 format
        if image.dtype != np.uint8:
            image = ((image - image.min()) * (255.0 / (image.max() - image.min()))).astype(np.uint8)
            
        return cv2.equalizeHist(image)

    @staticmethod
    def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
        """Apply Contrast Limited Adaptive Histogram Equalization"""
        if image is None or not isinstance(image, np.ndarray):
            return None
            
        # Ensure image is in uint8 format
        if image.dtype != np.uint8:
            image = ((image - image.min()) * (255.0 / (image.max() - image.min()))).astype(np.uint8)
            
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(image)

    @staticmethod
    def apply_adaptive_gamma(image):
        """Apply adaptive gamma correction based on image statistics"""
        if image is None or not isinstance(image, np.ndarray):
            return None
            
        # Normalize image to 0-1 range
        img_norm = (image - image.min()) / (image.max() - image.min())
        
        # Calculate mean brightness
        mean_brightness = np.mean(img_norm)
        
        # Adjust gamma based on mean brightness
        # If image is dark (low mean), use gamma < 1 to brighten
        # If image is bright (high mean), use gamma > 1 to darken
        if mean_brightness < 0.5:
            gamma = 0.5 + mean_brightness  # gamma will be between 0.5 and 1.0
        else:
            gamma = 1.0 + (mean_brightness - 0.5)  # gamma will be between 1.0 and 1.5
            
        # Apply gamma correction
        enhanced = exposure.adjust_gamma(img_norm, gamma)
        
        # Convert back to uint8
        return (enhanced * 255).astype(np.uint8)

class DraggableCanvas(FigureCanvas):
    def __init__(self, figure):
        super().__init__(figure)
        self.setParent(None)

        # Initialize drag parameters
        self.is_dragging = False
        self.last_mouse_pos = None  # Last known position of the mouse
        self.original_xlim = None
        self.original_ylim = None

        # Connect mouse events
        self.mpl_connect('button_press_event', self.on_mouse_press)
        self.mpl_connect('button_release_event', self.on_mouse_release)
        self.mpl_connect('motion_notify_event', self.on_mouse_move)

    def on_mouse_press(self, event):
        if event.inaxes and event.button == 1:  # Left click
            self.is_dragging = True
            self.last_mouse_pos = (event.x, event.y)
            self.setCursor(Qt.ClosedHandCursor)

            # Store the original axis limits
            ax = self.figure.axes[0]
            self.original_xlim = ax.get_xlim()
            self.original_ylim = ax.get_ylim()

    def on_mouse_release(self, event):
        if self.is_dragging:
            self.is_dragging = False
            self.last_mouse_pos = None
            self.setCursor(Qt.ArrowCursor)

    def on_mouse_move(self, event):
        if self.is_dragging and event.inaxes:
            if self.last_mouse_pos is None:
                return

            # Calculate pixel displacement
            dx_pixels = event.x - self.last_mouse_pos[0]
            dy_pixels = event.y - self.last_mouse_pos[1]

            # Update last mouse position
            self.last_mouse_pos = (event.x, event.y)

            # Convert pixel displacement to data coordinates
            ax = self.figure.axes[0]
            x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
            y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
            dx_data = -dx_pixels * (x_range / self.width())  # Negative for left-to-right drag
            dy_data = -dy_pixels * (y_range / self.height())  # Positive for bottom-to-top drag

            # Adjust axis limits incrementally
            ax.set_xlim(ax.get_xlim()[0] + dx_data, ax.get_xlim()[1] + dx_data)
            ax.set_ylim(ax.get_ylim()[0] + dy_data, ax.get_ylim()[1] + dy_data)

            # Redraw the canvas
            self.draw_idle()


class HistogramWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Histogram")
        self.setGeometry(800, 100, 600, 400)  # Position and size of histogram window
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Create matplotlib figure and canvas
        self.figure = Figure(figsize=(6, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        
        # Add canvas to layout
        layout.addWidget(self.canvas)
        
        # Add statistics label
        self.stats_label = QLabel()
        layout.addWidget(self.stats_label)

    def plot_histogram(self, image, title):
        """Plot histogram of the given image"""
        self.ax.clear()
        
        # Calculate histogram
        hist, bins = np.histogram(image.flatten(), bins=256, range=(0, 255))
        
        # Plot histogram
        self.ax.bar(bins[:-1], hist, width=1, color='blue', alpha=0.7)
        self.ax.set_title(f'Histogram - {title}')
        self.ax.set_xlabel('Pixel Value')
        self.ax.set_ylabel('Frequency')
        
        # Set reasonable y-axis limits
        max_freq = hist.max()
        self.ax.set_ylim(0, max_freq * 1.1)
        
        # Add grid
        self.ax.grid(True, alpha=0.3)
        
        # Calculate and display statistics
        mean_val = np.mean(image)
        median_val = np.median(image)
        std_val = np.std(image)
        min_val = np.min(image)
        max_val = np.max(image)
        
        stats_text = (f"Statistics:\n"
                     f"Mean: {mean_val:.2f}\n"
                     f"Median: {median_val:.2f}\n"
                     f"Std Dev: {std_val:.2f}\n"
                     f"Min: {min_val:.2f}\n"
                     f"Max: {max_val:.2f}")
        self.stats_label.setText(stats_text)
        
        # Refresh canvas
        self.canvas.draw()
        

class MedicalImageApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Medical Image Viewer")
        self.setGeometry(100, 100, 1400, 900)

        # Main widget and layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QGridLayout(self.main_widget)
        self.layout.setSpacing(5)  # Add spacing between grid cells

        # Create a frame for the control panel
        self.control_panel = QFrame()
        self.control_panel.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        self.control_panel.setFixedWidth(200)  # Set fixed width for control panel

        # Control panel layout
        self.control_layout = QVBoxLayout(self.control_panel)
        self.control_layout.setSpacing(5)  # Space between controls
        self.control_layout.setContentsMargins(10, 10, 10, 10)  # Margins within control panel
        self.status_label = QLabel("Select ROI Process")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.control_layout.addWidget(self.status_label)

        # Create histogram window
        self.histogram_window = HistogramWindow()
        
        # Add Show Histogram button
        self.show_histogram_button = QPushButton("Show Histogram")
        self.show_histogram_button.setStyleSheet("""
            QPushButton {
                min-height: 15px;
                padding: 3px;
                font-size: 10px;
                background-color: #f0f0f0;
                border: 1px solid #ccc;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
        """)

        # Add histogram type selector
        self.histogram_type = QComboBox()
        self.histogram_type.addItems(["Main Viewport", "Viewport 1", "Viewport 2"])
        self.histogram_type.setStyleSheet("""
            QComboBox {
                min-height: 15px;
                padding: 2px 5px;
                border: 1px solid #ccc;
                border-radius: 3px;
            }
        """)
        self.control_layout.addWidget(QLabel("Histogram Source:"))
        self.control_layout.addWidget(self.histogram_type)
        
        # Connect histogram type change to update
        self.histogram_type.currentIndexChanged.connect(self.update_histogram)

        # Buttons with consistent size and styling
        button_style = """
            QPushButton {
                min-height: 15px;
                padding: 1px;
                font-size: 12px;
                background-color: #f0f0f0;
                border: 1px solid #ccc;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
        """

        # Create and style buttons
        self.load_button = QPushButton("Load Image")
        self.select_signal_button = QPushButton("Select Signal ROI")
        self.select_signal2_button = QPushButton("Select Signal ROI 2")
        self.select_noise_button = QPushButton("Select Noise ROI")
        self.calculate_snr_button = QPushButton("Calculate SNR")
        self.calculate_cnr_button = QPushButton("Calculate CNR")
        self.reset_button = QPushButton("Reset")

        for button in [self.load_button, self.select_signal_button,
                       self.select_noise_button, self.calculate_snr_button,
                       self.reset_button, self.select_signal2_button, self.calculate_cnr_button]:
            button.setStyleSheet(button_style)
            self.control_layout.addWidget(button)

        # Add spacing between button group and controls
        self.control_layout.addSpacing(10)

        self.current_process_state = "idle"  # Start with idle state
        self.signal_roi = None  # Signal ROI array
        self.noise_roi = None  # Noise ROI array
        self.roi_rect = None  # Rectangle drawn on the image
        self.cid = None  # Connection ID for ROI selection
        self.signal2_roi = None  # Signal 2 ROI
        self.signal2_roi_rect = None  # Signal 2 Rectangle drawn on the image

        # FOV and Resolution section
        fov_frame = QFrame()
        fov_frame.setFrameStyle(QFrame.StyledPanel)
        fov_layout = QVBoxLayout(fov_frame)
        fov_layout.setSpacing(5)

        self.fov_label = QLabel("Adjust")
        self.fov_label.setAlignment(Qt.AlignCenter)

        # Style the dropdown and spinbox
        control_style = """
            QComboBox, QSpinBox {
                min-height: 15px;
                padding: 2px 5px;
                border: 1px solid #ccc;
                border-radius: 3px;
            }
        """

        self.resolution_dropdown = QComboBox()
        self.resolution_dropdown.addItems(["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"])
        self.resolution_dropdown.setStyleSheet(control_style)

        self.pixel_count_label = QLabel("Pixels per Sample:")
        self.pixel_count_label.setAlignment(Qt.AlignCenter)

        self.pixel_count_spinbox = QSpinBox()
        self.pixel_count_spinbox.setRange(1, 1000)
        self.pixel_count_spinbox.setValue(400)
        self.pixel_count_spinbox.setStyleSheet(control_style)

        # Add controls to FOV frame
        fov_layout.addWidget(self.fov_label)
        fov_layout.addWidget(self.resolution_dropdown)
        fov_layout.addWidget(self.pixel_count_label)
        fov_layout.addWidget(self.pixel_count_spinbox)
        self.control_layout.addWidget(fov_frame)

        # Viewport selector section
        viewport_frame = QFrame()
        viewport_frame.setFrameStyle(QFrame.StyledPanel)
        viewport_layout = QVBoxLayout(viewport_frame)
        viewport_layout.setSpacing(5)

        viewport_label = QLabel("View On:")
        viewport_label.setAlignment(Qt.AlignCenter)

        self.viewport_selector = QComboBox()
        self.viewport_selector.addItems(["Viewport 1", "Viewport 2"])
        self.viewport_selector.setStyleSheet(control_style)

        viewport_layout.addWidget(viewport_label)
        viewport_layout.addWidget(self.viewport_selector)
        self.control_layout.addWidget(viewport_frame)

        # Add the control panel to the main layout
        self.layout.addWidget(self.control_panel, 0, 0, 2, 1)

        # Create draggable canvases instead of regular ones
        self.figures = [Figure() for _ in range(3)]
        self.canvases = [DraggableCanvas(fig) for fig in self.figures]

        # Set size policies for canvases
        for canvas in self.canvases:
            canvas.setMinimumSize(400, 400)

        # Create scroll areas for each canvas
        self.scroll_areas = []
        for canvas in self.canvases:
            scroll_area = QScrollArea()
            scroll_area.setWidget(canvas)
            scroll_area.setWidgetResizable(True)
            self.scroll_areas.append(scroll_area)

        # Add scroll areas to the layout
        self.layout.addWidget(self.scroll_areas[0], 0, 1, 2, 1)  # Main viewport
        self.layout.addWidget(self.scroll_areas[1], 0, 2)  # Second viewport
        self.layout.addWidget(self.scroll_areas[2], 1, 2)  # Third viewport

        # Set column stretch factors
        self.layout.setColumnStretch(0, 0)  # Control panel (no stretch)
        self.layout.setColumnStretch(1, 2)  # Main viewport (stretch factor 2)
        self.layout.setColumnStretch(2, 1)  # Side viewports (stretch factor 1)

        # Connect signals
        self.load_button.clicked.connect(self.load_image)

        self.select_signal_button.clicked.connect(lambda: self.start_roi_selection("signal"))
        self.select_signal2_button.clicked.connect(lambda: self.start_roi_selection("signal2"))
        self.select_noise_button.clicked.connect(lambda: self.start_roi_selection("noise"))
        self.calculate_snr_button.clicked.connect(self.calculate_snr)
        self.calculate_cnr_button.clicked.connect(self.calculate_cnr)
        self.reset_button.clicked.connect(self.reset)
        self.resolution_dropdown.currentIndexChanged.connect(self.update_resolution)
        self.pixel_count_spinbox.valueChanged.connect(self.update_pixel_count)

        # Initialize other variables (keeping the rest of your original initialization)
        self.original_image = None
        self.adjusted_image = None
        self.image = None
        self.signal_roi = None
        self.noise_roi = None
        self.current_roi_type = None
        self.roi_rect = None
        self.axes = [fig.add_subplot(111) for fig in self.figures]
        self.ax_main = self.axes[0]
        self.cid = None

        # Add Zoom Slider
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(1, 10)
        self.zoom_slider.setValue(1)
        self.zoom_slider.setTickPosition(QSlider.TicksBelow)
        self.zoom_slider.setTickInterval(1)
        self.zoom_slider.valueChanged.connect(self.apply_zoom)
        self.control_layout.addWidget(self.zoom_slider)

        
        # Add Interpolation Dropdown
        self.interpolation_dropdown = QComboBox()
        self.interpolation_dropdown.addItems(["Nearest", "Bilinear", "Cubic"])
        self.interpolation_dropdown.setStyleSheet(control_style)
        self.control_layout.addWidget(QLabel("Interpolation Method:"))
        self.control_layout.addWidget(self.interpolation_dropdown)

        # Add Filter Section
        filter_frame = QFrame()
        filter_frame.setFrameStyle(QFrame.StyledPanel)
        filter_layout = QVBoxLayout(filter_frame)
        
        # Filter Type Selection
        self.filter_type = QComboBox()
        self.filter_type.addItems(["No Filter", "Lowpass", "Highpass"])
        self.filter_type.currentIndexChanged.connect(self.apply_filter)
        
        # Cutoff Frequency Slider
        self.cutoff_slider = QSlider(Qt.Horizontal)
        self.cutoff_slider.setRange(1, 100)
        self.cutoff_slider.setValue(50)
        self.cutoff_slider.valueChanged.connect(self.apply_filter)
        
        # Filter Order Spinner
        self.filter_order_spin = QSpinBox()
        self.filter_order_spin.setRange(1, 10)
        self.filter_order_spin.setValue(2)
        self.filter_order_spin.valueChanged.connect(self.apply_filter)
        
        # Add labels and widgets to filter layout
        filter_layout.addWidget(QLabel("Filter Type:"))
        filter_layout.addWidget(self.filter_type)
        filter_layout.addWidget(QLabel("Cutoff Frequency (%):"))
        filter_layout.addWidget(self.cutoff_slider)
        filter_layout.addWidget(QLabel("Filter Order:"))
        filter_layout.addWidget(self.filter_order_spin)
        
        self.control_layout.addWidget(filter_frame)

        # Add filter state tracking
        self.current_filter_type = "No Filter"
        self.current_cutoff = 50
        self.current_order = 2
        self.filtered_image = None
        
        # Connect filter changes to process_image
        self.filter_type.currentIndexChanged.connect(self.update_filter_params)
        self.cutoff_slider.valueChanged.connect(self.update_filter_params)
        self.filter_order_spin.valueChanged.connect(self.update_filter_params)

        # Add Show Histogram button
        self.show_histogram_button = QPushButton("Show Histogram")
        self.show_histogram_button.setStyleSheet(button_style)
        self.control_layout.addWidget(self.show_histogram_button)
        self.show_histogram_button.clicked.connect(self.show_histogram)

        # Add contrast enhancement controls
        self.setup_contrast_controls()

        self.setup_noise_controls()

        # Add viewport-specific image storage
        self.viewport_images = {1: None, 2: None}  # Store processed images for each viewport

    def setup_noise_controls(self):
        noise_frame = QFrame()
        noise_frame.setFrameStyle(QFrame.StyledPanel)
        noise_layout = QVBoxLayout(noise_frame)
        
        # Noise type selector
        self.noise_type = QComboBox()
        self.noise_type.addItems(["No Noise", "Gaussian", "Salt & Pepper", "Poisson"])
        
        # Denoising method selector
        self.denoise_method = QComboBox()
        self.denoise_method.addItems(["No Denoising", "Median", "Bilateral", "Non-local Means"])
        
        # Parameters for noise
        self.noise_strength = QSlider(Qt.Horizontal)
        self.noise_strength.setRange(1, 100)
        self.noise_strength.setValue(25)
        
        # Add widgets to layout
        noise_layout.addWidget(QLabel("Noise Type:"))
        noise_layout.addWidget(self.noise_type)
        noise_layout.addWidget(QLabel("Noise Strength:"))
        noise_layout.addWidget(self.noise_strength)
        noise_layout.addWidget(QLabel("Denoising Method:"))
        noise_layout.addWidget(self.denoise_method)
        
        self.control_layout.addWidget(noise_frame)
        
        # Connect signals
        self.noise_type.currentIndexChanged.connect(self.process_image)
        self.denoise_method.currentIndexChanged.connect(self.process_image)
        self.noise_strength.valueChanged.connect(self.process_image)
    
    def setup_contrast_controls(self):
        # Create contrast enhancement frame
        contrast_frame = QFrame()
        contrast_frame.setFrameStyle(QFrame.StyledPanel)
        contrast_layout = QVBoxLayout(contrast_frame)
        
        # Add contrast method selector
        contrast_label = QLabel("Contrast Enhancement:")
        self.contrast_method = QComboBox()
        self.contrast_method.addItems(["None", "Histogram Equalization", "CLAHE", "Adaptive Gamma"])
        
        # Add CLAHE parameters
        self.clahe_clip_label = QLabel("CLAHE Clip Limit:")
        self.clahe_clip = QSlider(Qt.Horizontal)
        self.clahe_clip.setRange(1, 50)  # 0.1 to 5.0
        self.clahe_clip.setValue(20)  # Default 2.0
        
        self.clahe_grid_label = QLabel("CLAHE Grid Size:")
        self.clahe_grid = QSlider(Qt.Horizontal)
        self.clahe_grid.setRange(2, 16)
        self.clahe_grid.setValue(8)  # Default 8x8
        
        # Add all widgets to layout
        contrast_layout.addWidget(contrast_label)
        contrast_layout.addWidget(self.contrast_method)
        contrast_layout.addWidget(self.clahe_clip_label)
        contrast_layout.addWidget(self.clahe_clip)
        contrast_layout.addWidget(self.clahe_grid_label)
        contrast_layout.addWidget(self.clahe_grid)
        
        # Add frame to control layout
        self.control_layout.addWidget(contrast_frame)
        
        # Connect signals
        self.contrast_method.currentIndexChanged.connect(self.apply_contrast_enhancement)
        self.clahe_clip.valueChanged.connect(self.apply_contrast_enhancement)
        self.clahe_grid.valueChanged.connect(self.apply_contrast_enhancement)

    def apply_contrast_enhancement(self):
        if self.original_image is None:
            return
            
        ###################################
        # Add noise reduction before enhancement
        denoised_image = cv2.GaussianBlur(self.original_image, (3,3), 0)
        
        # Get base image (after any filtering)
        if self.current_filter_type != "No Filter":
            base_image = self.apply_filter(self.original_image)
        else:
            base_image = self.original_image.copy()
        
        # Apply selected contrast enhancement
        method = self.contrast_method.currentText()
        if method == "Histogram Equalization":
            self.adjusted_image = ContrastEnhancement.apply_histogram_equalization(base_image)
        elif method == "CLAHE":
            clip_limit = self.clahe_clip.value() / 10.0  # Convert to 0.1-5.0 range
            grid_size = self.clahe_grid.value()
            self.adjusted_image = ContrastEnhancement.apply_clahe(base_image, 
                                                                clip_limit=clip_limit,
                                                                tile_grid_size=(grid_size, grid_size))
        elif method == "Adaptive Gamma":
            self.adjusted_image = ContrastEnhancement.apply_adaptive_gamma(base_image)
        else:  # None
            self.adjusted_image = base_image
            
        # Update display
        self.process_image()
        
        # Update histogram if visible
        if self.histogram_window.isVisible():
            self.update_histogram()


    def show_histogram(self):
        """Display histogram window and update its content"""
        self.update_histogram()
        self.histogram_window.show()
        self.histogram_window.raise_()  # Bring window to front
    
    def update_histogram(self):
        """Update the histogram display based on selected viewport"""
        viewport_selection = self.histogram_type.currentText()
        
        if viewport_selection == "Main Viewport" and self.original_image is not None:
            image_for_histogram = self.original_image
            title = "Main Viewport"
        elif viewport_selection == "Viewport 1" and self.viewport_images[1] is not None:
            image_for_histogram = self.viewport_images[1]
            title = "Viewport 1"
        elif viewport_selection == "Viewport 2" and self.viewport_images[2] is not None:
            image_for_histogram = self.viewport_images[2]
            title = "Viewport 2"
        else:
            return

        self.histogram_window.plot_histogram(image_for_histogram, title)
    
    def update_filter_params(self):
        """Update filter parameters and reprocess image"""
        self.current_filter_type = self.filter_type.currentText()
        self.current_cutoff = self.cutoff_slider.value() / 100.0
        self.current_order = self.filter_order_spin.value()
        self.process_image()
        
    def apply_filter(self, image):
        """Apply filter to an image and return the result"""
        if image is None or not isinstance(image, np.ndarray):
            return None
            
        if image.size == 0 or len(image.shape) < 2:
            return None
            
        # Convert image to float type for FFT
        image_float = image.astype(float)
        
        try:
            # Apply FFT
            f_transform = np.fft.fft2(image_float)
            f_shift = np.fft.fftshift(f_transform)
            
            rows, cols = image_float.shape
            
            # Create frequency grid
            u = np.linspace(-0.5, 0.5, cols)
            v = np.linspace(-0.5, 0.5, rows)
            U, V = np.meshgrid(u, v)
            D = np.sqrt(U**2 + V**2)
            
            # Create filter mask based on type
            if self.current_filter_type == "Lowpass":
                mask = 1 / (1 + (D / self.current_cutoff)**(2 * self.current_order))
            elif self.current_filter_type == "Highpass":
                mask = 1 - 1 / (1 + (D / self.current_cutoff)**(2 * self.current_order))
            else:  # No Filter
                return image
            
            # Apply filter
            f_shift_filtered = f_shift * mask
            f_ishift = np.fft.ifftshift(f_shift_filtered)
            img_back = np.fft.ifft2(f_ishift)
            filtered_image = np.abs(img_back)
            
            # Normalize the filtered image
            min_val = filtered_image.min()
            max_val = filtered_image.max()
            
            if max_val > min_val:  # Avoid division by zero
                filtered_image = ((filtered_image - min_val) * (255.0 / (max_val - min_val))).astype(np.uint8)
            else:
                filtered_image = np.zeros_like(filtered_image, dtype=np.uint8)
            
            return filtered_image
            
        except Exception as e:
            print(f"Error in apply_filter: {str(e)}")
            return image  # Return original image if filtering fails
    
    def process_image(self):
        """Process image with current zoom, FOV, filter, and contrast enhancement settings"""
        if self.original_image is None or not isinstance(self.original_image, np.ndarray):
            return

        try:
            # First apply resolution scaling
            scale = max(1, int(self.resolution_dropdown.currentText()))
            base_image = self.original_image[::scale, ::scale]
            
            # Make sure base_image is valid
            if not isinstance(base_image, np.ndarray) or base_image.size == 0 or len(base_image.shape) < 2:
                print("Error: Invalid image after scaling")
                return
            
            # Apply noise if selected
            noise_type = self.noise_type.currentText()
            strength = self.noise_strength.value()
            
            if noise_type == "Gaussian":
                base_image = NoiseGenerator.add_gaussian_noise(base_image, sigma=strength)
            elif noise_type == "Salt & Pepper":
                base_image = NoiseGenerator.add_salt_and_pepper(base_image, prob=strength/500)
            elif noise_type == "Poisson":
                base_image = NoiseGenerator.add_poisson_noise(base_image, scale=strength/25)
            
            # Apply denoising if selected
            denoise_method = self.denoise_method.currentText()
            if denoise_method == "Median":
                base_image = Denoiser.median_filter(base_image)
            elif denoise_method == "Bilateral":
                base_image = Denoiser.bilateral_filter(base_image)
            elif denoise_method == "Non-local Means":
                base_image = Denoiser.nlm_filter(base_image)
            
            # Apply filter to base image if needed
            if self.current_filter_type != "No Filter":
                filtered_image = self.apply_filter(base_image)
                if filtered_image is not None:
                    base_image = filtered_image
            
            # Apply contrast enhancement
            method = self.contrast_method.currentText()
            processed_image = base_image.copy()
            
            if method == "Histogram Equalization":
                base_image = ContrastEnhancement.apply_histogram_equalization(base_image)
            elif method == "CLAHE":
                clip_limit = self.clahe_clip.value() / 10.0  # Convert to 0.1-5.0 range
                grid_size = self.clahe_grid.value()
                base_image = ContrastEnhancement.apply_clahe(base_image, 
                                                        clip_limit=clip_limit,
                                                        tile_grid_size=(grid_size, grid_size))
            elif method == "Adaptive Gamma":
                base_image = ContrastEnhancement.apply_adaptive_gamma(base_image)
            
            # Apply zoom if needed
            zoom_factor = self.zoom_slider.value()
            if zoom_factor > 1:
                # Map interpolation methods
                interpolation_methods = {
                    "Nearest": Image.Resampling.NEAREST,
                    "Bilinear": Image.Resampling.BILINEAR,
                    "Cubic": Image.Resampling.BICUBIC
                }
                interpolation_method = interpolation_methods.get(
                    self.interpolation_dropdown.currentText(),
                    Image.Resampling.BILINEAR
                )

                # Ensure image is in uint8 format for PIL
                if base_image.dtype != np.uint8:
                    min_val = base_image.min()
                    max_val = base_image.max()
                    if max_val > min_val:
                        base_image = ((base_image - min_val) * (255.0 / (max_val - min_val))).astype(np.uint8)
                    else:
                        base_image = np.zeros_like(base_image, dtype=np.uint8)
                
                # Apply zoom
                processed_image = np.array(Image.fromarray(base_image).resize(
                    (base_image.shape[1] * zoom_factor, base_image.shape[0] * zoom_factor),
                    interpolation_method
                ))
            else:
                processed_image = base_image

            # Store the processed image for the current viewport
            target_viewport = self.viewport_selector.currentIndex() + 1
            self.viewport_images[target_viewport] = processed_image
            
            # Display in selected viewport
            target_index = self.viewport_selector.currentIndex() + 1
            ax = self.axes[target_index]
            canvas = self.canvases[target_index]
            
            # Display the processed image
            self.display_image(processed_image, ax, canvas)
            
            # Apply FOV settings
            h, w = processed_image.shape
            pixel_size = self.pixel_count_spinbox.value()
            
            # Calculate the centered FOV region
            x_start = max(0, (w - pixel_size) // 2)
            y_start = max(0, (h - pixel_size) // 2)
            x_end = min(w, x_start + pixel_size)
            y_end = min(h, y_start + pixel_size)
            
            # Set the axis limits for the zoomed view
            ax.set_xlim(x_start, x_end)
            ax.set_ylim(y_end, y_start)  # Inverted for correct orientation
            
            canvas.draw()

            # Update histogram if it's visible
            if (self.histogram_window.isVisible() and 
                self.histogram_type.currentText() == "Adjusted Image" and
                self.adjusted_image is not None):
                self.update_histogram()
            
        except Exception as e:
            print(f"Error in process_image: {str(e)}")
    
    def apply_zoom(self):
        """Trigger image processing when zoom changes"""
        self.process_image()
  
    def load_image(self):
        """Load an image file (DICOM or common image formats)."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Image File", "",
                                                   "Image Files (*.dcm *.png *.jpg *.jpeg *.bmp *.tiff);;All Files (*)")
        if file_path:
            if file_path.lower().endswith('.dcm'):
                dataset = pydicom.dcmread(file_path)
                self.original_image = dataset.pixel_array
            else:
                image = Image.open(file_path).convert('L')
                self.original_image = np.array(image)

            self.image = self.original_image.copy()
            self.display_image(self.image, self.ax_main, self.canvases[0])
            
            # Update histogram if window is visible
            if self.histogram_window.isVisible():
                self.update_histogram()

    def display_image(self, image, ax, canvas):
        """Display image in a specified viewport with proper aspect ratio."""
        ax.clear()
        if image is not None and image.size > 0 and image.shape[0] > 0 and image.shape[1] > 0:
            ax.imshow(image, cmap='gray')
            ax.axis('on')  # Show axes for better navigation

            # Reset the view limits to show the full image
            ax.set_xlim(0, image.shape[1])
            ax.set_ylim(image.shape[0], 0)  # Inverted for proper image orientation

            canvas.draw()

    def start_roi_selection(self, roi_type):
        """Start the process to select Signal or Noise ROI."""
        if roi_type == "signal" and self.current_process_state == "idle":
            self.status_label.setText("Select Signal ROI")
            self.current_process_state = "select_signal"
            self.select_signal_button.setEnabled(False)  # Disable the Signal ROI button
            self.select_noise_button.setEnabled(True)  # Enable the Noise ROI button
            self.cid = self.figures[0].canvas.mpl_connect("button_press_event", self.on_click)

        elif roi_type == "signal2" and self.current_process_state == "select_signal":
            self.status_label.setText("Select Signal ROI 2")
            self.current_process_state = "select_signal2"
            self.select_signal2_button.setEnabled(False)
            self.calculate_cnr_button.setEnabled(True)
            self.cid = self.figures[0].canvas.mpl_connect("button_press_event", self.on_click)

        elif roi_type == "noise" and self.current_process_state == "select_signal2":
            self.status_label.setText("Select Noise ROI")
            self.current_process_state = "select_noise"
            self.select_noise_button.setEnabled(False)
            self.calculate_snr_button.setEnabled(True)
            self.cid = self.figures[0].canvas.mpl_connect("button_press_event", self.on_click)

    def on_click(self, event):
        """Handle mouse click event for selecting ROI across all viewports."""
        if event.inaxes != self.ax_main:
            return

        # Handle the click based on the current process state
        if self.current_process_state == "select_signal":
            # Select the signal ROI
            x, y = int(event.xdata), int(event.ydata)
            width, height = 20, 20  # Fixed ROI size

            # Store ROI coordinates
            self.signal_roi_coords = (x, y, width, height)

            # Add rectangles to all viewports
            self.signal_roi_rects = []
            for ax in self.axes:
                rect = Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                self.signal_roi_rects.append(rect)

            # Store ROIs for all viewports
            self.signal_rois = []
            self.signal_rois.append(self.image[y:y + height, x:x + width])  # Original image
            if self.adjusted_image is not None:
                self.signal_rois.append(self.adjusted_image[y:y + height, x:x + width])  # Adjusted image 1
                self.signal_rois.append(self.adjusted_image[y:y + height, x:x + width])  # Adjusted image 2

            # Update all canvases
            for canvas in self.canvases:
                canvas.draw()

            self.current_process_state = "select_signal2"
            self.status_label.setText("Select Signal ROI 2")
            self.select_signal_button.setEnabled(False)
            self.select_signal2_button.setEnabled(True)

        elif self.current_process_state == "select_signal2":
            x, y = int(event.xdata), int(event.ydata)
            width, height = 20, 20

            # Store ROI coordinates
            self.signal2_roi_coords = (x, y, width, height)

            # Add rectangles to all viewports
            self.signal2_roi_rects = []
            for ax in self.axes:
                rect = Rectangle((x, y), width, height, linewidth=1, edgecolor='b', facecolor='none')
                ax.add_patch(rect)
                self.signal2_roi_rects.append(rect)

            # Store ROIs for all viewports
            self.signal2_rois = []
            self.signal2_rois.append(self.image[y:y + height, x:x + width])  # Original image
            if self.adjusted_image is not None:
                self.signal2_rois.append(self.adjusted_image[y:y + height, x:x + width])  # Adjusted image 1
                self.signal2_rois.append(self.adjusted_image[y:y + height, x:x + width])  # Adjusted image 2

            # Update all canvases
            for canvas in self.canvases:
                canvas.draw()

            self.current_process_state = "select_noise"
            self.status_label.setText("Select Noise ROI")
            self.select_signal2_button.setEnabled(False)
            self.select_noise_button.setEnabled(True)

        elif self.current_process_state == "select_noise":
            x, y = int(event.xdata), int(event.ydata)
            width, height = 20, 20

            # Store ROI coordinates
            self.noise_roi_coords = (x, y, width, height)

            # Add rectangles to all viewports
            self.noise_roi_rects = []
            for ax in self.axes:
                rect = Rectangle((x, y), width, height, linewidth=1, edgecolor='g', facecolor='none')
                ax.add_patch(rect)
                self.noise_roi_rects.append(rect)

            # Store ROIs for all viewports
            self.noise_rois = []
            self.noise_rois.append(self.image[y:y + height, x:x + width])  # Original image
            if self.adjusted_image is not None:
                self.noise_rois.append(self.adjusted_image[y:y + height, x:x + width])  # Adjusted image 1
                self.noise_rois.append(self.adjusted_image[y:y + height, x:x + width])  # Adjusted image 2

            # Update all canvases
            for canvas in self.canvases:
                canvas.draw()

            self.current_process_state = "calculate"
            self.status_label.setText("Calculate SNR/CNR")
            self.select_noise_button.setEnabled(False)
            self.calculate_snr_button.setEnabled(True)
            self.calculate_cnr_button.setEnabled(True)

    def calculate_snr(self):
        """Calculate and display the Signal-to-Noise Ratio (SNR) for all images"""
        if hasattr(self, 'signal_rois') and hasattr(self, 'noise_rois'):
            snr_text = "SNR Results:\n"
            
            # Calculate SNR for original image
            signal_mean = np.mean(self.signal_rois[0])
            noise_std = np.std(self.noise_rois[0])
            snr = signal_mean / noise_std if noise_std != 0 else 0
            snr_text += f"Original Image: {snr:.2f}\n"
            
            # Calculate SNR for viewport 1
            if self.viewport_images[1] is not None:
                # Get ROI coordinates
                x, y, width, height = self.signal_roi_coords
                noise_x, noise_y, noise_width, noise_height = self.noise_roi_coords
                
                # Extract ROIs from viewport 1's image
                viewport1_signal = self.viewport_images[1][y:y+height, x:x+width]
                viewport1_noise = self.viewport_images[1][noise_y:noise_y+noise_height, 
                                                        noise_x:noise_x+noise_width]
                
                signal_mean = np.mean(viewport1_signal)
                noise_std = np.std(viewport1_noise)
                snr = signal_mean / noise_std if noise_std != 0 else 0
                snr_text += f"Viewport 1: {snr:.2f}\n"
            
            # Calculate SNR for viewport 2
            if self.viewport_images[2] is not None:
                # Extract ROIs from viewport 2's image
                viewport2_signal = self.viewport_images[2][y:y+height, x:x+width]
                viewport2_noise = self.viewport_images[2][noise_y:noise_y+noise_height, 
                                                        noise_x:noise_x+noise_width]
                
                signal_mean = np.mean(viewport2_signal)
                noise_std = np.std(viewport2_noise)
                snr = signal_mean / noise_std if noise_std != 0 else 0
                snr_text += f"Viewport 2: {snr:.2f}\n"
            
            self.status_label.setText(snr_text)
        else:
            self.status_label.setText("Please select both signal and noise ROIs.")

    def calculate_cnr(self):
        """Calculate and display the Contrast-to-Noise Ratio (CNR) for all images"""
        if hasattr(self, 'signal_rois') and hasattr(self, 'signal2_rois') and hasattr(self, 'noise_rois'):
            cnr_text = "CNR Results:\n"
            
            # Calculate CNR for original image
            signal1_mean = np.mean(self.signal_rois[0])
            signal2_mean = np.mean(self.signal2_rois[0])
            noise_std = np.std(self.noise_rois[0])
            cnr = abs(signal1_mean - signal2_mean) / noise_std if noise_std != 0 else 0
            cnr_text += f"Original Image: {cnr:.2f}\n"
            
            # Calculate CNR for viewport 1
            if self.viewport_images[1] is not None:
                # Get ROI coordinates
                x1, y1, width1, height1 = self.signal_roi_coords
                x2, y2, width2, height2 = self.signal2_roi_coords
                noise_x, noise_y, noise_width, noise_height = self.noise_roi_coords
                
                # Extract ROIs from viewport 1's image
                viewport1_signal1 = self.viewport_images[1][y1:y1+height1, x1:x1+width1]
                viewport1_signal2 = self.viewport_images[1][y2:y2+height2, x2:x2+width2]
                viewport1_noise = self.viewport_images[1][noise_y:noise_y+noise_height, 
                                                        noise_x:noise_x+noise_width]
                
                signal1_mean = np.mean(viewport1_signal1)
                signal2_mean = np.mean(viewport1_signal2)
                noise_std = np.std(viewport1_noise)
                cnr = abs(signal1_mean - signal2_mean) / noise_std if noise_std != 0 else 0
                cnr_text += f"Viewport 1: {cnr:.2f}\n"
            
            # Calculate CNR for viewport 2
            if self.viewport_images[2] is not None:
                # Extract ROIs from viewport 2's image
                viewport2_signal1 = self.viewport_images[2][y1:y1+height1, x1:x1+width1]
                viewport2_signal2 = self.viewport_images[2][y2:y2+height2, x2:x2+width2]
                viewport2_noise = self.viewport_images[2][noise_y:noise_y+noise_height, 
                                                        noise_x:noise_x+noise_width]
                
                signal1_mean = np.mean(viewport2_signal1)
                signal2_mean = np.mean(viewport2_signal2)
                noise_std = np.std(viewport2_noise)
                cnr = abs(signal1_mean - signal2_mean) / noise_std if noise_std != 0 else 0
                cnr_text += f"Viewport 2: {cnr:.2f}\n"
            
            self.status_label.setText(cnr_text)
        else:
            self.status_label.setText("Please select all required ROIs.")

    def adjust_fov(self):
        """Adjust the Field of View (FOV) and display in the selected viewport without cropping."""
        if self.original_image is not None:
            # Get the scale factor from the resolution dropdown
            scale = max(1, int(self.resolution_dropdown.currentText()))
            base_image = self.original_image[::scale, ::scale]  
            
            # Apply current zoom if any
            zoom_factor = self.zoom_slider.value()
            if zoom_factor > 1:
                interpolation_methods = {
                    "Nearest": Image.Resampling.NEAREST,
                    "Bilinear": Image.Resampling.BILINEAR,
                    "Cubic": Image.Resampling.BICUBIC
                }
                interpolation_method = interpolation_methods.get(
                    self.interpolation_dropdown.currentText(), 
                    Image.Resampling.BILINEAR
                )
                
                # Convert to uint8 and ensure proper image format
                base_image_uint8 = ((base_image - base_image.min()) * (255.0 / (base_image.max() - base_image.min()))).astype(np.uint8)
                
                self.adjusted_image = np.array(Image.fromarray(base_image_uint8).resize(
                    (base_image.shape[1] * zoom_factor, base_image.shape[0] * zoom_factor),
                    interpolation_method
                ))
            else:
                self.adjusted_image = base_image

            # Get the selected viewport
            target_index = self.viewport_selector.currentIndex() + 1
            ax = self.axes[target_index]
            canvas = self.canvases[target_index]

            # Display the full adjusted image
            self.image = self.adjusted_image
            self.display_image(self.adjusted_image, ax, canvas)

            # Set an initial zoom region (FOV) centered on the image
            h, w = self.adjusted_image.shape
            pixel_size = self.pixel_count_spinbox.value()

            # Calculate the initial limits for the zoomed region (centered FOV)
            x_start = max(0, (w - pixel_size) // 2)
            y_start = max(0, (h - pixel_size) // 2)
            x_end = min(w, x_start + pixel_size)
            y_end = min(h, y_start + pixel_size)

            # Set the axis limits for the zoomed view
            ax.set_xlim(x_start, x_end)
            ax.set_ylim(y_end, y_start)  # Inverted for correct orientation

            canvas.draw()



    def update_resolution(self):
        """Update resolution and adjust FOV."""
        self.process_image()

    def update_pixel_count(self):
        """Update the number of pixels per sample."""
        self.process_image()

    def reset(self):
        """Reset all ROIs and calculations across all viewports"""
        # Clear stored ROIs
        self.signal_rois = []
        self.signal2_rois = []
        self.noise_rois = []
        
        # Remove rectangles from all viewports
        if hasattr(self, 'signal_roi_rects'):
            for rect in self.signal_roi_rects:
                rect.remove()
        if hasattr(self, 'signal2_roi_rects'):
            for rect in self.signal2_roi_rects:
                rect.remove()
        if hasattr(self, 'noise_roi_rects'):
            for rect in self.noise_roi_rects:
                rect.remove()
                
        # Reset coordinates
        self.signal_roi_coords = None
        self.signal2_roi_coords = None
        self.noise_roi_coords = None
        
        # Reset process state and buttons
        self.current_process_state = "idle"
        self.select_signal_button.setEnabled(True)
        self.select_signal2_button.setEnabled(False)
        self.select_noise_button.setEnabled(False)
        self.calculate_snr_button.setEnabled(False)
        self.calculate_cnr_button.setEnabled(False)
        self.status_label.setText("Select Signal ROI")
        
        # Update all canvases
        for canvas in self.canvases:
            canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MedicalImageApp()
    main_window.show()
    sys.exit(app.exec_())