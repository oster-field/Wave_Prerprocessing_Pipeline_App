import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QListWidget, QGroupBox, QMessageBox, QDialog,
                             QProgressBar, QTextEdit, QCheckBox, QLineEdit,
                             QSpinBox, QRadioButton, QAction)
from PyQt5.QtCore import Qt, pyqtSignal, QThread
from PyQt5.QtGui import QFont, QDragEnterEvent, QDropEvent
from pathlib import Path
import pandas as pd
import numpy as np
import datetime
import re
import matplotlib

matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import pyplot as plt

# ==============================================================================
# APPLICATION METADATA
# ==============================================================================
APP_NAME = "Wave Preprocessing Pipeline"
APP_VERSION = "1.2.0"
APP_AUTHOR = "Andrei Tregubov"
APP_YEAR = "2026"

# ==============================================================================
# SCIENTIFIC COLOR PALETTE (Viridis-inspired)
# ==============================================================================
COLORS = {
    'wave_data': '#3b82f6',  # Blue - primary wave data
    'dive_detect': '#ef4444',  # Red - dive detection
    'filtered': '#10b981',  # Green - filtered data
    'spectrum': '#8b5cf6',  # Purple - spectrum
    'grid': '#e5e7eb',  # Light gray - grid
    'text_primary': '#111827',  # Dark gray - text
    'text_secondary': '#6b7280',  # Medium gray - labels
}

# Viridis palette for multiple lines
VIRIDIS_COLORS = ['#440154', '#3b528b', '#21918c', '#5ec962', '#fde725']

# ==============================================================================
# MODERN MATPLOTLIB THEME (light background, refined colors)
# ==============================================================================
plt.rcParams.update({
    'figure.facecolor': '#ffffff',
    'axes.facecolor': '#f8fafc',
    'axes.edgecolor': '#e2e8f0',
    'axes.linewidth': 0.8,
    'axes.labelcolor': '#374151',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'axes.titlecolor': '#111827',
    'axes.titleweight': 'semibold',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.color': '#6b7280',
    'ytick.color': '#6b7280',
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'grid.color': '#e5e7eb',
    'grid.linewidth': 0.7,
    'grid.alpha': 1.0,
    'legend.facecolor': '#ffffff',
    'legend.edgecolor': '#e2e8f0',
    'legend.framealpha': 0.95,
    'legend.fontsize': 9,
    'legend.labelcolor': '#374151',
    'figure.titlesize': 13,
    'font.family': 'DejaVu Sans',
    'lines.antialiased': True,
})

# ==============================================================================
# PATHS
# ==============================================================================
SCRIPT_DIR = Path(__file__).parent
OUTPUT_FOLDER = SCRIPT_DIR / "Output"

# ==============================================================================
# VISUALIZATION CONFIGURATION
# ==============================================================================
# Optimal subsampling for FullHD displays (1920×1080)
# Based on Nyquist theorem: we need 2-3 points per pixel for smooth zoom
# Graph width ≈ 1800px → 1800 × 2.5 = 4500 points optimal
# We use 5000 for safety margin and smooth zooming
VISUALIZATION_TARGET_POINTS = 5000

# For spectrum visualization we want higher detail (100k points)
# because frequency domain requires finer resolution
SPECTRUM_TARGET_POINTS = 100000


# ==============================================================================

# ==============================================================================
# FOOTER FUNCTION FOR ALL WINDOWS
# ==============================================================================
def add_footer(window):
    """Add status bar footer to any QMainWindow with app info"""
    footer_text = f"{APP_NAME} v{APP_VERSION}  |  © {APP_YEAR} {APP_AUTHOR}"
    status_bar = window.statusBar()
    status_bar.showMessage(footer_text)
    status_bar.setStyleSheet("""
        QStatusBar {
            background-color: #f8f9fa;
            color: #6c757d;
            font-size: 10px;
            border-top: 1px solid #dee2e6;
            padding: 2px 8px;
        }
    """)


def create_progress_indicator(current_step):
    """Create compact progress indicator showing pipeline steps

    Args:
        current_step: 0=Load, 1=Plot, 2=Cut, 3=Fourier, 4=Analise

    Returns:
        QWidget with progress indicator
    """
    steps = [
        ("📁", "Load"),
        ("📊", "Plot"),
        ("✂️", "Cut"),
        ("🔄", "Fourier"),
        ("🎯", "Analise")
    ]

    widget = QWidget()
    widget.setFixedHeight(45)
    layout = QHBoxLayout(widget)
    layout.setContentsMargins(20, 5, 20, 5)
    layout.setSpacing(8)

    layout.addStretch()

    for i, (icon, name) in enumerate(steps):
        step_container = QWidget()
        step_container.setFixedSize(100, 32)
        step_layout = QHBoxLayout(step_container)
        step_layout.setContentsMargins(6, 4, 6, 4)
        step_layout.setSpacing(4)

        icon_label = QLabel(icon)
        icon_label.setFixedSize(16, 16)
        icon_label.setFont(QFont("Segoe UI", 9))
        icon_label.setAlignment(Qt.AlignCenter)

        text_label = QLabel(name)
        text_label.setFont(QFont("Segoe UI", 9))
        text_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        step_layout.addWidget(icon_label)
        step_layout.addWidget(text_label)
        step_layout.addStretch()

        if i < current_step:
            step_container.setStyleSheet("""
                QWidget {
                    background-color: #d1fae5;
                    border: 1px solid #10b981;
                    border-radius: 3px;
                }
                QLabel { color: #059669; font-weight: 600; }
            """)
        elif i == current_step:
            step_container.setStyleSheet("""
                QWidget {
                    background-color: #dbeafe;
                    border: 2px solid #3b82f6;
                    border-radius: 3px;
                }
                QLabel { color: #2563eb; font-weight: 700; }
            """)
        else:
            step_container.setStyleSheet("""
                QWidget {
                    background-color: #f9fafb;
                    border: 1px solid #e5e7eb;
                    border-radius: 3px;
                }
                QLabel { color: #9ca3af; font-weight: 400; }
            """)

        layout.addWidget(step_container)

        if i < len(steps) - 1:
            arrow = QLabel("›")
            arrow.setFixedWidth(12)
            arrow.setFont(QFont("Segoe UI", 14, QFont.Bold))
            arrow.setStyleSheet("color: #d1d5db;")
            arrow.setAlignment(Qt.AlignCenter)
            layout.addWidget(arrow)

    layout.addStretch()

    widget.setStyleSheet("""
        QWidget {
            background-color: #ffffff;
            border-bottom: 1px solid #e5e7eb;
        }
    """)

    return widget


def read_sensor_freq_from_csv(csv_path, default=None):
    """Read sensor frequency from CSV comment header. Returns int Hz.
    Raises ValueError if not found and no default provided."""
    try:
        with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if not line.startswith('#'):
                    break
                m = re.search(r'[Ss]ensor\s+frequenc[a-z]*[^:]*:\s*(\d+)', line)
                if m:
                    return int(m.group(1))
    except Exception:
        pass
    if default is not None:
        return default
    raise ValueError(
        f"Could not find \'Sensor frequency: N Hz\' in header of:\n{csv_path}\n"
        "Re-run the pipeline from Step 1 to regenerate the file."
    )


def _lbl(text, style=""):
    """Convenience: create a plain QLabel with optional stylesheet."""
    w = QLabel(text)
    if style:
        w.setStyleSheet(style)
    return w


class ProcessingThread(QThread):
    """Background thread for data processing"""
    progress = pyqtSignal(int, str)  # percentage, message
    finished = pyqtSignal(bool, object)  # success, result_dataframe

    def __init__(self, info_file, data_files):
        super().__init__()
        self.info_file = info_file
        self.data_files = data_files
        self.should_stop = False

    def stop(self):
        self.should_stop = True

    def run(self):
        try:
            # Step 1: Read INFO file
            self.progress.emit(5, "Reading INFO file...")
            metadata = self.read_info_file()

            # Step 2: Read all data files at once
            self.progress.emit(10, f"Loading {len(self.data_files)} files...")

            all_surface_displacement_data = []

            for i, file_path in enumerate(self.data_files):
                if self.should_stop:
                    return

                # Read data from file
                data = self.read_data_file(file_path)
                all_surface_displacement_data.append(data)

                # Update progress
                progress_pct = 10 + int((i + 1) / len(self.data_files) * 30)
                self.progress.emit(progress_pct, f"Loaded {i + 1}/{len(self.data_files)}")

            if self.should_stop:
                return

            # Step 3: Concatenate all data into single array
            self.progress.emit(45, "Combining data...")
            all_data = np.concatenate(all_surface_displacement_data)

            # Step 4: Split into 20-minute readings
            self.progress.emit(50, "Splitting into 20-min readings...")
            points_per_reading = metadata['sensor_frequency'] * 1200

            # Only keep complete readings
            num_complete_readings = len(all_data) // points_per_reading
            all_data = all_data[:num_complete_readings * points_per_reading]

            # Step 5: Create reading numbers (vectorized!)
            self.progress.emit(60, "Creating reading numbers...")
            reading_numbers = np.repeat(np.arange(1, num_complete_readings + 1), points_per_reading)

            # Step 6: Generate timestamps (vectorized!)
            self.progress.emit(70, "Generating timestamps...")
            start_time = datetime.datetime.combine(metadata['date_start'], datetime.time())

            # Calculate frequency in milliseconds
            time_delta_milliseconds = 1000.0 / metadata['sensor_frequency']  # ms per point

            # Create timestamps using milliseconds
            timestamps = pd.date_range(
                start=start_time,
                periods=len(all_data),
                freq=f'{time_delta_milliseconds}ms'
            )

            # Step 7: Create DataFrame (single operation!)
            self.progress.emit(85, "Creating DataFrame...")
            final_df = pd.DataFrame({
                'timestamp': timestamps,
                'surface_displacement': all_data,
                'reading_number': reading_numbers
            })

            # Add metadata as attributes
            final_df.attrs['description'] = 'Raw data immediately after transfer from .dat files'
            final_df.attrs['sensor_frequency_hz'] = metadata['sensor_frequency']
            final_df.attrs['recording_start'] = str(metadata['date_start'])
            final_df.attrs['recording_end'] = str(metadata['date_end'])
            final_df.attrs['points_per_reading'] = points_per_reading
            final_df.attrs['total_readings'] = num_complete_readings

            # Step 8: Save to CSV
            self.progress.emit(90, "Saving to CSV file...")

            # Get output path - Output folder next to the script
            output_folder = OUTPUT_FOLDER
            output_folder.mkdir(exist_ok=True)  # Create if doesn't exist
            output_file = output_folder / "Step1_TXTtoCSV.csv"

            # Save with metadata as comments in header
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("# STEP 1: TXT to CSV - Raw Data\n")
                f.write("# ==========================================\n")
                f.write("# Description: Raw data immediately after transfer from .dat files\n")
                f.write(f"# Sensor frequency: {metadata['sensor_frequency']} Hz\n")
                f.write(f"# Recording start: {metadata['date_start']}\n")
                f.write(f"# Recording end: {metadata['date_end']}\n")
                f.write(f"# Points per reading (20 min): {points_per_reading}\n")
                f.write(f"# Total readings: {num_complete_readings}\n")
                f.write(f"# Total data points: {len(final_df)}\n")
                f.write(f"# Files processed: {len(self.data_files)}\n")
                f.write("# ==========================================\n")

            # Append actual data
            final_df.to_csv(output_file, mode='a', index=False)

            self.progress.emit(100, f"Complete! Saved to {output_file.name}")
            self.finished.emit(True, final_df)

        except Exception as e:
            import traceback
            error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
            self.progress.emit(0, error_msg)
            self.finished.emit(False, None)

    def read_info_file(self):
        """Read metadata from INFO file using regex - flexible parsing"""
        # Try every encoding until we find the frequency — never fall back to default
        sensor_frequency = None
        content = None
        for encoding in ['utf-8', 'windows-1251', 'cp1251', 'latin-1']:
            try:
                with open(self.info_file, 'r', encoding=encoding, errors='ignore') as f:
                    content = f.read()
                freq_match = re.search(r'[Чч]астота\s+опроса[^:]*:\s*(\d+)', content) \
                             or re.search(r'[Ff]requency[^:]*:\s*(\d+)', content)
                if freq_match:
                    sensor_frequency = int(freq_match.group(1))
                    break  # Found — stop trying encodings
            except Exception:
                continue
        if sensor_frequency is None:
            raise ValueError(
                "Could not find sensor frequency in INFO file.\n"
                "Expected a line like: 'Частота опроса: 8 Гц' or 'Frequency: 8 Hz'"
            )

        # Datetimes: "2022.10.18 11:47:48.000"
        dt_pattern = re.compile(r'(\d{4}\.\d{2}\.\d{2}\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?)')
        found = dt_pattern.findall(content)

        dt_start = dt_end = None
        date_start = date_end = None
        for s in found:
            for fmt in ('%Y.%m.%d %H:%M:%S.%f', '%Y.%m.%d %H:%M:%S'):
                try:
                    dt = datetime.datetime.strptime(s.strip(), fmt)
                    if dt_start is None:
                        dt_start = dt
                        date_start = dt.date()
                    else:
                        dt_end = dt
                        date_end = dt.date()
                    break
                except ValueError:
                    continue

        return {
            'date_start': date_start,
            'date_end': date_end,
            'dt_start': dt_start,
            'sensor_frequency': sensor_frequency,
        }

    def read_data_file(self, file_path):
        """Read data from .dat/.txt/.npy file"""
        file_ext = Path(file_path).suffix.lower()

        if file_ext == '.npy':
            return np.load(file_path)
        elif file_ext in ['.dat', '.txt']:
            # Read raw bytes and decode
            with open(file_path, 'rb') as f:
                content = f.read().decode('utf-8', errors='ignore')

            # Replace comma decimal separator with dot (Russian locale files)
            content = content.replace(',', '.')

            # Parse line by line, skip empty and non-numeric lines
            values = []
            for line in content.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    values.append(float(line))
                except ValueError:
                    continue  # skip header/text lines

            return np.array(values, dtype=np.float64)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")


class FileDropZone(QLabel):
    """Drop zone widget for dragging and dropping files."""
    files_dropped = pyqtSignal(list)

    def __init__(self, text="", allowed_extensions=None):
        super().__init__(text)
        self.allowed_extensions = allowed_extensions or []
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumHeight(120)
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #cbd5e1;
                border-radius: 10px;
                background-color: #f8fafc;
                color: #94a3b8;
                font-size: 13px;
                padding: 20px;
                font-family: 'Segoe UI', sans-serif;
            }
            QLabel:hover {
                background-color: #f0f9ff;
                border-color: #38bdf8;
                color: #0369a1;
            }
        """)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet(
                self.styleSheet().replace('#f8fafc', '#f0fdf4').replace('dashed #cbd5e1', 'solid #4ade80'))

    def dragLeaveEvent(self, event):
        self.setStyleSheet(self.styleSheet().replace('#f0fdf4', '#f8fafc').replace('solid #4ade80', 'dashed #cbd5e1'))

    def dropEvent(self, event: QDropEvent):
        files = [url.toLocalFile() for url in event.mimeData().urls()]

        # Filter by allowed extensions if specified
        if self.allowed_extensions:
            valid_files = [f for f in files if any(f.endswith(ext) for ext in self.allowed_extensions)]
        else:
            valid_files = files

        if valid_files:
            self.files_dropped.emit(valid_files)

        self.setStyleSheet(self.styleSheet().replace('#f0fdf4', '#f8fafc').replace('solid #4ade80', 'dashed #cbd5e1'))


class ProgressDialog(QDialog):
    """Dialog showing processing progress"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Processing Data")
        self.setModal(True)
        self.setFixedSize(500, 250)

        layout = QVBoxLayout(self)

        # Title
        title = QLabel("Processing Wave Data")
        title.setFont(QFont("Segoe UI", 13, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #111827; padding: 8px; letter-spacing: 0.2px;")
        layout.addWidget(title)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: none;
                border-radius: 6px;
                text-align: center;
                height: 26px;
                background-color: #f1f5f9;
                color: #64748b;
                font-size: 11px;
                font-family: 'Segoe UI', sans-serif;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #0ea5e9, stop:1 #38bdf8);
                border-radius: 6px;
            }
        """)
        layout.addWidget(self.progress_bar)

        # Status message
        self.status_label = QLabel("Starting...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet(
            "color: #6b7280; padding: 8px; font-size: 12px; font-family: 'Segoe UI', sans-serif;")
        layout.addWidget(self.status_label)

        # Log window
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(80)
        self.log_text.setStyleSheet("""
            QTextEdit {
                border: 1px solid #e5e7eb;
                border-radius: 6px;
                background-color: #f9fafb;
                color: #6b7280;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 10px;
                padding: 4px;
            }
        """)
        layout.addWidget(self.log_text)

    def update_progress(self, percentage, message):
        """Update progress bar and message"""
        self.progress_bar.setValue(percentage)
        self.status_label.setText(message)
        self.log_text.append(f"[{percentage}%] {message}")


class MainWindow(QMainWindow):
    """Main application window — file loading and pipeline entry point."""

    def __init__(self):
        super().__init__()
        self.info_file = None
        self.data_files = []
        self.init_ui()

    def init_ui(self):
        """Initialize the main window UI."""
        self.setWindowTitle("🌊 Wave data preprocessing pipeline")

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Header
        header = QLabel("🌊  Wave Data Preprocessing Pipeline")
        header.setFont(QFont("Segoe UI", 20, QFont.Bold))
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("color: #111827; padding: 24px 20px 6px 20px; letter-spacing: -0.3px;")
        layout.addWidget(header)

        # Instruction
        instruction = QLabel("Load metadata and data files to begin processing")
        instruction.setAlignment(Qt.AlignCenter)
        instruction.setStyleSheet(
            "color: #9ca3af; font-size: 13px; padding-bottom: 16px; font-family: 'Segoe UI', sans-serif;")
        layout.addWidget(instruction)

        # INFO file section
        info_group = self.create_info_section()
        layout.addWidget(info_group)

        # Data files section
        data_group = self.create_data_section()
        layout.addWidget(data_group)

        # Continue button
        self.btn_continue = QPushButton("▶  Continue to Step 1 — Plot Raw Data")
        self.btn_continue.setEnabled(False)
        self.btn_continue.setStyleSheet("""
            QPushButton {
                background-color: #0ea5e9;
                color: white;
                font-size: 15px;
                font-weight: 600;
                padding: 13px 20px;
                border-radius: 8px;
                margin-top: 14px;
                border: none;
                font-family: 'Segoe UI', sans-serif;
                letter-spacing: 0.2px;
            }
            QPushButton:hover:enabled {
                background-color: #0284c7;
            }
            QPushButton:pressed:enabled {
                background-color: #0369a1;
            }
            QPushButton:disabled {
                background-color: #e5e7eb;
                color: #9ca3af;
            }
        """)
        self.btn_continue.clicked.connect(self.on_continue)
        layout.addWidget(self.btn_continue)

        # Status
        self.status_label = QLabel("Waiting for files...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet(
            "color: #9ca3af; padding: 10px; font-family: 'Segoe UI', sans-serif; font-size: 12px;")
        layout.addWidget(self.status_label)

        layout.addStretch()

        self.apply_global_styles()

        # Add footer BEFORE showing window
        add_footer(self)

        # Show maximized AFTER UI is fully built
        self.showMaximized()

    def create_info_section(self):
        """Section for INFO file"""
        group = QGroupBox("INFO File")
        layout = QVBoxLayout()

        # Drop zone for INFO
        self.info_drop = FileDropZone(
            "🎯 Drag & Drop INFO.dat file here\nor click button below",
            allowed_extensions=['.dat', '.txt']
        )
        self.info_drop.files_dropped.connect(self.on_info_dropped)
        layout.addWidget(self.info_drop)

        # Buttons
        btn_layout = QHBoxLayout()

        btn_browse_info = QPushButton("📂 Browse INFO File")
        btn_browse_info.clicked.connect(self.browse_info_file)
        btn_layout.addWidget(btn_browse_info)

        self.btn_clear_info = QPushButton("🗑️ Clear")
        self.btn_clear_info.clicked.connect(self.clear_info)
        self.btn_clear_info.setEnabled(False)
        btn_layout.addWidget(self.btn_clear_info)

        layout.addLayout(btn_layout)

        # Info about loaded file
        self.info_label = QLabel("No file loaded")
        self.info_label.setStyleSheet(
            "color: #9ca3af; font-style: italic; padding: 5px 8px; font-size: 12px; font-family: 'Segoe UI', sans-serif;")
        layout.addWidget(self.info_label)

        group.setLayout(layout)
        return group

    def create_data_section(self):
        """Section for data files"""
        group = QGroupBox("Data Files")
        layout = QVBoxLayout()

        # Drop zone for data
        self.data_drop = FileDropZone(
            "🎯 Drag & Drop data files here (.dat, .txt, .npy)\nor click button below",
            allowed_extensions=['.dat', '.txt', '.npy']
        )
        self.data_drop.files_dropped.connect(self.on_data_dropped)
        layout.addWidget(self.data_drop)

        # Buttons
        btn_layout = QHBoxLayout()

        btn_browse_data = QPushButton("📂 Browse Data Files")
        btn_browse_data.clicked.connect(self.browse_data_files)
        btn_layout.addWidget(btn_browse_data)

        self.btn_clear_data = QPushButton("🗑️ Clear All")
        self.btn_clear_data.clicked.connect(self.clear_data)
        self.btn_clear_data.setEnabled(False)
        btn_layout.addWidget(self.btn_clear_data)

        layout.addLayout(btn_layout)

        # List of loaded files
        self.data_list = QListWidget()
        self.data_list.setMaximumHeight(200)
        self.data_list.setStyleSheet("""
            QListWidget {
                border: 1px solid #e5e7eb;
                border-radius: 8px;
                background-color: #ffffff;
                padding: 4px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 12px;
                color: #374151;
            }
            QListWidget::item {
                padding: 4px 8px;
                border-radius: 4px;
            }
            QListWidget::item:selected {
                background-color: #e0f2fe;
                color: #0369a1;
            }
            QListWidget::item:hover {
                background-color: #f8fafc;
            }
        """)
        layout.addWidget(self.data_list)

        # File counter
        self.data_count_label = QLabel("Files loaded: 0")
        self.data_count_label.setStyleSheet(
            "color: #9ca3af; padding: 4px 6px; font-size: 12px; font-family: 'Segoe UI', sans-serif;")
        layout.addWidget(self.data_count_label)

        group.setLayout(layout)
        return group

    def on_info_dropped(self, files):
        """Handle INFO file drop event."""
        if files:
            # Take the first file only
            self.set_info_file(files[0])

    def on_data_dropped(self, files):
        """Handle data files drop event."""
        if files:
            self.add_data_files(files)

    def browse_info_file(self):
        """Browse for INFO file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select INFO File",
            "",
            "Data Files (*.dat *.txt);;All Files (*.*)"
        )
        if file_path:
            self.set_info_file(file_path)

    def browse_data_files(self):
        """Browse for data files"""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Data Files",
            "",
            "Data Files (*.dat *.txt *.npy);;All Files (*.*)"
        )
        if files:
            self.add_data_files(files)

    def set_info_file(self, file_path):
        """Set INFO file and display parsed metadata"""
        self.info_file = file_path
        filename = Path(file_path).name

        try:
            meta = self._parse_info_file(file_path)

            lines = [f"✅ {filename}"]
            lines.append(f"📡 Frequency: {meta['sensor_frequency']} Hz")
            if meta['dt_start']:
                lines.append(f"🕐 Start:  {meta['dt_start'].strftime('%Y-%m-%d  %H:%M:%S')}")
            if meta['dt_end']:
                lines.append(f"🕑 End:    {meta['dt_end'].strftime('%Y-%m-%d  %H:%M:%S')}")
            if meta['recording_duration']:
                lines.append(f"⏱  Duration: {meta['recording_duration']}")
            if meta['total_measurements']:
                lines.append(f"📊 Measurements: {meta['total_measurements']:,}")

            self.info_label.setText("\n".join(lines))
            self.info_label.setStyleSheet(
                "color: #16a34a; font-weight: 600; padding: 5px 8px; font-family: 'Consolas', 'Courier New', monospace; font-size: 12px; line-height: 1.6;"
            )
        except Exception as e:
            self.info_label.setText(f"Loaded: {filename}\n⚠  Parse error: {str(e)}")
            self.info_label.setStyleSheet(
                "color: #d97706; font-weight: 600; padding: 5px 8px; font-size: 12px; font-family: 'Segoe UI', sans-serif;")

        self.btn_clear_info.setEnabled(True)
        self.update_status()

    def _parse_info_file(self, file_path):
        """Parse INFO file with regex - handles any encoding, flexible format"""
        # Try every encoding until we find the frequency — never fall back to default
        sensor_frequency = None
        content = None
        for encoding in ['utf-8', 'windows-1251', 'cp1251', 'latin-1']:
            try:
                with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                    content = f.read()
                freq_match = re.search(r'[Чч]астота\s+опроса[^:]*:\s*(\d+)', content) \
                             or re.search(r'[Ff]requency[^:]*:\s*(\d+)', content)
                if freq_match:
                    sensor_frequency = int(freq_match.group(1))
                    break  # Found — stop trying encodings
            except Exception:
                continue
        if sensor_frequency is None:
            raise ValueError(
                "Could not find sensor frequency in INFO file.\n"
                "Expected a line like: 'Частота опроса: 8 Гц' or 'Frequency: 8 Hz'"
            )

        # Datetimes: "2022.10.18 11:47:48.000"
        dt_pattern = re.compile(r'(\d{4}\.\d{2}\.\d{2}\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?)')
        found = dt_pattern.findall(content)

        dt_start = dt_end = None
        for s in found:
            for fmt in ('%Y.%m.%d %H:%M:%S.%f', '%Y.%m.%d %H:%M:%S'):
                try:
                    dt = datetime.datetime.strptime(s.strip(), fmt)
                    if dt_start is None:
                        dt_start = dt
                    else:
                        dt_end = dt
                    break
                except ValueError:
                    continue

        # Duration
        recording_duration = None
        if dt_start and dt_end:
            delta = dt_end - dt_start
            total_s = int(delta.total_seconds())
            days = delta.days
            hours = (total_s % 86400) // 3600
            mins = (total_s % 3600) // 60
            secs = total_s % 60
            recording_duration = (
                f"{days}d {hours:02d}h {mins:02d}m {secs:02d}s"
                if days > 0 else
                f"{hours:02d}h {mins:02d}m {secs:02d}s"
            )

        # Total measurements
        meas_match = re.search(r'[Кк]оличество\s+измерений[^:]*:\s*(\d+)', content)
        total_measurements = int(meas_match.group(1)) if meas_match else None

        return {
            'sensor_frequency': sensor_frequency,
            'dt_start': dt_start,
            'dt_end': dt_end,
            'date_start': dt_start.date() if dt_start else None,
            'date_end': dt_end.date() if dt_end else None,
            'recording_duration': recording_duration,
            'total_measurements': total_measurements,
        }

    def add_data_files(self, files):
        """Add data files"""
        # Filter duplicates
        new_files = [f for f in files if f not in self.data_files]

        if not new_files:
            return

        self.data_files.extend(new_files)

        # Update list
        self.data_list.clear()
        for file_path in self.data_files:
            self.data_list.addItem(Path(file_path).name)

        # Update counter
        self.data_count_label.setText(f"Files loaded: {len(self.data_files)}")
        self.btn_clear_data.setEnabled(True)

        self.update_status()

    def clear_info(self):
        """Clear INFO file"""
        self.info_file = None
        self.info_label.setText("No file loaded")
        self.info_label.setStyleSheet(
            "color: #ef4444; font-style: italic; padding: 5px 8px; font-size: 12px; font-family: 'Segoe UI', sans-serif;")
        self.btn_clear_info.setEnabled(False)
        self.update_status()

    def clear_data(self):
        """Clear all data files"""
        self.data_files = []
        self.data_list.clear()
        self.data_count_label.setText("Files loaded: 0")
        self.btn_clear_data.setEnabled(False)
        self.update_status()

    def update_status(self):
        """Update status and continue button availability"""
        if self.info_file and self.data_files:
            self.status_label.setText(f"Ready to process — INFO + {len(self.data_files)} data files loaded")
            self.status_label.setStyleSheet(
                "color: #16a34a; font-weight: 600; padding: 10px; font-family: 'Segoe UI', sans-serif; font-size: 12px;")
            self.btn_continue.setEnabled(True)
        elif self.info_file:
            self.status_label.setText("Load data files to continue")
            self.status_label.setStyleSheet(
                "color: #d97706; padding: 10px; font-family: 'Segoe UI', sans-serif; font-size: 12px;")
            self.btn_continue.setEnabled(False)
        elif self.data_files:
            self.status_label.setText("Load INFO file to continue")
            self.status_label.setStyleSheet(
                "color: #d97706; padding: 10px; font-family: 'Segoe UI', sans-serif; font-size: 12px;")
            self.btn_continue.setEnabled(False)
        else:
            self.status_label.setText("Waiting for files...")
            self.status_label.setStyleSheet(
                "color: #9ca3af; padding: 10px; font-family: 'Segoe UI', sans-serif; font-size: 12px;")
            self.btn_continue.setEnabled(False)

    def on_continue(self):
        """Continue to processing"""
        # Show progress dialog
        self.progress_dialog = ProgressDialog(self)

        # Create processing thread
        self.processing_thread = ProcessingThread(self.info_file, self.data_files)
        self.processing_thread.progress.connect(self.progress_dialog.update_progress)
        self.processing_thread.finished.connect(self.on_processing_finished)

        # Start processing
        self.processing_thread.start()
        self.progress_dialog.exec_()

    def on_processing_finished(self, success, result_df):
        """Called when processing is complete"""
        if hasattr(self, 'progress_dialog'):
            self.progress_dialog.close()

        if success and result_df is not None:
            # Get output file path
            output_folder = OUTPUT_FOLDER
            output_file = output_folder / "Step1_TXTtoCSV.csv"
            viz_cache_file = output_folder / "Step1_Visualization.csv"

            # Create visualization cache (subsampled data)
            viz_data = result_df.copy()
            if len(viz_data) > 10000:
                step = len(viz_data) // 10000
                viz_data = viz_data.iloc[::step].reset_index(drop=True)

            # Save cache
            with open(viz_cache_file, 'w', encoding='utf-8') as f:
                f.write("# VISUALIZATION CACHE - Subsampled data for fast plotting\n")
                f.write("# ==========================================\n")
                f.write(f"# Sensor frequency: {result_df.attrs.get('sensor_frequency_hz', 'N/A')} Hz\n")
                f.write(f"# Recording start: {result_df.attrs.get('recording_start', 'N/A')}\n")
                f.write(f"# Recording end: {result_df.attrs.get('recording_end', 'N/A')}\n")
                f.write(f"# Sampled points: {len(viz_data)}\n")
                f.write(f"# Original points: {len(result_df)}\n")
                f.write("# ==========================================\n")

            viz_data.to_csv(viz_cache_file, mode='a', index=False)

            # Show success message
            QMessageBox.information(
                self,
                "Success!",
                f"✅ Data processed and saved!\n\n"
                f"📊 Statistics:\n"
                f"  • Total points: {len(result_df):,}\n"
                f"  • Total readings (20-min): {result_df['reading_number'].max()}\n"
                f"  • Sensor frequency: {result_df.attrs.get('sensor_frequency_hz', 'N/A')} Hz\n"
                f"  • Recording period: {result_df.attrs.get('recording_start')} to {result_df.attrs.get('recording_end')}\n\n"
            )

            # Store result for visualization - use subsampled data
            self.processed_data = viz_data

            # Close this window and open visualization
            self.open_visualization_window()

        else:
            # Get error message from progress dialog log
            error_text = ""
            if hasattr(self, 'progress_dialog'):
                error_text = self.progress_dialog.log_text.toPlainText()

            # Show detailed error
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle("Processing Error")
            msg.setText("❌ Processing failed!")
            msg.setInformativeText("Check the detailed error below:")
            msg.setDetailedText(error_text)
            msg.exec_()

    def open_visualization_window(self):
        """Open visualization window"""
        # Create BEFORE closing self — store on QApplication to prevent GC
        viz = VisualizationWindow(self.processed_data)
        QApplication.instance()._viz_window = viz
        viz.show()
        self.close()

    def apply_global_styles(self):
        """Apply global stylesheet."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f9fafb;
            }
            QWidget {
                background-color: #f9fafb;
                font-family: 'Segoe UI', sans-serif;
            }
            QGroupBox {
                font-weight: 600;
                font-size: 13px;
                border: 1px solid #e5e7eb;
                border-radius: 10px;
                margin-top: 14px;
                padding-top: 16px;
                background-color: #ffffff;
                color: #374151;
            }
            QGroupBox::title {
                color: #374151;
                subcontrol-origin: margin;
                left: 14px;
                padding: 0 8px;
                background-color: #ffffff;
            }
            QPushButton {
                background-color: #f3f4f6;
                color: #374151;
                border: 1px solid #d1d5db;
                padding: 8px 18px;
                border-radius: 6px;
                font-size: 13px;
                font-weight: 600;
                font-family: 'Segoe UI', sans-serif;
            }
            QPushButton:hover {
                background-color: #e5e7eb;
                border-color: #9ca3af;
            }
            QPushButton:pressed {
                background-color: #d1d5db;
            }
            QPushButton:disabled {
                background-color: #f3f4f6;
                color: #9ca3af;
                border-color: #e5e7eb;
            }
        """)


def process_zero_mean(step2_file, output_folder, progress_bar, status):
    """
    Shared Step 2 post-processing:
    1. Subtract global mean from all surface_displacement values → Step2_Zero_Mean.csv
    2. Save per-reading averages → Parameters.csv
    3. Save subsampled cache → Step2_Visualization.csv

    Called by both VisualizationWindow (skip path) and ManualRemovalWindow (trim path).
    """
    status.setText("Reading Step2_Initial_Cut.csv...")
    QApplication.processEvents()

    data = pd.read_csv(step2_file, comment='#')

    # Step 1: Global mean (Avg_Depth_FullRec)
    progress_bar.setValue(85)
    status.setText("Calculating global average (Avg_Depth_FullRec)...")
    QApplication.processEvents()

    avg_depth_full_rec = data['surface_displacement'].mean()

    # Step 2: Per-reading averages
    progress_bar.setValue(88)
    status.setText("Calculating averages for each 20-min reading...")
    QApplication.processEvents()

    reading_averages = data.groupby('reading_number')['surface_displacement'].mean().reset_index()
    reading_averages.columns = ['reading_number', 'average_depth']

    # Step 3: Subtract global mean
    progress_bar.setValue(92)
    status.setText("Creating Zero Mean data...")
    QApplication.processEvents()

    zero_mean_data = data.copy()
    zero_mean_data['surface_displacement'] = zero_mean_data['surface_displacement'] - avg_depth_full_rec

    # Step 4: Save Step2_Zero_Mean.csv
    progress_bar.setValue(95)
    status.setText("Saving Step2_Zero_Mean.csv...")
    QApplication.processEvents()

    zero_mean_file = output_folder / "Step2_Zero_Mean.csv"
    with open(zero_mean_file, 'w', encoding='utf-8') as f:
        f.write("# STEP 2: Zero Mean - Global average subtracted\n")
        f.write("# ==========================================\n")
        f.write(f"# Sensor frequency: {read_sensor_freq_from_csv(step2_file)} Hz\n")
        f.write(f"# Average Depth (Full Record): {avg_depth_full_rec:.6f}\n")
        f.write(f"# All surface_displacement values have this subtracted\n")
        f.write("# ==========================================\n")

    zero_mean_data.to_csv(zero_mean_file, mode='a', index=False)

    # Step 5: Save subsampled visualization cache
    progress_bar.setValue(96)
    status.setText("Creating Step2_Visualization.csv...")
    QApplication.processEvents()

    subsample_step = max(1, len(zero_mean_data) // VISUALIZATION_TARGET_POINTS)
    viz_data = zero_mean_data.iloc[::subsample_step].copy()

    step2_viz_file = output_folder / "Step2_Visualization.csv"
    with open(step2_viz_file, 'w', encoding='utf-8') as f:
        f.write("# STEP 2: Visualization Cache - Subsampled Zero Mean data\n")
        f.write("# ==========================================\n")
        f.write(f"# Sampled points: {len(viz_data)}\n")
        f.write(f"# Original points: {len(zero_mean_data)}\n")
        f.write("# ==========================================\n")

    viz_data.to_csv(step2_viz_file, mode='a', index=False)

    # Step 6: Save Parameters.csv
    progress_bar.setValue(98)
    status.setText("Saving Parameters.csv...")
    QApplication.processEvents()

    parameters_file = output_folder / "Parameters.csv"
    with open(parameters_file, 'w', encoding='utf-8') as f:
        f.write("# PARAMETERS - 20-minute readings and their characteristics\n")
        f.write("# ==========================================\n")
        f.write(f"# Average Depth (Full Record): {avg_depth_full_rec:.6f}\n")
        f.write("# ==========================================\n")

    reading_averages.to_csv(parameters_file, mode='a', index=False)


class VisualizationWindow(QMainWindow):
    """Window for visualizing processed data"""

    def __init__(self, data_df):
        super().__init__()
        self.data_df = data_df
        self.init_ui()

    def init_ui(self):
        """Initialize visualization window"""
        self.setWindowTitle("🌊 Wave Data Visualization")
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(0, self.showMaximized)  # defer until event loop is running

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Progress indicator
        progress = create_progress_indicator(current_step=1)
        layout.addWidget(progress)

        # Header
        header = QLabel("Raw Data Visualization")
        header.setFont(QFont("Segoe UI", 17, QFont.Bold))
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("color: #111827; padding: 14px; letter-spacing: -0.2px;")
        layout.addWidget(header)

        # Info label
        info_text = (f"Readings: {self.data_df['reading_number'].max()} | "
                     f"Frequency: {self.data_df.attrs.get('sensor_frequency_hz', 'N/A')} Hz")
        info_label = QLabel(info_text)
        info_label.setAlignment(Qt.AlignCenter)
        info_label.setStyleSheet("color: #6b7280; font-size: 12px; padding: 4px; font-family: 'Consolas', monospace;")
        layout.addWidget(info_label)

        # Plot canvas with interactive toolbar
        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
        self.canvas = self.create_plot()
        toolbar = NavigationToolbar2QT(self.canvas, self)
        full_data_action = QAction('📊 Build all data points (slow)', self)
        full_data_action.triggered.connect(self.build_full_data_step1)
        toolbar.addAction(full_data_action)
        layout.addWidget(toolbar)
        layout.addWidget(self.canvas)

        # Buttons
        btn_layout = QHBoxLayout()

        self.btn_skip = QPushButton("Continue WITHOUT manual removal")
        self.btn_skip.setStyleSheet("""
            QPushButton {
                background-color: #fff1f2;
                color: #e11d48;
                font-size: 14px;
                font-weight: 600;
                padding: 13px 20px;
                border-radius: 8px;
                border: 1px solid #fecdd3;
                font-family: 'Segoe UI', sans-serif;
            }
            QPushButton:hover {
                background-color: #ffe4e6;
                border-color: #fda4af;
            }
        """)
        self.btn_skip.clicked.connect(self.on_skip_removal)
        btn_layout.addWidget(self.btn_skip)

        self.btn_manual = QPushButton("✂️  Proceed with Manual Data Removal")
        self.btn_manual.setStyleSheet("""
            QPushButton {
                background-color: #0ea5e9;
                color: white;
                font-size: 14px;
                font-weight: 600;
                padding: 13px 20px;
                border-radius: 8px;
                border: none;
                font-family: 'Segoe UI', sans-serif;
            }
            QPushButton:hover {
                background-color: #0284c7;
            }
            QPushButton:disabled {
                background-color: #e5e7eb;
                color: #9ca3af;
            }
        """)
        self.btn_manual.clicked.connect(self.on_manual_removal)
        btn_layout.addWidget(self.btn_manual)

        self._update_manual_btn_state()  # btn_manual exists now; reads _has_dives from create_plot

        layout.addLayout(btn_layout)

        self.apply_styles()

        # Add footer
        add_footer(self)

    def create_plot(self):
        """Create matplotlib plot - optimized for speed"""

        # Get screen resolution for adaptive subsampling
        try:
            screen_width = QApplication.primaryScreen().size().width()
        except Exception:
            screen_width = 1920  # safe fallback

        # Adaptive point count based on screen resolution
        # Base: 5000 points for FullHD (optimal by Nyquist theorem)
        # Scale proportionally for higher resolutions
        if screen_width >= 2560:  # 4K (2.67x more pixels)
            target_points = VISUALIZATION_TARGET_POINTS * 3  # 15000
        elif screen_width >= 1920:  # Full HD
            target_points = VISUALIZATION_TARGET_POINTS * 2  # 10000 (more for initial view)
        else:  # HD or lower
            target_points = VISUALIZATION_TARGET_POINTS  # 5000

        # Create figure
        fig = Figure(figsize=(14, 6), dpi=100)
        canvas = FigureCanvas(fig)

        ax = fig.add_subplot(111)

        # Subsample data for visualization
        data_to_plot = self.data_df.copy()

        if len(data_to_plot) > target_points:
            step = len(data_to_plot) // target_points
            data_to_plot = data_to_plot.iloc[::step].reset_index(drop=True)

        # Detect dives
        dive_mask = self.detect_dives(data_to_plot['surface_displacement'].values)

        # Save result for button state update
        self._has_dives = bool(dive_mask.sum() > 0)

        # Convert timestamps
        timestamps = pd.to_datetime(data_to_plot['timestamp'], errors='coerce')
        surface_displacement = data_to_plot['surface_displacement'].values

        # FIRST: Draw complete blue line (no gaps)
        ax.plot(timestamps, surface_displacement,
                linewidth=0.5, color=COLORS['wave_data'], alpha=0.7, label='Wave data',
                marker='o', markersize=1.5, markerfacecolor=COLORS['wave_data'],
                markeredgewidth=0, zorder=2)

        # SECOND: Overlay red segments on top (no connecting lines between segments)
        if dive_mask.sum() > 0:
            dive_indices = np.where(dive_mask)[0]

            # Vectorised segment extraction: find index gaps > 1
            breaks = np.where(np.diff(dive_indices) > 1)[0]
            starts = np.concatenate([[dive_indices[0]], dive_indices[breaks + 1]])
            ends = np.concatenate([dive_indices[breaks], [dive_indices[-1]]])
            segments = list(zip(starts, ends))

            # Plot each segment separately (prevents connecting lines)
            for i, (start, end) in enumerate(segments):
                label = 'Sensor deployment/retrieval' if i == 0 else None
                ax.plot(timestamps[start:end + 1], surface_displacement[start:end + 1],
                        linewidth=1.0, color=COLORS['dive_detect'], alpha=0.9, label=label,
                        marker='o', markersize=1.5, markerfacecolor=COLORS['dive_detect'],
                        markeredgewidth=0, zorder=3)

        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Surface displacement (meters)', fontsize=12)
        ax.set_title(f'Raw Data - sensor deployment and/or retrieval will be automatically detected',
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, zorder=0)
        ax.legend(loc='upper right')

        # Format x-axis with dates
        import matplotlib.dates as mdates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%y'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        fig.autofmt_xdate()

        fig.tight_layout()

        return canvas

    def detect_dives(self, surface_displacement, sensitivity=3.0):
        """
        Dive detector based on gradient analysis.

        Key rules:
        1. GUARD: if first point >= 2.0  → no beginning leg.
                  if last  point >= 2.0  → no ending leg.
           (Deployment always starts in air/shallow <2, retrieval ends there.)

        2. BEGINNING LEG — search in first 40% of data:
           Find ALL positive gradient spikes where:
             - gradient > sensitivity * grad_std  (significant spike)
             - surface displacement BEFORE spike < 2.0        (coming from air/shallow)
             - surface displacement AFTER  spike >= 2.0       (arrived at depth)
           These are true water-entry events. The leg ends after the
           LAST such spike (sensor may re-surface several times before
           settling at deployment depth).
           Mark [0 : last_jump_settled] as dive leg.

        3. ENDING LEG — search in last 40% of data:
           Find ALL negative gradient spikes where:
             - gradient < -sensitivity * grad_std (significant drop)
             - surface displacement BEFORE spike >= 2.0       (was at depth)
             - surface displacement AFTER  spike < 2.0        (left the water)
           The leg starts at the FIRST such spike.
           Mark [first_drop_start : n] as dive leg.

        Returns:
            Boolean mask where True = dive/retrieval section to remove.
        """
        n = len(surface_displacement)
        dive_mask = np.zeros(n, dtype=bool)

        if n < 100:
            return dive_mask

        gradient = np.gradient(surface_displacement)
        gradient_abs = np.abs(gradient)
        grad_std = np.std(gradient)
        threshold = sensitivity * grad_std

        # ── BEGINNING LEG ──────────────────────────────────────────────────
        if surface_displacement[0] < 2.0:
            search_end = min(int(n * 0.4), 4000)

            # True dive entry: large positive gradient AND crosses the 2.0 boundary
            jump_indices = []
            for i in range(1, search_end):
                if gradient[i] > threshold:
                    p_before = surface_displacement[i - 1]
                    p_after = surface_displacement[min(i + 1, n - 1)]
                    # Must be a genuine air→water crossing
                    if p_before < 2.0 and p_after >= 2.0:
                        jump_indices.append(i)

            if jump_indices:
                # Use the LAST genuine entry — leg ends after all re-entries
                last_jump = jump_indices[-1]

                # Walk forward from last spike until gradient calms
                leg_end = last_jump
                for i in range(last_jump + 1, min(last_jump + 200, search_end)):
                    if gradient_abs[i] < grad_std:
                        leg_end = i
                        break

                # 10 % safety margin
                leg_end = min(leg_end + max(1, leg_end // 10), n - 1)
                dive_mask[0:leg_end] = True

        # ── ENDING LEG ─────────────────────────────────────────────────────
        if surface_displacement[-1] < 2.0:
            search_start = max(int(n * 0.6), n - 4000)

            # True retrieval exit: large negative gradient AND crosses 2.0 boundary
            drop_indices = []
            for i in range(search_start + 1, n):
                if gradient[i] < -threshold:
                    p_before = surface_displacement[i - 1]
                    p_after = surface_displacement[min(i + 1, n - 1)]
                    # Must be a genuine water→air crossing
                    if p_before >= 2.0 and p_after < 2.0:
                        drop_indices.append(i)

            if drop_indices:
                # Use the FIRST genuine exit — leg starts at first surface crossing
                first_drop = drop_indices[0]

                # Walk backward until gradient calms
                leg_start = first_drop
                for i in range(first_drop - 1, max(first_drop - 200, search_start), -1):
                    if gradient_abs[i] < grad_std:
                        leg_start = i
                        break

                # 10 % safety margin (go a bit earlier)
                leg_start = max(leg_start - max(1, (n - leg_start) // 10), 0)
                dive_mask[leg_start:] = True

        return dive_mask

    def _update_manual_btn_state(self):
        """Enable btn_manual only if dive detector found at least one leg.
        Reads self._has_dives set by create_plot() to avoid duplicate detection."""
        has_dives = getattr(self, '_has_dives', True)  # True = enabled by default if plot not yet run
        self.btn_manual.setEnabled(bool(has_dives))
        if not has_dives:
            self.btn_manual.setToolTip(
                "Dive detector found no deployment/retrieval legs in this dataset."
            )

    def on_manual_removal(self):
        """Handle manual removal button click"""
        # Store on app instance to survive self destruction
        _w = ManualRemovalWindow(self.data_df)
        QApplication.instance()._manual_window = _w
        _w.show()
        self.close()

    def on_skip_removal(self):
        """Handle skip button click - copy Step1 to Step2 with progress bar and Zero Mean processing"""
        output_folder = OUTPUT_FOLDER
        step1_file = output_folder / "Step1_TXTtoCSV.csv"
        step2_file = output_folder / "Step2_Initial_Cut.csv"

        # Show progress dialog
        progress_dialog = QDialog(self)
        progress_dialog.setWindowTitle("Processing Data")
        progress_dialog.setModal(True)
        progress_dialog.setFixedSize(500, 150)

        layout = QVBoxLayout(progress_dialog)

        label = QLabel("Processing data...")
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)

        progress_bar = QProgressBar()
        progress_bar.setRange(0, 100)
        layout.addWidget(progress_bar)

        status = QLabel("Copying file...")
        status.setAlignment(Qt.AlignCenter)
        status.setStyleSheet("color: #6b7280; font-family: 'Segoe UI', sans-serif; font-size: 12px;")
        layout.addWidget(status)

        progress_dialog.show()
        QApplication.processEvents()

        try:
            # Step 1: Copy file
            progress_bar.setValue(10)
            status.setText("Copying Step1 to Step2...")
            QApplication.processEvents()

            import shutil
            shutil.copy(step1_file, step2_file)

            # Process Zero Mean
            self.process_zero_mean(step2_file, output_folder, progress_bar, status)

            progress_bar.setValue(100)
            status.setText("Complete!")
            QApplication.processEvents()

            progress_dialog.close()

            # Open Step 3 window — store on app to survive self destruction
            _w = Step3FourierWindow()
            QApplication.instance()._step3_window = _w
            _w.show()
            self.close()

        except Exception as e:
            progress_dialog.close()
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to process:\n{str(e)}"
            )

    def process_zero_mean(self, step2_file, output_folder, progress_bar, status):
        """Delegate to the shared module-level implementation."""
        process_zero_mean(step2_file, output_folder, progress_bar, status)

    def build_full_data_step1(self):
        """Load and plot full Step1 data in new window"""
        progress = QDialog(self);
        progress.setWindowTitle('Loading Full Data')
        progress.setModal(True);
        progress.setFixedSize(400, 100)
        _l = QVBoxLayout(progress);
        _l.addWidget(QLabel('Loading Step1_TXTtoCSV.csv...'))
        pb = QProgressBar();
        pb.setRange(0, 0);
        _l.addWidget(pb)
        progress.show();
        QApplication.processEvents()
        try:
            df = pd.read_csv(OUTPUT_FOLDER / 'Step1_TXTtoCSV.csv', comment='#')
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            progress.close()
            _w = FullDataWindow(df, 'Step 1: Full Raw Data')
            self._full_window = _w;
            _w.show()
        except Exception as e:
            progress.close();
            QMessageBox.critical(self, 'Error', f'Could not load full data:\n{str(e)}')

    def apply_styles(self):
        """Apply global styles"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f9fafb;
            }
            QWidget {
                background-color: #f9fafb;
                font-family: 'Segoe UI', sans-serif;
            }
            QPushButton {
                background-color: #f3f4f6;
                color: #374151;
                border: 1px solid #d1d5db;
                padding: 8px 18px;
                border-radius: 6px;
                font-size: 13px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #e5e7eb;
                border-color: #9ca3af;
            }
        """)


class ManualRemovalWindow(QMainWindow):
    """Window for manually selecting dive regions to remove"""

    def __init__(self, viz_data_df):
        super().__init__()
        self.viz_data_df = viz_data_df  # Subsampled visualization data
        self.cut_timestamps = {'beginning': None,
                               'ending': None}  # Store cut timestamps (work for both viz and full data)
        self.cut_lines = {}  # Store cut line references for each graph
        self.shaded_regions = {}  # Store shaded region references
        self.init_ui()

    def init_ui(self):
        """Initialize manual removal window"""
        self.setWindowTitle("🌊 Manual Dive Removal")

        # Open maximized — deferred so it works both from checkpoint and normal flow
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(0, self.showMaximized)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Progress indicator
        progress = create_progress_indicator(current_step=2)
        layout.addWidget(progress)

        # Header
        header = QLabel("Manual Dive Section Removal")
        header.setFont(QFont("Segoe UI", 17, QFont.Bold))
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("color: #111827; padding: 14px; letter-spacing: -0.2px;")
        layout.addWidget(header)

        # Instructions
        instructions = QLabel(
            "Double-click on the graph to mark cut point. "
            "Deployment: removes everything BEFORE double-click. "
            "Retrieval: removes everything AFTER double-click."
        )
        instructions.setAlignment(Qt.AlignCenter)
        instructions.setStyleSheet(
            "color: #6b7280; font-size: 12px; padding: 4px; font-family: 'Segoe UI', sans-serif;")
        layout.addWidget(instructions)

        # Detect dives on visualization data
        self.detect_dive_legs()

        # Beginning dive plot (DEPLOYMENT)
        if self.beginning_data is not None:
            beginning_group = QGroupBox("Sensor Deployment")
            beginning_layout = QVBoxLayout()
            self.beginning_canvas = self.create_interactive_plot(
                self.beginning_data,
                "Deployment - Double-click to mark cut point",
                'beginning'
            )
            beginning_layout.addWidget(self.beginning_canvas)
            beginning_group.setLayout(beginning_layout)
            layout.addWidget(beginning_group, stretch=1)

        # Ending dive plot (RETRIEVAL)
        if self.ending_data is not None:
            ending_group = QGroupBox("Sensor Retrieval")
            ending_layout = QVBoxLayout()
            self.ending_canvas = self.create_interactive_plot(
                self.ending_data,
                "Retrieval - Double-click to mark cut point",
                'ending'
            )
            ending_layout.addWidget(self.ending_canvas)
            ending_group.setLayout(ending_layout)
            layout.addWidget(ending_group, stretch=1)

        # Buttons
        btn_layout = QHBoxLayout()

        btn_save = QPushButton("▶  Continue with trimmed data")
        btn_save.setStyleSheet("""
            QPushButton {
                background-color: #0ea5e9;
                color: white;
                font-size: 14px;
                font-weight: 600;
                padding: 13px 20px;
                border-radius: 8px;
                border: none;
                font-family: 'Segoe UI', sans-serif;
            }
            QPushButton:hover {
                background-color: #0284c7;
            }
        """)
        btn_save.clicked.connect(self.save_trimmed_data)
        btn_layout.addWidget(btn_save)

        layout.addLayout(btn_layout)

        self.apply_styles()

        # Add footer
        add_footer(self)

    def detect_dive_legs(self):
        """Detect dive legs on visualization (subsampled) data"""
        surface_displacement_viz = self.viz_data_df['surface_displacement'].values

        # Detect dives (same algorithm as VisualizationWindow)
        dive_mask = self.detect_dives(surface_displacement_viz)

        # Find beginning and ending legs
        dive_indices = np.where(dive_mask)[0]

        self.beginning_data = None
        self.ending_data = None
        self.beginning_viz_range = None  # (start_idx, end_idx) in viz data
        self.ending_viz_range = None

        if len(dive_indices) > 0:
            # Find segments
            diff = np.diff(dive_indices)
            breaks = np.where(diff > 100)[0]  # Gap > 100 points = different segments

            if len(breaks) == 0:
                # Only one segment
                if dive_indices[0] < len(surface_displacement_viz) // 2:
                    # Beginning
                    self.beginning_viz_range = (0, dive_indices[-1])
                else:
                    # Ending
                    self.ending_viz_range = (dive_indices[0], len(surface_displacement_viz) - 1)
            else:
                # Two segments
                # Beginning segment
                begin_end = dive_indices[breaks[0]]
                self.beginning_viz_range = (0, begin_end)

                # Ending segment
                end_start = dive_indices[breaks[0] + 1]
                self.ending_viz_range = (end_start, len(surface_displacement_viz) - 1)

            # Add +10% safety margin
            if self.beginning_viz_range:
                start, end = self.beginning_viz_range
                margin = int((end - start) * 0.1)
                end = min(end + margin, len(surface_displacement_viz) - 1)
                self.beginning_viz_range = (start, end)
                self.beginning_data = self.viz_data_df.iloc[start:end + 1].copy()

            if self.ending_viz_range:
                start, end = self.ending_viz_range
                margin = int((end - start) * 0.1)
                start = max(start - margin, 0)
                self.ending_viz_range = (start, end)
                self.ending_data = self.viz_data_df.iloc[start:end + 1].copy()

    def detect_dives(self, surface_displacement, sensitivity=3.0):
        """
        Dive detector — identical to VisualizationWindow.detect_dives.

        BEGINNING LEG: search first 40%, find ALL positive gradient spikes
        crossing the 2.0 m boundary (air→water). Mark [0:last_jump_settled].

        ENDING LEG: search last 40%, find ALL negative gradient spikes
        crossing 2.0 m boundary (water→air). Mark [first_drop_start:n].

        Guards: if surface_displacement[0] >= 2.0 → no beginning leg;
                if surface_displacement[-1] >= 2.0 → no ending leg.
        """
        n = len(surface_displacement)
        dive_mask = np.zeros(n, dtype=bool)

        if n < 100:
            return dive_mask

        gradient = np.gradient(surface_displacement)
        gradient_abs = np.abs(gradient)
        grad_std = np.std(gradient)
        threshold = sensitivity * grad_std

        # ── BEGINNING LEG ──────────────────────────────────────────────────
        if surface_displacement[0] < 2.0:
            search_end = min(int(n * 0.4), 4000)

            jump_indices = []
            for i in range(1, search_end):
                if gradient[i] > threshold:
                    p_before = surface_displacement[i - 1]
                    p_after = surface_displacement[min(i + 1, n - 1)]
                    if p_before < 2.0 and p_after >= 2.0:
                        jump_indices.append(i)

            if jump_indices:
                last_jump = jump_indices[-1]

                leg_end = last_jump
                for i in range(last_jump + 1, min(last_jump + 200, search_end)):
                    if gradient_abs[i] < grad_std:
                        leg_end = i
                        break

                leg_end = min(leg_end + max(1, leg_end // 10), n - 1)
                dive_mask[0:leg_end] = True

        # ── ENDING LEG ─────────────────────────────────────────────────────
        if surface_displacement[-1] < 2.0:
            search_start = max(int(n * 0.6), n - 4000)

            drop_indices = []
            for i in range(search_start + 1, n):
                if gradient[i] < -threshold:
                    p_before = surface_displacement[i - 1]
                    p_after = surface_displacement[min(i + 1, n - 1)]
                    if p_before >= 2.0 and p_after < 2.0:
                        drop_indices.append(i)

            if drop_indices:
                first_drop = drop_indices[0]

                leg_start = first_drop
                for i in range(first_drop - 1, max(first_drop - 200, search_start), -1):
                    if gradient_abs[i] < grad_std:
                        leg_start = i
                        break

                leg_start = max(leg_start - max(1, (n - leg_start) // 10), 0)
                dive_mask[leg_start:] = True

        return dive_mask

    def create_interactive_plot(self, data, title, leg_type):
        """Create interactive matplotlib plot with double-click handler - full data with zoom"""
        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT

        # Taller figure for vertical layout
        fig = Figure(figsize=(14, 5), dpi=100)
        canvas = FigureCanvas(fig)

        ax = fig.add_subplot(111)

        # Plot FULL viz data
        full_timestamps = self.viz_data_df['timestamp']
        full_surface_displacement = self.viz_data_df['surface_displacement'].values

        # Plot complete data
        ax.plot(full_timestamps, full_surface_displacement, linewidth=0.5, color=COLORS['wave_data'], alpha=0.7,
                marker='o', markersize=1.5, markerfacecolor=COLORS['wave_data'], markeredgewidth=0, zorder=2)

        # Highlight the detected dive section in red
        if leg_type == 'beginning' and self.beginning_viz_range:
            start, end = self.beginning_viz_range
            dive_timestamps = self.viz_data_df['timestamp'].iloc[start:end + 1]
            dive_surface_displacement = full_surface_displacement[start:end + 1]
            ax.plot(dive_timestamps, dive_surface_displacement, linewidth=0.8, color=COLORS['dive_detect'], alpha=0.9,
                    label='Detected dive',
                    marker='o', markersize=1.5, markerfacecolor=COLORS['dive_detect'], markeredgewidth=0, zorder=3)
        elif leg_type == 'ending' and self.ending_viz_range:
            start, end = self.ending_viz_range
            dive_timestamps = self.viz_data_df['timestamp'].iloc[start:end + 1]
            dive_surface_displacement = full_surface_displacement[start:end + 1]
            ax.plot(dive_timestamps, dive_surface_displacement, linewidth=0.8, color=COLORS['dive_detect'], alpha=0.9,
                    label='Detected dive',
                    marker='o', markersize=1.5, markerfacecolor=COLORS['dive_detect'], markeredgewidth=0, zorder=3)

        # No axis labels, only tick values
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, zorder=0)
        ax.legend(loc='upper right', fontsize=9)

        # Format dates - horizontal, no rotation
        import matplotlib.dates as mdates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m %H:%M'))
        # Keep ticks horizontal
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center', fontsize=9)

        # Set initial zoom on the detected dive section
        if leg_type == 'beginning' and self.beginning_viz_range:
            start, end = self.beginning_viz_range
            margin_points = int((end - start) * 0.2)
            zoom_start = max(0, start - margin_points)
            zoom_end = min(len(full_timestamps) - 1, end + margin_points)

            ax.set_xlim(full_timestamps.iloc[zoom_start], full_timestamps.iloc[zoom_end])

        elif leg_type == 'ending' and self.ending_viz_range:
            start, end = self.ending_viz_range
            margin_points = int((end - start) * 0.2)
            zoom_start = max(0, start - margin_points)
            zoom_end = min(len(full_timestamps) - 1, end + margin_points)

            ax.set_xlim(full_timestamps.iloc[zoom_start], full_timestamps.iloc[zoom_end])

        # Initialize cut line and shaded region storage
        self.cut_lines[leg_type] = None
        self.shaded_regions[leg_type] = None

        # Double-click handler
        def on_click(event):
            if event.inaxes == ax and event.button == 1 and event.dblclick:  # Left double-click
                # Safely remove previous cut line — object may be stale after ax.cla()
                if self.cut_lines[leg_type] is not None:
                    try:
                        self.cut_lines[leg_type].remove()
                    except Exception:
                        pass
                    self.cut_lines[leg_type] = None

                # Safely remove previous shading
                if self.shaded_regions[leg_type] is not None:
                    try:
                        self.shaded_regions[leg_type].remove()
                    except Exception:
                        pass
                    self.shaded_regions[leg_type] = None

                # Draw vertical line at double-click position
                self.cut_lines[leg_type] = ax.axvline(
                    event.xdata, color=COLORS['filtered'], linewidth=2,
                    linestyle='--', label='Cut point', zorder=10
                )

                # Convert clicked x-position to datetime (make tz-naive)
                clicked_time = pd.Timestamp(mdates.num2date(event.xdata)).tz_localize(None)

                # Use current axis limits for shading — works for both viz and full-res graph
                cur_xlim = ax.get_xlim()
                if leg_type == 'beginning':
                    # Shade everything BEFORE the cut (left side)
                    self.shaded_regions[leg_type] = ax.axvspan(
                        cur_xlim[0], event.xdata,
                        alpha=0.3, color='red', zorder=1, label='To be removed'
                    )
                else:  # ending
                    # Shade everything AFTER the cut (right side)
                    self.shaded_regions[leg_type] = ax.axvspan(
                        event.xdata, cur_xlim[1],
                        alpha=0.3, color='red', zorder=1, label='To be removed'
                    )

                canvas.draw()

                # Store cut timestamp — works for both subsampled and full-resolution graphs
                self.cut_timestamps[leg_type] = clicked_time
                print(f"{leg_type} cut at timestamp: {clicked_time}")

        canvas.mpl_connect('button_press_event', on_click)  # dblclick fires button_press_event with event.dblclick=True

        fig.tight_layout()

        # Add navigation toolbar for zoom/pan
        toolbar = NavigationToolbar2QT(canvas, self)
        full_data_action = QAction('📊 Build all data points (slow)', self)

        # Capture ax/canvas/fig/leg_type in closure — redraws THIS graph in place
        def _build_full_in_place(checked=False, _ax=ax, _canvas=canvas, _fig=fig, _lt=leg_type):
            import matplotlib.dates as _mdates
            full_file = OUTPUT_FOLDER / 'Step1_TXTtoCSV.csv'
            if not full_file.exists():
                QMessageBox.warning(self, 'Not found', f'Could not find:\n{full_file}')
                return
            # Progress cursor
            QApplication.setOverrideCursor(Qt.WaitCursor)
            QApplication.processEvents()
            try:
                df_full = pd.read_csv(full_file, comment='#')
                df_full['timestamp'] = pd.to_datetime(df_full['timestamp'], errors='coerce')
                ts = df_full['timestamp']
                prs = df_full['surface_displacement'].values

                # Remember current view state before clearing
                cur_xlim = _ax.get_xlim()
                cur_ylim = _ax.get_ylim()
                cur_title = _ax.get_title().split(' —')[0]  # strip old suffix

                # Save user's cut line and shading (we'll re-add them after)
                saved_cut_line_x = None
                if self.cut_lines.get(_lt):
                    try:
                        saved_cut_line_x = self.cut_lines[_lt].get_xdata()[0]
                    except Exception:
                        pass
                saved_shading = None
                if self.shaded_regions.get(_lt):
                    try:
                        verts = self.shaded_regions[_lt].get_paths()[0].vertices
                        saved_shading = (float(verts[:, 0].min()), float(verts[:, 0].max()))
                    except Exception:
                        pass

                # Fully clear the axes and redraw from scratch
                _ax.cla()

                # Draw full (non-subsampled) data
                _ax.plot(ts, prs, linewidth=0.3, color=COLORS['wave_data'], alpha=0.8,
                         marker='o', markersize=1.2, markerfacecolor=COLORS['wave_data'], markeredgewidth=0, zorder=2)

                # Re-draw dive highlight
                if _lt == 'beginning' and self.beginning_viz_range:
                    s, e = self.beginning_viz_range
                    viz_ts = self.viz_data_df['timestamp']
                    t0 = viz_ts.iloc[s];
                    t1 = viz_ts.iloc[e]
                    mask = (ts >= t0) & (ts <= t1)
                    _ax.plot(ts[mask], prs[mask], linewidth=0.6,
                             color=COLORS['dive_detect'], alpha=0.9, label='Detected dive',
                             marker='o', markersize=1.2, markerfacecolor=COLORS['dive_detect'], markeredgewidth=0,
                             zorder=3)
                elif _lt == 'ending' and self.ending_viz_range:
                    s, e = self.ending_viz_range
                    viz_ts = self.viz_data_df['timestamp']
                    t0 = viz_ts.iloc[s];
                    t1 = viz_ts.iloc[e]
                    mask = (ts >= t0) & (ts <= t1)
                    _ax.plot(ts[mask], prs[mask], linewidth=0.6,
                             color=COLORS['dive_detect'], alpha=0.9, label='Detected dive',
                             marker='o', markersize=1.2, markerfacecolor=COLORS['dive_detect'], markeredgewidth=0,
                             zorder=3)

                # Restore user's cut line and shading if they had set one
                if saved_cut_line_x is not None:
                    self.cut_lines[_lt] = _ax.axvline(
                        saved_cut_line_x, color=COLORS['filtered'], linewidth=2,
                        linestyle='--', label='Cut point', zorder=10
                    )
                if saved_shading is not None:
                    x0, x1 = saved_shading
                    self.shaded_regions[_lt] = _ax.axvspan(
                        x0, x1, alpha=0.3, color='red', zorder=1,
                        label='To be removed'
                    )

                # Restore formatting
                import matplotlib.dates as _mdates2
                _ax.xaxis.set_major_formatter(_mdates2.DateFormatter('%d-%m %H:%M'))
                _ax.grid(True, alpha=0.3, zorder=0)
                _ax.legend(loc='upper right', fontsize=9)
                _ax.set_title(
                    f'{cur_title} — {len(prs):,} pts (full resolution)',
                    fontsize=12, fontweight='bold'
                )
                _ax.set_xlim(cur_xlim)
                _canvas.draw()
            except Exception as ex:
                QMessageBox.critical(self, 'Error', f'Could not load full data:\n{ex}')
            finally:
                QApplication.restoreOverrideCursor()

        full_data_action.triggered.connect(_build_full_in_place)
        toolbar.addAction(full_data_action)

        # Container widget
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.addWidget(toolbar)
        container_layout.addWidget(canvas)

        return container

    def save_trimmed_data(self):
        """Save trimmed data with progress bar - convert viz indices to full data"""
        # Show progress dialog
        progress_dialog = QDialog(self)
        progress_dialog.setWindowTitle("Saving Trimmed Data")
        progress_dialog.setModal(True)
        progress_dialog.setFixedSize(500, 150)

        layout = QVBoxLayout(progress_dialog)

        label = QLabel("Processing and saving data...")
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)

        progress_bar = QProgressBar()
        progress_bar.setRange(0, 100)
        layout.addWidget(progress_bar)

        status = QLabel("Loading full data...")
        status.setAlignment(Qt.AlignCenter)
        status.setStyleSheet("color: #6b7280; font-family: 'Segoe UI', sans-serif; font-size: 12px;")
        layout.addWidget(status)

        progress_dialog.show()
        QApplication.processEvents()

        try:
            # Load full CSV
            progress_bar.setValue(10)
            status.setText("Reading full CSV file...")
            QApplication.processEvents()

            csv_file = OUTPUT_FOLDER / "Step1_TXTtoCSV.csv"

            progress_bar.setValue(20)
            status.setText(f"Loading full dataset ({csv_file.name})...")
            QApplication.processEvents()

            # Read full data — the heavy operation; progress stays at 20 % while
            # pandas reads the file, then jumps to 45 % when done.
            full_data = pd.read_csv(csv_file, comment='#')

            progress_bar.setValue(40)
            status.setText("Parsing timestamps...")
            QApplication.processEvents()

            full_data['timestamp'] = pd.to_datetime(full_data['timestamp'], errors='coerce')

            progress_bar.setValue(45)
            status.setText("Building timestamp index for binary search...")
            QApplication.processEvents()

            # Timestamps are strictly monotonically increasing (generated via
            # pd.date_range in Step 1), so binary search via searchsorted is
            # O(log N) — completes in microseconds even for 50 M rows.
            _ts_array = full_data['timestamp'].values  # numpy datetime64 array — sorted

            def _ts_to_full_idx(ts):
                needle = np.datetime64(ts)
                pos = int(np.searchsorted(_ts_array, needle))
                pos = max(0, min(pos, len(_ts_array) - 1))
                if pos > 0:
                    d_prev = abs((_ts_array[pos - 1] - needle).astype('int64'))
                    d_curr = abs((_ts_array[pos] - needle).astype('int64'))
                    if d_prev < d_curr:
                        pos -= 1
                return pos

            def _viz_idx_to_full_idx(viz_idx):
                viz_ts = self.viz_data_df['timestamp'].iloc[viz_idx]
                return _ts_to_full_idx(viz_ts)

            # Determine cut points in full data
            beginning_full_idx = None
            ending_full_idx = None

            if self.cut_timestamps['beginning'] is not None:
                progress_bar.setValue(50)
                status.setText("Locating deployment cut point...")
                QApplication.processEvents()
                beginning_full_idx = _ts_to_full_idx(self.cut_timestamps['beginning'])
            elif self.beginning_viz_range:
                progress_bar.setValue(50)
                status.setText("Locating deployment boundary...")
                QApplication.processEvents()
                beginning_full_idx = _viz_idx_to_full_idx(self.beginning_viz_range[1])

            if self.cut_timestamps['ending'] is not None:
                progress_bar.setValue(55)
                status.setText("Locating retrieval cut point...")
                QApplication.processEvents()
                ending_full_idx = _ts_to_full_idx(self.cut_timestamps['ending'])
            elif self.ending_viz_range:
                progress_bar.setValue(55)
                status.setText("Locating retrieval boundary...")
                QApplication.processEvents()
                ending_full_idx = _viz_idx_to_full_idx(self.ending_viz_range[0])

            progress_bar.setValue(60)
            status.setText("Trimming data...")
            QApplication.processEvents()

            # Apply cuts
            trimmed_data = full_data.copy()

            if beginning_full_idx is not None:
                trimmed_data = trimmed_data.iloc[beginning_full_idx:]

            if ending_full_idx is not None:
                end_relative = ending_full_idx - trimmed_data.index[0]
                trimmed_data = trimmed_data.iloc[:end_relative]

            progress_bar.setValue(70)
            status.setText(f"Trimmed to {len(trimmed_data):,} points. Saving to Step2_Initial_Cut.csv...")
            QApplication.processEvents()

            # Save to Step2
            output_folder = OUTPUT_FOLDER
            step2_file = output_folder / "Step2_Initial_Cut.csv"

            # Write with metadata
            with open(step2_file, 'w', encoding='utf-8') as f:
                f.write("# STEP 2: Initial Cut - Manual dive removal\n")
                f.write("# ==========================================\n")
                f.write(f"# Sensor frequency: {read_sensor_freq_from_csv(csv_file)} Hz\n")
                f.write(f"# Original points: {len(full_data):,}\n")
                f.write(f"# Trimmed points: {len(trimmed_data):,}\n")
                f.write(f"# Points removed: {len(full_data) - len(trimmed_data):,}\n")
                f.write("# ==========================================\n")

            status.setText(f"Writing {len(trimmed_data):,} rows to disk...")
            QApplication.processEvents()

            trimmed_data.to_csv(step2_file, mode='a', index=False)

            progress_bar.setValue(80)
            status.setText("Processing Zero Mean...")
            QApplication.processEvents()

            # Process Zero Mean
            self.process_zero_mean(step2_file, output_folder, progress_bar, status)

            progress_bar.setValue(100)
            status.setText("Complete!")
            QApplication.processEvents()

            progress_dialog.close()

            # Open Step 3 window — store on app to survive self destruction
            _w = Step3FourierWindow()
            QApplication.instance()._step3_window = _w
            _w.show()
            self.close()

        except Exception as e:
            progress_dialog.close()
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to save trimmed data:\n{str(e)}"
            )

    def process_zero_mean(self, step2_file, output_folder, progress_bar, status):
        """Delegate to the shared module-level implementation."""
        process_zero_mean(step2_file, output_folder, progress_bar, status)

    def build_full_data_step2(self):
        """Load and plot full Step2 data in new window"""
        progress = QDialog(self);
        progress.setWindowTitle('Loading Full Data')
        progress.setModal(True);
        progress.setFixedSize(400, 100)
        _l = QVBoxLayout(progress);
        _l.addWidget(QLabel('Loading Step2_Initial_Cut.csv...'))
        pb = QProgressBar();
        pb.setRange(0, 0);
        _l.addWidget(pb)
        progress.show();
        QApplication.processEvents()
        try:
            df = pd.read_csv(OUTPUT_FOLDER / 'Step2_Initial_Cut.csv', comment='#')
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            progress.close()
            _w = FullDataWindow(df, 'Step 2: Full Initial Cut Data')
            self._full_window = _w;
            _w.show()
        except Exception as e:
            progress.close();
            QMessageBox.critical(self, 'Error', f'Could not load full data:\n{str(e)}')

    def apply_styles(self):
        """Apply global styles"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f9fafb;
            }
            QWidget {
                background-color: #f9fafb;
                font-family: 'Segoe UI', sans-serif;
            }
            QGroupBox {
                font-weight: 600;
                border: 1px solid #e5e7eb;
                border-radius: 10px;
                margin-top: 12px;
                padding-top: 12px;
                background-color: #ffffff;
                color: #374151;
            }
            QGroupBox::title {
                color: #374151;
                subcontrol-origin: margin;
                left: 14px;
                padding: 0 8px;
                background-color: #ffffff;
            }
            QPushButton {
                background-color: #f3f4f6;
                color: #374151;
                border: 1px solid #d1d5db;
                padding: 8px 18px;
                border-radius: 6px;
                font-size: 13px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #e5e7eb;
                border-color: #9ca3af;
            }
        """)


class Step3FourierWindow(QMainWindow):
    """Window for Step 3: Fourier Transform - Remove low frequencies"""

    def __init__(self):
        super().__init__()
        self.spectrum_full = None  # complex FFT — for apply_transform
        self.frequencies_full = None
        self.cutoff_freq = None
        self.data_step2 = None  # cached Step2_Zero_Mean DataFrame (timestamps + reading_number)
        self.init_ui()
        self.load_and_transform()

    def init_ui(self):
        """Initialize Step 3 Fourier window"""
        self.setWindowTitle("🌊 Step 3: Fourier Transform")
        # Don't show maximized here - will show after data loads

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Progress indicator
        progress = create_progress_indicator(current_step=3)
        layout.addWidget(progress)

        # ── Top bar: spectrogram parameters + button ─────────────────────
        # Outer container with card-style background
        top_card = QWidget()
        top_card.setStyleSheet("""
            QWidget {
                background-color: #f8fafc;
                border: 1px solid #e5e7eb;
                border-radius: 8px;
            }
        """)
        top_card.setFixedHeight(56)
        top_bar = QHBoxLayout(top_card)
        top_bar.setContentsMargins(16, 0, 16, 0)
        top_bar.setSpacing(8)

        lbl_style = (
            "font-size: 13px; color: #6b7280; background: transparent; border: none; font-family: 'Segoe UI', sans-serif;"
        )
        spin_style = """
            QSpinBox {
                font-size: 13px;
                font-weight: 600;
                color: #111827;
                background: #ffffff;
                padding: 3px 6px;
                border: 1px solid #d1d5db;
                border-radius: 6px;
                min-width: 72px;
                font-family: 'Segoe UI', sans-serif;
            }
            QSpinBox:focus { border-color: #0ea5e9; }
            QSpinBox::up-button, QSpinBox::down-button {
                background: #f3f4f6;
                border-radius: 3px;
            }
        """

        # Window size
        top_bar.addWidget(_lbl("Window FT:", lbl_style))
        self.spin_window = QSpinBox()
        self.spin_window.setRange(1, 120)
        self.spin_window.setValue(10)
        self.spin_window.setSuffix(" min")
        self.spin_window.setStyleSheet(spin_style)
        self.spin_window.setFixedHeight(32)
        top_bar.addWidget(self.spin_window)

        top_bar.addWidget(_lbl("shift", lbl_style))
        self.spin_delta = QSpinBox()
        self.spin_delta.setRange(1, 600)
        self.spin_delta.setValue(60)
        self.spin_delta.setSuffix(" sec")
        self.spin_delta.setStyleSheet(spin_style)
        self.spin_delta.setFixedHeight(32)
        top_bar.addWidget(self.spin_delta)

        top_bar.addWidget(_lbl("spectrum", lbl_style))
        self.spin_part = QSpinBox()
        self.spin_part.setRange(1, 100)
        self.spin_part.setValue(20)
        self.spin_part.setSuffix(" %")
        self.spin_part.setStyleSheet(spin_style)
        self.spin_part.setFixedHeight(32)
        top_bar.addWidget(self.spin_part)

        # Spectrogram button — right after the last spinbox
        btn_spectrogram = QPushButton("  📊  Plot Spectrogram")
        btn_spectrogram.setFixedHeight(36)
        btn_spectrogram.setStyleSheet("""
            QPushButton {
                background-color: #fff7ed;
                color: #c2410c;
                font-size: 13px;
                font-weight: 600;
                padding: 0 18px;
                border-radius: 6px;
                border: 1px solid #fed7aa;
                font-family: 'Segoe UI', sans-serif;
            }
            QPushButton:hover  { background-color: #ffedd5; border-color: #fdba74; }
            QPushButton:pressed{ background-color: #fee2c8; }
        """)
        btn_spectrogram.clicked.connect(self.plot_spectrogram)
        top_bar.addWidget(btn_spectrogram)

        top_bar.addStretch()
        layout.addWidget(top_card)

        # Graph placeholder (will hold 2 graphs)
        self.graph_layout = QVBoxLayout()
        layout.addLayout(self.graph_layout)

        # Apply button at bottom
        btn_layout = QHBoxLayout()

        self.btn_continue = QPushButton("▶  Apply and Continue")
        self.btn_continue.setEnabled(False)  # Disabled until cutoff selected
        self.btn_continue.setStyleSheet("""
            QPushButton {
                background-color: #0ea5e9;
                color: white;
                font-size: 14px;
                font-weight: 600;
                padding: 13px 20px;
                border-radius: 8px;
                border: none;
                font-family: 'Segoe UI', sans-serif;
            }
            QPushButton:hover:enabled {
                background-color: #0284c7;
            }
            QPushButton:disabled {
                background-color: #e5e7eb;
                color: #9ca3af;
            }
        """)
        self.btn_continue.clicked.connect(self.apply_transform)
        btn_layout.addWidget(self.btn_continue)

        layout.addLayout(btn_layout)

        # Add footer
        add_footer(self)

    def load_and_transform(self):
        """Load Step2 data and perform FFT"""
        output_folder = OUTPUT_FOLDER

        step2_file = output_folder / "Step2_Zero_Mean.csv"
        step3_spectrum = output_folder / "Step3_Spectrum.csv"
        step3_spectrum_viz = output_folder / "Step3_Spectrum_Visualization.csv"

        # Check if spectrum already exists
        if step3_spectrum_viz.exists():
            # Show progress for loading cache
            progress_dialog = QDialog(None)
            progress_dialog.setWindowTitle("Loading Cached Spectrum")
            progress_dialog.setModal(True)
            progress_dialog.setFixedSize(500, 150)
            progress_dialog.setWindowFlags(
                progress_dialog.windowFlags() | Qt.WindowStaysOnTopHint
            )

            layout = QVBoxLayout(progress_dialog)

            label = QLabel("Loading cached spectrum...")
            label.setAlignment(Qt.AlignCenter)
            layout.addWidget(label)

            progress_bar = QProgressBar()
            progress_bar.setRange(0, 100)
            layout.addWidget(progress_bar)

            status = QLabel("Loading visualization...")
            status.setAlignment(Qt.AlignCenter)
            status.setStyleSheet("color: #6b7280; font-family: 'Segoe UI', sans-serif; font-size: 12px;")
            layout.addWidget(status)

            progress_dialog.show()
            QApplication.processEvents()

            # Load cached spectrum
            progress_bar.setValue(30)
            QApplication.processEvents()

            spectrum_df = pd.read_csv(step3_spectrum_viz, comment='#')
            self.frequencies_viz = spectrum_df['frequency'].values
            self.spectrum_viz_real = spectrum_df['real'].values
            self.spectrum_viz_imag = spectrum_df['imag'].values

            progress_bar.setValue(60)
            status.setText("Loading full spectrum...")
            QApplication.processEvents()

            # Load full spectrum
            spectrum_full_df = pd.read_csv(step3_spectrum, comment='#')
            self.frequencies_full = spectrum_full_df['frequency'].values
            self.spectrum_full = spectrum_full_df['real'].values + 1j * spectrum_full_df['imag'].values

            progress_bar.setValue(90)
            status.setText("Creating plot...")
            QApplication.processEvents()

            progress_dialog.close()

            self.create_spectrum_plot()

            # Show window maximized — deferred for checkpoint-start compatibility
            from PyQt5.QtCore import QTimer
            QTimer.singleShot(0, self.showMaximized)
            return

        # Show progress dialog
        progress_dialog = QDialog(None)
        progress_dialog.setWindowTitle("Computing Fourier Transform")
        progress_dialog.setModal(True)
        progress_dialog.setFixedSize(500, 150)
        progress_dialog.setWindowFlags(
            progress_dialog.windowFlags() | Qt.WindowStaysOnTopHint
        )

        layout = QVBoxLayout(progress_dialog)

        label = QLabel("Processing Fourier Transform...")
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)

        progress_bar = QProgressBar()
        progress_bar.setRange(0, 100)
        layout.addWidget(progress_bar)

        status = QLabel("Loading data...")
        status.setAlignment(Qt.AlignCenter)
        status.setStyleSheet("color: #6b7280; font-family: 'Segoe UI', sans-serif; font-size: 12px;")
        layout.addWidget(status)

        progress_dialog.show()
        QApplication.processEvents()

        try:
            # Load full Step2 data
            progress_bar.setValue(10)
            status.setText("Loading Step2_Zero_Mean.csv...")
            QApplication.processEvents()

            data = pd.read_csv(step2_file, comment='#')
            y = data['surface_displacement'].values

            # Cache the full DataFrame — apply_transform needs timestamps and
            # reading_number but must not re-read the file (it may be hundreds of MB).
            self.data_step2 = data

            # Read sensor frequency from Step2 metadata header
            sensor_freq = read_sensor_freq_from_csv(step2_file)

            progress_bar.setValue(30)
            status.setText(f"Computing FFT for {len(y):,} points (sensor freq: {sensor_freq} Hz)...")
            QApplication.processEvents()

            # Perform full FFT (two-sided, original approach)
            from scipy.fftpack import fft, fftfreq

            s = fft(y)
            x = fftfreq(len(y), (1 / sensor_freq) / (2 * np.pi))  # Angular frequency ω [rad/s]

            progress_bar.setValue(60)
            status.setText("Saving full spectrum...")
            QApplication.processEvents()

            # Save full spectrum (real + imag kept for irfft in apply_transform)
            spectrum_df = pd.DataFrame({
                'frequency': x,
                'real': s.real,
                'imag': s.imag
            })

            with open(step3_spectrum, 'w', encoding='utf-8') as f:
                f.write("# STEP 3: Full Fourier Spectrum\n")
                f.write("# ==========================================\n")
                f.write(f"# Total points: {len(x)}\n")
                f.write(f"# Sensor frequency: {sensor_freq} Hz\n")
                f.write("# ==========================================\n")

            spectrum_df.to_csv(step3_spectrum, mode='a', index=False)

            self.spectrum_full = s  # complex FFT — used by apply_transform
            self.frequencies_full = x

            progress_bar.setValue(75)
            status.setText("Creating visualization...")
            QApplication.processEvents()

            # Subsample for visualization
            target_points = SPECTRUM_TARGET_POINTS
            step = max(1, len(x) // target_points)

            x_viz = x[::step]
            s_viz = s[::step]

            # Save visualization
            viz_df = pd.DataFrame({
                'frequency': x_viz,
                'real': s_viz.real,
                'imag': s_viz.imag
            })

            progress_bar.setValue(85)
            status.setText("Saving spectrum visualization...")
            QApplication.processEvents()

            with open(step3_spectrum_viz, 'w', encoding='utf-8') as f:
                f.write("# STEP 3: Spectrum Visualization (subsampled)\n")
                f.write("# ==========================================\n")
                f.write(f"# Sampled points: {len(x_viz)}\n")
                f.write(f"# Original points: {len(x)}\n")
                f.write("# ==========================================\n")

            viz_df.to_csv(step3_spectrum_viz, mode='a', index=False)

            self.frequencies_viz = x_viz
            self.spectrum_viz_real = s_viz.real
            self.spectrum_viz_imag = s_viz.imag

            progress_bar.setValue(95)
            status.setText("Preparing plot...")
            QApplication.processEvents()

            progress_dialog.close()

            # Create plot with its own progress
            self.create_spectrum_plot()

            # Show window maximized — deferred for checkpoint-start compatibility
            from PyQt5.QtCore import QTimer
            QTimer.singleShot(0, self.showMaximized)

        except Exception as e:
            progress_dialog.close()
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to compute FFT:\n{str(e)}"
            )

    def create_spectrum_plot(self):
        """Create two separate independent spectrum plots"""
        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
        from matplotlib import ticker as mticker

        # ── S(ω) = 2·|FFT|² / (N · max(ω))  [m²/s]  ───────────────────────
        # Two-sided FFT stored → take positive freqs only and multiply by 2
        # to recover the one-sided (physical) spectral density.
        # Factor 2 is equivalent to dividing by N/2 instead of N.
        N = len(self.spectrum_full)
        omega_max = np.max(np.abs(self.frequencies_full))

        # Top graph: subsampled, ω > 0.1 (exclude DC and large low-freq harmonics)
        pos_viz = self.frequencies_viz > 0.1
        freq_viz = self.frequencies_viz[pos_viz]
        mag_viz = np.sqrt(self.spectrum_viz_real[pos_viz] ** 2 + self.spectrum_viz_imag[pos_viz] ** 2)
        s_viz = (mag_viz ** 2) / ((N / 2) * omega_max)

        # Bottom graph: full resolution, 0 < ω ≤ 0.1
        pos_full = self.frequencies_full >= 0
        freq_full = self.frequencies_full[pos_full]
        mag_full = np.sqrt(self.spectrum_full.real[pos_full] ** 2 + self.spectrum_full.imag[pos_full] ** 2)
        s_full = (mag_full ** 2) / ((N / 2) * omega_max)
        zoom_idx = (freq_full > 0) & (freq_full <= 0.1)
        freq_zoom = freq_full[zoom_idx]
        s_zoom = s_full[zoom_idx]

        # ==============================================================================
        # TOP GRAPH: overview, ω ∈ [0, 3]
        # ==============================================================================
        fig_top = Figure(figsize=(16, 4), dpi=100)
        canvas_top = FigureCanvas(fig_top)
        ax_top = fig_top.add_subplot(111)

        ax_top.plot(freq_viz, s_viz, linewidth=0.8, color=COLORS['spectrum'], alpha=0.9, zorder=2)
        ax_top.set_xlabel('ω, [rad/s]', fontsize=11)
        ax_top.set_ylabel('S(ω), [m²/s]', fontsize=11)
        ax_top.grid(True, alpha=0.3, zorder=0)
        ax_top.set_xlim(0, 3.0)
        ax_top.set_ylim(0, None)

        fig_top.tight_layout()
        toolbar_top = NavigationToolbar2QT(canvas_top, self)
        full_time_action = QAction('📊 Build full spectrum (slow)', self)
        full_time_action.triggered.connect(self.build_full_spectrum_linear)
        toolbar_top.addAction(full_time_action)

        # ==============================================================================
        # BOTTOM GRAPH: 0 < ω ≤ 0.1, log Y, interactive cutoff
        # ==============================================================================
        fig_bottom = Figure(figsize=(16, 5), dpi=100)
        canvas_bottom = FigureCanvas(fig_bottom)
        ax_bottom = fig_bottom.add_subplot(111)

        ax_bottom.plot(freq_zoom, s_zoom, linewidth=0.8, color=COLORS['spectrum'], alpha=0.9, zorder=2)
        ax_bottom.set_xlabel('ω, [rad/s]', fontsize=11)
        ax_bottom.set_ylabel('S(ω), [m²/s]', fontsize=11)
        ax_bottom.set_xlim(0, 0.1)
        ax_bottom.grid(True, alpha=0.3, which='both', zorder=0)
        # Log scale — set AFTER plot so matplotlib auto-sets ylim from data (no warning)
        ax_bottom.set_yscale('log')

        fig_bottom.tight_layout()
        toolbar_bottom = NavigationToolbar2QT(canvas_bottom, self)
        full_spectrum_action = QAction('📊 Build full spectrum (slow)', self)
        full_spectrum_action.triggered.connect(self.build_full_spectrum)
        toolbar_bottom.addAction(full_spectrum_action)

        # ==============================================================================
        # DOUBLE-CLICK HANDLER (only bottom graph)
        # Double-click sets cutoff - everything BELOW (left of) cutoff is removed
        # ==============================================================================

        self.cutoff_line = None
        self.shaded_region = None
        self.cutoff_text = None  # Text annotation on top graph

        def on_click(event):
            if event.inaxes == ax_bottom and event.button == 1 and event.dblclick:  # Left double-click
                # Remove previous
                if self.cutoff_line:
                    self.cutoff_line.remove()
                if self.shaded_region:
                    self.shaded_region.remove()
                if self.cutoff_text:
                    self.cutoff_text.remove()

                clicked_omega = event.xdata

                # Draw vertical line at cutoff
                self.cutoff_line = ax_bottom.axvline(clicked_omega, color='red',
                                                     linewidth=2, linestyle='--',
                                                     zorder=10)

                # Shade region to be removed (LEFT of cutoff = low frequencies)
                self.shaded_region = ax_bottom.axvspan(0, clicked_omega,
                                                       alpha=0.3, color='red', zorder=1)

                # Convert omega (rad/s) to period (minutes + seconds)
                # T = 2π/ω (in seconds)
                period_seconds = (2 * np.pi) / clicked_omega
                period_minutes = int(period_seconds // 60)
                period_secs = int(period_seconds % 60)

                # Add text annotation to BOTTOM graph (upper right corner)
                cutoff_text = f"Cut-off > {period_minutes} min {period_secs} sec"
                self.cutoff_text = ax_bottom.text(0.98, 0.95, cutoff_text,
                                                  transform=ax_bottom.transAxes,
                                                  fontsize=12,
                                                  verticalalignment='top',
                                                  horizontalalignment='right',
                                                  bbox=dict(boxstyle='round',
                                                            facecolor='white',
                                                            edgecolor='black',
                                                            alpha=0.9))

                canvas_bottom.draw()

                # Store cutoff frequency
                self.cutoff_freq = clicked_omega
                self.btn_continue.setEnabled(True)

                print(
                    f"Cutoff frequency set to: {self.cutoff_freq:.4f} rad/s (Period: {period_minutes} min {period_secs} sec)")

        canvas_bottom.mpl_connect('button_press_event', on_click)

        # ==============================================================================
        # ADD TO LAYOUT (like Step 2 - two independent graphs)
        # ==============================================================================

        # Clear previous
        for i in reversed(range(self.graph_layout.count())):
            self.graph_layout.itemAt(i).widget().setParent(None)

        # Add top graph
        self.graph_layout.addWidget(toolbar_top)
        self.graph_layout.addWidget(canvas_top)

        # Add bottom graph
        self.graph_layout.addWidget(toolbar_bottom)
        self.graph_layout.addWidget(canvas_bottom)

        # Store references
        self.fig_top = fig_top
        self.fig_bottom = fig_bottom
        self.ax_top = ax_top
        self.ax_bottom = ax_bottom
        self.canvas_top = canvas_top
        self.canvas_bottom = canvas_bottom

    def apply_transform(self):
        """Apply cutoff and perform inverse FFT"""
        if self.cutoff_freq is None:
            QMessageBox.warning(self, "Warning", "Please select cutoff frequency first!")
            return

        output_folder = OUTPUT_FOLDER

        # Show progress dialog
        progress_dialog = QDialog(self)
        progress_dialog.setWindowTitle("Applying Transform")
        progress_dialog.setModal(True)
        progress_dialog.setFixedSize(500, 150)

        layout = QVBoxLayout(progress_dialog)

        label = QLabel("Applying inverse Fourier Transform...")
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)

        progress_bar = QProgressBar()
        progress_bar.setRange(0, 100)
        layout.addWidget(progress_bar)

        status = QLabel("Filtering spectrum...")
        status.setAlignment(Qt.AlignCenter)
        status.setStyleSheet("color: #6b7280; font-family: 'Segoe UI', sans-serif; font-size: 12px;")
        layout.addWidget(status)

        progress_dialog.show()
        QApplication.processEvents()

        try:
            # Apply cutoff filter on the complex FFT spectrum
            progress_bar.setValue(20)
            status.setText("Applying frequency filter...")
            QApplication.processEvents()

            s_filtered = self.spectrum_full.copy()

            # Zero out frequencies below cutoff — vectorised, O(N) not O(N) with Python overhead
            s_filtered[np.abs(self.frequencies_full) < self.cutoff_freq] = 0j

            progress_bar.setValue(40)
            status.setText("Computing inverse FFT...")
            QApplication.processEvents()

            # Inverse FFT
            from scipy.fftpack import ifft

            y_transformed = ifft(s_filtered).real

            progress_bar.setValue(60)
            status.setText("Saving transformed data...")
            QApplication.processEvents()

            # Reuse the DataFrame cached during FFT computation.
            # If loading from checkpoint (cache path), data_step2 is None → read once.
            step2_file = output_folder / "Step2_Zero_Mean.csv"
            if self.data_step2 is not None:
                data_orig = self.data_step2
            else:
                data_orig = pd.read_csv(step2_file, comment='#')

            # Create transformed dataframe
            data_transformed = data_orig.copy()
            data_transformed['surface_displacement'] = y_transformed

            progress_bar.setValue(65)
            status.setText("Removing edge readings...")
            QApplication.processEvents()

            # Remove first 2 and last 2 readings (20-minute recordings)
            reading_numbers = data_transformed['reading_number'].unique()
            if len(reading_numbers) > 4:
                # Get reading numbers to keep (exclude first 2 and last 2)
                readings_to_keep = reading_numbers[2:-2]
                data_transformed = data_transformed[data_transformed['reading_number'].isin(readings_to_keep)]
                data_transformed = data_transformed.reset_index(drop=True)

            progress_bar.setValue(70)
            status.setText("Saving transformed data...")
            QApplication.processEvents()

            # Save Step3_Transformed
            step3_file = output_folder / "Step3_Transformed.csv"

            with open(step3_file, 'w', encoding='utf-8') as f:
                f.write("# STEP 3: Transformed Data - Low frequencies removed\n")
                f.write("# ==========================================\n")
                f.write(f"# Sensor frequency: {read_sensor_freq_from_csv(step2_file)} Hz\n")
                f.write(f"# Cutoff frequency: {self.cutoff_freq:.6f} rad/s\n")
                f.write(f"# Original readings: {len(reading_numbers)}\n")
                f.write(
                    f"# Readings after edge removal: {len(readings_to_keep) if len(reading_numbers) > 4 else len(reading_numbers)}\n")
                f.write(f"# First 2 and last 2 readings removed\n")
                f.write(f"# Total points: {len(data_transformed)}\n")
                f.write("# ==========================================\n")

            data_transformed.to_csv(step3_file, mode='a', index=False)

            progress_bar.setValue(80)
            status.setText("Creating visualization...")
            QApplication.processEvents()

            # Create visualization
            target_points = VISUALIZATION_TARGET_POINTS  # 5000 points optimal for FullHD
            step = max(1, len(data_transformed) // target_points)

            data_viz = data_transformed.iloc[::step].copy()

            step3_viz_file = output_folder / "Step3_Visualization.csv"
            _viz_sensor_freq = read_sensor_freq_from_csv(output_folder / "Step2_Zero_Mean.csv")

            with open(step3_viz_file, 'w', encoding='utf-8') as f:
                f.write("# STEP 3: Visualization - Transformed data (subsampled)\n")
                f.write("# ==========================================\n")
                f.write(f"# Sampled points: {len(data_viz)}\n")
                f.write(f"# Original points: {len(data_transformed)}\n")
                f.write(f"# Sensor frequency: {_viz_sensor_freq} Hz\n")
                f.write("# ==========================================\n")

            data_viz.to_csv(step3_viz_file, mode='a', index=False)

            progress_bar.setValue(100)
            status.setText("Complete!")
            QApplication.processEvents()

            # Keep progress dialog open, update for comparison
            progress_bar.setValue(0)
            status.setText("Loading Step2 data for comparison...")
            QApplication.processEvents()

            # Show comparison window with progress
            self.show_comparison(data_viz, output_folder, progress_dialog, progress_bar, status)

        except Exception as e:
            progress_dialog.close()
            QMessageBox.critical(
                self,
                "Error",
                f"Transform failed:\n{str(e)}"
            )

    def plot_spectrogram(self):
        """
        Compute and display windowed Fourier Transform (Spectrogram)
        Based on 8_WindowFT.py algorithm
        """
        output_folder = OUTPUT_FOLDER
        step2_file = output_folder / "Step2_Zero_Mean.csv"

        # Show progress dialog
        progress_dialog = QDialog(self)
        progress_dialog.setWindowTitle("Computing Spectrogram")
        progress_dialog.setModal(True)
        progress_dialog.setFixedSize(500, 150)

        layout = QVBoxLayout(progress_dialog)

        label = QLabel("Computing windowed Fourier Transform...")
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)

        progress_bar = QProgressBar()
        progress_bar.setRange(0, 100)
        layout.addWidget(progress_bar)

        status = QLabel("Loading data...")
        status.setAlignment(Qt.AlignCenter)
        status.setStyleSheet("color: #6b7280; font-family: 'Segoe UI', sans-serif; font-size: 12px;")
        layout.addWidget(status)

        progress_dialog.show()
        QApplication.processEvents()

        try:
            # Load Step2_Zero_Mean data
            progress_bar.setValue(10)
            status.setText("Loading Step2_Zero_Mean.csv...")
            QApplication.processEvents()

            data = pd.read_csv(step2_file, comment='#')
            y = data['surface_displacement'].values

            # Parameters from UI spinboxes
            WindowSize = self.spin_window.value()  # minutes
            DeltaWindow = self.spin_delta.value()  # seconds
            part = self.spin_part.value()  # percent of spectrum to keep
            Sensor_Frequency = read_sensor_freq_from_csv(step2_file)

            progress_bar.setValue(20)
            status.setText("Preparing window parameters...")
            QApplication.processEvents()

            # Compute window parameters
            window = WindowSize * 60 * Sensor_Frequency  # window size in samples
            # Number of windows that fit with the given shift
            n = int((len(y) - window) / (DeltaWindow * Sensor_Frequency))

            from scipy.fftpack import rfft, rfftfreq
            from scipy.signal.windows import hann

            # Angular frequency axis in rad/s
            w = rfftfreq(window, (1 / Sensor_Frequency) / (2 * np.pi))
            # Number of spectrum points to keep (slice [0:spec_len] is always valid)
            # spec_idx is used only for w[spec_idx] — must be < len(w)
            spec_len = int(len(w) * 0.01 * part)  # equals len(w) when part=100
            spec_idx = min(spec_len, len(w) - 1)  # safe index for w[] — never out of bounds
            z = []

            progress_bar.setValue(30)
            status.setText(f"Computing {n} windows...")
            QApplication.processEvents()

            # Windowed FFT loop
            for i in range(n):
                # Update progress
                if i % 10 == 0:
                    progress_pct = 30 + int((i / n) * 60)
                    progress_bar.setValue(progress_pct)
                    status.setText(f"Processing window {i + 1}/{n}...")
                    QApplication.processEvents()

                # Extract window — offset in samples = window index × shift_samples
                shift_samples = DeltaWindow * Sensor_Frequency
                arr = y[i * shift_samples: window + i * shift_samples]

                # Apply Hann window
                mask = hann(len(arr))

                # Compute FFT
                s = np.abs(rfft(arr * mask))[0:spec_len]

                # Spectral density: (s²) / (len(arr) * max(w))
                s = (s ** 2) / (len(arr) * np.max(w))

                # Log scale
                s = np.log10(s)

                z.append(np.flip(s))

            z = np.asarray(z)

            progress_bar.setValue(95)
            status.setText("Creating spectrogram plot...")
            QApplication.processEvents()

            progress_dialog.close()

            # Create spectrogram window
            self.show_spectrogram(z, w, WindowSize, DeltaWindow, n, spec_len, spec_idx)

        except Exception as e:
            progress_dialog.close()
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to compute spectrogram:\n{str(e)}"
            )

    def show_spectrogram(self, z, w, WindowSize, DeltaWindow, n, spec_len, spec_idx):
        """Display spectrogram in separate window with fullscreen capability"""
        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT

        spectrogram_window = QDialog(self)
        spectrogram_window.setWindowTitle("Spectrogram - Windowed Fourier Transform")
        spectrogram_window.showMaximized()

        layout = QVBoxLayout(spectrogram_window)

        fig = Figure(figsize=(16, 10), dpi=100)
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)

        # Plot spectrogram (log scale, gist_heat colormap, vmin=-10)
        img = ax.imshow(
            np.flip(np.flip(z).T),
            extent=[0, WindowSize / 60 + n * DeltaWindow / 3600,
                    0, w[spec_idx]],  # spec_idx = min(spec_len, len(w)-1)
            cmap='gist_heat',
            vmin=-10,
            aspect='auto'
        )

        # Colorbar — use fig.colorbar, not plt.colorbar, to avoid cross-figure warning
        colorbar = fig.colorbar(img, ax=ax, shrink=0.75)
        colorbar.ax.set_ylabel('Spectral density, [m²/s]', size=16)
        colorbar.ax.tick_params(labelsize=14)

        # Set aspect ratio (from Article_WindowFT.py)
        ratio = 0.25
        x_left, x_right = ax.get_xlim()
        y_low, y_high = ax.get_ylim()
        ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)

        # Labels with larger fonts
        ax.tick_params(labelsize=14)
        ax.set_xlabel('t, [hours]', fontsize=16)
        ax.set_ylabel('ω, [rad/s]', fontsize=16)

        fig.tight_layout()

        # Add toolbar
        toolbar = NavigationToolbar2QT(canvas, spectrogram_window)

        layout.addWidget(toolbar)
        layout.addWidget(canvas)

        # Show window (non-modal, already maximized, can toggle fullscreen with F11)
        spectrogram_window.setModal(False)
        spectrogram_window.show()

    def show_comparison(self, data_transformed_viz, output_folder, progress_dialog, progress_bar, status):
        """Show before/after comparison"""
        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT

        # Load Step2 visualization (BEFORE Fourier transform)
        progress_bar.setValue(20)
        status.setText("Loading Step2 visualization...")
        QApplication.processEvents()

        step2_viz = output_folder / "Step2_Visualization.csv"
        data_before = pd.read_csv(step2_viz, comment='#')
        data_before['timestamp'] = pd.to_datetime(data_before['timestamp'], errors='coerce')
        data_transformed_viz['timestamp'] = pd.to_datetime(data_transformed_viz['timestamp'], errors='coerce')

        # Drop rows with invalid timestamps
        data_before = data_before.dropna(subset=['timestamp'])
        data_transformed_viz = data_transformed_viz.dropna(subset=['timestamp'])

        progress_bar.setValue(50)
        status.setText("Creating comparison plot...")
        QApplication.processEvents()

        # Create new window
        comparison_window = QDialog(self)
        comparison_window.setWindowTitle("Before/After Comparison")
        comparison_window.setGeometry(100, 100, 1400, 700)  # Normal size, not maximized
        comparison_window.setModal(True)

        layout = QVBoxLayout(comparison_window)

        # Header
        header = QLabel("Fourier Transform Applied — Before vs After")
        header.setFont(QFont("Segoe UI", 15, QFont.Bold))
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("color: #111827; padding: 12px; letter-spacing: -0.2px;")
        layout.addWidget(header)

        progress_bar.setValue(70)
        status.setText("Rendering plot...")
        QApplication.processEvents()

        # Create plot
        fig = Figure(figsize=(14, 6), dpi=100)
        canvas = FigureCanvas(fig)

        ax = fig.add_subplot(111)

        # Plot before (orange, more transparent per your request)
        ax.plot(data_before['timestamp'], data_before['surface_displacement'],
                linewidth=0.5, color='#fb923c', alpha=0.6, label='Before',
                marker='o', markersize=1.5, markerfacecolor='#fb923c', markeredgewidth=0, zorder=1)

        # Plot after (blue, less transparent per your request)
        ax.plot(data_transformed_viz['timestamp'], data_transformed_viz['surface_displacement'],
                linewidth=0.5, color=COLORS['wave_data'], alpha=0.7, label='After',
                marker='o', markersize=1.5, markerfacecolor=COLORS['wave_data'], markeredgewidth=0, zorder=2)

        # Horizontal line at y=0 (on top)
        ax.axhline(y=0, color='black', linewidth=2, linestyle='-', zorder=10)

        ax.set_title('Before/After Fourier Transform', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, zorder=0)
        ax.legend(loc='upper right')

        # Format dates
        import matplotlib.dates as mdates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m %H:%M'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center', fontsize=9)

        fig.tight_layout()

        progress_bar.setValue(90)
        status.setText("Finalizing...")
        QApplication.processEvents()

        toolbar = NavigationToolbar2QT(canvas, comparison_window)

        layout.addWidget(toolbar)
        layout.addWidget(canvas)

        progress_bar.setValue(100)
        status.setText("Complete!")
        QApplication.processEvents()

        # Close progress dialog
        progress_dialog.close()

        # Continue button
        btn_continue = QPushButton("▶  Continue to Step 4")
        btn_continue.setStyleSheet("""
            QPushButton {
                background-color: #0ea5e9;
                color: white;
                font-size: 14px;
                font-weight: 600;
                padding: 13px 20px;
                border-radius: 8px;
                border: none;
                font-family: 'Segoe UI', sans-serif;
            }
            QPushButton:hover {
                background-color: #0284c7;
            }
        """)

        def go_to_step4():
            comparison_window.close()  # ends exec_() → returns below
            # Defer Step4 creation to AFTER exec_() fully unwinds,
            # so the Qt event loop is clean before Step3 is destroyed.
            from PyQt5.QtCore import QTimer
            def _open_step4():
                step3_ref = self  # keep Step3 alive a moment longer
                win = Step4ProcessingWindow()
                QApplication.instance()._step4_window = win
                win.show()
                step3_ref.close()

            QTimer.singleShot(0, _open_step4)

        btn_continue.clicked.connect(go_to_step4)
        layout.addWidget(btn_continue)

        comparison_window.exec_()

    def _build_full_spectrum(self, log_scale: bool):
        """Load Step3_Spectrum.csv and display it in a FullSpectrumWindow."""
        progress = QDialog(self)
        progress.setWindowTitle('Loading Full Spectrum')
        progress.setModal(True)
        progress.setFixedSize(400, 100)
        _l = QVBoxLayout(progress)
        _l.addWidget(QLabel('Loading Step3_Spectrum.csv...'))
        pb = QProgressBar()
        pb.setRange(0, 0)
        _l.addWidget(pb)
        progress.show()
        QApplication.processEvents()
        try:
            df = pd.read_csv(OUTPUT_FOLDER / 'Step3_Spectrum.csv', comment='#')
            progress.close()
            _w = FullSpectrumWindow(df, log_scale=log_scale)
            attr = '_full_spectrum_window' if log_scale else '_full_spectrum_linear_window'
            setattr(QApplication.instance(), attr, _w)
            _w.show()
        except Exception as e:
            progress.close()
            QMessageBox.critical(self, 'Error', f'Could not load full spectrum:\n{str(e)}')

    def build_full_spectrum_linear(self):
        """Show full spectrum in linear scale (top graph toolbar button)."""
        self._build_full_spectrum(log_scale=False)

    def build_full_spectrum(self):
        """Show full spectrum in log scale (bottom graph toolbar button)."""
        self._build_full_spectrum(log_scale=True)


class Step4ProcessingWindow(QMainWindow):
    """Window for Step 4: Spike removal and RMS filtering"""

    def __init__(self):
        super().__init__()
        self.current_reading = 0  # Track which reading is being processed
        self.init_ui()
        # Load after event loop starts so the window is visible first
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(0, self.load_and_visualize)

    def init_ui(self):
        """Initialize Step 4 window"""
        self.setWindowTitle("🌊 Step 4: Spike Removal & RMS Filtering")
        # Don't show maximized here - will show after data loads

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Progress indicator
        progress = create_progress_indicator(current_step=4)
        layout.addWidget(progress)

        # Header
        header = QLabel("Step 4: Data Quality Processing")
        header.setFont(QFont("Segoe UI", 17, QFont.Bold))
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("color: #111827; padding: 14px; letter-spacing: -0.2px;")
        layout.addWidget(header)

        # Graph placeholder
        self.graph_layout = QVBoxLayout()
        layout.addLayout(self.graph_layout)

        # Controls
        controls_group = QGroupBox("Processing Options")
        controls_layout = QVBoxLayout()

        # Checkbox 1: Remove spikes
        self.cb_remove_spikes = QCheckBox("Remove spikes")
        controls_layout.addWidget(self.cb_remove_spikes)

        # Checkbox 2: Remove low RMS recordings
        rms_layout = QHBoxLayout()
        self.cb_remove_low_rms = QCheckBox("Remove recordings with RMS <")
        rms_layout.addWidget(self.cb_remove_low_rms)

        self.rms_input = QLineEdit("0.015")
        self.rms_input.setMaxLength(10)
        self.rms_input.setFixedWidth(80)
        self.rms_input.setPlaceholderText("0.015")
        rms_layout.addWidget(self.rms_input)

        rms_layout.addWidget(QLabel("meters"))
        rms_layout.addStretch()
        controls_layout.addLayout(rms_layout)

        # Checkbox 3: Spline interpolation (UI only, not functional yet)
        spline_layout = QHBoxLayout()
        self.cb_spline = QCheckBox("Spline interpolation  from")
        spline_layout.addWidget(self.cb_spline)

        # Current frequency label — will be updated after data loads
        self.lbl_freq_from = QLabel("? Hz")
        self.lbl_freq_from.setStyleSheet("font-weight: 600; color: #0284c7; font-family: 'Consolas', monospace;")
        spline_layout.addWidget(self.lbl_freq_from)

        spline_layout.addWidget(QLabel("→"))

        self.spline_freq_input = QSpinBox()
        self.spline_freq_input.setRange(1, 1000)
        self.spline_freq_input.setValue(8)
        self.spline_freq_input.setSuffix(" Hz")
        self.spline_freq_input.setFixedWidth(90)
        self.spline_freq_input.setStyleSheet("""
            QSpinBox {
                font-size: 13px; padding: 2px 6px;
                border: 1px solid #d1d5db; border-radius: 6px;
                background: #ffffff; color: #111827;
                font-family: 'Segoe UI', sans-serif;
            }
            QSpinBox:focus { border-color: #0ea5e9; }
        """)
        spline_layout.addWidget(self.spline_freq_input)
        spline_layout.addStretch()
        controls_layout.addLayout(spline_layout)

        # ── Legend ──────────────────────────────────────────────────────────
        controls_layout.addSpacing(8)
        legend_layout = QHBoxLayout()
        legend_layout.setSpacing(18)

        def _legend_item(color, label_text, style='fill'):
            """Helper: colored square/circle icon + label."""
            row = QHBoxLayout()
            row.setSpacing(5)
            icon = QLabel()
            icon.setFixedSize(18, 18)
            if style == 'fill':
                icon.setStyleSheet(
                    f"background-color: {color}; border-radius: 3px; opacity: 0.6;"
                )
            else:  # dashed circle
                icon.setStyleSheet(
                    f"border: 2px dashed {color}; border-radius: 9px; background: transparent;"
                )
            row.addWidget(icon)
            row.addWidget(QLabel(label_text))
            return row

        for row in [
            _legend_item("#22c55e", "Processed recording", 'fill'),
            _legend_item("#e74c3c", "Removed recording", 'fill'),
            _legend_item("#e74c3c", "Spike", 'circle'),
        ]:
            legend_layout.addLayout(row)
        legend_layout.addStretch()
        controls_layout.addLayout(legend_layout)

        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)

        # Buttons
        btn_layout = QHBoxLayout()

        self.btn_start = QPushButton("▶  Start Processing")
        self.btn_start.setEnabled(True)  # Always enabled
        self.btn_start.setStyleSheet("""
            QPushButton {
                background-color: #0ea5e9;
                color: white;
                font-size: 14px;
                font-weight: 600;
                padding: 13px 20px;
                border-radius: 8px;
                border: none;
                font-family: 'Segoe UI', sans-serif;
            }
            QPushButton:hover {
                background-color: #0284c7;
            }
        """)
        self.btn_start.clicked.connect(self.start_processing)
        btn_layout.addWidget(self.btn_start)

        layout.addLayout(btn_layout)

        # Add footer
        add_footer(self)

    def load_and_visualize(self):
        """Load Step3_Visualization and create plot"""
        self.hide()  # Hide window while loading — shown maximized after data is ready
        output_folder = OUTPUT_FOLDER

        step3_viz = output_folder / "Step3_Visualization.csv"

        # Parent=None so the dialog is visible even before self is shown
        progress_dialog = QDialog(None)
        progress_dialog.setWindowTitle("Opening Step 4...")
        progress_dialog.setModal(True)
        progress_dialog.setFixedSize(500, 150)
        progress_dialog.setWindowFlags(
            progress_dialog.windowFlags() | Qt.WindowStaysOnTopHint
        )

        _lay = QVBoxLayout(progress_dialog)

        _lbl = QLabel("Step 4: Spike Removal & RMS Filtering")
        _lbl.setAlignment(Qt.AlignCenter)
        _lbl.setFont(QFont("Segoe UI", 11, QFont.Bold))
        _lbl.setStyleSheet("color: #111827;")
        _lay.addWidget(_lbl)

        progress_bar = QProgressBar()
        progress_bar.setRange(0, 100)
        _lay.addWidget(progress_bar)

        status = QLabel("Loading Step3_Visualization.csv...")
        status.setAlignment(Qt.AlignCenter)
        status.setStyleSheet("color: #6b7280; font-family: 'Segoe UI', sans-serif; font-size: 12px;")
        _lay.addWidget(status)

        progress_dialog.show()
        QApplication.processEvents()

        try:
            progress_bar.setValue(20)
            QApplication.processEvents()

            # Read sensor frequency from this file's header
            viz_sensor_freq = read_sensor_freq_from_csv(step3_viz)
            self.lbl_freq_from.setText(f"{viz_sensor_freq} Hz")
            self.spline_freq_input.setMinimum(viz_sensor_freq)
            self.spline_freq_input.setValue(max(viz_sensor_freq, self.spline_freq_input.value()))

            progress_bar.setValue(40)
            status.setText("Reading data...")
            QApplication.processEvents()

            df = pd.read_csv(step3_viz, comment='#')
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

            progress_bar.setValue(70)
            status.setText("Opening window...")
            QApplication.processEvents()

            # Store data for deferred draw, then show maximized BEFORE building
            # the plot so the canvas already has its final pixel size when
            # matplotlib renders (avoids a second full render after resize).
            self._pending_plot_df = df
            progress_bar.setValue(90)
            progress_dialog.close()

            self.showMaximized()
            # One event-loop tick later the window has its real dimensions
            from PyQt5.QtCore import QTimer
            QTimer.singleShot(0, self._draw_initial_plot)

        except Exception as e:
            progress_dialog.close()
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to load data:\n{str(e)}"
            )

    def _draw_initial_plot(self):
        """Deferred: called after showMaximized so canvas has final pixel size."""
        df = getattr(self, '_pending_plot_df', None)
        if df is not None:
            self.create_interactive_plot(df)
            self._pending_plot_df = None

    def create_interactive_plot(self, data):
        """Create interactive matplotlib plot with ALL reading boundaries"""
        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT

        fig = Figure(dpi=100)
        canvas = FigureCanvas(fig)
        canvas.setSizePolicy(canvas.sizePolicy().Expanding,
                             canvas.sizePolicy().Expanding)

        ax = fig.add_subplot(111)

        # Plot data
        timestamps = data['timestamp']
        surface_displacement = data['surface_displacement'].values

        ax.plot(timestamps, surface_displacement, linewidth=0.5, color=COLORS['wave_data'], alpha=0.7,
                marker='o', markersize=1.5, markerfacecolor=COLORS['wave_data'], markeredgewidth=0, zorder=2)

        # Add horizontal line at y=0 (thick black dashed)
        ax.axhline(y=0, color='black', linewidth=1.8, linestyle='--', dashes=(6, 4),
                   alpha=0.7, zorder=5)

        # Add vertical lines for ALL reading boundaries
        reading_numbers = data['reading_number'].unique()

        for reading_num in reading_numbers:
            # Find first timestamp of this reading
            reading_data = data[data['reading_number'] == reading_num]
            if len(reading_data) > 0:
                reading_start = reading_data['timestamp'].iloc[0]
                ax.axvline(reading_start, color='gray', linestyle='--',
                           linewidth=0.5, alpha=0.3)

        ax.set_title('Step 3: Transformed Data (after Fourier Transform, with 20-min reading boundaries)',
                     fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, zorder=0)

        # Tight x-axis: first to last timestamp
        ax.set_xlim(timestamps.iloc[0], timestamps.iloc[-1])

        # Format dates
        import matplotlib.dates as mdates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m %H:%M'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center', fontsize=9)

        fig.tight_layout()

        # Store for later access during processing
        self.fig = fig
        self.ax = ax
        self.canvas = canvas

        # Add navigation toolbar
        toolbar = NavigationToolbar2QT(canvas, self)
        full_data_action = QAction('📊 Build all data points (slow)', self)
        full_data_action.triggered.connect(self.build_full_data_step3)
        toolbar.addAction(full_data_action)

        # Clear previous graph and add new one
        for i in reversed(range(self.graph_layout.count())):
            self.graph_layout.itemAt(i).widget().setParent(None)

        self.graph_layout.addWidget(toolbar)
        self.graph_layout.addWidget(canvas, stretch=1)

    def start_processing(self):
        """
        Start spike removal and/or RMS filtering with real-time visualization
        Calculate ALL wave parameters in single pass
        """
        output_folder = OUTPUT_FOLDER

        step2_file = output_folder / "Step3_Transformed.csv"
        step2_viz = output_folder / "Step3_Visualization.csv"
        parameters_file = output_folder / "Parameters.csv"

        # Get options
        remove_spikes = self.cb_remove_spikes.isChecked()
        remove_low_rms = self.cb_remove_low_rms.isChecked()
        use_spline = self.cb_spline.isChecked()
        spline_target_hz = self.spline_freq_input.value()  # target Hz from spinbox

        # Parse RMS threshold
        rms_threshold = 0.015
        if remove_low_rms:
            try:
                rms_text = self.rms_input.text().strip()
                rms_threshold = float(rms_text)
            except Exception:
                rms_threshold = 0.015

        # Helper functions for wave analysis
        def kh_solver(h, Tz):
            """Solve kh equation: x * tanh(x) = (4π²h)/(g*Tz²)"""
            from scipy.optimize import fsolve
            import warnings
            g = 9.81
            target = (4 * np.pi ** 2 * h) / (g * Tz ** 2)

            def equation(x):
                return x * np.tanh(x) - target

            # Adaptive initial guess based on target magnitude
            if target < 1.0:
                x0 = target  # small target → solution near 0
            elif target < 10.0:
                x0 = np.sqrt(target)  # moderate target
            else:
                x0 = target  # large target → solution is large

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = fsolve(equation, x0, full_output=False)[0]

            return max(0.01, result)  # Avoid division by zero

        def calculate_spectrum_params(arr, sensor_freq):
            """Calculate spectral parameters: Q, nu, eps_width, rho"""
            from scipy.fftpack import rfft, rfftfreq
            from scipy.signal.windows import hann

            # FFT with Hann window
            win = hann(len(arr))
            s_full = np.abs(rfft(arr * win))
            # Frequency grid must be built from the original signal length,
            # NOT from len(s_full) which is already the one-sided rfft output.
            w = rfftfreq(len(arr), (1 / sensor_freq) / (2 * np.pi))

            # Filter frequencies: use half-Nyquist as upper bound
            # (sensor_freq * π is the Nyquist in rad/s; half gives a practical limit)
            n = sensor_freq * np.pi / 2
            ind = w < n
            s = s_full[ind]
            w = w[ind]

            if len(w) < 2:
                return 0, 0, 0, 0

            dx = w[1] - w[0]

            # Spectral moments
            m0 = np.trapezoid(s, dx=dx)
            m1 = np.trapezoid(w * s, dx=dx)
            m2 = np.trapezoid((w ** 2) * s, dx=dx)
            m4 = np.trapezoid((w ** 4) * s, dx=dx)

            if m0 == 0:
                return 0, 0, 0, 0

            # Q (Goda parameter)
            Q = np.trapezoid(w * s ** 2, dx=dx) / (m0 ** 2)

            # nu (narrowness)
            if m1 != 0:
                nu = np.sqrt(((m0 * m2) / (m1 ** 2)) - 1)
            else:
                nu = 0

            # eps_width
            if m0 * m4 != 0:
                eps_width = np.sqrt(1 - (m2 ** 2) / (m0 * m4))
            else:
                eps_width = 0

            # rho - amplitude to extrema ratio
            from PyAstronomy import pyaC
            from scipy.fftpack import irfft

            try:
                y = irfft(rfft(arr)[ind])
                t = np.arange(len(y))
                tc, ti = pyaC.zerocross1d(t, y, getIndices=True)

                tnew = np.sort(np.append(t, tc))
                for c1 in range(1, len(tnew)):
                    if tnew[c1] in tc:
                        tzm1 = np.where(tnew == tnew[c1 - 1])[0]
                        yzm1 = np.where(y == y[tzm1])[0]
                        y = np.insert(y, yzm1 + 1, [0])

                amplitudes = 0
                extremas = 0
                q = np.arange(0)

                for j in y:
                    if j == 0:
                        q = np.abs(q)
                        q = np.append(q, 0)
                        amplitudes += 1
                        for c2 in range(1, len(q) - 1):
                            if q[c2] > q[c2 - 1] and q[c2] > q[c2 + 1]:
                                extremas += 1
                        q = np.arange(0)
                    q = np.append(q, j)

                if extremas > 0:
                    rho = amplitudes / extremas
                    if rho > 1:
                        rho = 1
                else:
                    rho = 0
            except Exception:
                rho = 0

            return Q, nu, eps_width, rho

        def calculate_gamma(kh):
            """Calculate gamma function for BFI"""
            try:
                v = 1 + (2 * kh) / (np.sinh(2 * kh))
                a = -v ** 2 + 2 + 8 * (kh ** 2) * ((np.cosh(2 * kh)) / ((np.sinh(2 * kh)) ** 2))
                b = ((np.cosh(4 * kh) + 8 - 2 * (np.tanh(kh)) ** 2) / (8 * (np.sinh(kh)) ** 4) -
                     ((2 * (np.cosh(kh)) ** 2 + 0.5 * v) ** 2) /
                     ((np.sinh(2 * kh)) ** 2 * ((kh / (np.tanh(kh))) - (v / 2) ** 2)))

                if a < 0:
                    print(f"Warning: a < 0 for kh={kh}")

                gamma = v * np.sqrt(np.abs(b) / abs(a))
                return gamma
            except Exception:
                return 0

        def calculate_Tz(arr, sensor_freq):
            """Calculate mean zero-crossing period"""
            from PyAstronomy import pyaC

            try:
                x = np.arange(len(arr))
                xc, xi = pyaC.zerocross1d(x, arr, getIndices=True)
                if len(xc) > 0:
                    Tz = ((len(arr) / sensor_freq) / len(xc)) * 2
                else:
                    Tz = 10.0  # Default
                return Tz
            except Exception:
                return 10.0

        # Show progress dialog
        progress_dialog = QDialog(self)
        progress_dialog.setWindowTitle("Processing Step 3 + Calculating Wave Parameters")
        progress_dialog.setModal(True)
        progress_dialog.setFixedSize(550, 150)

        layout = QVBoxLayout(progress_dialog)

        label = QLabel("Processing data and calculating wave parameters...")
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)

        progress_bar = QProgressBar()
        progress_bar.setRange(0, 100)
        layout.addWidget(progress_bar)

        status = QLabel("Loading data...")
        status.setAlignment(Qt.AlignCenter)
        status.setStyleSheet("color: #6b7280; font-family: 'Segoe UI', sans-serif; font-size: 12px;")
        layout.addWidget(status)

        progress_dialog.show()
        QApplication.processEvents()

        try:
            # Load full data
            progress_bar.setValue(5)
            status.setText("Loading Step3_Transformed.csv...")
            QApplication.processEvents()

            data = pd.read_csv(step2_file, comment='#')
            data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')

            # Load existing parameters
            existing_params = pd.read_csv(parameters_file, comment='#')

            # Load viz data for mapping
            viz_data = pd.read_csv(step2_viz, comment='#')
            viz_data['timestamp'] = pd.to_datetime(viz_data['timestamp'], errors='coerce')

            progress_bar.setValue(10)
            status.setText("Starting processing...")
            QApplication.processEvents()

            # Group by reading_number
            grouped = data.groupby('reading_number')
            total_readings = len(grouped)

            removed_readings = []
            spike_locations = []

            # Storage for ALL new parameters
            new_params = []

            import matplotlib.dates as mdates

            # Batch visualization
            batch_size = 10
            batch_readings = []
            batch_has_removal = False

            # Read sensor frequency from CSV header (Step3 inherits it from Step2)
            sensor_freq = read_sensor_freq_from_csv(step2_file)

            # ==========================
            # MAIN PROCESSING LOOP - OPTIMIZED
            # ==========================

            # Pre-compute all arrays for speed (avoid repeated pandas indexing)
            reading_arrays = {}
            reading_info = {}

            for reading_num, reading_data in grouped:
                reading_arrays[reading_num] = {
                    'surface_displacement': reading_data['surface_displacement'].values,
                    'indices': reading_data.index.values,
                    'timestamps': reading_data['timestamp'].values,
                    'start': reading_data['timestamp'].iloc[0],
                    'end': reading_data['timestamp'].iloc[-1]
                }

                # Get depth
                depth_row = existing_params[existing_params['reading_number'] == reading_num]
                if len(depth_row) > 0:
                    depth = depth_row['average_depth'].values[0]
                else:
                    depth = 10.0

                reading_info[reading_num] = {'depth': depth}

            # Process in batches for visualization
            reading_nums = list(reading_arrays.keys())

            # Visualization: batch size 100 for speed
            batch_size = 100
            batch_readings = []
            removed_in_batch = []  # Track which specific readings were removed

            for idx, reading_num in enumerate(reading_nums):
                progress_pct = 10 + int((idx / total_readings) * 75)
                progress_bar.setValue(progress_pct)
                status.setText(f"Processing reading {reading_num} ({idx + 1}/{total_readings})...")

                if idx % 5 == 0:
                    QApplication.processEvents()

                arr_data = reading_arrays[reading_num]
                arr = arr_data[
                    'surface_displacement'].copy()  # .copy() required — arr is modified in-place during spike removal
                arr_indices = arr_data['indices']
                arr_timestamps = arr_data['timestamps']
                reading_start = arr_data['start']
                reading_end = arr_data['end']

                depth = reading_info[reading_num]['depth']

                # ========== STEP 1: CHECK RMS ==========
                should_remove = False
                rms_value = np.sqrt(np.mean(arr ** 2))

                if remove_low_rms and rms_value < rms_threshold:
                    should_remove = True
                    removed_readings.append(reading_num)

                # Track for batch coloring
                batch_readings.append({
                    'start': reading_start,
                    'end': reading_end,
                    'reading_num': reading_num,
                    'removed': should_remove
                })

                if should_remove:
                    removed_in_batch.append({
                        'start': reading_start,
                        'end': reading_end
                    })
                    continue

                # ========== STEP 2: SPIKE REMOVAL (if enabled) ==========
                if remove_spikes:
                    variance = np.var(arr)
                    threshold_spike = 6 * np.sqrt(variance)

                    for j in range(len(arr) - 1):
                        if np.abs(arr[j + 1] - arr[j]) > threshold_spike:
                            spike_idx = arr_indices[j + 1]
                            spike_time = arr_timestamps[j + 1]
                            spike_value = arr[j + 1]

                            spike_locations.append({
                                'time': spike_time,
                                'value': spike_value
                            })

                            if j + 2 < len(arr):
                                new_value = (arr[j] + arr[j + 2]) / 2
                            else:
                                new_value = arr[j]

                            data.loc[spike_idx, 'surface_displacement'] = new_value
                            arr[j + 1] = new_value

                # ========== STEP 3: CALCULATE ALL WAVE PARAMETERS ==========

                # --- Spline interpolation for amplitude metrics (As, Hs) ---
                # If enabled: resample arr to spline_target_hz via cubic spline.
                # All spectral/integral parameters (Q, nu, eps, Tz…) still use
                # the original arr at sensor_freq.
                if use_spline and spline_target_hz > sensor_freq:
                    from scipy.interpolate import CubicSpline
                    n_orig = len(arr)
                    t_orig = np.arange(n_orig) / sensor_freq  # seconds
                    cs = CubicSpline(t_orig, arr)
                    # New time grid at target frequency
                    t_new = np.arange(0, t_orig[-1], 1.0 / spline_target_hz)
                    arr_amp = cs(t_new)  # interpolated
                else:
                    arr_amp = arr  # same as original

                variance = np.var(arr)  # from ORIGINAL — for spectral params
                sigma = np.sqrt(np.var(arr_amp))  # from interpolated — for amplitudes
                As = 2 * sigma
                Hs = 4 * sigma

                Tz = calculate_Tz(arr, sensor_freq)
                kh = kh_solver(depth, Tz)
                k = kh / depth
                eps = (k * Hs) / 4
                a = As / depth
                Ur = (3 * k * Hs) / ((2 * k * depth) ** 3)

                Q, nu, eps_width, rho = calculate_spectrum_params(arr, sensor_freq)

                if rho > 0 and rho < 1:
                    eps_rho = (2 * np.sqrt(1 - rho)) / (2 - rho)
                else:
                    eps_rho = 0

                gamma = calculate_gamma(kh)

                sqrt_2pi = np.sqrt(2 * np.pi)
                BFI_proper = sqrt_2pi * eps * Q * gamma
                BFI_goda = sqrt_2pi * eps * Q
                BFI_goda_divide = eps / Q if Q != 0 else 0
                BFI_nu = eps / nu if nu != 0 else 0
                BFI_eps = eps / eps_width if eps_width != 0 else 0
                BFI_rho = eps / eps_rho if eps_rho != 0 else 0

                new_params.append({
                    'reading_number': reading_num,
                    'average_depth': depth,
                    'rms': rms_value,
                    'As': As,
                    'Hs': Hs,
                    'Tz': Tz,
                    'kh': kh,
                    'k': k,
                    'eps': eps,
                    'a': a,
                    'Ur': Ur,
                    'Q_goda': Q,
                    'nu': nu,
                    'eps_width': eps_width,
                    'rho': rho,
                    'eps_rho': eps_rho,
                    'gamma': gamma,
                    'BFI_proper': BFI_proper,
                    'BFI_goda': BFI_goda,
                    'BFI_goda_divide': BFI_goda_divide,
                    'BFI_nu': BFI_nu,
                    'BFI_eps': BFI_eps,
                    'BFI_rho': BFI_rho
                })

                # ========== VISUALIZATION (OPTIMIZED) ==========
                if (idx + 1) % batch_size == 0 or idx == total_readings - 1:
                    # Color entire batch GREEN first
                    if len(batch_readings) > 0:
                        batch_start = batch_readings[0]['start']
                        batch_end = batch_readings[-1]['end']

                        x_start = mdates.date2num(batch_start)
                        x_end = mdates.date2num(batch_end)

                        y_min, y_max = self.ax.get_ylim()

                        # Green for entire batch
                        self.ax.axvspan(x_start, x_end, alpha=0.35, color='green', zorder=2)

                        # Red ONLY for removed readings (overlay on top)
                        for removed_reading in removed_in_batch:
                            x_rem_start = mdates.date2num(removed_reading['start'])
                            x_rem_end = mdates.date2num(removed_reading['end'])
                            self.ax.axvspan(x_rem_start, x_rem_end, alpha=0.35, color='red', zorder=3)

                    batch_readings = []
                    removed_in_batch = []

                    self.canvas.draw()
                    QApplication.processEvents()

            # Draw black circles for spikes
            progress_bar.setValue(87)
            status.setText("Marking spike locations...")
            QApplication.processEvents()

            for spike_info in spike_locations:
                spike_time = spike_info['time']
                spike_value = spike_info['value']

                spike_x = mdates.date2num(spike_time)

                # Dashed unfilled circle marker — always round regardless of axis scale
                self.ax.plot(spike_x, spike_value, 'o',
                             markersize=9,
                             markerfacecolor='none',
                             markeredgecolor='red',
                             markeredgewidth=1.2,
                             linestyle='none',
                             zorder=15)

            self.canvas.draw()
            QApplication.processEvents()

            # Remove filtered readings
            progress_bar.setValue(90)
            status.setText("Finalizing data...")
            QApplication.processEvents()

            if removed_readings:
                data_filtered = data[~data['reading_number'].isin(removed_readings)]
            else:
                data_filtered = data

            # Save Step4 file
            progress_bar.setValue(93)
            status.setText("Saving Step4_Filtered.csv...")
            QApplication.processEvents()

            step4_file = output_folder / "Step4_Filtered.csv"

            with open(step4_file, 'w', encoding='utf-8') as f:
                f.write("# STEP 4: Filtered Data - Spike removal & RMS filtering\n")
                f.write("# ==========================================\n")
                f.write(f"# Spike removal: {remove_spikes}\n")
                f.write(f"# RMS filtering: {remove_low_rms}\n")
                if remove_low_rms:
                    f.write(f"# RMS threshold: {rms_threshold} meters\n")
                f.write(f"# Spikes found and corrected: {len(spike_locations)}\n")
                f.write(f"# Readings removed: {len(removed_readings)}\n")
                f.write(f"# Readings remaining: {total_readings - len(removed_readings)}\n")
                f.write("# ==========================================\n")

            data_filtered.to_csv(step4_file, mode='a', index=False)

            # Update Parameters.csv with ALL new parameters
            progress_bar.setValue(96)
            status.setText("Updating Parameters.csv...")
            QApplication.processEvents()

            new_params_df = pd.DataFrame(new_params)

            # Save updated Parameters.csv
            parameters_updated = output_folder / "Parameters.csv"

            with open(parameters_updated, 'w', encoding='utf-8') as f:
                f.write("# PARAMETERS - 20-minute readings and wave characteristics\n")
                f.write("# ==========================================\n")
                f.write("# All wave parameters calculated in Step 3\n")
                f.write("# ==========================================\n")

            new_params_df.to_csv(parameters_updated, mode='a', index=False)

            progress_bar.setValue(100)
            status.setText("Complete!")
            QApplication.processEvents()

            progress_dialog.close()

            # Calculate statistics for display
            if len(new_params) > 0:
                mean_Hs = np.mean([p['Hs'] for p in new_params])
                mean_Tz = np.mean([p['Tz'] for p in new_params])
                mean_eps = np.mean([p['eps'] for p in new_params])
                mean_BFI = np.mean([p['BFI_goda'] for p in new_params])
            else:
                mean_Hs = mean_Tz = mean_eps = mean_BFI = 0

            # Show pipeline complete window
            stats = {
                'total_readings': total_readings,
                'removed_rms': len(removed_readings),
                'spikes_corrected': len(spike_locations),
                'remaining': total_readings - len(removed_readings),
                'mean_Hs': mean_Hs,
                'mean_Tz': mean_Tz,
            }
            _cw = PipelineCompleteWindow(output_folder, stats)
            QApplication.instance()._complete_window = _cw
            _cw.show()

        except Exception as e:
            progress_dialog.close()
            import traceback
            QMessageBox.critical(
                self,
                "Error",
                f"Processing failed:\n{str(e)}\n\n{traceback.format_exc()}"
            )

    def build_full_data_step3(self):
        """Load and plot full Step3 data in new window"""
        progress = QDialog(self);
        progress.setWindowTitle('Loading Full Data')
        progress.setModal(True);
        progress.setFixedSize(400, 100)
        _l = QVBoxLayout(progress);
        _l.addWidget(QLabel('Loading Step3_Transformed.csv...'))
        pb = QProgressBar();
        pb.setRange(0, 0);
        _l.addWidget(pb)
        progress.show();
        QApplication.processEvents()
        try:
            df = pd.read_csv(OUTPUT_FOLDER / 'Step3_Transformed.csv', comment='#')
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            progress.close()
            _w = FullDataWindow(df, 'Step 3: Full Transformed Data')
            self._full_window = _w;
            _w.show()
        except Exception as e:
            progress.close();
            QMessageBox.critical(self, 'Error', f'Could not load full data:\n{str(e)}')


class FullDataWindow(QMainWindow):
    """
    Full-resolution data viewer.

    Deferred-draw pattern: showMaximized() first, then QTimer(0) → _draw().
    The canvas already has its final pixel size when matplotlib renders —
    no double-render even for 50 M points.
    """

    def __init__(self, data_df, title):
        super().__init__()
        self.setWindowTitle(title)
        self._data_df = data_df
        self._title   = title

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Empty figure — no figsize; stretches to fill the window
        self._fig    = Figure(dpi=100)
        self._canvas = FigureCanvas(self._fig)
        self._canvas.setSizePolicy(
            self._canvas.sizePolicy().Expanding,
            self._canvas.sizePolicy().Expanding,
        )

        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
        toolbar = NavigationToolbar2QT(self._canvas, self)
        layout.addWidget(toolbar)
        layout.addWidget(self._canvas, stretch=1)

        # Slim bottom bar: info label + close button
        bottom_bar = QWidget()
        bottom_bar.setMaximumHeight(36)
        bottom_bar.setStyleSheet("background:#f8fafc; border-top:1px solid #e5e7eb;")
        bar_layout = QHBoxLayout(bottom_bar)
        bar_layout.setContentsMargins(12, 4, 12, 4)

        info_lbl = QLabel(
            f"Total points: {len(data_df):,}  |  "
            f"Memory: ~{len(data_df) * 24 / 1024 / 1024:.1f} MB"
        )
        info_lbl.setStyleSheet(
            "font-size:11px; color:#6b7280;"
            " font-family:'Consolas',monospace; background:transparent;"
        )
        bar_layout.addWidget(info_lbl)
        bar_layout.addStretch()

        close_btn = QPushButton("Close")
        close_btn.setFixedHeight(26)
        close_btn.setStyleSheet(
            "QPushButton{background:#f3f4f6;color:#374151;border:1px solid #d1d5db;"
            "border-radius:4px;padding:2px 14px;font-size:12px;font-weight:600;}"
            "QPushButton:hover{background:#e5e7eb;}"
        )
        close_btn.clicked.connect(self.close)
        bar_layout.addWidget(close_btn)

        layout.addWidget(bottom_bar)

        self.showMaximized()
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(0, self._draw)

    def _draw(self):
        """Render into the already-maximized, correctly-sized canvas."""
        import matplotlib.dates as mdates

        ax = self._fig.add_subplot(111)

        ts = self._data_df['timestamp']
        sd = self._data_df['surface_displacement'].values

        ax.plot(ts, sd,
                linewidth=0.4, color=COLORS['wave_data'], alpha=0.85,
                marker='o', markersize=1.2,
                markerfacecolor=COLORS['wave_data'], markeredgewidth=0,
                zorder=2)

        # Horizontal reference line at y=0
        ax.axhline(y=0, color='black', linewidth=1.8, linestyle='--',
                   dashes=(6, 4), alpha=0.7, zorder=4)

        ax.set_xlabel('Date', fontsize=11)
        ax.set_ylabel('Surface displacement (m)', fontsize=11)
        ax.set_title(
            f'{self._title} — {len(self._data_df):,} points',
            fontsize=13, fontweight='bold',
        )
        ax.grid(True, alpha=0.3, zorder=0)

        # Tight x-axis: clamp to first and last timestamp
        ax.set_xlim(ts.iloc[0], ts.iloc[-1])

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%y'))
        self._fig.autofmt_xdate()
        self._fig.tight_layout(pad=1.5)

        self._canvas.draw()


class FullSpectrumWindow(QMainWindow):
    """
    Full-resolution spectrum viewer (log or linear scale).

    Same deferred-draw pattern: showMaximized() first → QTimer(0) → _draw().
    """

    def __init__(self, spectrum_df, log_scale=True):
        super().__init__()
        title_mode = "log scale" if log_scale else "linear scale"
        self.setWindowTitle(f"Full Spectrum — {title_mode}")

        self._spectrum_df = spectrum_df
        self._log_scale   = log_scale

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Empty figure — stretches to fill the window
        self._fig    = Figure(dpi=100)
        self._canvas = FigureCanvas(self._fig)
        self._canvas.setSizePolicy(
            self._canvas.sizePolicy().Expanding,
            self._canvas.sizePolicy().Expanding,
        )

        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
        toolbar = NavigationToolbar2QT(self._canvas, self)
        layout.addWidget(toolbar)
        layout.addWidget(self._canvas, stretch=1)

        # Slim bottom bar
        bottom_bar = QWidget()
        bottom_bar.setMaximumHeight(36)
        bottom_bar.setStyleSheet("background:#f8fafc; border-top:1px solid #e5e7eb;")
        bar_layout = QHBoxLayout(bottom_bar)
        bar_layout.setContentsMargins(12, 4, 12, 4)

        info_lbl = QLabel(f"Total frequency points: {len(spectrum_df):,}")
        info_lbl.setStyleSheet(
            "font-size:11px; color:#6b7280;"
            " font-family:'Consolas',monospace; background:transparent;"
        )
        bar_layout.addWidget(info_lbl)
        bar_layout.addStretch()

        close_btn = QPushButton("Close")
        close_btn.setFixedHeight(26)
        close_btn.setStyleSheet(
            "QPushButton{background:#f3f4f6;color:#374151;border:1px solid #d1d5db;"
            "border-radius:4px;padding:2px 14px;font-size:12px;font-weight:600;}"
            "QPushButton:hover{background:#e5e7eb;}"
        )
        close_btn.clicked.connect(self.close)
        bar_layout.addWidget(close_btn)

        layout.addWidget(bottom_bar)

        self.showMaximized()
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(0, self._draw)

    def _draw(self):
        """Render into the already-maximized canvas."""
        freq = self._spectrum_df['frequency'].values
        real = self._spectrum_df['real'].values
        imag = self._spectrum_df['imag'].values

        N         = len(freq)
        omega_max = np.max(np.abs(freq)) if N > 0 else 1
        s         = (real ** 2 + imag ** 2) / ((N / 2) * omega_max)

        if self._log_scale:
            f_min        = freq[freq > 0].min() if np.any(freq > 0) else 0
            mask         = freq > f_min
            title_suffix = "full, log scale, DC removed"
        else:
            mask         = freq > 0.05
            title_suffix = "ω > 0.05, linear scale"

        freq_plot = freq[mask]
        s_plot    = s[mask]

        ax = self._fig.add_subplot(111)
        ax.plot(freq_plot, s_plot,
                linewidth=0.8, color=COLORS['dive_detect'], zorder=2)

        ax.set_xlabel('ω, [rad/s]', fontsize=11)
        ax.set_ylabel('S(ω), [m²/s]', fontsize=11)
        ax.set_title(
            f'Full Spectrum — {mask.sum():,} points — {title_suffix}',
            fontsize=13, fontweight='bold',
        )
        ax.grid(True, alpha=0.3, which='both', zorder=0)

        # Clamp negative axes at initial view
        if len(freq_plot) > 0:
            ax.set_xlim(left=0)

        if self._log_scale:
            ax.set_yscale('log')
            s_pos = s_plot[s_plot > 0]
            if len(s_pos) > 0:
                ax.set_ylim(bottom=s_pos.min() * 0.5)
        else:
            ax.set_ylim(bottom=0)

        self._fig.tight_layout(pad=1.5)
        self._canvas.draw()


class PipelineCompleteWindow(QDialog):
    """Final window shown after Step 4 completes — export / cleanup / exit."""

    # Files to KEEP (never deleted by "Clear cache")
    KEEP_FILES = {'Parameters.csv', 'Step4_Filtered.csv'}

    # All intermediate files (cache)
    CACHE_FILES = [
        'Step1_TXTtoCSV.csv',
        'Step1_Visualization.csv',
        'Step2_Initial_Cut.csv',
        'Step2_Visualization.csv',
        'Step2_Zero_Mean.csv',
        'Step3_Spectrum.csv',
        'Step3_Spectrum_Visualization.csv',
        'Step3_Transformed.csv',
        'Step3_Visualization.csv',
    ]

    def __init__(self, output_folder, stats):
        super().__init__(None)
        self.output_folder = Path(output_folder)
        self.stats = stats
        self.setWindowTitle("🎉 Pipeline Complete")
        self.setFixedWidth(460)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(24, 20, 24, 20)

        # ── Header ──────────────────────────────────────────────────────────
        title = QLabel("✓  Step 4 Complete!")
        title.setFont(QFont("Segoe UI", 15, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #16a34a; padding-bottom: 4px; letter-spacing: -0.2px;")
        layout.addWidget(title)

        # ── Stats ────────────────────────────────────────────────────────────
        s = self.stats
        stats_text = (
            f"<b>Readings processed:</b> {s['total_readings']}<br>"
            f"<b>Removed (low RMS):</b> {s['removed_rms']}<br>"
            f"<b>Spikes corrected:</b> {s['spikes_corrected']}<br>"
            f"<b>Remaining readings:</b> {s['remaining']}<br>"
        )
        stats_lbl = QLabel(stats_text)
        stats_lbl.setStyleSheet(
            "background:#f8fafc; border-radius:8px; padding:12px 16px;"
            "font-size:13px; color:#374151; border: 1px solid #e5e7eb;"
            "font-family: 'Segoe UI', sans-serif; line-height: 1.6;"
        )
        stats_lbl.setTextFormat(Qt.RichText)
        layout.addWidget(stats_lbl)

        # ── Kept files note ──────────────────────────────────────────────────
        kept_lbl = QLabel(
            "<b>Output files:</b><br>"
            "• <tt>Parameters.csv</tt> — wave parameters for all readings<br>"
            "• <tt>Step4_Filtered.csv</tt> — filtered surface_displacement time-series"
        )
        kept_lbl.setTextFormat(Qt.RichText)
        kept_lbl.setStyleSheet("font-size:12px; color:#6b7280; padding: 4px 0; font-family: 'Segoe UI', sans-serif;")
        layout.addWidget(kept_lbl)

        layout.addSpacing(4)

        # ── Buttons ──────────────────────────────────────────────────────────
        # 1. Clear cache
        btn_cache = QPushButton("🗑  Clear cache  (keep Parameters & Filtered)")
        btn_cache.setStyleSheet("""
            QPushButton {
                background:#f3f4f6; color:#374151; font-size:13px;
                padding:10px; border-radius:6px; border:1px solid #d1d5db;
                font-family: 'Segoe UI', sans-serif;
            }
            QPushButton:hover { background:#e5e7eb; border-color:#9ca3af; }
        """)
        btn_cache.clicked.connect(self._clear_cache)
        layout.addWidget(btn_cache)

        # 2. Save / export
        btn_save = QPushButton("💾  Export data…")
        btn_save.setStyleSheet("""
            QPushButton {
                background:#0ea5e9; color:white; font-size:13px;
                padding:10px; border-radius:6px; border: none;
                font-family: 'Segoe UI', sans-serif; font-weight: 600;
            }
            QPushButton:hover { background:#0284c7; }
        """)
        btn_save.clicked.connect(self._export_data)
        layout.addWidget(btn_save)

        # 3. Exit
        btn_exit = QPushButton("✖  Exit")
        btn_exit.setStyleSheet("""
            QPushButton {
                background:#fff1f2; color:#e11d48; font-size:13px;
                font-weight:600; padding:10px; border-radius:6px;
                border: 1px solid #fecdd3; font-family: 'Segoe UI', sans-serif;
            }
            QPushButton:hover { background:#ffe4e6; border-color:#fda4af; }
        """)
        btn_exit.clicked.connect(self.on_exit_with_rename)
        layout.addWidget(btn_exit)

    # ── Actions ──────────────────────────────────────────────────────────────

    def on_exit_with_rename(self):
        """Exit application and rename Output folder with current timestamp"""
        try:
            output_folder = self.output_folder

            if output_folder.exists():
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                new_name = f"Output_{timestamp}"
                new_path = output_folder.parent / new_name

                # Rename
                output_folder.rename(new_path)

                QMessageBox.information(
                    self,
                    "Folder Renamed",
                    f"Output folder renamed to:\n{new_name}"
                )
        except Exception as e:
            print(f"Could not rename: {e}")

        QApplication.quit()

    def _clear_cache(self):
        deleted, missing = [], []
        for fname in self.CACHE_FILES:
            fp = self.output_folder / fname
            if fp.exists():
                try:
                    fp.unlink()
                    deleted.append(fname)
                except Exception as e:
                    missing.append(f"{fname} ({e})")
            else:
                missing.append(f"{fname} (not found)")

        msg = f"Deleted {len(deleted)} file(s)."
        if deleted:
            msg += "\n\nRemoved:\n" + "\n".join(f"  • {f}" for f in deleted)
        if missing:
            msg += "\n\nSkipped:\n" + "\n".join(f"  • {f}" for f in missing)
        QMessageBox.information(self, "Cache cleared", msg)

    def _export_data(self):
        """Export Step4_Filtered.csv and Parameters.csv in chosen format."""
        fmt_dialog = QDialog(self)
        fmt_dialog.setWindowTitle("Export format")
        fmt_dialog.setFixedWidth(320)
        fl = QVBoxLayout(fmt_dialog)
        fl.addWidget(QLabel("Choose export format:"))

        formats = [
            ("Text (.txt) — tab-separated", "txt"),
            ("MATLAB (.mat) — scipy.io.savemat", "mat"),
        ]
        radios = []
        for label, key in formats:
            rb = QRadioButton(label)
            fl.addWidget(rb)
            radios.append((rb, key))
        radios[0][0].setChecked(True)

        btn_row = QHBoxLayout()
        ok_btn = QPushButton("Export");
        ok_btn.clicked.connect(fmt_dialog.accept)
        cxl_btn = QPushButton("Cancel");
        cxl_btn.clicked.connect(fmt_dialog.reject)
        btn_row.addWidget(ok_btn);
        btn_row.addWidget(cxl_btn)
        fl.addLayout(btn_row)

        if fmt_dialog.exec_() != QDialog.Accepted:
            return

        chosen = next(key for rb, key in radios if rb.isChecked())

        dest_dir = QFileDialog.getExistingDirectory(
            self, "Select destination folder", str(self.output_folder)
        )
        if not dest_dir:
            return
        dest_dir = Path(dest_dir)

        # ── Progress dialog ──────────────────────────────────────────────────
        prog = QDialog(self)
        prog.setWindowTitle("Exporting…")
        prog.setModal(True)
        prog.setFixedSize(420, 110)
        _pl = QVBoxLayout(prog)
        prog_lbl = QLabel("Preparing…")
        prog_lbl.setAlignment(Qt.AlignCenter)
        _pl.addWidget(prog_lbl)
        prog_bar = QProgressBar()
        prog_bar.setRange(0, 0)  # indeterminate spinner
        _pl.addWidget(prog_bar)
        prog.show()
        QApplication.processEvents()

        exported = []
        errors = []
        files = ("Step4_Filtered.csv", "Parameters.csv")

        for idx, fname in enumerate(files):
            prog_lbl.setText(f"Exporting {fname}…")
            QApplication.processEvents()

            src = self.output_folder / fname
            if not src.exists():
                errors.append(f"{fname} not found");
                continue
            try:
                df = pd.read_csv(src, comment='#')
                stem = src.stem

                if chosen == "txt":
                    out = dest_dir / f"{stem}.txt"
                    df.to_csv(out, sep='\t', index=False)
                    exported.append(out.name)




                elif chosen == "mat":
                    from scipy.io import savemat
                    out = dest_dir / f"{stem}.mat"
                    mat_dict = {}
                    for col in df.columns:

                        safe = col.replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '').replace('/',
                                                                                                                 '_')

                        vals = df[col].values
                        if col == 'timestamp':
                            try:
                                ts = pd.to_datetime(vals, errors='coerce')
                                epoch = pd.Timestamp('0000-01-01')
                                days_since_epoch = (ts - epoch).dt.total_seconds() / 86400.0
                                mat_dict['timestamp'] = days_since_epoch.values.astype(float)
                            except Exception:
                                continue
                            continue

                        if vals.dtype == object or str(vals.dtype).startswith('<U'):
                            try:
                                vals = pd.to_numeric(vals, errors='coerce')
                                if np.all(np.isnan(vals)):
                                    continue

                            except Exception:
                                continue

                        mat_dict[safe] = vals.astype(float)

                    if mat_dict:
                        savemat(str(out), mat_dict, do_compression=True)
                        exported.append(out.name)
                    else:

                        errors.append(f"{fname}: No numeric data")

            except Exception as e:
                errors.append(f"{fname}: {e}")

        prog.close()

        msg = f"Exported {len(exported)} file(s) to:\n{dest_dir}"
        if exported:
            msg += "\n\n" + "\n".join(f"  • {f}" for f in exported)
        if errors:
            msg += "\n\nErrors:\n" + "\n".join(f"  • {e}" for e in errors)
        QMessageBox.information(self, "Export complete", msg)


def clear_output_folder(output_folder):
    """Delete all pipeline output files so user starts from scratch."""
    all_files = [
        'Step1_TXTtoCSV.csv',
        'Step1_Visualization.csv',
        'Step2_Initial_Cut.csv',
        'Step2_Visualization.csv',
        'Step2_Zero_Mean.csv',
        'Step3_Spectrum.csv',
        'Step3_Spectrum_Visualization.csv',
        'Step3_Transformed.csv',
        'Step3_Visualization.csv',
        'Parameters.csv',
        'Step4_Filtered.csv',
    ]
    for fname in all_files:
        p = Path(output_folder) / fname
        try:
            if p.exists():
                p.unlink()
        except Exception:
            pass


def main():
    """Launch application"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    # Global modern stylesheet — light theme with refined custom controls
    app.setStyleSheet("""
        QDialog, QWidget {
            background-color: #ffffff;
            color: #111827;
            font-family: 'Segoe UI', sans-serif;
        }
        QProgressBar {
            border: none;
            border-radius: 6px;
            text-align: center;
            height: 22px;
            background-color: #f1f5f9;
            color: #6b7280;
            font-size: 11px;
            font-family: 'Segoe UI', sans-serif;
        }
        QProgressBar::chunk {
            background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #0ea5e9, stop:1 #38bdf8);
            border-radius: 6px;
        }
        QScrollBar:vertical {
            background: #f8fafc;
            width: 8px;
            border-radius: 4px;
            margin: 0;
        }
        QScrollBar::handle:vertical {
            background: #cbd5e1;
            border-radius: 4px;
            min-height: 30px;
        }
        QScrollBar::handle:vertical:hover {
            background: #94a3b8;
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
        QScrollBar:horizontal {
            background: #f8fafc;
            height: 8px;
            border-radius: 4px;
        }
        QScrollBar::handle:horizontal {
            background: #cbd5e1;
            border-radius: 4px;
        }
        QScrollBar::handle:horizontal:hover { background: #94a3b8; }
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
        QCheckBox {
            color: #374151;
            spacing: 8px;
            font-size: 13px;
            font-family: 'Segoe UI', sans-serif;
        }
        QCheckBox::indicator {
            width: 16px;
            height: 16px;
            border: 1.5px solid #d1d5db;
            border-radius: 4px;
            background: #ffffff;
        }
        QCheckBox::indicator:checked {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #0ea5e9, stop:1 #0284c7);
            border-color: #0284c7;
            image: none;
        }
        QCheckBox::indicator:hover {
            border-color: #0ea5e9;
        }
        QLineEdit {
            background: #ffffff;
            color: #111827;
            border: 1px solid #d1d5db;
            border-radius: 6px;
            padding: 4px 8px;
            font-size: 13px;
            font-family: 'Segoe UI', sans-serif;
        }
        QLineEdit:focus {
            border-color: #0ea5e9;
            outline: none;
        }
        QToolBar {
            background: #f8fafc;
            border: none;
            border-bottom: 1px solid #e5e7eb;
            spacing: 2px;
            padding: 2px 4px;
        }
        QToolButton {
            background: transparent;
            color: #374151;
            border-radius: 4px;
            padding: 3px 6px;
            font-size: 12px;
        }
        QToolButton:hover {
            background: #e5e7eb;
        }
        QToolButton:pressed {
            background: #d1d5db;
        }
        QGroupBox {
            font-weight: 600;
            font-size: 13px;
            border: 1px solid #e5e7eb;
            border-radius: 10px;
            margin-top: 14px;
            padding-top: 16px;
            background-color: #ffffff;
            color: #374151;
        }
        QGroupBox::title {
            color: #374151;
            subcontrol-origin: margin;
            left: 14px;
            padding: 0 8px;
            background-color: #ffffff;
        }
        QPushButton {
            background-color: #f3f4f6;
            color: #374151;
            border: 1px solid #d1d5db;
            padding: 8px 18px;
            border-radius: 6px;
            font-size: 13px;
            font-weight: 600;
            font-family: 'Segoe UI', sans-serif;
        }
        QPushButton:hover {
            background-color: #e5e7eb;
            border-color: #9ca3af;
        }
        QPushButton:pressed {
            background-color: #d1d5db;
        }
        QPushButton:disabled {
            background-color: #f3f4f6;
            color: #9ca3af;
            border-color: #e5e7eb;
        }
    """)

    output_folder = OUTPUT_FOLDER

    # CHECKPOINT 4: Check if Step4_Filtered exists (Pipeline complete)
    step4_filtered = output_folder / "Step4_Filtered.csv"
    parameters = output_folder / "Parameters.csv"

    if step4_filtered.exists() and parameters.exists():
        reply = QMessageBox.question(
            None,
            "Pipeline Complete!",
            f"Found completed pipeline:\n"
            f"• Step4_Filtered.csv\n"
            f"• Parameters.csv\n\n"
            "View results or start from scratch?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )

        if reply == QMessageBox.Yes:
            # Read basic stats from files for display
            try:
                params_df = pd.read_csv(parameters, comment='#')
                total_readings = len(params_df)
                mean_Hs = params_df['Hs'].mean() if 'Hs' in params_df.columns else 0.0
                mean_Tz = params_df['Tz'].mean() if 'Tz' in params_df.columns else 0.0

                stats = {
                    'total_readings': total_readings,
                    'removed_rms': 0,  # Not tracked in checkpoint
                    'spikes_corrected': 0,  # Not tracked in checkpoint
                    'remaining': total_readings,
                    'mean_Hs': mean_Hs,
                    'mean_Tz': mean_Tz
                }

                complete_window = PipelineCompleteWindow(output_folder, stats)
                complete_window.exec_()
                sys.exit(0)
            except Exception as e:
                QMessageBox.warning(
                    None,
                    "Error",
                    f"Could not read results:\n{str(e)}\n\nStarting from Step 4 instead."
                )
                # Fall through to checkpoint 3
        else:
            # User wants to start from scratch
            clear_output_folder(output_folder)
            window = MainWindow()
            QApplication.instance()._main_window = window
            window.show()
            sys.exit(app.exec_())

    # CHECKPOINT 3: Check if Step3_Transformed exists
    step3_transformed = output_folder / "Step3_Transformed.csv"
    step3_viz = output_folder / "Step3_Visualization.csv"

    if step3_transformed.exists() and step3_viz.exists():
        reply = QMessageBox.question(
            None,
            "Step 3 Complete - Continue?",
            f"Found processed Step 3 data:\n"
            f"• Step3_Transformed.csv\n"
            f"• Step3_Visualization.csv\n"
            f"• Parameters.csv\n\n"
            "Continue to Step 4 (Spike removal & RMS filtering) or start from scratch?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )

        if reply == QMessageBox.Yes:
            step4_window = Step4ProcessingWindow()
            app.instance()._step4_window = step4_window  # keep alive — prevent GC
            step4_window.show()
            sys.exit(app.exec_())
        else:
            # User wants to start from scratch — delete all output files
            clear_output_folder(output_folder)
            window = MainWindow()
            QApplication.instance()._main_window = window
            window.show()
            sys.exit(app.exec_())

    # CHECKPOINT 2: Check if Step2_Zero_Mean exists (Step 2 complete)
    step2_zero_mean = output_folder / "Step2_Zero_Mean.csv"
    step2_viz = output_folder / "Step2_Visualization.csv"
    parameters_file = output_folder / "Parameters.csv"

    if step2_zero_mean.exists() and step2_viz.exists() and parameters_file.exists():
        reply = QMessageBox.question(
            None,
            "Step 2 Complete - Continue?",
            f"Found processed Step 2 data:\n"
            f"• Step2_Zero_Mean.csv\n"
            f"• Step2_Visualization.csv\n"
            f"• Parameters.csv\n\n"
            "Continue to Step 3 (Fourier Transform) or start from scratch?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )

        if reply == QMessageBox.Yes:
            # Open Step 3 Fourier window
            step3_window = Step3FourierWindow()
            app.instance()._step3_window = step3_window  # prevent GC
            step3_window.show()
            sys.exit(app.exec_())
        else:
            # User wants to start from scratch — delete all output files
            clear_output_folder(output_folder)
            window = MainWindow()
            QApplication.instance()._main_window = window
            window.show()
            sys.exit(app.exec_())

    # CHECKPOINT 1: Check if Step1 CSV already exists
    csv_file = output_folder / "Step1_TXTtoCSV.csv"
    viz_cache_file = output_folder / "Step1_Visualization.csv"

    if csv_file.exists():
        # Ask user if they want to load existing data
        reply = QMessageBox.question(
            None,
            "Existing Data Found",
            f"Found existing processed data:\n{csv_file}\n\n"
            "Load this data directly (fast) or start from scratch?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )

        if reply == QMessageBox.No:
            # User wants to start from scratch — delete all output files
            clear_output_folder(output_folder)
            window = MainWindow()
            QApplication.instance()._main_window = window
            window.show()
            sys.exit(app.exec_())

        if reply == QMessageBox.Yes:
            try:
                # Check if visualization cache exists - INSTANT LOAD
                if viz_cache_file.exists():
                    # Super fast path - just load pre-sampled data
                    df = pd.read_csv(viz_cache_file, comment='#')
                    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

                    # Read metadata
                    with open(viz_cache_file, 'r') as f:
                        for line in f:
                            if line.startswith('# Sensor frequency:'):
                                df.attrs['sensor_frequency_hz'] = int(line.split(':')[1].strip().split()[0])
                            elif line.startswith('# Recording start:'):
                                df.attrs['recording_start'] = line.split(':', 1)[1].strip()
                            elif line.startswith('# Recording end:'):
                                df.attrs['recording_end'] = line.split(':', 1)[1].strip()

                    # Open visualization instantly
                    viz_window = VisualizationWindow(df)
                    viz_window.show()
                    sys.exit(app.exec_())

                # No cache - need to create it
                # Show progress dialog for loading
                progress_dialog = QDialog()
                progress_dialog.setWindowTitle("Loading Data")
                progress_dialog.setModal(True)
                progress_dialog.setFixedSize(400, 150)

                layout = QVBoxLayout(progress_dialog)

                label = QLabel("Loading CSV file...")
                label.setAlignment(Qt.AlignCenter)
                layout.addWidget(label)

                progress = QProgressBar()
                progress.setRange(0, 0)  # Indeterminate
                layout.addWidget(progress)

                status = QLabel("Reading file...")
                status.setAlignment(Qt.AlignCenter)
                status.setStyleSheet("color: #6b7280; font-family: 'Segoe UI', sans-serif; font-size: 12px;")
                layout.addWidget(status)

                progress_dialog.show()
                app.processEvents()

                # Adaptive subsample: count newlines in one fast binary read,
                # then pick step so we keep ~10 000 rows for visualization.
                # This is much cheaper than the old line-by-line generator because
                # bytes.count() is a single C-level scan with no Python objects per line.
                status.setText("Scanning file size...")
                app.processEvents()

                with open(csv_file, 'rb') as f:
                    raw = f.read()
                # Subtract comment lines and the one column-header line (-2 total)
                total_lines = raw.count(b'\n') - raw.count(b'\n#') - 2
                del raw  # free memory immediately

                target_rows = 10000
                sample_step = max(1, total_lines // target_rows)

                status.setText(f"Loading {total_lines:,} records (keeping 1 of every {sample_step} rows)...")
                app.processEvents()

                # Read in chunks and subsample on the fly
                chunk_size = 100000
                sampled_data = []
                row_counter = 0

                for chunk in pd.read_csv(csv_file, comment='#', chunksize=chunk_size):
                    keep_indices = [i for i in range(len(chunk)) if (row_counter + i) % sample_step == 0]
                    if keep_indices:
                        sampled_data.append(chunk.iloc[keep_indices])
                    row_counter += len(chunk)
                    progress_pct = min(100, int(row_counter / total_lines * 100))
                    status.setText(f"Loading... {progress_pct}% ({row_counter:,} / {total_lines:,})")
                    app.processEvents()

                # Combine all sampled data
                df = pd.concat(sampled_data, ignore_index=True)

                status.setText("Converting timestamps...")
                app.processEvents()

                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

                # Read metadata from original file
                with open(csv_file, 'r') as f:
                    for line in f:
                        if line.startswith('# Sensor frequency:'):
                            freq = int(line.split(':')[1].strip().split()[0])
                            df.attrs['sensor_frequency_hz'] = freq
                        elif line.startswith('# Recording start:'):
                            df.attrs['recording_start'] = line.split(':', 1)[1].strip()
                        elif line.startswith('# Recording end:'):
                            df.attrs['recording_end'] = line.split(':', 1)[1].strip()

                # Save visualization cache for next time
                status.setText("Saving visualization cache...")
                app.processEvents()

                with open(viz_cache_file, 'w', encoding='utf-8') as f:
                    f.write("# VISUALIZATION CACHE - Subsampled data for fast plotting\n")
                    f.write("# ==========================================\n")
                    f.write(f"# Sensor frequency: {df.attrs.get('sensor_frequency_hz', 'N/A')} Hz\n")
                    f.write(f"# Recording start: {df.attrs.get('recording_start', 'N/A')}\n")
                    f.write(f"# Recording end: {df.attrs.get('recording_end', 'N/A')}\n")
                    f.write(f"# Sampled points: {len(df)}\n")
                    f.write("# ==========================================\n")

                df.to_csv(viz_cache_file, mode='a', index=False)

                progress_dialog.close()

                # Open visualization
                viz_window = VisualizationWindow(df)
                viz_window.show()
                sys.exit(app.exec_())

            except Exception as e:
                if 'progress_dialog' in locals():
                    progress_dialog.close()
                QMessageBox.warning(
                    None,
                    "Error Loading",
                    f"Could not load existing data:\n{str(e)}\n\nStarting from scratch."
                )

    # Normal start - show file loading window
    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()