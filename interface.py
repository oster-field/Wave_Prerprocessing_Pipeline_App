"""
Sakhalin Wave Processor - Новый интерфейс
Простое окно для загрузки файлов с drag & drop
"""

import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QListWidget, QGroupBox, QMessageBox, QDialog,
                             QProgressBar, QTextEdit, QCheckBox, QLineEdit,
                             QSpinBox)
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


def read_sensor_freq_from_csv(csv_path, default=8):
    """Read sensor frequency from CSV comment header. Returns int Hz."""
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
    return default


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

            all_pressure_data = []

            for i, file_path in enumerate(self.data_files):
                if self.should_stop:
                    return

                # Read data from file
                data = self.read_data_file(file_path)
                all_pressure_data.append(data)

                # Update progress
                progress_pct = 10 + int((i + 1) / len(self.data_files) * 30)
                self.progress.emit(progress_pct, f"Loaded {i+1}/{len(self.data_files)}")

            if self.should_stop:
                return

            # Step 4: Concatenate all data into single array (fast!)
            self.progress.emit(45, "Combining data...")
            all_data = np.concatenate(all_pressure_data)

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
                'pressure': all_data,
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
            script_dir = Path(__file__).parent if '__file__' in globals() else Path.cwd()
            output_folder = script_dir / "Output"
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
        content = None
        for encoding in ['windows-1251', 'cp1251', 'utf-8', 'latin-1']:
            try:
                with open(self.info_file, 'r', encoding=encoding, errors='ignore') as f:
                    content = f.read()
                break
            except Exception:
                continue
        if content is None:
            raise ValueError("Could not read INFO file with any encoding")

        # Sensor frequency
        freq_match = re.search(r'[Чч]астота\s+опроса[^:]*:\s*(\d+)', content) \
                  or re.search(r'[Ff]requency[^:]*:\s*(\d+)', content)
        sensor_frequency = int(freq_match.group(1)) if freq_match else 8

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
            'date_start':       date_start,
            'date_end':         date_end,
            'dt_start':         dt_start,
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
    """Зона для перетаскивания файлов"""
    files_dropped = pyqtSignal(list)

    def __init__(self, text="", allowed_extensions=None):
        super().__init__(text)
        self.allowed_extensions = allowed_extensions or []
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumHeight(120)
        self.setStyleSheet("""
            QLabel {
                border: 3px dashed #3498db;
                border-radius: 10px;
                background-color: #ecf0f1;
                color: #2c3e50;
                font-size: 14px;
                padding: 20px;
            }
            QLabel:hover {
                background-color: #d5dbdb;
                border-color: #2980b9;
            }
        """)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet(self.styleSheet().replace('#ecf0f1', '#a8e6cf'))

    def dragLeaveEvent(self, event):
        self.setStyleSheet(self.styleSheet().replace('#a8e6cf', '#ecf0f1'))

    def dropEvent(self, event: QDropEvent):
        files = [url.toLocalFile() for url in event.mimeData().urls()]

        # Фильтруем по расширениям если заданы
        if self.allowed_extensions:
            valid_files = [f for f in files if any(f.endswith(ext) for ext in self.allowed_extensions)]
        else:
            valid_files = files

        if valid_files:
            self.files_dropped.emit(valid_files)

        self.setStyleSheet(self.styleSheet().replace('#a8e6cf', '#ecf0f1'))


class ProgressDialog(QDialog):
    """Dialog showing processing progress"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Processing Data")
        self.setModal(True)
        self.setFixedSize(500, 250)

        layout = QVBoxLayout(self)

        # Title
        title = QLabel("🔄 Processing Wave Data")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #3498db;
                border-radius: 5px;
                text-align: center;
                height: 30px;
            }
            QProgressBar::chunk {
                background-color: #3498db;
            }
        """)
        layout.addWidget(self.progress_bar)

        # Status message
        self.status_label = QLabel("Starting...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: #7f8c8d; padding: 10px;")
        layout.addWidget(self.status_label)

        # Log window
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(80)
        self.log_text.setStyleSheet("""
            QTextEdit {
                border: 1px solid #bdc3c7;
                border-radius: 3px;
                background-color: #ecf0f1;
                font-family: monospace;
                font-size: 10px;
            }
        """)
        layout.addWidget(self.log_text)

    def update_progress(self, percentage, message):
        """Update progress bar and message"""
        self.progress_bar.setValue(percentage)
        self.status_label.setText(message)
        self.log_text.append(f"[{percentage}%] {message}")


class MainWindow(QMainWindow):
    """Главное окно приложения"""

    def __init__(self):
        super().__init__()
        self.info_file = None
        self.data_files = []
        self.check_existing_data()

    def check_existing_data(self):
        """Check if processed data already exists"""
        # Look for Output/Step1_TXTtoCSV.csv in current directory
        current_dir = Path.cwd()
        output_file = current_dir / "Output" / "Step1_TXTtoCSV.csv"

        if output_file.exists():
            # Ask user if they want to continue from previous session
            reply = QMessageBox.question(
                None,
                "Previous Session Found",
                f"Found existing processed data:\n{output_file}\n\n"
                "Do you want to continue from previous session?\n\n"
                "Yes - Load existing data and show visualization\n"
                "No - Start fresh (will overwrite)",
                QMessageBox.Yes | QMessageBox.No
            )

            if reply == QMessageBox.Yes:
                # Load existing data and go directly to visualization
                try:
                    df = pd.read_csv(output_file, comment='#')

                    # Read metadata from file
                    with open(output_file, 'r') as f:
                        for line in f:
                            if line.startswith('# Sensor frequency:'):
                                freq = int(line.split(':')[1].strip().split()[0])
                                df.attrs['sensor_frequency_hz'] = freq
                            elif line.startswith('# Recording start:'):
                                df.attrs['recording_start'] = line.split(':', 1)[1].strip()
                            elif line.startswith('# Recording end:'):
                                df.attrs['recording_end'] = line.split(':', 1)[1].strip()

                    # Show visualization directly
                    self.show_visualization_directly(df)
                    return
                except Exception as e:
                    QMessageBox.warning(
                        None,
                        "Load Error",
                        f"Could not load existing file:\n{str(e)}\n\nStarting fresh."
                    )

        # If no existing data or user chose to start fresh, show normal UI
        self.init_ui()

    def show_visualization_directly(self, df):
        """Show visualization window directly without main window"""
        self.viz_window = VisualizationWindow(df)
        self.viz_window.show()
        # Don't show the main window
        self.hide()

    def init_ui(self):
        """Инициализация интерфейса"""
        self.setWindowTitle("🌊 Sakhalin Wave Processor")

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Header
        header = QLabel("🌊 Sakhalin Wave Data Processor")
        header.setFont(QFont("Arial", 20, QFont.Bold))
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("color: #2c3e50; padding: 20px;")
        layout.addWidget(header)

        # Instruction
        instruction = QLabel("📁 Load files for wave data processing")
        instruction.setAlignment(Qt.AlignCenter)
        instruction.setStyleSheet("color: #7f8c8d; font-size: 13px; padding-bottom: 10px;")
        layout.addWidget(instruction)

        # Секция INFO файла
        info_group = self.create_info_section()
        layout.addWidget(info_group)

        # Секция файлов данных
        data_group = self.create_data_section()
        layout.addWidget(data_group)

        # Continue button
        self.btn_continue = QPushButton("▶️ Continue to Processing")
        self.btn_continue.setEnabled(False)
        self.btn_continue.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                font-size: 16px;
                font-weight: bold;
                padding: 15px;
                border-radius: 8px;
                margin-top: 20px;
            }
            QPushButton:hover:enabled {
                background-color: #229954;
            }
            QPushButton:disabled {
                background-color: #95a5a6;
            }
        """)
        self.btn_continue.clicked.connect(self.on_continue)
        layout.addWidget(self.btn_continue)

        # Status
        self.status_label = QLabel("⏳ Waiting for files...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: #7f8c8d; padding: 10px;")
        layout.addWidget(self.status_label)

        layout.addStretch()

        self.apply_global_styles()

        # Show maximized AFTER UI is fully built
        self.showMaximized()

    def create_info_section(self):
        """Section for INFO file"""
        group = QGroupBox("📋 INFO File")
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
        self.info_label.setStyleSheet("color: #e74c3c; font-style: italic; padding: 5px;")
        layout.addWidget(self.info_label)

        group.setLayout(layout)
        return group

    def create_data_section(self):
        """Section for data files"""
        group = QGroupBox("📊 Data Files")
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
                border: 1px solid #bdc3c7;
                border-radius: 5px;
                background-color: white;
                padding: 5px;
            }
        """)
        layout.addWidget(self.data_list)

        # File counter
        self.data_count_label = QLabel("Files loaded: 0")
        self.data_count_label.setStyleSheet("color: #7f8c8d; padding: 5px;")
        layout.addWidget(self.data_count_label)

        group.setLayout(layout)
        return group

    def on_info_dropped(self, files):
        """Обработка перетаскивания INFO файла"""
        if files:
            # Берём первый файл
            self.set_info_file(files[0])

    def on_data_dropped(self, files):
        """Обработка перетаскивания файлов данных"""
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
                "color: #27ae60; font-weight: bold; padding: 5px; font-family: monospace;"
            )
        except Exception as e:
            self.info_label.setText(f"✅ Loaded: {filename}\n⚠️ Parse error: {str(e)}")
            self.info_label.setStyleSheet("color: #e67e22; font-weight: bold; padding: 5px;")

        self.btn_clear_info.setEnabled(True)
        self.update_status()

    def _parse_info_file(self, file_path):
        """Parse INFO file with regex - handles any encoding, flexible format"""
        content = None
        for encoding in ['windows-1251', 'cp1251', 'utf-8', 'latin-1']:
            try:
                with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                    content = f.read()
                break
            except Exception:
                continue
        if content is None:
            raise ValueError("Cannot read file")

        # Sensor frequency
        freq_match = re.search(r'[Чч]астота\s+опроса[^:]*:\s*(\d+)', content) \
                  or re.search(r'[Ff]requency[^:]*:\s*(\d+)', content)
        sensor_frequency = int(freq_match.group(1)) if freq_match else 8

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
            days    = delta.days
            hours   = (total_s % 86400) // 3600
            mins    = (total_s % 3600)  // 60
            secs    = total_s % 60
            recording_duration = (
                f"{days}d {hours:02d}h {mins:02d}m {secs:02d}s"
                if days > 0 else
                f"{hours:02d}h {mins:02d}m {secs:02d}s"
            )

        # Total measurements
        meas_match = re.search(r'[Кк]оличество\s+измерений[^:]*:\s*(\d+)', content)
        total_measurements = int(meas_match.group(1)) if meas_match else None

        return {
            'sensor_frequency':   sensor_frequency,
            'dt_start':           dt_start,
            'dt_end':             dt_end,
            'date_start':         dt_start.date() if dt_start else None,
            'date_end':           dt_end.date()   if dt_end   else None,
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
        self.info_label.setStyleSheet("color: #e74c3c; font-style: italic; padding: 5px;")
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
            self.status_label.setText(f"✅ Ready to process: INFO + {len(self.data_files)} data files")
            self.status_label.setStyleSheet("color: #27ae60; font-weight: bold; padding: 10px;")
            self.btn_continue.setEnabled(True)
        elif self.info_file:
            self.status_label.setText("⏳ Load data files to continue")
            self.status_label.setStyleSheet("color: #f39c12; padding: 10px;")
            self.btn_continue.setEnabled(False)
        elif self.data_files:
            self.status_label.setText("⏳ Load INFO file to continue")
            self.status_label.setStyleSheet("color: #f39c12; padding: 10px;")
            self.btn_continue.setEnabled(False)
        else:
            self.status_label.setText("⏳ Waiting for files...")
            self.status_label.setStyleSheet("color: #7f8c8d; padding: 10px;")
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
            script_dir = Path(__file__).parent if '__file__' in globals() else Path.cwd()
            output_folder = script_dir / "Output"
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
                f"💾 Saved to:\n"
                f"  {output_file}\n"
                f"  {viz_cache_file} (visualization cache)\n\n"
                f"Next: Visualization"
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
        # Close current window
        self.close()

        # Open visualization window
        self.viz_window = VisualizationWindow(self.processed_data)
        self.viz_window.show()

    def apply_global_styles(self):
        """Применить глобальные стили"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f6fa;
            }
            QGroupBox {
                font-weight: bold;
                font-size: 14px;
                border: 2px solid #bdc3c7;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 15px;
                background-color: white;
            }
            QGroupBox::title {
                color: #2c3e50;
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 8px;
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #21618c;
            }
            QPushButton:disabled {
                background-color: #bdc3c7;
                color: #7f8c8d;
            }
        """)


class VisualizationWindow(QMainWindow):
    """Window for visualizing processed data"""

    def __init__(self, data_df):
        super().__init__()
        self.data_df = data_df
        self.init_ui()

    def init_ui(self):
        """Initialize visualization window"""
        self.setWindowTitle("🌊 Wave Data Visualization")
        self.showMaximized()  # Fullscreen

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Header
        header = QLabel("📊 Raw Data Visualization")
        header.setFont(QFont("Arial", 18, QFont.Bold))
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("color: #2c3e50; padding: 15px;")
        layout.addWidget(header)

        # Info label
        info_text = (f"Total points: {len(self.data_df):,} | "
                    f"Readings: {self.data_df['reading_number'].max()} | "
                    f"Frequency: {self.data_df.attrs.get('sensor_frequency_hz', 'N/A')} Hz")
        info_label = QLabel(info_text)
        info_label.setAlignment(Qt.AlignCenter)
        info_label.setStyleSheet("color: #7f8c8d; font-size: 12px; padding: 5px;")
        layout.addWidget(info_label)

        # Plot canvas with interactive toolbar
        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
        self.canvas = self.create_plot()
        toolbar = NavigationToolbar2QT(self.canvas, self)
        layout.addWidget(toolbar)
        layout.addWidget(self.canvas)

        # Buttons
        btn_layout = QHBoxLayout()

        self.btn_skip = QPushButton("⏭️ Continue without Manual Removal")
        self.btn_skip.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                font-size: 14px;
                font-weight: bold;
                padding: 15px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
        self.btn_skip.clicked.connect(self.on_skip_removal)
        btn_layout.addWidget(self.btn_skip)

        self.btn_manual = QPushButton("✏️ Proceed with Manual Data Removal")
        self.btn_manual.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                font-size: 14px;
                font-weight: bold;
                padding: 15px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #229954;
            }
        """)
        self.btn_manual.clicked.connect(self.on_manual_removal)
        btn_layout.addWidget(self.btn_manual)

        layout.addLayout(btn_layout)

        self.apply_styles()

    def create_plot(self):
        """Create matplotlib plot - optimized for speed"""
        from PyQt5.QtWidgets import QDesktopWidget

        # Get screen resolution for adaptive subsampling
        screen = QDesktopWidget().screenGeometry()
        screen_width = screen.width()

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
        dive_mask = self.detect_dives(data_to_plot['pressure'].values)

        # Convert timestamps
        timestamps = pd.to_datetime(data_to_plot['timestamp'], errors='coerce')
        pressure = data_to_plot['pressure'].values

        # FIRST: Draw complete blue line (no gaps)
        ax.plot(timestamps, pressure,
               linewidth=0.5, color='#3498db', alpha=0.7, label='Wave data', zorder=1)

        # SECOND: Overlay red segments on top (no connecting lines between segments)
        if dive_mask.sum() > 0:
            # Find continuous dive segments
            dive_indices = np.where(dive_mask)[0]

            # Split into continuous segments
            segments = []
            if len(dive_indices) > 0:
                segment_start = dive_indices[0]
                for i in range(1, len(dive_indices)):
                    if dive_indices[i] != dive_indices[i-1] + 1:
                        # End of segment
                        segments.append((segment_start, dive_indices[i-1]))
                        segment_start = dive_indices[i]
                # Last segment
                segments.append((segment_start, dive_indices[-1]))

            # Plot each segment separately (prevents connecting lines)
            for i, (start, end) in enumerate(segments):
                label = 'Sensor deployment/retrieval' if i == 0 else None
                ax.plot(timestamps[start:end+1], pressure[start:end+1],
                       linewidth=1.0, color='#e74c3c', alpha=0.9, label=label, zorder=2)

        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Pressure', fontsize=12)
        ax.set_title(f'Raw Wave Data - {len(self.data_df):,} total points ({len(data_to_plot):,} displayed at {screen_width}px)',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')

        # Format x-axis with dates
        import matplotlib.dates as mdates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%y'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        fig.autofmt_xdate()

        fig.tight_layout()

        return canvas

    def detect_dives(self, pressure, sensitivity=3.0):
        """
        Dive detector based on gradient analysis.

        Key rules:
        1. GUARD: if first point >= 2.0  → no beginning leg.
                  if last  point >= 2.0  → no ending leg.
           (Deployment always starts in air/shallow <2, retrieval ends there.)

        2. BEGINNING LEG — search in first 40% of data:
           Find ALL positive gradient spikes where:
             - gradient > sensitivity * grad_std  (significant spike)
             - pressure BEFORE spike < 2.0        (coming from air/shallow)
             - pressure AFTER  spike >= 2.0       (arrived at depth)
           These are true water-entry events. The leg ends after the
           LAST such spike (sensor may re-surface several times before
           settling at deployment depth).
           Mark [0 : last_jump_settled] as dive leg.

        3. ENDING LEG — search in last 40% of data:
           Find ALL negative gradient spikes where:
             - gradient < -sensitivity * grad_std (significant drop)
             - pressure BEFORE spike >= 2.0       (was at depth)
             - pressure AFTER  spike < 2.0        (left the water)
           The leg starts at the FIRST such spike.
           Mark [first_drop_start : n] as dive leg.

        Returns:
            Boolean mask where True = dive/retrieval section to remove.
        """
        n = len(pressure)
        dive_mask = np.zeros(n, dtype=bool)

        if n < 100:
            return dive_mask

        gradient     = np.gradient(pressure)
        gradient_abs = np.abs(gradient)
        grad_std     = np.std(gradient)
        threshold    = sensitivity * grad_std

        # ── BEGINNING LEG ──────────────────────────────────────────────────
        if pressure[0] < 2.0:
            search_end = min(int(n * 0.4), 4000)

            # True dive entry: large positive gradient AND crosses the 2.0 boundary
            jump_indices = []
            for i in range(1, search_end):
                if gradient[i] > threshold:
                    p_before = pressure[i - 1]
                    p_after  = pressure[min(i + 1, n - 1)]
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
        if pressure[-1] < 2.0:
            search_start = max(int(n * 0.6), n - 4000)

            # True retrieval exit: large negative gradient AND crosses 2.0 boundary
            drop_indices = []
            for i in range(search_start + 1, n):
                if gradient[i] < -threshold:
                    p_before = pressure[i - 1]
                    p_after  = pressure[min(i + 1, n - 1)]
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


    def on_manual_removal(self):
        """Handle manual removal button click"""
        # Close current window and open manual removal window
        self.manual_window = ManualRemovalWindow(self.data_df)
        self.manual_window.show()
        self.close()

    def on_skip_removal(self):
        """Handle skip button click - copy Step1 to Step2 with progress bar and Zero Mean processing"""
        script_dir = Path(__file__).parent if '__file__' in globals() else Path.cwd()
        output_folder = script_dir / "Output"
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
        status.setStyleSheet("color: #7f8c8d;")
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

            # Open Step 3 window
            self.step3_window = Step3FourierWindow()
            self.step3_window.show()
            self.close()

        except Exception as e:
            progress_dialog.close()
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to process:\n{str(e)}"
            )

    def process_zero_mean(self, step2_file, output_folder, progress_bar, status):
        """
        Process Zero Mean - same as in ManualRemovalWindow
        """
        # Read Step2 data
        status.setText("Reading Step2_Initial_Cut.csv...")
        QApplication.processEvents()

        data = pd.read_csv(step2_file, comment='#')

        # Calculate global average
        progress_bar.setValue(85)
        status.setText("Calculating global average (Avg_Depth_FullRec)...")
        QApplication.processEvents()

        avg_depth_full_rec = data['pressure'].mean()

        # Calculate average for each reading
        progress_bar.setValue(88)
        status.setText("Calculating averages for each 20-min reading...")
        QApplication.processEvents()

        reading_averages = data.groupby('reading_number')['pressure'].mean().reset_index()
        reading_averages.columns = ['reading_number', 'average_depth']

        # Create Zero Mean data
        progress_bar.setValue(92)
        status.setText("Creating Zero Mean data...")
        QApplication.processEvents()

        zero_mean_data = data.copy()
        zero_mean_data['pressure'] = zero_mean_data['pressure'] - avg_depth_full_rec

        # Save Zero Mean file
        progress_bar.setValue(95)
        status.setText("Saving Step2_Zero_Mean.csv...")
        QApplication.processEvents()

        zero_mean_file = output_folder / "Step2_Zero_Mean.csv"
        with open(zero_mean_file, 'w', encoding='utf-8') as f:
            f.write("# STEP 2: Zero Mean - Global average subtracted\n")
            f.write("# ==========================================\n")
            f.write(f"# Sensor frequency: {read_sensor_freq_from_csv(step2_file)} Hz\n")
            f.write(f"# Average Depth (Full Record): {avg_depth_full_rec:.6f}\n")
            f.write(f"# All pressure values have this subtracted\n")
            f.write("# ==========================================\n")

        zero_mean_data.to_csv(zero_mean_file, mode='a', index=False)

        # Create Step2_Visualization.csv (subsampled for caching)
        progress_bar.setValue(96)
        status.setText("Creating Step2_Visualization.csv...")
        QApplication.processEvents()

        target_points = VISUALIZATION_TARGET_POINTS  # 5000 points optimal for FullHD
        subsample_step = max(1, len(zero_mean_data) // target_points)
        viz_data = zero_mean_data.iloc[::subsample_step].copy()

        step2_viz_file = output_folder / "Step2_Visualization.csv"
        with open(step2_viz_file, 'w', encoding='utf-8') as f:
            f.write("# STEP 2: Visualization Cache - Subsampled Zero Mean data\n")
            f.write("# ==========================================\n")
            f.write(f"# Sampled points: {len(viz_data)}\n")
            f.write(f"# Original points: {len(zero_mean_data)}\n")
            f.write("# ==========================================\n")

        viz_data.to_csv(step2_viz_file, mode='a', index=False)

        # Save Parameters file
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

    def apply_styles(self):
        """Apply global styles"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f6fa;
            }
        """)


class ManualRemovalWindow(QMainWindow):
    """Window for manually selecting dive regions to remove"""

    def __init__(self, viz_data_df):
        super().__init__()
        self.viz_data_df = viz_data_df  # Subsampled visualization data
        self.cut_indices = {'beginning': None, 'ending': None}  # Store cut points in viz data
        self.cut_lines = {}  # Store cut line references for each graph
        self.shaded_regions = {}  # Store shaded region references
        self.init_ui()

    def init_ui(self):
        """Initialize manual removal window"""
        self.setWindowTitle("🌊 Manual Dive Removal")

        # Open maximized (not fullscreen, but maximized window)
        self.showMaximized()

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Header
        header = QLabel("✂️ Manual Dive Section Removal")
        header.setFont(QFont("Arial", 18, QFont.Bold))
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("color: #2c3e50; padding: 15px;")
        layout.addWidget(header)

        # Instructions
        instructions = QLabel(
            "Click on the graph to mark cut point. "
            "Deployment: removes everything BEFORE click. "
            "Retrieval: removes everything AFTER click."
        )
        instructions.setAlignment(Qt.AlignCenter)
        instructions.setStyleSheet("color: #7f8c8d; font-size: 12px; padding: 5px;")
        layout.addWidget(instructions)

        # Detect dives on visualization data
        self.detect_dive_legs()

        # Beginning dive plot (DEPLOYMENT)
        if self.beginning_data is not None:
            beginning_group = QGroupBox("🔻 Sensor Deployment")
            beginning_layout = QVBoxLayout()
            self.beginning_canvas = self.create_interactive_plot(
                self.beginning_data,
                "Deployment - Click to mark cut point",
                'beginning'
            )
            beginning_layout.addWidget(self.beginning_canvas)
            beginning_group.setLayout(beginning_layout)
            layout.addWidget(beginning_group, stretch=1)

        # Ending dive plot (RETRIEVAL)
        if self.ending_data is not None:
            ending_group = QGroupBox("🔺 Sensor Retrieval")
            ending_layout = QVBoxLayout()
            self.ending_canvas = self.create_interactive_plot(
                self.ending_data,
                "Retrieval - Click to mark cut point",
                'ending'
            )
            ending_layout.addWidget(self.ending_canvas)
            ending_group.setLayout(ending_layout)
            layout.addWidget(ending_group, stretch=1)

        # Buttons
        btn_layout = QHBoxLayout()

        btn_save = QPushButton("💾 Save Trimmed Data")
        btn_save.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                font-size: 14px;
                font-weight: bold;
                padding: 15px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #229954;
            }
        """)
        btn_save.clicked.connect(self.save_trimmed_data)
        btn_layout.addWidget(btn_save)

        layout.addLayout(btn_layout)

        self.apply_styles()

    def detect_dive_legs(self):
        """Detect dive legs on visualization (subsampled) data"""
        pressure_viz = self.viz_data_df['pressure'].values

        # Detect dives (same algorithm as VisualizationWindow)
        dive_mask = self.detect_dives(pressure_viz)

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
                if dive_indices[0] < len(pressure_viz) // 2:
                    # Beginning
                    self.beginning_viz_range = (0, dive_indices[-1])
                else:
                    # Ending
                    self.ending_viz_range = (dive_indices[0], len(pressure_viz) - 1)
            else:
                # Two segments
                # Beginning segment
                begin_end = dive_indices[breaks[0]]
                self.beginning_viz_range = (0, begin_end)

                # Ending segment
                end_start = dive_indices[breaks[0] + 1]
                self.ending_viz_range = (end_start, len(pressure_viz) - 1)

            # Add +10% safety margin
            if self.beginning_viz_range:
                start, end = self.beginning_viz_range
                margin = int((end - start) * 0.1)
                end = min(end + margin, len(pressure_viz) - 1)
                self.beginning_viz_range = (start, end)
                self.beginning_data = self.viz_data_df.iloc[start:end+1].copy()

            if self.ending_viz_range:
                start, end = self.ending_viz_range
                margin = int((end - start) * 0.1)
                start = max(start - margin, 0)
                self.ending_viz_range = (start, end)
                self.ending_data = self.viz_data_df.iloc[start:end+1].copy()

    def detect_dives(self, pressure, sensitivity=3.0):
        """Same dive detection as VisualizationWindow"""
        n = len(pressure)
        dive_mask = np.zeros(n, dtype=bool)

        if n < 100:
            return dive_mask

        gradient = np.gradient(pressure)
        gradient_abs = np.abs(gradient)
        grad_std = np.std(gradient)

        # Beginning leg
        search_end = min(n // 3, 2000)
        beginning_grad = gradient[:search_end]
        max_jump_idx = beginning_grad.argmax()
        max_jump_value = beginning_grad[max_jump_idx]

        if max_jump_value > sensitivity * grad_std:
            jump_end = max_jump_idx
            for i in range(max_jump_idx + 1, min(max_jump_idx + 100, search_end)):
                if gradient_abs[i] < grad_std:
                    jump_end = i
                    break
            leg_length = jump_end
            safety_margin = int(leg_length * 0.1)
            jump_end = min(jump_end + safety_margin, n - 1)
            dive_mask[0:jump_end] = True

        # Ending leg
        search_start = max(2 * n // 3, n - 2000)
        wave_mean = np.mean(pressure[n//3:2*n//3])
        low_threshold = wave_mean * 0.3

        ending_section = pressure[search_start:]
        drop_start = None
        for i in range(len(ending_section) - 50):
            if np.all(ending_section[i:i+50] < low_threshold):
                drop_start = search_start + i
                break

        if drop_start is not None:
            leg_length = n - drop_start
            safety_margin = int(leg_length * 0.1)
            drop_start = max(drop_start - safety_margin, 0)
            dive_mask[drop_start:] = True

        return dive_mask

    def create_interactive_plot(self, data, title, leg_type):
        """Create interactive matplotlib plot with click handler - full data with zoom"""
        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT

        # Taller figure for vertical layout
        fig = Figure(figsize=(14, 5), dpi=100)
        canvas = FigureCanvas(fig)

        ax = fig.add_subplot(111)

        # Plot FULL viz data
        full_timestamps = self.viz_data_df['timestamp']
        full_pressure = self.viz_data_df['pressure'].values

        # Plot complete data
        ax.plot(full_timestamps, full_pressure, linewidth=0.5, color='#3498db', alpha=0.7)

        # Highlight the detected dive section in red
        if leg_type == 'beginning' and self.beginning_viz_range:
            start, end = self.beginning_viz_range
            dive_timestamps = self.viz_data_df['timestamp'].iloc[start:end+1]
            dive_pressure = full_pressure[start:end+1]
            ax.plot(dive_timestamps, dive_pressure, linewidth=0.8, color='#e74c3c', alpha=0.9, label='Detected dive')
        elif leg_type == 'ending' and self.ending_viz_range:
            start, end = self.ending_viz_range
            dive_timestamps = self.viz_data_df['timestamp'].iloc[start:end+1]
            dive_pressure = full_pressure[start:end+1]
            ax.plot(dive_timestamps, dive_pressure, linewidth=0.8, color='#e74c3c', alpha=0.9, label='Detected dive')

        # No axis labels, only tick values
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
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

        # Click handler
        def on_click(event):
            if event.inaxes == ax and event.button == 1:  # Left click
                # Remove previous cut line and shading
                if self.cut_lines[leg_type]:
                    self.cut_lines[leg_type].remove()
                    self.cut_lines[leg_type] = None

                if self.shaded_regions[leg_type]:
                    self.shaded_regions[leg_type].remove()
                    self.shaded_regions[leg_type] = None

                # Draw vertical line at click (event.xdata is already a number)
                self.cut_lines[leg_type] = ax.axvline(event.xdata, color='green',
                                                       linewidth=2, linestyle='--',
                                                       label='Cut point', zorder=10)

                # Convert clicked x-position to datetime (make tz-naive)
                clicked_time = pd.Timestamp(mdates.num2date(event.xdata)).tz_localize(None)

                # Find closest index in viz data
                time_diff = np.abs((full_timestamps - clicked_time).dt.total_seconds())
                viz_idx = time_diff.argmin()

                # Add red shading for region to be removed
                # Convert timestamps to matplotlib numbers for axvspan
                if leg_type == 'beginning':
                    # Shade everything BEFORE the cut (left side)
                    x_start = mdates.date2num(full_timestamps.iloc[0])
                    x_end = event.xdata  # Already a number
                    self.shaded_regions[leg_type] = ax.axvspan(
                        x_start,
                        x_end,
                        alpha=0.3,
                        color='red',
                        zorder=1,
                        label='To be removed'
                    )
                else:  # ending
                    # Shade everything AFTER the cut (right side)
                    x_start = event.xdata  # Already a number
                    x_end = mdates.date2num(full_timestamps.iloc[-1])
                    self.shaded_regions[leg_type] = ax.axvspan(
                        x_start,
                        x_end,
                        alpha=0.3,
                        color='red',
                        zorder=1,
                        label='To be removed'
                    )

                canvas.draw()

                # Store cut index
                self.cut_indices[leg_type] = viz_idx
                print(f"{leg_type} cut at viz data index: {viz_idx}")

        canvas.mpl_connect('button_press_event', on_click)

        fig.tight_layout()

        # Add navigation toolbar for zoom/pan
        toolbar = NavigationToolbar2QT(canvas, self)

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
        status.setStyleSheet("color: #7f8c8d;")
        layout.addWidget(status)

        progress_dialog.show()
        QApplication.processEvents()

        try:
            # Load full CSV
            progress_bar.setValue(10)
            status.setText("Reading full CSV file...")
            QApplication.processEvents()

            script_dir = Path(__file__).parent if '__file__' in globals() else Path.cwd()
            csv_file = script_dir / "Output" / "Step1_TXTtoCSV.csv"

            # Count lines for progress
            with open(csv_file, 'rb') as f:
                total_lines = sum(1 for _ in f if not _.startswith(b'#')) - 1

            progress_bar.setValue(20)
            status.setText(f"Loading {total_lines:,} rows...")
            QApplication.processEvents()

            # Read full data
            full_data = pd.read_csv(csv_file, comment='#')
            full_data['timestamp'] = pd.to_datetime(full_data['timestamp'], errors='coerce')

            progress_bar.setValue(40)
            status.setText("Converting indices from visualization to full data...")
            QApplication.processEvents()

            # Calculate subsampling step (how we created viz data)
            subsample_step = len(full_data) // len(self.viz_data_df)

            # Convert viz indices to full data indices
            beginning_full_idx = None
            ending_full_idx = None

            if self.cut_indices['beginning'] is not None:
                # User clicked - use that point
                beginning_full_idx = self.cut_indices['beginning'] * subsample_step
            elif self.beginning_viz_range:
                # No click - use end of detected leg
                beginning_full_idx = self.beginning_viz_range[1] * subsample_step

            if self.cut_indices['ending'] is not None:
                # User clicked - use that point
                ending_full_idx = self.cut_indices['ending'] * subsample_step
            elif self.ending_viz_range:
                # No click - use start of detected leg
                ending_full_idx = self.ending_viz_range[0] * subsample_step

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

            progress_bar.setValue(80)
            status.setText("Saving to Step2_Initial_Cut.csv...")
            QApplication.processEvents()

            # Save to Step2
            output_folder = script_dir / "Output"
            step2_file = output_folder / "Step2_Initial_Cut.csv"

            # Write with metadata
            with open(step2_file, 'w', encoding='utf-8') as f:
                f.write("# STEP 2: Initial Cut - Manual dive removal\n")
                f.write("# ==========================================\n")
                f.write(f"# Original points: {len(full_data):,}\n")
                f.write(f"# Trimmed points: {len(trimmed_data):,}\n")
                f.write(f"# Points removed: {len(full_data) - len(trimmed_data):,}\n")
                f.write("# ==========================================\n")

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

            # Open Step 3 window
            self.step3_window = Step3FourierWindow()
            self.step3_window.show()
            self.close()

        except Exception as e:
            progress_dialog.close()
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to save trimmed data:\n{str(e)}"
            )

    def process_zero_mean(self, step2_file, output_folder, progress_bar, status):
        """
        Process Zero Mean:
        1. Calculate Avg_Depth_FullRec (mean of all pressure values)
        2. Create Step2_Zero_Mean.csv (all values - Avg_Depth_FullRec)
        3. Create Parameters.csv with reading means and metadata
        """
        # Read Step2 data
        status.setText("Reading Step2_Initial_Cut.csv...")
        QApplication.processEvents()

        data = pd.read_csv(step2_file, comment='#')

        # Step 1: Calculate global average
        progress_bar.setValue(85)
        status.setText("Calculating global average (Avg_Depth_FullRec)...")
        QApplication.processEvents()

        avg_depth_full_rec = data['pressure'].mean()

        # Step 2: Calculate average for each reading
        progress_bar.setValue(88)
        status.setText("Calculating averages for each 20-min reading...")
        QApplication.processEvents()

        reading_averages = data.groupby('reading_number')['pressure'].mean().reset_index()
        reading_averages.columns = ['reading_number', 'average_depth']

        # Step 4: Create Zero Mean data (subtract global average from all points)
        progress_bar.setValue(92)
        status.setText("Creating Zero Mean data...")
        QApplication.processEvents()

        zero_mean_data = data.copy()
        zero_mean_data['pressure'] = zero_mean_data['pressure'] - avg_depth_full_rec

        # Step 4: Save Zero Mean file
        progress_bar.setValue(95)
        status.setText("Saving Step2_Zero_Mean.csv...")
        QApplication.processEvents()

        zero_mean_file = output_folder / "Step2_Zero_Mean.csv"
        with open(zero_mean_file, 'w', encoding='utf-8') as f:
            f.write("# STEP 2: Zero Mean - Global average subtracted\n")
            f.write("# ==========================================\n")
            f.write(f"# Sensor frequency: {read_sensor_freq_from_csv(step2_file)} Hz\n")
            f.write(f"# Average Depth (Full Record): {avg_depth_full_rec:.6f}\n")
            f.write(f"# All pressure values have this subtracted\n")
            f.write("# ==========================================\n")

        zero_mean_data.to_csv(zero_mean_file, mode='a', index=False)

        # Create Step2_Visualization.csv (subsampled for caching)
        progress_bar.setValue(96)
        status.setText("Creating Step2_Visualization.csv...")
        QApplication.processEvents()

        target_points = VISUALIZATION_TARGET_POINTS  # 5000 points optimal for FullHD
        subsample_step = max(1, len(zero_mean_data) // target_points)
        viz_data = zero_mean_data.iloc[::subsample_step].copy()

        step2_viz_file = output_folder / "Step2_Visualization.csv"
        with open(step2_viz_file, 'w', encoding='utf-8') as f:
            f.write("# STEP 2: Visualization Cache - Subsampled Zero Mean data\n")
            f.write("# ==========================================\n")
            f.write(f"# Sampled points: {len(viz_data)}\n")
            f.write(f"# Original points: {len(zero_mean_data)}\n")
            f.write("# ==========================================\n")

        viz_data.to_csv(step2_viz_file, mode='a', index=False)

        # Step 5: Save Parameters file
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

    def apply_styles(self):
        """Apply global styles"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f6fa;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #bdc3c7;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                color: #2c3e50;
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)


class Step3FourierWindow(QMainWindow):
    """Window for Step 3: Fourier Transform - Remove low frequencies"""

    def __init__(self):
        super().__init__()
        self.spectrum_full    = None   # complex FFT — for apply_transform
        self.frequencies_full = None
        self.cutoff_freq      = None
        self.init_ui()
        self.load_and_transform()

    def init_ui(self):
        """Initialize Step 3 Fourier window"""
        self.setWindowTitle("🌊 Step 3: Fourier Transform")
        # Don't show maximized here - will show after data loads

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # ── Top bar: spectrogram parameters + button ─────────────────────
        # Outer container with card-style background
        top_card = QWidget()
        top_card.setStyleSheet("""
            QWidget {
                background-color: #f0f3f7;
                border: 1px solid #d5dce8;
                border-radius: 8px;
            }
        """)
        top_card.setFixedHeight(56)
        top_bar = QHBoxLayout(top_card)
        top_bar.setContentsMargins(16, 0, 16, 0)
        top_bar.setSpacing(8)

        lbl_style = (
            "font-size: 13px; color: #34495e; background: transparent; border: none;"
        )
        spin_style = """
            QSpinBox {
                font-size: 13px;
                font-weight: bold;
                color: #2c3e50;
                background: white;
                padding: 3px 6px;
                border: 1.5px solid #b0bec5;
                border-radius: 5px;
                min-width: 72px;
            }
            QSpinBox:focus { border-color: #ff8c00; }
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
                background-color: #ff8c00;
                color: white;
                font-size: 13px;
                font-weight: bold;
                padding: 0 18px;
                border-radius: 6px;
                border: none;
            }
            QPushButton:hover  { background-color: #e67e00; }
            QPushButton:pressed{ background-color: #cc6f00; }
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

        self.btn_continue = QPushButton("▶️ Apply and Continue")
        self.btn_continue.setEnabled(False)  # Disabled until cutoff selected
        self.btn_continue.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                font-size: 14px;
                font-weight: bold;
                padding: 15px;
                border-radius: 5px;
            }
            QPushButton:hover:enabled {
                background-color: #229954;
            }
            QPushButton:disabled {
                background-color: #95a5a6;
            }
        """)
        self.btn_continue.clicked.connect(self.apply_transform)
        btn_layout.addWidget(self.btn_continue)

        layout.addLayout(btn_layout)

    def load_and_transform(self):
        """Load Step2 data and perform FFT"""
        script_dir = Path(__file__).parent if '__file__' in globals() else Path.cwd()
        output_folder = script_dir / "Output"

        step2_file = output_folder / "Step2_Zero_Mean.csv"
        step3_spectrum = output_folder / "Step3_Spectrum.csv"
        step3_spectrum_viz = output_folder / "Step3_Spectrum_Visualization.csv"

        # Check if spectrum already exists
        if step3_spectrum_viz.exists():
            # Show progress for loading cache
            progress_dialog = QDialog(self)
            progress_dialog.setWindowTitle("Loading Cached Spectrum")
            progress_dialog.setModal(True)
            progress_dialog.setFixedSize(500, 150)

            layout = QVBoxLayout(progress_dialog)

            label = QLabel("Loading cached spectrum...")
            label.setAlignment(Qt.AlignCenter)
            layout.addWidget(label)

            progress_bar = QProgressBar()
            progress_bar.setRange(0, 100)
            layout.addWidget(progress_bar)

            status = QLabel("Loading visualization...")
            status.setAlignment(Qt.AlignCenter)
            status.setStyleSheet("color: #7f8c8d;")
            layout.addWidget(status)

            progress_dialog.show()
            QApplication.processEvents()

            # Load cached spectrum
            progress_bar.setValue(30)
            QApplication.processEvents()

            spectrum_df = pd.read_csv(step3_spectrum_viz, comment='#')
            self.frequencies_viz   = spectrum_df['frequency'].values
            self.spectrum_viz_real = spectrum_df['real'].values
            self.spectrum_viz_imag = spectrum_df['imag'].values

            progress_bar.setValue(60)
            status.setText("Loading full spectrum...")
            QApplication.processEvents()

            # Load full spectrum
            spectrum_full_df = pd.read_csv(step3_spectrum, comment='#')
            self.frequencies_full = spectrum_full_df['frequency'].values
            self.spectrum_full    = spectrum_full_df['real'].values + 1j * spectrum_full_df['imag'].values

            progress_bar.setValue(90)
            status.setText("Creating plot...")
            QApplication.processEvents()

            progress_dialog.close()

            self.create_spectrum_plot()

            # Show window maximized AFTER data is loaded
            self.showMaximized()
            return

        # Show progress dialog
        progress_dialog = QDialog(self)
        progress_dialog.setWindowTitle("Computing Fourier Transform")
        progress_dialog.setModal(True)
        progress_dialog.setFixedSize(500, 150)

        layout = QVBoxLayout(progress_dialog)

        label = QLabel("Processing Fourier Transform...")
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)

        progress_bar = QProgressBar()
        progress_bar.setRange(0, 100)
        layout.addWidget(progress_bar)

        status = QLabel("Loading data...")
        status.setAlignment(Qt.AlignCenter)
        status.setStyleSheet("color: #7f8c8d;")
        layout.addWidget(status)

        progress_dialog.show()
        QApplication.processEvents()

        try:
            # Load full Step2 data
            progress_bar.setValue(10)
            status.setText("Loading Step2_Zero_Mean.csv...")
            QApplication.processEvents()

            data = pd.read_csv(step2_file, comment='#')
            y = data['pressure'].values

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

            self.spectrum_full    = s   # complex FFT — used by apply_transform
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

            self.frequencies_viz   = x_viz
            self.spectrum_viz_real = s_viz.real
            self.spectrum_viz_imag = s_viz.imag

            progress_bar.setValue(95)
            status.setText("Preparing plot...")
            QApplication.processEvents()

            progress_dialog.close()

            # Create plot with its own progress
            self.create_spectrum_plot()

            # Show window maximized AFTER data is loaded
            self.showMaximized()

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

        # ── S(ω) = |FFT|² / (N · max(ω))  [m²/s]  ──────────────────────────
        N = len(self.spectrum_full)
        omega_max = np.max(np.abs(self.frequencies_full))

        # Top graph: subsampled, ω > 0.05 (exclude DC and large low-freq harmonics)
        pos_viz = self.frequencies_viz > 0.05
        freq_viz = self.frequencies_viz[pos_viz]
        mag_viz  = np.sqrt(self.spectrum_viz_real[pos_viz]**2 + self.spectrum_viz_imag[pos_viz]**2)
        s_viz    = (mag_viz**2) / (N * omega_max)

        # Bottom graph: full resolution, positive, 0 < ω ≤ 0.1
        pos_full = self.frequencies_full >= 0
        freq_full = self.frequencies_full[pos_full]
        mag_full  = np.sqrt(self.spectrum_full.real[pos_full]**2 + self.spectrum_full.imag[pos_full]**2)
        s_full    = (mag_full**2) / (N * omega_max)
        zoom_idx  = (freq_full > 0) & (freq_full <= 0.1)
        freq_zoom = freq_full[zoom_idx]
        s_zoom    = s_full[zoom_idx]

        # ==============================================================================
        # TOP GRAPH: overview, ω ∈ [0, 3]
        # ==============================================================================
        fig_top = Figure(figsize=(16, 4), dpi=100)
        canvas_top = FigureCanvas(fig_top)
        ax_top = fig_top.add_subplot(111)

        ax_top.plot(freq_viz, s_viz, linewidth=0.8, color='green', alpha=0.9)
        ax_top.set_xlabel('ω, [rad/s]', fontsize=11)
        ax_top.set_ylabel('S(ω), [m²/s]', fontsize=11)
        ax_top.grid(True, alpha=0.3)
        ax_top.set_xlim(0, 3.0)
        ax_top.set_ylim(0, None)

        fig_top.tight_layout()
        toolbar_top = NavigationToolbar2QT(canvas_top, self)

        # ==============================================================================
        # BOTTOM GRAPH: 0 < ω ≤ 0.1, log Y, interactive cutoff
        # ==============================================================================
        fig_bottom = Figure(figsize=(16, 5), dpi=100)
        canvas_bottom = FigureCanvas(fig_bottom)
        ax_bottom = fig_bottom.add_subplot(111)

        ax_bottom.plot(freq_zoom, s_zoom, linewidth=0.8, color='green', alpha=0.9)
        ax_bottom.set_xlabel('ω, [rad/s]', fontsize=11)
        ax_bottom.set_ylabel('S(ω), [m²/s]', fontsize=11)
        ax_bottom.set_xlim(0, 0.1)
        ax_bottom.grid(True, alpha=0.3, which='both')
        # Log scale — set AFTER plot so matplotlib auto-sets ylim from data (no warning)
        ax_bottom.set_yscale('log')

        fig_bottom.tight_layout()
        toolbar_bottom = NavigationToolbar2QT(canvas_bottom, self)

        # ==============================================================================
        # CLICK HANDLER (only bottom graph)
        # Click sets cutoff - everything BELOW (left of) cutoff is removed
        # ==============================================================================

        self.cutoff_line = None
        self.shaded_region = None
        self.cutoff_text = None  # Text annotation on top graph

        def on_click(event):
            if event.inaxes == ax_bottom and event.button == 1:  # Left click
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

                print(f"Cutoff frequency set to: {self.cutoff_freq:.4f} rad/s (Period: {period_minutes} min {period_secs} sec)")

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

        script_dir = Path(__file__).parent if '__file__' in globals() else Path.cwd()
        output_folder = script_dir / "Output"

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
        status.setStyleSheet("color: #7f8c8d;")
        layout.addWidget(status)

        progress_dialog.show()
        QApplication.processEvents()

        try:
            # Apply cutoff filter on the complex FFT spectrum
            progress_bar.setValue(20)
            status.setText("Applying frequency filter...")
            QApplication.processEvents()

            s_filtered = self.spectrum_full.copy()

            # Remove frequencies below cutoff
            for i in range(len(self.frequencies_full)):
                if abs(self.frequencies_full[i]) < self.cutoff_freq:
                    s_filtered[i] = 0 + 0j

            progress_bar.setValue(40)
            status.setText("Computing inverse FFT...")
            QApplication.processEvents()

            # Inverse FFT
            from scipy.fftpack import ifft

            y_transformed = ifft(s_filtered).real

            progress_bar.setValue(60)
            status.setText("Saving transformed data...")
            QApplication.processEvents()

            # Load original data to get timestamps
            step2_file = output_folder / "Step2_Zero_Mean.csv"
            data_orig = pd.read_csv(step2_file, comment='#')

            # Create transformed dataframe
            data_transformed = data_orig.copy()
            data_transformed['pressure'] = y_transformed

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
                f.write(f"# Cutoff frequency: {self.cutoff_freq:.6f} rad/s\n")
                f.write(f"# Original readings: {len(reading_numbers)}\n")
                f.write(f"# Readings after edge removal: {len(readings_to_keep) if len(reading_numbers) > 4 else len(reading_numbers)}\n")
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
        script_dir = Path(__file__).parent if '__file__' in globals() else Path.cwd()
        output_folder = script_dir / "Output"
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
        status.setStyleSheet("color: #7f8c8d;")
        layout.addWidget(status)

        progress_dialog.show()
        QApplication.processEvents()

        try:
            # Load Step2_Zero_Mean data
            progress_bar.setValue(10)
            status.setText("Loading Step2_Zero_Mean.csv...")
            QApplication.processEvents()

            data = pd.read_csv(step2_file, comment='#')
            y = data['pressure'].values

            # Parameters from UI spinboxes
            WindowSize = self.spin_window.value()   # минут
            DeltaWindow = self.spin_delta.value()   # секунд
            part = self.spin_part.value()            # процент от спектра
            Sensor_Frequency = read_sensor_freq_from_csv(step2_file)

            progress_bar.setValue(20)
            status.setText("Preparing window parameters...")
            QApplication.processEvents()

            # Compute window parameters
            window = WindowSize * 60 * Sensor_Frequency  # размер окна в точках
            n = int((len(y) - window) / (DeltaWindow * Sensor_Frequency))  # число окон

            from scipy.fftpack import rfft, rfftfreq
            from scipy.signal.windows import hann

            # ВАЖНО: rfftfreq с угловой частотой (рад/с)
            w = rfftfreq(window, (1 / Sensor_Frequency) / (2 * np.pi))
            # Number of spectrum points to keep (slice [0:spec_len] is always valid)
            # spec_idx is used only for w[spec_idx] — must be < len(w)
            spec_len = int(len(w) * 0.01 * part)          # может равняться len(w) при 100%
            spec_idx = min(spec_len, len(w) - 1)           # безопасный индекс для w[]
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
                    status.setText(f"Processing window {i+1}/{n}...")
                    QApplication.processEvents()

                # Extract window
                arr = y[i*DeltaWindow : window + i*DeltaWindow]

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
                    0, w[spec_idx]],          # spec_idx = min(spec_len, len(w)-1)
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
        header = QLabel("Fourier Transform Applied - Before vs After")
        header.setFont(QFont("Arial", 16, QFont.Bold))
        header.setAlignment(Qt.AlignCenter)
        layout.addWidget(header)

        progress_bar.setValue(70)
        status.setText("Rendering plot...")
        QApplication.processEvents()

        # Create plot
        fig = Figure(figsize=(14, 6), dpi=100)
        canvas = FigureCanvas(fig)

        ax = fig.add_subplot(111)

        # Plot before (orange, more transparent per your request)
        ax.plot(data_before['timestamp'], data_before['pressure'],
               linewidth=0.5, color='#FFA500', alpha=0.6, label='Before', zorder=1)

        # Plot after (blue, less transparent per your request)
        ax.plot(data_transformed_viz['timestamp'], data_transformed_viz['pressure'],
               linewidth=0.5, color='#3498db', alpha=0.7, label='After', zorder=2)

        # Horizontal line at y=0 (on top)
        ax.axhline(y=0, color='black', linewidth=2, linestyle='-', zorder=10)

        ax.set_title('Before/After Fourier Transform', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
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
        btn_continue = QPushButton("▶️ Continue to Step 4")
        btn_continue.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                font-size: 14px;
                font-weight: bold;
                padding: 15px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #229954;
            }
        """)

        def go_to_step4():
            comparison_window.close()
            self.close()
            self.step4_window = Step4ProcessingWindow()
            self.step4_window.show()

        btn_continue.clicked.connect(go_to_step4)
        layout.addWidget(btn_continue)

        comparison_window.exec_()
class Step4ProcessingWindow(QMainWindow):
    """Window for Step 4: Spike removal and RMS filtering"""

    def __init__(self):
        super().__init__()
        self.current_reading = 0  # Track which reading is being processed
        self.init_ui()
        # Don't show yet - wait until data is loaded
        self.load_and_visualize()

    def init_ui(self):
        """Initialize Step 3 window"""
        self.setWindowTitle("🌊 Step 4: Spike Removal & RMS Filtering")
        # Don't show maximized here - will show after data loads

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Header
        header = QLabel("Step 4: Data Quality Processing")
        header.setFont(QFont("Arial", 18, QFont.Bold))
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("color: #2c3e50; padding: 15px;")
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
        self.lbl_freq_from.setStyleSheet("font-weight: bold; color: #2c3e50;")
        spline_layout.addWidget(self.lbl_freq_from)

        spline_layout.addWidget(QLabel("→"))

        self.spline_freq_input = QSpinBox()
        self.spline_freq_input.setRange(1, 1000)
        self.spline_freq_input.setValue(8)
        self.spline_freq_input.setSuffix(" Hz")
        self.spline_freq_input.setFixedWidth(90)
        self.spline_freq_input.setStyleSheet("""
            QSpinBox {
                font-size: 13px; padding: 2px 4px;
                border: 1.5px solid #b0bec5; border-radius: 4px;
            }
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
            _legend_item("#27ae60", "Processed recording", 'fill'),
            _legend_item("#e74c3c", "Removed recording",   'fill'),
            _legend_item("#e74c3c", "Spike",               'circle'),
        ]:
            legend_layout.addLayout(row)
        legend_layout.addStretch()
        controls_layout.addLayout(legend_layout)

        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)

        # Buttons
        btn_layout = QHBoxLayout()

        self.btn_start = QPushButton("▶️ Start Processing")
        self.btn_start.setEnabled(True)  # Always enabled
        self.btn_start.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                font-size: 14px;
                font-weight: bold;
                padding: 15px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #229954;
            }
        """)
        self.btn_start.clicked.connect(self.start_processing)
        btn_layout.addWidget(self.btn_start)

        layout.addLayout(btn_layout)

    def load_and_visualize(self):
        """Load Step3_Visualization and create plot"""
        script_dir = Path(__file__).parent if '__file__' in globals() else Path.cwd()
        output_folder = script_dir / "Output"

        step3_viz = output_folder / "Step3_Visualization.csv"

        # Show progress dialog
        progress_dialog = QDialog(self)
        progress_dialog.setWindowTitle("Loading Data")
        progress_dialog.setModal(True)
        progress_dialog.setFixedSize(500, 150)

        layout = QVBoxLayout(progress_dialog)

        label = QLabel("Loading Step3_Visualization.csv...")
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)

        progress_bar = QProgressBar()
        progress_bar.setRange(0, 100)
        layout.addWidget(progress_bar)

        status = QLabel("Loading visualization...")
        status.setAlignment(Qt.AlignCenter)
        status.setStyleSheet("color: #7f8c8d;")
        layout.addWidget(status)

        progress_dialog.show()
        QApplication.processEvents()

        try:
            # Load Step3_Visualization directly (already subsampled)
            progress_bar.setValue(30)
            status.setText("Reading Step3_Visualization.csv...")
            QApplication.processEvents()

            # Read sensor frequency from this file's header
            viz_sensor_freq = read_sensor_freq_from_csv(step3_viz)
            self.lbl_freq_from.setText(f"{viz_sensor_freq} Hz")
            self.spline_freq_input.setMinimum(viz_sensor_freq)
            self.spline_freq_input.setValue(max(viz_sensor_freq, self.spline_freq_input.value()))

            df = pd.read_csv(step3_viz, comment='#')
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

            progress_bar.setValue(70)
            status.setText("Creating plot...")
            QApplication.processEvents()

            progress_dialog.close()

            # Create plot
            self.create_interactive_plot(df)

            # Show window maximized AFTER data is loaded
            self.showMaximized()

        except Exception as e:
            progress_dialog.close()
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to load data:\n{str(e)}"
            )

    def create_interactive_plot(self, data):
        """Create interactive matplotlib plot with ALL reading boundaries"""
        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT

        fig = Figure(figsize=(14, 6), dpi=100)
        canvas = FigureCanvas(fig)

        ax = fig.add_subplot(111)

        # Plot data
        timestamps = data['timestamp']
        pressure = data['pressure'].values

        ax.plot(timestamps, pressure, linewidth=0.5, color='#3498db', alpha=0.7)

        # Add horizontal line at y=0 (thick black)
        ax.axhline(y=0, color='black', linewidth=2, linestyle='-', zorder=5)

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
        ax.grid(True, alpha=0.3)

        # Set x-axis limits to data boundaries (tight zoom)
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

        # Clear previous graph and add new one
        for i in reversed(range(self.graph_layout.count())):
            self.graph_layout.itemAt(i).widget().setParent(None)

        self.graph_layout.addWidget(toolbar)
        self.graph_layout.addWidget(canvas)

    def start_processing(self):
        """
        Start spike removal and/or RMS filtering with real-time visualization
        Calculate ALL wave parameters in single pass
        """
        script_dir = Path(__file__).parent if '__file__' in globals() else Path.cwd()
        output_folder = script_dir / "Output"

        step2_file = output_folder / "Step3_Transformed.csv"
        step2_viz = output_folder / "Step3_Visualization.csv"
        parameters_file = output_folder / "Parameters.csv"

        # Get options
        remove_spikes    = self.cb_remove_spikes.isChecked()
        remove_low_rms   = self.cb_remove_low_rms.isChecked()
        use_spline       = self.cb_spline.isChecked()
        spline_target_hz = self.spline_freq_input.value()  # target Hz from spinbox

        # Parse RMS threshold
        rms_threshold = 0.015
        if remove_low_rms:
            try:
                rms_text = self.rms_input.text().strip()
                rms_threshold = float(rms_text)
            except:
                rms_threshold = 0.015

        # Helper functions for wave analysis
        def kh_solver(h, Tz):
            """Solve kh equation: x * tanh(x) = (4π²h)/(g*Tz²)"""
            from scipy.optimize import fsolve
            g = 9.81
            target = (4 * np.pi**2 * h) / (g * Tz**2)

            def equation(x):
                return x * np.tanh(x) - target

            # Initial guess
            x0 = 1.0
            result = fsolve(equation, x0)[0]
            return max(0.01, result)  # Avoid division by zero

        def calculate_spectrum_params(arr, sensor_freq):
            """Calculate spectral parameters: Q, nu, eps_width, rho"""
            from scipy.fftpack import rfft, rfftfreq
            from scipy.signal.windows import hann

            # FFT with Hann window
            mask = hann(len(arr))
            s = np.abs(rfft(arr * mask))
            w = rfftfreq(len(s), (1 / sensor_freq) / (2 * np.pi))

            # Filter frequencies
            if sensor_freq == 8:
                n = 5.75
            else:
                n = np.pi
            ind = w < n
            s = s[ind]
            w = w[ind]

            if len(w) < 2:
                return 0, 0, 0, 0

            dx = w[1] - w[0]

            # Spectral moments
            m0 = np.trapezoid(s, dx=dx)
            m1 = np.trapezoid(w * s, dx=dx)
            m2 = np.trapezoid((w**2) * s, dx=dx)
            m4 = np.trapezoid((w**4) * s, dx=dx)

            if m0 == 0:
                return 0, 0, 0, 0

            # Q (Goda parameter)
            Q = np.trapezoid(w * s**2, dx=dx) / (m0**2)

            # nu (narrowness)
            if m1 != 0:
                nu = np.sqrt(((m0 * m2) / (m1**2)) - 1)
            else:
                nu = 0

            # eps_width
            if m0 * m4 != 0:
                eps_width = np.sqrt(1 - (m2**2)/(m0 * m4))
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
            except:
                rho = 0

            return Q, nu, eps_width, rho

        def calculate_gamma(kh):
            """Calculate gamma function for BFI"""
            try:
                v = 1 + (2 * kh) / (np.sinh(2 * kh))
                a = -v**2 + 2 + 8 * (kh**2) * ((np.cosh(2 * kh)) / ((np.sinh(2 * kh))**2))
                b = ((np.cosh(4 * kh) + 8 - 2 * (np.tanh(kh))**2) / (8 * (np.sinh(kh))**4) -
                     ((2 * (np.cosh(kh))**2 + 0.5 * v)**2) /
                     ((np.sinh(2 * kh))**2 * ((kh / (np.tanh(kh))) - (v / 2)**2)))

                if a < 0:
                    print(f"Warning: a < 0 for kh={kh}")

                gamma = v * np.sqrt(np.abs(b) / abs(a))
                return gamma
            except:
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
            except:
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
        status.setStyleSheet("color: #7f8c8d;")
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
                    'pressure': reading_data['pressure'].values,
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
                status.setText(f"Processing reading {reading_num} ({idx+1}/{total_readings})...")

                if idx % 5 == 0:
                    QApplication.processEvents()

                arr_data = reading_arrays[reading_num]
                arr = arr_data['pressure'].copy()  # ВАЖНО: .copy() для возможности модификации!
                arr_indices = arr_data['indices']
                arr_timestamps = arr_data['timestamps']
                reading_start = arr_data['start']
                reading_end = arr_data['end']

                depth = reading_info[reading_num]['depth']

                # ========== STEP 1: CHECK RMS ==========
                should_remove = False
                rms_value = np.sqrt(np.mean(arr**2))

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

                            data.loc[spike_idx, 'pressure'] = new_value
                            arr[j + 1] = new_value

                # ========== STEP 3: CALCULATE ALL WAVE PARAMETERS ==========

                # --- Spline interpolation for amplitude metrics (As, Hs) ---
                # If enabled: resample arr to spline_target_hz via cubic spline.
                # All spectral/integral parameters (Q, nu, eps, Tz…) still use
                # the original arr at sensor_freq.
                if use_spline and spline_target_hz > sensor_freq:
                    from scipy.interpolate import CubicSpline
                    n_orig   = len(arr)
                    t_orig   = np.arange(n_orig) / sensor_freq          # seconds
                    cs       = CubicSpline(t_orig, arr)
                    # New time grid at target frequency
                    t_new    = np.arange(0, t_orig[-1], 1.0 / spline_target_hz)
                    arr_amp  = cs(t_new)                                # interpolated
                else:
                    arr_amp = arr                                        # same as original

                variance = np.var(arr)          # from ORIGINAL — for spectral params
                sigma    = np.sqrt(np.var(arr_amp))   # from interpolated — for amplitudes
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

            QMessageBox.information(
                self,
                "Success!",
                f"✅ Step 3 processing complete!\n\n"
                f"📊 Statistics:\n"
                f"Total readings: {total_readings}\n"
                f"Removed (low RMS): {len(removed_readings)}\n"
                f"Spikes corrected: {len(spike_locations)}\n"
                f"Remaining: {total_readings - len(removed_readings)}\n\n"
                f"Files saved:\n"
                f"• Step4_Filtered.csv\n"
                f"• Parameters.csv (updated with 23 parameters)"
            )

        except Exception as e:
            progress_dialog.close()
            import traceback
            QMessageBox.critical(
                self,
                "Error",
                f"Processing failed:\n{str(e)}\n\n{traceback.format_exc()}"
            )


def main():
    """Launch application"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    script_dir = Path(__file__).parent if '__file__' in globals() else Path.cwd()
    output_folder = script_dir / "Output"

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
            step4_window.show()
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
            step3_window.show()
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
                status.setStyleSheet("color: #7f8c8d;")
                layout.addWidget(status)

                progress_dialog.show()
                app.processEvents()

                # Count total lines (fast)
                status.setText("Counting lines...")
                app.processEvents()

                with open(csv_file, 'rb') as f:
                    total_lines = sum(1 for _ in f if not _.startswith(b'#')) - 1  # -1 for header

                status.setText(f"Loading {total_lines:,} records (sampling for speed)...")
                app.processEvents()

                # Calculate how many rows to sample
                target_rows = 10000
                sample_step = max(1, total_lines // target_rows)

                status.setText(f"Reading file in chunks (keeping 1 of every {sample_step} rows)...")
                app.processEvents()

                # Read in chunks and subsample on the fly
                chunk_size = 100000
                sampled_data = []
                row_counter = 0

                for chunk in pd.read_csv(csv_file, comment='#', chunksize=chunk_size):
                    # Sample from this chunk
                    chunk_indices = range(row_counter, row_counter + len(chunk))
                    keep_indices = [i for i in chunk_indices if i % sample_step == 0]

                    if keep_indices:
                        local_indices = [i - row_counter for i in keep_indices]
                        sampled_data.append(chunk.iloc[local_indices])

                    row_counter += len(chunk)

                    # Update progress
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