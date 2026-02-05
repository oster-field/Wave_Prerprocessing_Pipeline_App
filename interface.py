"""
Sakhalin Wave Processor - Новый интерфейс
Простое окно для загрузки файлов с drag & drop
"""

import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QListWidget, QGroupBox, QMessageBox, QDialog,
                             QProgressBar, QTextEdit)
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

            # Step 3: Concatenate all data into single array (fast!)
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

            # Get output path - create Output folder
            output_folder = Path(self.data_files[0]).parent / "Output"
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
        """Read metadata from INFO file"""
        # Try different encodings
        for encoding in ['windows-1251', 'utf-8', 'cp1251']:
            try:
                with open(self.info_file, 'r', encoding=encoding, errors='ignore') as f:
                    lines = f.readlines()
                break
            except:
                continue
        else:
            with open(self.info_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()

        # Read dates (lines 5 and 7 - lines after "начала записи" and "окончания записи")
        date_start = None
        date_end = None
        sensor_frequency = None

        try:
            # Line 6 (0-indexed: 5) should have start date
            if len(lines) > 5:
                date_line = lines[5].strip()
                if date_line and date_line[0].isdigit():
                    date_start = datetime.datetime.strptime(date_line, '%Y.%m.%d %H:%M:%S.%f').date()

            # Line 8 (0-indexed: 7) should have end date
            if len(lines) > 7:
                date_line = lines[7].strip()
                if date_line and date_line[0].isdigit():
                    date_end = datetime.datetime.strptime(date_line, '%Y.%m.%d %H:%M:%S.%f').date()
        except Exception as e:
            self.progress.emit(0, f"Warning: Could not parse dates - {str(e)}")

        # Read frequency from line 3 (0-indexed: 2)
        if len(lines) > 2:
            numbers = re.findall(r'\d+', lines[2])
            if numbers:
                sensor_frequency = int(numbers[0])

        if sensor_frequency is None:
            sensor_frequency = 8  # Default

        return {
            'date_start': date_start,
            'date_end': date_end,
            'sensor_frequency': sensor_frequency
        }

    def read_data_file(self, file_path):
        """Read data from .dat/.txt/.npy file"""
        file_ext = Path(file_path).suffix.lower()

        if file_ext == '.npy':
            return np.load(file_path)
        elif file_ext in ['.dat', '.txt']:
            # Read as text first to handle comma decimal separator
            try:
                # Try loading directly (if using dot separator)
                return np.loadtxt(file_path)
            except:
                # If that fails, replace commas with dots
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    data_text = f.read().replace(',', '.')
                # Parse the numbers
                values = [float(x.strip()) for x in data_text.split() if x.strip()]
                return np.array(values)
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

        # Cancel button
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
        layout.addWidget(self.btn_cancel)

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
        self.init_ui()

    def init_ui(self):
        """Инициализация интерфейса"""
        self.setWindowTitle("🌊 Sakhalin Wave Processor")
        self.setGeometry(100, 100, 900, 700)

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
        """Set INFO file"""
        self.info_file = file_path
        filename = Path(file_path).name

        # Update UI
        self.info_label.setText(f"✅ Loaded: {filename}")
        self.info_label.setStyleSheet("color: #27ae60; font-weight: bold; padding: 5px;")
        self.btn_clear_info.setEnabled(True)

        # Try to read frequency from INFO
        try:
            frequency = self.read_frequency_from_info(file_path)
            self.info_label.setText(f"✅ Loaded: {filename}\n📡 Sensor frequency: {frequency} Hz")
        except Exception as e:
            self.info_label.setText(f"✅ Loaded: {filename}\n⚠️ Could not read frequency: {str(e)}")

        self.update_status()

    def read_frequency_from_info(self, file_path):
        """Read sensor frequency from INFO file"""
        import re

        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        # Look for line with frequency (usually in line 2-3)
        for line in lines[:10]:  # Check first 10 lines
            # Find numbers in line
            numbers = re.findall(r'\d+', line)
            # Look for keywords
            if any(keyword in line.lower() for keyword in ['частота', 'frequency', 'герц', 'hz', 'гц']):
                if numbers:
                    return int(numbers[0])

        # If not found by keywords, try line 3 (as in original)
        if len(lines) >= 3:
            numbers = re.findall(r'\d+', lines[2])
            if numbers:
                return int(numbers[0])

        raise ValueError("Could not find sensor frequency in file")

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

        # Connect cancel button
        self.progress_dialog.btn_cancel.clicked.connect(self.cancel_processing)

        # Start processing
        self.processing_thread.start()
        self.progress_dialog.exec_()

    def cancel_processing(self):
        """Cancel the processing"""
        if hasattr(self, 'processing_thread'):
            self.processing_thread.stop()
        if hasattr(self, 'progress_dialog'):
            self.progress_dialog.close()

    def on_processing_finished(self, success, result_df):
        """Called when processing is complete"""
        if hasattr(self, 'progress_dialog'):
            self.progress_dialog.close()

        if success and result_df is not None:
            # Get output file path
            output_folder = Path(self.data_files[0]).parent / "Output"
            output_file = output_folder / "Step1_TXTtoCSV.csv"

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
                f"  {output_file}\n\n"
                f"Next: Visualization"
            )

            # Store result for visualization
            self.processed_data = result_df

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
        self.setGeometry(100, 100, 1400, 900)

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

        # Plot canvas
        self.canvas = self.create_plot()
        layout.addWidget(self.canvas)

        # Buttons
        btn_layout = QHBoxLayout()

        self.btn_manual = QPushButton("✏️ Proceed with Manual Data Removal")
        self.btn_manual.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                font-size: 14px;
                font-weight: bold;
                padding: 15px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        self.btn_manual.clicked.connect(self.on_manual_removal)
        btn_layout.addWidget(self.btn_manual)

        self.btn_skip = QPushButton("⏭️ Continue without Manual Removal")
        self.btn_skip.setStyleSheet("""
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
        self.btn_skip.clicked.connect(self.on_skip_removal)
        btn_layout.addWidget(self.btn_skip)

        layout.addLayout(btn_layout)

        self.apply_styles()

    def create_plot(self):
        """Create matplotlib plot of all data"""
        # Create figure
        fig = Figure(figsize=(14, 6), dpi=100)
        canvas = FigureCanvas(fig)

        ax = fig.add_subplot(111)

        # Plot data - subsample if too many points for display
        data_to_plot = self.data_df
        if len(data_to_plot) > 50000:
            # Sample every Nth point for visualization
            step = len(data_to_plot) // 50000
            data_to_plot = data_to_plot.iloc[::step]

        # Plot pressure vs time
        ax.plot(data_to_plot.index, data_to_plot['pressure'],
               linewidth=0.5, color='#3498db', alpha=0.7)

        ax.set_xlabel('Data Point Index', fontsize=12)
        ax.set_ylabel('Pressure', fontsize=12)
        ax.set_title('Raw Wave Data - All Readings', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Format x-axis
        ax.ticklabel_format(style='plain', axis='x')

        fig.tight_layout()

        return canvas

    def on_manual_removal(self):
        """Handle manual removal button click"""
        QMessageBox.information(
            self,
            "Manual Removal",
            "Manual data removal feature will be implemented here.\n\n"
            "This will allow you to select and remove unwanted data sections."
        )
        print("Manual removal clicked - to be implemented")

    def on_skip_removal(self):
        """Handle skip button click"""
        QMessageBox.information(
            self,
            "Continue",
            "Continuing without manual removal.\n\n"
            "Next processing steps will be implemented here."
        )
        print("Skip removal clicked - to be implemented")

    def apply_styles(self):
        """Apply global styles"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f6fa;
            }
        """)


def main():
    """Запуск приложения"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()