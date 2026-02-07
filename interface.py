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
from matplotlib import pyplot as plt


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
        """Read data from .dat/.txt/.npy file - optimized version"""
        file_ext = Path(file_path).suffix.lower()

        if file_ext == '.npy':
            return np.load(file_path)
        elif file_ext in ['.dat', '.txt']:
            # Optimized reading - use numpy's faster methods
            try:
                # Try direct load with comma as decimal separator
                return np.genfromtxt(file_path, delimiter='\n', encoding='utf-8')
            except:
                # Fallback: read and replace commas
                with open(file_path, 'rb') as f:
                    content = f.read().decode('utf-8', errors='ignore')
                content = content.replace(',', '.')
                # Use fromstring for speed
                return np.fromstring(content, sep='\n')
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
        """Create matplotlib plot - optimized for speed"""
        from PyQt5.QtWidgets import QDesktopWidget

        # Get screen resolution for adaptive subsampling
        screen = QDesktopWidget().screenGeometry()
        screen_width = screen.width()

        # Adaptive point count based on screen resolution
        # Higher resolution = more points for better quality
        if screen_width >= 2560:  # 4K
            target_points = 15000
        elif screen_width >= 1920:  # Full HD
            target_points = 12000
        else:  # HD or lower
            target_points = 8000

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
        timestamps = pd.to_datetime(data_to_plot['timestamp'])
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
        NEW dive detector based on gradient analysis

        Logic:
        - Beginning leg: Low values + VERTICAL JUMP (large positive gradient)
        - Ending leg: VERTICAL DROP (large negative gradient) + low values

        The "leg" includes:
        1. Pre-jump noise near zero
        2. The vertical jump/drop itself
        3. Stops when stable wave oscillations begin
        4. +10% safety margin to ensure complete coverage

        Args:
            pressure: array of pressure values
            sensitivity: threshold multiplier for gradient detection

        Returns:
            Boolean mask where True = dive section
        """
        n = len(pressure)
        dive_mask = np.zeros(n, dtype=bool)

        if n < 100:
            return dive_mask

        # Calculate gradient (rate of change)
        gradient = np.gradient(pressure)
        gradient_abs = np.abs(gradient)

        # Statistics
        grad_std = np.std(gradient)

        # === BEGINNING LEG ===
        # Look in first 30% of data
        search_end = min(n // 3, 2000)

        # Find the largest positive jump in beginning
        beginning_grad = gradient[:search_end]
        max_jump_idx = beginning_grad.argmax()
        max_jump_value = beginning_grad[max_jump_idx]

        # If there's a significant jump (> sensitivity * std)
        if max_jump_value > sensitivity * grad_std:
            # Beginning leg goes from start to end of jump
            # Find where the jump transition completes
            jump_end = max_jump_idx

            # Extend a bit past the jump to include transition
            for i in range(max_jump_idx + 1, min(max_jump_idx + 100, search_end)):
                # Stop when gradient becomes small (stable oscillations)
                if gradient_abs[i] < grad_std:
                    jump_end = i
                    break

            # Add 10% safety margin
            leg_length = jump_end
            safety_margin = int(leg_length * 0.1)
            jump_end = min(jump_end + safety_margin, n - 1)

            # Mark from start to end of jump
            dive_mask[0:jump_end] = True

        # === ENDING LEG ===
        # Look in last 30% of data
        search_start = max(2 * n // 3, n - 2000)

        # Find where pressure starts dropping significantly
        ending_section = pressure[search_start:]
        ending_grad = gradient[search_start:]

        # Find the point where pressure drops below wave level and stays low
        wave_mean = np.mean(pressure[n//3:2*n//3])  # Middle section = waves
        low_threshold = wave_mean * 0.3  # Much lower than waves

        # Find where it drops below threshold
        drop_start = None
        for i in range(len(ending_section) - 50):
            # Check if next 50 points are all below threshold
            if np.all(ending_section[i:i+50] < low_threshold):
                drop_start = search_start + i
                break

        if drop_start is not None:
            # Add 10% safety margin (go back earlier)
            leg_length = n - drop_start
            safety_margin = int(leg_length * 0.1)
            drop_start = max(drop_start - safety_margin, 0)

            # Mark from drop start to end
            dive_mask[drop_start:] = True

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

            QMessageBox.information(
                self,
                "Success",
                f"✅ Processing complete!\n\n"
                f"Files created:\n"
                f"• Step2_Initial_Cut.csv\n"
                f"• Step2_Zero_Mean.csv\n"
                f"• Parameters.csv"
            )
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
            f.write(f"# Average Depth (Full Record): {avg_depth_full_rec:.6f}\n")
            f.write(f"# All pressure values have this subtracted\n")
            f.write("# ==========================================\n")

        zero_mean_data.to_csv(zero_mean_file, mode='a', index=False)

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
            layout.addWidget(beginning_group, stretch=1)  # Equal vertical space

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
            layout.addWidget(ending_group, stretch=1)  # Equal vertical space

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

        btn_cancel = QPushButton("❌ Cancel")
        btn_cancel.setStyleSheet("""
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
        btn_cancel.clicked.connect(self.close)
        btn_layout.addWidget(btn_cancel)

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
            full_data['timestamp'] = pd.to_datetime(full_data['timestamp'])

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

            QMessageBox.information(
                self,
                "Success!",
                f"✅ Data trimmed and processed!\n\n"
                f"Original points: {len(full_data):,}\n"
                f"Removed: {len(full_data) - len(trimmed_data):,}\n"
                f"Remaining: {len(trimmed_data):,}\n\n"
                f"Files created:\n"
                f"• Step2_Initial_Cut.csv\n"
                f"• Step2_Zero_Mean.csv\n"
                f"• Parameters.csv"
            )
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

        # Step 3: Create Zero Mean data (subtract global average from all points)
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
            f.write(f"# Average Depth (Full Record): {avg_depth_full_rec:.6f}\n")
            f.write(f"# All pressure values have this subtracted\n")
            f.write("# ==========================================\n")

        zero_mean_data.to_csv(zero_mean_file, mode='a', index=False)

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


class Step3ProcessingWindow(QMainWindow):
    """Window for Step 3: Spike removal and RMS filtering"""

    def __init__(self):
        super().__init__()
        self.current_reading = 0  # Track which reading is being processed
        self.init_ui()
        self.load_and_visualize()

    def init_ui(self):
        """Initialize Step 3 window"""
        self.setWindowTitle("🌊 Step 3: Spike Removal & RMS Filtering")
        self.showMaximized()

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Header
        header = QLabel("Step 3: Data Quality Processing")
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
        self.cb_remove_spikes.stateChanged.connect(self.check_start_button)
        controls_layout.addWidget(self.cb_remove_spikes)

        # Checkbox 2: Remove low RMS recordings
        rms_layout = QHBoxLayout()
        self.cb_remove_low_rms = QCheckBox("Remove recordings with low")
        self.cb_remove_low_rms.stateChanged.connect(self.check_start_button)
        rms_layout.addWidget(self.cb_remove_low_rms)

        # RMS input field - formatted as 0,000
        self.rms_input = QLineEdit("0,015")
        self.rms_input.setMaxLength(5)
        self.rms_input.setFixedWidth(60)
        self.rms_input.textChanged.connect(self.format_rms_input)
        rms_layout.addWidget(self.rms_input)

        rms_layout.addWidget(QLabel("meters RMS"))
        rms_layout.addStretch()
        controls_layout.addLayout(rms_layout)

        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)

        # Buttons
        btn_layout = QHBoxLayout()

        self.btn_start = QPushButton("▶️ Start Processing")
        self.btn_start.setEnabled(False)  # Disabled until checkbox checked
        self.btn_start.setStyleSheet("""
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
        self.btn_start.clicked.connect(self.start_processing)
        btn_layout.addWidget(self.btn_start)

        btn_skip = QPushButton("⏭️ Skip")
        btn_skip.setStyleSheet("""
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
        btn_skip.clicked.connect(self.skip_processing)
        btn_layout.addWidget(btn_skip)

        layout.addLayout(btn_layout)

    def format_rms_input(self, text):
        """Format RMS input as 0,000"""
        # Remove все кроме цифр
        digits = ''.join(c for c in text if c.isdigit())

        if len(digits) == 0:
            return

        # Pad with zeros if needed
        digits = digits.zfill(4)  # Minimum 4 digits
        digits = digits[:4]  # Maximum 4 digits

        # Format as 0,000
        formatted = digits[0] + ',' + digits[1:]

        # Update field without triggering signal again
        if formatted != text:
            cursor_pos = self.rms_input.cursorPosition()
            self.rms_input.blockSignals(True)
            self.rms_input.setText(formatted)
            self.rms_input.setCursorPosition(min(cursor_pos, len(formatted)))
            self.rms_input.blockSignals(False)

    def check_start_button(self):
        """Enable start button only if at least one checkbox is checked"""
        enabled = self.cb_remove_spikes.isChecked() or self.cb_remove_low_rms.isChecked()
        self.btn_start.setEnabled(enabled)

    def load_and_visualize(self):
        """Load Step2_Zero_Mean and create visualization with reading boundaries"""
        # TODO: Implement loading with progress bar and visualization
        pass

    def start_processing(self):
        """Start spike removal and/or RMS filtering"""
        # TODO: Implement processing logic
        pass

    def skip_processing(self):
        """Skip Step 3 processing"""
        QMessageBox.information(
            self,
            "Skipped",
            "Step 3 processing skipped."
        )
        self.close()


def main():
    """Launch application"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    script_dir = Path(__file__).parent if '__file__' in globals() else Path.cwd()
    output_folder = script_dir / "Output"

    # CHECKPOINT 2: Check if Step2_Zero_Mean exists
    step2_zero_mean = output_folder / "Step2_Zero_Mean.csv"
    parameters_file = output_folder / "Parameters.csv"

    if step2_zero_mean.exists() and parameters_file.exists():
        reply = QMessageBox.question(
            None,
            "Step 2 Complete - Continue?",
            f"Found processed Step 2 data:\n"
            f"• Step2_Zero_Mean.csv\n"
            f"• Parameters.csv\n\n"
            "Continue to Step 3 (Spike removal & RMS filtering) or start from scratch?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )

        if reply == QMessageBox.Yes:
            # Open Step 3 processing window
            from PyQt5.QtWidgets import QCheckBox, QLineEdit
            step3_window = Step3ProcessingWindow()
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
                    df['timestamp'] = pd.to_datetime(df['timestamp'])

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

                df['timestamp'] = pd.to_datetime(df['timestamp'])

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