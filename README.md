# Wave Preprocessing Pipeline App

## Project Description
This project implements a wave preprocessing pipeline designed to handle and prepare wave data for further analysis. It focuses on optimizing data collection, cleaning, and transformation processes, ensuring high-quality and ready-to-use datasets for researchers and analysts in the field of wave studies.

## Recent Improvements (v1.1.0)

### Code Quality Enhancements
- ✅ **Fixed PEP 8 Violations**: Replaced all bare `except:` clauses with specific exception types
- ✅ **Optimized Imports**: Moved all imports to top-level for better performance and organization
- ✅ **Added Docstrings**: Improved code documentation throughout the project
- ✅ **Better Error Messages**: More descriptive error messages for users

### Robustness & Security
- ✅ **File Validation**: Added comprehensive file path and size validation
  - Checks file existence, readability, and size before processing
  - Prevents loading corrupted or oversized files
  - Default max file size: 500 MB (configurable)
- ✅ **Input Validation**: Validates all user inputs to prevent crashes
- ✅ **Error Handling**: Improved exception handling with specific error types

### Performance Optimizations
- ✅ **Memory Optimization**: Better handling of large files with chunked reading
- ✅ **Efficient Data Loading**: Optimized numpy operations for faster processing
- ✅ **Caching**: Smart caching of visualization data for faster reopening

### Testing & Quality Assurance
- ✅ **Comprehensive Test Suite**: Added `test_bugs.py` with extensive edge case testing
  - File validation tests
  - Data processing edge cases
  - CSV operations
  - NumPy operations
  - Error handling scenarios

### Configuration System
- ✅ **Config File Support**: Added `config.json` for easy customization
- ✅ **Configuration Loader**: Flexible configuration management with `config_loader.py`
- ✅ **Customizable Settings**: All major parameters now configurable without code changes

### Dependencies
- ✅ **Updated requirements.txt**: Added missing scipy dependency
- ✅ **Optional Dependencies**: Graceful handling of optional packages (PyAstronomy)

## Installation Instructions

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/oster-field/Wave_Prerprocessing_Pipeline_App.git
   cd Wave_Prerprocessing_Pipeline_App
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Install PyAstronomy for advanced wave analysis:
   ```bash
   pip install PyAstronomy
   ```

## Usage Instructions

### Basic Usage
1. Navigate to the project directory.
2. Run the preprocessing pipeline using:
   ```bash
   python interface.py
   ```

3. Use the drag-and-drop interface to:
   - Load the INFO file (contains metadata)
   - Load data files (.dat, .txt, or .npy)
   - Click "Continue to Processing"

4. The processed data will be saved in the `Output/` directory.

### Configuration

Edit `config.json` to customize application behavior:

```json
{
  "data_processing": {
    "max_file_size_mb": 500,
    "visualization_target_points": 5000,
    "default_sensor_frequency": 8
  },
  "output": {
    "directory": "Output",
    "step1_filename": "Step1_TXTtoCSV.csv"
  }
}
```

### Running Tests

To verify the installation and test edge cases:

```bash
python test_bugs.py
```

This runs comprehensive tests including:
- File validation
- Data processing edge cases
- CSV operations
- NumPy operations
- Error handling

## Project Structure

```
Wave_Prerprocessing_Pipeline_App/
├── interface.py           # Main application with GUI
├── config.json           # Configuration file
├── config_loader.py      # Configuration management
├── test_bugs.py          # Comprehensive test suite
├── requirements.txt      # Python dependencies
├── README.md            # This file
└── Output/              # Generated output files
    ├── Step1_TXTtoCSV.csv
    └── Step1_Visualization.csv
```

## Features

### Data Processing Pipeline
1. **Step 1: Data Loading**
   - Reads INFO file for metadata (sensor frequency, dates)
   - Loads multiple data files (.dat, .txt, .npy)
   - Validates file integrity and size
   - Merges data into single dataset

2. **Step 2: Data Transformation**
   - Splits data into 20-minute readings
   - Generates timestamps based on sensor frequency
   - Creates structured DataFrame with metadata

3. **Step 3: Visualization**
   - Interactive plots with zoom and pan
   - Subsampled data for fast rendering
   - Wave parameter calculation
   - Export to various formats

### File Validation
- **Size Check**: Prevents loading files larger than configured limit
- **Existence Check**: Verifies files exist and are readable
- **Encoding Handling**: Supports multiple encodings (UTF-8, Windows-1251, CP1251)
- **Format Validation**: Checks file extensions and content

### Error Handling
- **Specific Exceptions**: All exceptions properly typed
- **User-Friendly Messages**: Clear error descriptions
- **Graceful Degradation**: Continues operation when possible
- **Logging**: Detailed error logs for debugging

## Performance Characteristics

### Memory Usage
- **Chunked Reading**: Processes large files in chunks
- **Smart Caching**: Caches visualization data (10k points max)
- **Memory Limits**: Configurable memory constraints

### Speed Optimizations
- **NumPy Vectorization**: All operations use vectorized numpy
- **Efficient I/O**: Optimized file reading with pandas
- **Parallel Ready**: Architecture supports parallel processing

### Scalability
- **Large Files**: Handles files up to 500 MB (configurable)
- **Multiple Files**: Processes multiple input files efficiently
- **Long Recordings**: Optimized for long-duration recordings

## Known Limitations

1. **PyAstronomy Optional**: Some wave analysis features require PyAstronomy
2. **GUI Framework**: Requires PyQt5 (not compatible with headless systems)
3. **Memory**: Very large files (>1GB) may require system with adequate RAM
4. **Platform**: Tested on Windows and Linux; macOS support not verified

## Troubleshooting

### Issue: "Module not found" errors
**Solution**: Install all dependencies:
```bash
pip install -r requirements.txt
```

### Issue: File too large error
**Solution**: Increase max file size in `config.json`:
```json
{"data_processing": {"max_file_size_mb": 1000}}
```

### Issue: Encoding errors
**Solution**: The application automatically tries multiple encodings. If issues persist, convert files to UTF-8.

### Issue: Out of memory
**Solution**:
1. Close other applications
2. Process files in smaller batches
3. Reduce chunk size in config

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Run tests: `python test_bugs.py`
4. Submit a pull request

## Bug Reports

Found a bug? Please report it with:
- Steps to reproduce
- Expected vs actual behavior
- System information (OS, Python version)
- Error messages and logs

## License

This project is open source. See LICENSE file for details.

## Changelog

### v1.1.0 (Current)
- Added comprehensive input validation
- Improved error handling with specific exception types
- Added configuration file support
- Created extensive test suite
- Optimized imports and code structure
- Added scipy to dependencies
- Enhanced documentation

### v1.0.0
- Initial release
- Basic data processing pipeline
- PyQt5 GUI interface
- Drag-and-drop file loading
- Visualization features

## Contact & Support

For questions or support, please open an issue on GitHub.

---

Made with ❤️ for wave data analysis
