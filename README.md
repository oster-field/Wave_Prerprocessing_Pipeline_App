# Wave Data Preprocessing Pipeline

A desktop GUI application for end-to-end processing of bottom-mounted pressure sensor data. The pipeline converts raw water column height recordings into a comprehensive set of wave parameters — fully automated, no MATLAB or Python scripting required.

![Platform](https://img.shields.io/badge/platform-Windows%2010%2F11-blue)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-proprietary-lightgrey)

---

## What it does

Raw pressure sensor recordings contain tidal drift, deployment/retrieval transients, electronic spikes, and infragravity noise — none of which belong in wave statistics. This pipeline handles all of that automatically and produces clean, publication-ready wave parameters for each 20-minute segment of a deployment. Processed data will be used for ML model building.

The full processing chain:

1. **Load & Convert** — reads binary `.dat` files and a `.INFO` metadata file, merges them into a single timestamped time series, and splits it into 20-minute statistically homogeneous segments
2. **Visualize & Cut** — plots the full record, auto-detects deployment/retrieval legs using a gradient-based dive detector, and lets you manually correct the trim boundaries if needed
3. **Fourier Filter** — computes the full Fourier spectrum, lets you interactively choose a high-pass cutoff frequency, and applies it with mirror-padding to suppress Gibbs-effect edge artifacts
4. **Analyze** — walks through every 20-minute segment and computes 20+ wave parameters, with optional RMS filtering, spike removal, and cubic spline upsampling

---

## Output parameters

For each 20-minute segment, `Parameters.csv` contains:

| Parameter | Description |
|---|---|
| `RMS` | Root mean square surface displacement, m |
| `As`, `Hs` | Significant amplitude and wave height (4σ), m |
| `As_1/3`, `Hs_1/3` | Mean of the highest 1/3 amplitudes/heights, m |
| `Tz` | Mean zero-crossing period, s |
| `kh`, `k` | Dimensionless depth and wavenumber (dispersion equation) |
| `ε` | Wave steepness |
| `a` | Relative amplitude (As/h) |
| `Ur` | Ursell number |
| `Q` | Goda peakedness parameter |
| `ν` | Spectral width (moment-based) |
| `ε_w` | Spectral width (moment-based, alternative) |
| `ρ` | Regularity parameter (waves / extrema ratio) |
| `ε_ρ` | Spectral width derived from ρ |
| `γ` | Nonlinear coefficient (function of kh) |
| `BFI_proper`, `BFI_goda` | Benjamin–Feir Index (full and simplified) |

In addition to the parameters table, the pipeline produces:

- **`Step4_Filtered.csv`** — the cleaned surface displacement time series
- **`Amplitudes_Heights.csv`** — per-segment arrays of individual wave amplitudes and heights

All three files export to `.txt` (tab-separated) or `.mat` (MATLAB v5) format directly from the app.

---

## Installation

**Requirements:** Python 3.10+, Windows 10/11

```bash
git clone https://github.com/your-username/wave-preprocessing-pipeline.git
cd wave-preprocessing-pipeline
pip install -r requirements.txt
python Data_Preprocessing_Pipeline.py
```

### Running without Python (standalone .exe)

A pre-built Windows executable is available on the [Releases](../../releases) page. Download `Data_Preprocessing_Pipeline.exe` and run it directly — no Python installation needed.

To build the executable yourself:

```bash
pip install pyinstaller
pyinstaller --onefile --windowed --icon=assets/icon.ico ^
  --hidden-import=scipy.fftpack ^
  --hidden-import=scipy.signal.windows ^
  --hidden-import=scipy.optimize ^
  --hidden-import=scipy.interpolate ^
  --hidden-import=scipy.io ^
  --hidden-import=PyAstronomy.pyaC ^
  --hidden-import=matplotlib.backends.backend_qt5agg ^
  --name="Data_Preprocessing_Pipeline" ^
  Data_Preprocessing_Pipeline.py
```

---

## Input format

The app expects:

- One or more **`.dat` files** — plain text, one value per line, representing water column height in metres. Both `.` and `,` decimal separators are accepted. Non-numeric lines are silently skipped.
- One **`.INFO` file** — free-form text with metadata. The parser uses regex to extract:
  - Sampling frequency (e.g. `Частота опроса: 8 Гц` or `Frequency: 8 Hz`)
  - Start datetime in `YYYY.MM.DD HH:MM:SS` format
  - End datetime (used for integrity check only)

> **Note:** If your sensor records pressure in Pa or mmHg, convert to metres of water column before loading.

Files can be drag-and-dropped into the app window or added via the file browser. Multiple `.dat` files are merged chronologically by filename.

---

## Checkpoint system

Every step writes its result to the `Output/` folder as a CSV file. On next launch, the app detects which steps are already complete and offers to resume from where you left off — useful for long deployments where re-running Step 3 from scratch would be slow.

```
Output/
├── Step1_TXTtoCSV.csv
├── Step2_Zero_Mean.csv
├── Step3_Transformed.csv
├── Parameters.csv
├── Step4_Filtered.csv
└── Amplitudes_Heights.csv
```

---

## Project structure

```
wave-preprocessing-pipeline/
├── Data_Preprocessing_Pipeline.py   # Main application (single-file)
├── assets/
│   └── icon.ico
├── requirements.txt
├── .gitignore
├── Manual_EN.pdf
├── Manual_DE.pdf
├── Manual_RU.pdf
└── README.md
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `PyQt5` | GUI framework |
| `matplotlib` | All plots and interactive spectrum viewer |
| `numpy` | Array operations |
| `pandas` | CSV I/O and data management |
| `scipy` | FFT, filtering, dispersion solver, spline interpolation, MATLAB export |
| `PyAstronomy` | Zero-crossing detection (`pyaC.zerocross1d`) |

---

## Loading results in MATLAB

```matlab
% Wave parameters — one row per 20-min segment
data = load('Parameters.mat');
Hs   = data.Hs;
Tz   = data.Tz;
BFI  = data.BFI_proper;

% Surface displacement time series
ts = load('Step4_Filtered.mat');
eta = ts.surface_displacement;
t   = datetime(ts.timestamp + 1, 'ConvertFrom', 'datenum'); % +1 day offset

% Individual wave amplitudes and heights (cell arrays, variable length)
ah = load('Amplitudes_Heights.mat');
H1 = ah.heights{1};   % heights for segment 1
histogram(vertcat(ah.heights{:}), 30);
```

## Loading results in Python

```python
import pandas as pd
import scipy.io as sio
import numpy as np

params = pd.read_csv('Parameters.txt', sep='\t')

mat = sio.loadmat('Amplitudes_Heights.mat')
heights_r1 = mat['heights'][0, 0].flatten()
```



## Author

© Andrei Tregubov, 2026