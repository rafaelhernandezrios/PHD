# EEG Experimentation Platform - Cognitive Load

Complete platform for real-time EEG experimentation with cognitive load analysis using the AURA device.

## 📋 Description

EEG signal acquisition and analysis system designed for cognitive load experiments. The platform allows:

- **Real-time acquisition** of EEG signals via LSL (Lab Streaming Layer)
- **Signal processing** with digital filtering and spectral analysis
- **Structured experimental protocol** with multiple phases
- **Real-time visualization** of signals and cognitive load metrics
- **Integrated cognitive tasks** (Stroop, passive reading)
- **Data logging** organized by user

## 🎯 Main Features

### Acquisition and Processing
- LSL connection to AURA device (8 channels, 250 Hz)
- Digital filtering: Notch 60 Hz + Bandpass 1-40 Hz
- Real-time spectral analysis (Welch's method)
- Bandpower calculation: Theta (4-7 Hz) and Alpha (8-12 Hz)
- Cognitive load index: Theta_Fz / Alpha_Pz ratio

### Experimental Protocol
1. **Setup**: Signal quality verification
2. **Baseline**: 90s eyes open + 90s eyes closed
3. **Low Load**: Passive text reading (3 min)
4. **High Load**: Stroop task (3 min)
5. **Analysis**: Processing and visualization of results

### Graphical Interface
- Scientific dashboard with dark theme
- Visualization of 8 individual EEG channels
- Real-time cognitive load ratio graph
- Integrated cognitive tasks with immediate feedback

## 🚀 Installation

### Requirements
- Python 3.9 or higher
- AURA device with drivers installed
- LSL (Lab Streaming Layer) configured

### Installation Steps

1. **Clone the repository**
```bash
git clone https://github.com/tu-usuario/Cognitive-load.git
cd Cognitive-load
```

2. **Create virtual environment**
```bash
python -m venv venv
```

3. **Activate virtual environment**

**Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
venv\Scripts\activate.bat
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

4. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 📖 Usage

### Run the Applications

**Stroop Task (Original Experiment):**
```bash
python apps/main.py
```

**Go/No-Go Task:**
```bash
python apps/main_go_nogo.py
```

**Baseline Recording:**
```bash
python apps/main_baseline.py
```

### Usage Flow

1. **Connect Device**
   - Make sure the AURA device is on and transmitting via LSL
   - Click "Connect AURA"
   - Verify that "Connected" appears in green

2. **Start Setup**
   - Click "Start Setup"
   - Enter the user name or ID when prompted
   - Check signal quality in the 8 channel plots

3. **Execute Protocol**
   - **Baseline**: Click "Start Baseline"
     - Keep your eyes open for 90 seconds
     - Then close your eyes for 90 seconds
   - **Low Load**: Click "Start Low Load"
     - Read the text on screen passively
   - **High Load**: Click "Start High Load"
     - Perform the Stroop task:
       - Press **R** for Red
       - Press **B** for Blue
       - Press **G** for Green
       - Press **Y** for Yellow
       - Identify the **COLOR of the ink**, not the written word

4. **Save Data**
   - When finished, click "Save Data"
   - Data will be saved in `data_[user_name]/eeg_data_[timestamp].csv`

## 📁 Project Structure

```
Cognitive-load/
├── apps/                         # Main application entry points
│   ├── main.py                   # Stroop task (original experiment)
│   ├── main_go_nogo.py          # Go/No-Go task application
│   └── main_baseline.py         # Baseline recording application
│
├── core/                         # Core modules (reusable)
│   ├── signal_worker.py         # LSL acquisition and signal processing
│   └── waaf_filter.py           # WAAF artifact removal (optional)
│
├── tasks/                        # Experimental task widgets
│   ├── ui_main.py               # Stroop task UI
│   ├── experiment_logic.py      # Stroop experiment logic
│   ├── go_nogo_task.py          # Go/No-Go task widget
│   └── baseline_task.py         # Baseline recording widget
│
├── analysis/                     # Data analysis
│   ├── pipeline/                # Main analysis pipelines (use these)
│   │   ├── step1_explore_data.py
│   │   ├── step2_analyze_individual_subjects.py
│   │   ├── step3_detect_artifacts.py
│   │   ├── step4_cognitive_load_cleaned.py    # Rafa: cognitive load + hypothesis
│   │   ├── step5_analysis_cumplen_incluibles.py  # Rafa: only includible + cumplen
│   │   ├── step_jeronimo_segment_by_events.py # Jeronimo: segment by Event
│   │   ├── step_jeronimo_cognitive_load.py    # Jeronimo: load vs baseline
│   │   ├── step_jeronimo_cumplen_hipotesis.py # Jeronimo: cumplen + Rafa/Joss
│   │   ├── step_edwin_segment_by_events.py    # Edwin: segment Laberinto
│   │   ├── step_edwin_cognitive_load.py       # Edwin: load vs baseline (Rafa)
│   │   ├── step_edwin_cumplen_hipotesis.py    # Edwin: cumplen + Joss (Rafa excluded)
│   │   └── README.md
│   ├── utils/                    # Diagnostic utilities
│   ├── archive/                  # Legacy scripts (reference only)
│   └── DOCUMENTACION_DATOS.md
│
├── debug/                        # Debug scripts
│   ├── debug_edwin.py
│   └── debug_edgar.py
│
├── docs/                         # Documentation
│   ├── README_GO_NOGO.md
│   ├── README_BASELINE.md
│   └── FLUJO_CARGA_COGNITIVA.md
│
├── data/                         # Experimental data
│   ├── Data-Experimento-Rafa/   # Go/No-Go, baseline eyes open/closed
│   ├── Data-Experimento-Jeronimo/  # CanineQuest (events, baselines in data_*)
│   └── Data-Experimento-Edwin/    # Laberinto (AURA_RAW, no baseline folders)
│
├── output/                       # Analysis results
│   ├── analysis_output/         # Rafa pipeline (step4, step5)
│   ├── jeronimo_segmented/      # Jeronimo segmented CSVs
│   ├── jeronimo_analysis/       # Jeronimo cognitive load + plots
│   ├── edwin_segmented/         # Edwin segmented CSVs
│   └── edwin_analysis/          # Edwin cognitive load + plots
│
├── requirements.txt
├── README.md                     # This file
└── .gitignore
```

## 📈 Analysis Pipelines

Post-processing pipelines for cognitive load (Theta/Alpha ratio) and hypothesis testing. Run from the project root.

### Pipeline Rafa (Data-Experimento-Rafa)

Data: baseline (eyes open/closed), low load, high load. Hypothesis: **High Load > Low Load**.

| Step | Script | Description |
|------|--------|-------------|
| 1 | `step1_explore_data.py` | Explore CSVs, sample counts per phase |
| 2 | `step2_analyze_individual_subjects.py` | Per-subject signal check (Fz, Pz) |
| 3 | `step3_detect_artifacts.py` | Artifact detection and suppression |
| 4 | `step4_cognitive_load_cleaned.py` | Cognitive load (cleaned), hypothesis check, **exclusion criteria** (artifact % > 50% or &lt; 2 windows in low/high) |
| 5 | `step5_analysis_cumplen_incluibles.py` | Analysis and plots **only for includible subjects that meet hypothesis** |

Output: `output/analysis_output/` (CSVs, comparison plots, cumplen incluibles).

### Pipeline Jeronimo (Data-Experimento-Jeronimo)

Data: CanineQuest variants (keyboard, gamepad, haptico). No explicit phase labels; **Event** column marks segment start/end. Folders starting with `data` = baseline.

| Step | Script | Description |
|------|--------|-------------|
| 1 | `step_jeronimo_segment_by_events.py` | Segment AURA_RAW by Event → `pre`, `segment_4`, …; baselines from `data_*` folders |
| 2 | `step_jeronimo_cognitive_load.py` | Theta/Alpha per phase, **normalized to baseline** (baselines from Jeronimo or Rafa summary for Eli/Jeronimo) |
| 3 | `step_jeronimo_cumplen_hipotesis.py` | Include sessions with **any phase > baseline** + **all Rafa and Joss** sessions; generate plots |

Output: `output/jeronimo_segmented/`, `output/jeronimo_analysis/`.

### Pipeline Edwin (Data-Experimento-Edwin)

Data: **Laberinto** only (AURA_RAW). No baseline folders; baselines from **Rafa** experiment summary.

| Step | Script | Description |
|------|--------|-------------|
| 1 | `step_edwin_segment_by_events.py` | Segment AURA_RAW by Event → `pre`, `segment_*` |
| 2 | `step_edwin_cognitive_load.py` | Theta/Alpha per phase, normalized to **Rafa baseline** (Dani→Daniel, Eli→eliza, etc.) |
| 3 | `step_edwin_cumplen_hipotesis.py` | Include sessions with **any phase > baseline** + **all Joss**; **Rafa excluded**; generate plots |

Output: `output/edwin_segmented/`, `output/edwin_analysis/`.

### Quick run (examples)

```bash
# Rafa (full pipeline)
python analysis/pipeline/step1_explore_data.py
python analysis/pipeline/step2_analyze_individual_subjects.py
python analysis/pipeline/step3_detect_artifacts.py
python analysis/pipeline/step4_cognitive_load_cleaned.py
python analysis/pipeline/step5_analysis_cumplen_incluibles.py

# Jeronimo
python analysis/pipeline/step_jeronimo_segment_by_events.py
python analysis/pipeline/step_jeronimo_cognitive_load.py
python analysis/pipeline/step_jeronimo_cumplen_hipotesis.py

# Edwin
python analysis/pipeline/step_edwin_segment_by_events.py
python analysis/pipeline/step_edwin_cognitive_load.py
python analysis/pipeline/step_edwin_cumplen_hipotesis.py
```

## 📊 Data Format

The saved CSV files contain the following columns:

- `timestamp`: Sample timestamp
- `phase`: Experiment phase (setup, baseline_eyes_open, etc.)
- `label`: Descriptive phase label (setup, baseline_eyes_open, low_cognitive_load, high_cognitive_load, etc.)
- `channel_0` to `channel_7`: Values of the 8 EEG channels (filtered, in device units)

## 🔧 Technical Configuration

### System Parameters

| Parameter | Value |
|-----------|-------|
| Sampling rate | 250 Hz |
| EEG Channels | 8 |
| Notch Filter | 60 Hz, Q=30 |
| Bandpass Filter | 1-40 Hz, order 4 |
| Analysis window | 2 seconds (500 samples) |
| Theta Band | 4-7 Hz (Channel 3 - Fz) |
| Alpha Band | 8-12 Hz (Channel 6 - Pz) |
| Calculation frequency | 1 Hz |

### Channel Mapping

- **Channel 0**: Fp1 (Frontopolar) 
- **Channel 1**: Fp2 (Frontopolar)
- **Channel 2**: F3 (Frontal)
- **Channel 3**: Fz (Frontal) - Used for Theta analysis (4-7 Hz)
- **Channel 4**: F4 (Frontal)
- **Channel 5**: P3 (Parietal)
- **Channel 6**: Pz (Parietal) - Used for Alpha analysis (8-12 Hz)
- **Channel 7**: P4 (Parietal)

## 📚 Documentation

For detailed technical information, see:
- [FLUJO_CARGA_COGNITIVA.md](docs/FLUJO_CARGA_COGNITIVA.md) - Complete system documentation
- [README_GO_NOGO.md](docs/README_GO_NOGO.md) - Go/No-Go task documentation
- [README_BASELINE.md](docs/README_BASELINE.md) - Baseline recording documentation
- [analysis/pipeline/README.md](analysis/pipeline/README.md) - Rafa pipeline (step1–step4) details

## 🛠️ Technologies Used

- **Python 3.9+**
- **PyQt5** - Graphical interface
- **pyqtgraph** - Real-time visualization
- **NumPy** - Numerical processing
- **SciPy** - Digital filters and spectral analysis
- **Pandas** - Data handling and CSV export
- **pylsl** - Communication with Lab Streaming Layer

## 📝 License

This project is under the MIT license. See `LICENSE` file for more details.

## 👥 Contributions

Contributions are welcome. Please:

1. Fork the project
2. Create a branch for your feature (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Contact

For questions or support, please open an issue in the repository.

## 🙏 Acknowledgments

- AURA device for EEG signal acquisition
- Lab Streaming Layer (LSL) community
- Neurotechnology scientific community

---

**Version:** 1.1  
**Last update:** January 2026
