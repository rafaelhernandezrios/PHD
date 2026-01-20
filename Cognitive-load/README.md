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

### Run the Application

```bash
python main.py
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
├── main.py                      # Application entry point
├── signal_worker.py             # Worker thread for LSL acquisition
├── experiment_logic.py           # Protocol state machine
├── ui_main.py                   # PyQt5 graphical interface
├── requirements.txt             # Project dependencies
├── FLUJO_CARGA_COGNITIVA.md    # Detailed technical documentation
├── README.md                    # This file
├── .gitignore                   # Files ignored by Git
└── data_*/                      # User data folders (not versioned)
    └── eeg_data_*.csv           # CSV files with experimental data
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

For detailed technical information about the processing flow, see:
- [FLUJO_CARGA_COGNITIVA.md](FLUJO_CARGA_COGNITIVA.md) - Complete system documentation

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

**Version:** 1.0  
**Last update:** December 2024
