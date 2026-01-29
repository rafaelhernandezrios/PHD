"""
main.py
Application entry point.
Connects all modules: SignalWorker, ExperimentLogic and MainWindow.
"""

import sys
import pandas as pd
import numpy as np
import os
from datetime import datetime
from PyQt5.QtWidgets import QApplication, QMessageBox, QInputDialog
from PyQt5.QtCore import QTimer

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.signal_worker import SignalWorker
from tasks.experiment_logic import ExperimentLogic, ExperimentPhase
from tasks.ui_main import MainWindow


class EEGExperimentApp:
    """
    Main class that coordinates all application components.
    """
    
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.window = MainWindow()
        
        # Main components
        self.signal_worker = SignalWorker(sample_rate=250, n_channels=8, buffer_duration=2.0)
        self.experiment_logic = ExperimentLogic()
        
        # Data for logging
        self.data_log = []
        self._log_buffer = []  # Buffer for batch processing
        self._log_buffer_size = 50  # Process in batches of 50 samples
        self._log_counter = 0  # Initialize counter for subsampling (1 every 5 samples = 50 Hz)
        self._subsampling_factor = 5  # Save every 5th sample (250 Hz -> 50 Hz)
        self.current_phase_name = "idle"
        self.is_logging = False
        self.current_user = None
        self.user_folder = None
        
        # Text for low cognitive load phase
        self.low_load_text = """
        Bad Bunny, whose real name is Benito Antonio Martínez Ocasio, was born on 
        March 10, 1994, in Vega Baja, Puerto Rico. He grew up in a working-class 
        family and developed a passion for music at a young age. Before becoming 
        a global superstar, he worked as a supermarket bagger and studied audiovisual 
        communication at the University of Puerto Rico.
        
        His breakthrough came in 2016 when he uploaded his song "Diles" to SoundCloud. 
        The track caught the attention of DJ Luian, who signed him to his label. 
        Bad Bunny's unique style, which blends reggaeton, trap, and Latin pop, quickly 
        gained popularity. His distinctive voice, creative lyrics, and bold fashion 
        choices set him apart from other artists in the genre.
        
        In 2018, he released his debut album "X 100pre" which included hits like 
        "Mía" and "Solo de Mí". The album was a massive success, establishing him 
        as one of the leading figures in Latin music. He continued to break records 
        with subsequent albums like "YHLQMDLG" and "El Último Tour del Mundo", 
        becoming the first Spanish-language artist to top the Billboard 200 chart.
        
        Beyond music, Bad Bunny has made a significant cultural impact. He challenges 
        traditional gender norms through his fashion choices and public statements, 
        advocating for LGBTQ+ rights and social justice. His influence extends beyond 
        music, making him one of the most important cultural figures of his generation.
        """
        
        # Connect signals
        self._connect_signals()
        
        # Timer to update plots (reduced for better performance)
        self.plot_timer = QTimer()
        self.plot_timer.timeout.connect(self.update_plots)
        self.plot_timer.start(200)  # Update every 200ms (~5 FPS, reduced to avoid saturation)
        
        # Timer to calculate ratio (less frequent to reduce CPU load)
        self.ratio_timer = QTimer()
        self.ratio_timer.timeout.connect(self.calculate_and_update_ratio)
        self.ratio_timer.start(2000)  # Calculate ratio every 2 seconds (reduced for performance)
    
    def _connect_signals(self):
        """Connects all signals between components."""
        
        # SignalWorker -> UI (plotting - subsampled for performance)
        self.signal_worker.raw_data_ready.connect(self.window.update_raw_plot)
        self.signal_worker.plot_data_ready.connect(self.window.update_filtered_plot)
        self.signal_worker.connection_status.connect(self.on_connection_status)
        
        # ExperimentLogic -> UI
        self.experiment_logic.phase_changed.connect(self.on_phase_changed)
        self.experiment_logic.timer_update.connect(self.window.update_timer)
        self.experiment_logic.instruction_update.connect(self.window.instruction_label.setText)
        
        # UI -> App
        self.window.connect_btn.clicked.connect(self.on_connect_clicked)
        self.window.start_setup_btn.clicked.connect(self.on_start_setup)
        self.window.start_baseline_btn.clicked.connect(self.on_start_baseline)
        self.window.start_low_load_btn.clicked.connect(self.on_start_low_load)
        self.window.start_high_load_btn.clicked.connect(self.on_start_high_load)
        self.window.save_data_btn.clicked.connect(self.on_save_data)
        
        # SignalWorker -> Data logging
        self.signal_worker.data_ready.connect(self.log_data_sample)
    
    def on_connect_clicked(self):
        """Handles the connection button click."""
        if not self.signal_worker.isRunning():
            self.window.connect_btn.setText("Connecting...")
            self.window.connect_btn.setEnabled(False)
            self.signal_worker.connect_to_stream()
            if self.signal_worker.inlet:
                self.signal_worker.start()
                self.window.start_setup_btn.setEnabled(True)
        else:
            self.signal_worker.stop()
            self.signal_worker.wait()
            self.window.connect_btn.setText("Connect AURA")
            self.window.start_setup_btn.setEnabled(False)
            self.window.start_baseline_btn.setEnabled(False)
            self.window.start_low_load_btn.setEnabled(False)
            self.window.start_high_load_btn.setEnabled(False)
    
    def on_connection_status(self, connected, message):
        """Handles connection status changes."""
        if connected:
            self.window.connect_btn.setText("Disconnect")
            self.window.connect_btn.setEnabled(True)
            self.window.update_status("Connected", "#00ff88")
        else:
            self.window.connect_btn.setText("Connect AURA")
            self.window.connect_btn.setEnabled(True)
            self.window.update_status("Disconnected", "#ff4444")
            if message:
                QMessageBox.warning(self.window, "Connection Error", message)
    
    def on_start_setup(self):
        """Starts the Setup phase."""
        # Request user name if not set
        if self.current_user is None:
            user_name, ok = QInputDialog.getText(
                self.window, 
                "Experiment User", 
                "Enter the user name or ID:"
            )
            if not ok or not user_name.strip():
                QMessageBox.warning(self.window, "User Required", 
                                  "You must enter a user name to continue.")
                return
            
            self.current_user = user_name.strip()
            # Create folder for the user
            self.user_folder = f"data_{self.current_user}"
            os.makedirs(self.user_folder, exist_ok=True)
            self.window.update_status(f"User: {self.current_user}", "#00ff88")
        
        self.experiment_logic.start_setup()
        self.current_phase_name = "setup"
        self.window.show_instructions(
            "Check the EEG signal quality in the plots. "
            "Make sure all channels show activity without excessive artifacts. "
            "When ready, press 'Start Baseline'."
        )
        self.window.start_baseline_btn.setEnabled(True)
        self.is_logging = True
        # Reset log counter when starting new experiment
        self._log_counter = 0
        print(f"[LOGGING] Logging started. Subsampling factor: {self._subsampling_factor} (target: 50 Hz)")
    
    def on_start_baseline(self):
        """Starts the Baseline phase."""
        self.experiment_logic.start_baseline()
        self.current_phase_name = "baseline_eyes_open"
        self.window.show_instructions(
            "Keep your eyes open and relax. "
            "Stare at the center point of the screen."
        )
        self.window.start_baseline_btn.setEnabled(False)
    
    def on_start_low_load(self):
        """Starts the Low Cognitive Load phase."""
        self.experiment_logic.start_low_load()
        self.current_phase_name = "low_load"
        self.window.show_text_reading(self.low_load_text)
        self.window.start_low_load_btn.setEnabled(False)
        self.window.start_high_load_btn.setEnabled(True)
    
    def on_start_high_load(self):
        """Starts the High Cognitive Load phase (Stroop)."""
        self.experiment_logic.start_high_load()
        self.current_phase_name = "high_load"
        stroop_widget = self.window.show_stroop_task()
        # Connect Stroop response signal if needed
        self.window.start_high_load_btn.setEnabled(False)
        self.window.save_data_btn.setEnabled(True)
    
    def on_phase_changed(self, phase, message):
        """Handles experiment phase changes."""
        # IMPORTANTE: NO resetear el contador al cambiar de fase
        # El contador debe persistir para mantener el subsampling consistente
        if phase == "baseline_eyes_closed":
            self.current_phase_name = "baseline_eyes_closed"
            self.window.show_instructions(
                "Close your eyes and relax completely. "
                "Do not move and maintain calm breathing."
            )
        elif phase == "baseline_completed":
            self.current_phase_name = "baseline_completed"
            self.window.start_low_load_btn.setEnabled(True)
        elif phase == "low_load_completed":
            self.current_phase_name = "low_load_completed"
            self.window.start_high_load_btn.setEnabled(True)
        elif phase == "analysis":
            self.current_phase_name = "analysis"
            self.window.show_instructions(
                "Data analysis in progress. "
                "Results are being processed."
            )
        elif phase == "completed":
            self.current_phase_name = "completed"
            self.window.show_instructions(
                "Experiment completed successfully. "
                "You can save the data and close the application."
            )
    
    def log_data_sample(self, data, timestamp):
        """
        Logs a data sample for later saving.
        Uses subsampling and batch processing to avoid memory saturation.
        
        Args:
            data: Array with filtered data from 8 channels
            timestamp: Sample timestamp
        """
        if not self.is_logging:
            return
        
        # Subsampling: save every 5 samples (~50 Hz instead of 250 Hz)
        # This significantly reduces memory usage
        self._log_counter += 1
        
        # Skip if not time to log yet (early return for performance)
        if self._log_counter % self._subsampling_factor != 0:
            return
        
        # Debug: Print first few logged samples to verify subsampling
        if not hasattr(self, '_log_debug_counter'):
            self._log_debug_counter = 0
        if self._log_debug_counter < 5:
            print(f"[LOGGING] Sample {self._log_debug_counter}: counter={self._log_counter}, logged=True, timestamp={timestamp:.3f}")
            self._log_debug_counter += 1
        
        # Phase mapping to more descriptive labels (cached to avoid dict lookup overhead)
        if not hasattr(self, '_phase_labels'):
            self._phase_labels = {
                'idle': 'idle',
                'setup': 'setup',
                'baseline_eyes_open': 'baseline_eyes_open',
                'baseline_eyes_closed': 'baseline_eyes_closed',
                'baseline_completed': 'baseline_completed',
                'low_load': 'low_cognitive_load',
                'low_load_completed': 'low_load_completed',
                'high_load': 'high_cognitive_load',
                'high_load_completed': 'high_load_completed',
                'analysis': 'analysis',
                'completed': 'completed'
            }
        
        # Add to buffer (more efficient than creating dict immediately)
        self._log_buffer.append((data, timestamp, self.current_phase_name))
        
        # Process buffer in batches to reduce overhead
        if len(self._log_buffer) >= self._log_buffer_size:
            self._flush_log_buffer()
    
    def _flush_log_buffer(self):
        """Flushes the log buffer to data_log (batch processing for efficiency)."""
        if not self._log_buffer:
            return
        
        # Process all items in buffer at once
        for data, timestamp, phase_name in self._log_buffer:
            record = {
                'timestamp': timestamp,
                'phase': phase_name,
                'label': self._phase_labels.get(phase_name, phase_name),
                'channel_0': data[0] if len(data) > 0 else np.nan,
                'channel_1': data[1] if len(data) > 1 else np.nan,
                'channel_2': data[2] if len(data) > 2 else np.nan,
                'channel_3': data[3] if len(data) > 3 else np.nan,
                'channel_4': data[4] if len(data) > 4 else np.nan,
                'channel_5': data[5] if len(data) > 5 else np.nan,
                'channel_6': data[6] if len(data) > 6 else np.nan,
                'channel_7': data[7] if len(data) > 7 else np.nan,
            }
            self.data_log.append(record)
        
        self._log_buffer.clear()
    
    def calculate_and_update_ratio(self):
        """Calculates and updates the cognitive load ratio."""
        if not self.signal_worker.isRunning():
            return
        result = self.signal_worker.get_cognitive_load_ratio()
        if result is not None:
            ratio, theta_power, alpha_power = result
            self.window.update_ratio_plot(ratio, theta_power, alpha_power)
    
    def update_plots(self):
        """Updates the plots (called by timer)."""
        # Plots are updated automatically via signals
        # This function can be used for additional updates if needed
        pass
    
    def on_save_data(self):
        """Saves logged data to a CSV file."""
        # Flush any remaining data in buffer before saving
        if hasattr(self, '_log_buffer') and self._log_buffer:
            self._flush_log_buffer()
        
        if not self.data_log:
            QMessageBox.warning(self.window, "No Data", "No data to save.")
            return
        
        if self.current_user is None:
            QMessageBox.warning(self.window, "User Required", 
                              "No user has been set. Data will not be saved.")
            return
        
        try:
            # Create DataFrame
            df = pd.DataFrame(self.data_log)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"eeg_data_{timestamp}.csv"
            
            # Save in user folder
            filepath = os.path.join(self.user_folder, filename)
            df.to_csv(filepath, index=False)
            
            QMessageBox.information(
                self.window, 
                "Data Saved", 
                f"Data has been saved successfully to:\n{filepath}"
            )
            
        except Exception as e:
            QMessageBox.critical(
                self.window, 
                "Save Error", 
                f"Error saving data:\n{str(e)}"
            )
    
    def run(self):
        """Runs the application."""
        self.window.show()
        return self.app.exec()


def main():
    """Main function."""
    app = EEGExperimentApp()
    sys.exit(app.run())


if __name__ == "__main__":
    main()

