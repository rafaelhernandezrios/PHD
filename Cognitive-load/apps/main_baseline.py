"""
main_baseline.py
Standalone application for baseline recording with EEG data.
Phase 1: Eyes open (1.5 minutes) - staring at fixation point
Phase 2: Eyes closed (1.5 minutes)
"""

import sys
import pandas as pd
import numpy as np
import os
from datetime import datetime
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QMessageBox, QInputDialog
from PyQt5.QtCore import QTimer

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.signal_worker import SignalWorker
from tasks.baseline_task import BaselineTask


class BaselineApp(QMainWindow):
    """
    Main application window for baseline recording with EEG.
    """
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Baseline Recording - EEG")
        self.setGeometry(100, 100, 1200, 800)
        
        # Main components
        self.signal_worker = SignalWorker(sample_rate=250, n_channels=8, buffer_duration=2.0)
        self.baseline_task = BaselineTask(duration_seconds=90)  # 1.5 minutes = 90 seconds
        
        # Data logging
        self.data_log = []
        self.raw_data_log = []
        self.is_logging = False
        self.current_user = None
        self.user_folder = None
        self.current_phase_label = "idle"
        
        # Setup UI
        self.init_ui()
        
        # Connect signals
        self._connect_signals()
        
        # Timer to update plots (if needed)
        self.plot_timer = QTimer()
        self.plot_timer.timeout.connect(self.update_plots)
        self.plot_timer.start(200)  # Update every 200ms
    
    def init_ui(self):
        """Initializes the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout()
        central_widget.setLayout(layout)
        
        # Add baseline task widget
        layout.addWidget(self.baseline_task)
        
        # Status bar
        self.statusBar().showMessage("Ready to connect")
        
        # Menu bar
        menubar = self.menuBar()
        file_menu = menubar.addMenu('File')
        
        connect_action = file_menu.addAction('Connect to EEG')
        connect_action.triggered.connect(self.on_connect_clicked)
        
        set_user_action = file_menu.addAction('Set User Name')
        set_user_action.triggered.connect(self.on_set_user)
        
        save_action = file_menu.addAction('Save Data')
        save_action.triggered.connect(self.on_save_data)
        
        exit_action = file_menu.addAction('Exit')
        exit_action.triggered.connect(self.close)
    
    def _connect_signals(self):
        """Connects all signals between components."""
        # SignalWorker signals
        self.signal_worker.data_ready.connect(self.on_data_ready)
        self.signal_worker.raw_data_ready.connect(self.on_raw_data_ready)
        self.signal_worker.connection_status.connect(self.on_connection_status)
        
        # Baseline task signals
        self.baseline_task.phase_changed_signal.connect(self.on_phase_changed)
        self.baseline_task.task_complete_signal.connect(self.on_task_complete)
    
    def on_connect_clicked(self):
        """Handles the connection button click."""
        if not self.signal_worker.isRunning():
            self.statusBar().showMessage("Connecting to EEG stream...")
            self.signal_worker.start()
        else:
            self.statusBar().showMessage("Already connected")
    
    def on_connection_status(self, status, message):
        """Handles connection status updates."""
        if status:
            self.statusBar().showMessage(f"Connected: {message}", 5000)
        else:
            self.statusBar().showMessage(f"Connection failed: {message}", 5000)
    
    def on_set_user(self):
        """Sets the current user name for data saving."""
        user_name, ok = QInputDialog.getText(
            self, 
            'Set User Name', 
            'Enter user name:'
        )
        
        if ok and user_name:
            self.current_user = user_name.strip()
            # Create user folder
            self.user_folder = os.path.join(os.getcwd(), f"data_{self.current_user}")
            if not os.path.exists(self.user_folder):
                os.makedirs(self.user_folder)
            self.statusBar().showMessage(f"User set to: {self.current_user}", 3000)
    
    def on_data_ready(self, data, timestamp):
        """Handles filtered data from SignalWorker."""
        if not self.is_logging:
            return
        
        # Subsampling: log every 5th sample (~50 Hz instead of 250 Hz)
        if not hasattr(self, '_filtered_log_counter'):
            self._filtered_log_counter = 0
        
        self._filtered_log_counter += 1
        if self._filtered_log_counter % 5 == 0:
            record = {
                'timestamp': timestamp,
                'phase_label': self.current_phase_label,
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
            
            # Debug: print first few samples
            if len(self.data_log) <= 3:
                print(f"[DEBUG] Logged sample {len(self.data_log)}: phase={self.current_phase_label}, timestamp={timestamp}")
    
    def on_raw_data_ready(self, data, timestamp):
        """Handles raw data from SignalWorker."""
        if not self.is_logging:
            return
        
        # Log all raw data (full 250 Hz)
        record = {
            'timestamp': timestamp,
            'phase_label': self.current_phase_label,
            'channel_0': data[0] if len(data) > 0 else np.nan,
            'channel_1': data[1] if len(data) > 1 else np.nan,
            'channel_2': data[2] if len(data) > 2 else np.nan,
            'channel_3': data[3] if len(data) > 3 else np.nan,
            'channel_4': data[4] if len(data) > 4 else np.nan,
            'channel_5': data[5] if len(data) > 5 else np.nan,
            'channel_6': data[6] if len(data) > 6 else np.nan,
            'channel_7': data[7] if len(data) > 7 else np.nan,
        }
        self.raw_data_log.append(record)
    
    def on_phase_changed(self, phase_name):
        """Handles phase changes from baseline task."""
        # Map phase names to labels
        phase_labels = {
            'eyes_open': 'baseline_eyes_open',
            'eyes_closed': 'baseline_eyes_closed'
        }
        self.current_phase_label = phase_labels.get(phase_name, 'idle')
        # Ensure logging continues during phase changes
        if not self.is_logging and self.baseline_task.is_running:
            print(f"[DEBUG] Phase changed but logging was off, restarting...")
            self.start_logging()
        print(f"[DEBUG] Phase changed to: {phase_name}, label: {self.current_phase_label}")
        self.statusBar().showMessage(f"Phase: {self.current_phase_label}", 3000)
    
    def on_task_complete(self):
        """Handles task completion."""
        print(f"[DEBUG] Task complete. Data log length: {len(self.data_log)}, Raw log length: {len(self.raw_data_log)}")
        self.stop_logging()
        # Auto-save after task completion
        QTimer.singleShot(1000, self.on_save_data)  # Delay to ensure all data is logged
    
    def start_logging(self):
        """Starts data logging."""
        self.is_logging = True
        self.current_phase_label = "baseline_start"
        # Reset counters
        if hasattr(self, '_filtered_log_counter'):
            self._filtered_log_counter = 0
        print(f"[DEBUG] Logging started. SignalWorker running: {self.signal_worker.isRunning()}")
        self.statusBar().showMessage("Logging started")
    
    def stop_logging(self):
        """Stops data logging."""
        self.is_logging = False
        self.current_phase_label = "idle"
        self.statusBar().showMessage("Logging stopped")
    
    def update_plots(self):
        """Updates plots if needed."""
        # Can be used for real-time visualization if needed
        pass
    
    def on_save_data(self):
        """Saves logged data to CSV files."""
        print(f"[DEBUG] Save data called. Filtered: {len(self.data_log)}, Raw: {len(self.raw_data_log)}")
        if not self.data_log and not self.raw_data_log:
            QMessageBox.warning(
                self, 
                "No Data", 
                f"No data to save.\nFiltered samples: {len(self.data_log)}\nRaw samples: {len(self.raw_data_log)}\nIs logging: {self.is_logging}\nSignalWorker running: {self.signal_worker.isRunning() if hasattr(self, 'signal_worker') else 'N/A'}"
            )
            return
        
        if self.current_user is None:
            QMessageBox.warning(
                self, 
                "User Required", 
                "Please set a user name before saving data."
            )
            return
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save filtered data (subsampled)
            if self.data_log:
                df_filtered = pd.DataFrame(self.data_log)
                filename_filtered = f"baseline_filtered_{timestamp}.csv"
                filepath_filtered = os.path.join(self.user_folder, filename_filtered)
                df_filtered.to_csv(filepath_filtered, index=False)
            
            # Save raw data (full rate)
            if self.raw_data_log:
                df_raw = pd.DataFrame(self.raw_data_log)
                filename_raw = f"baseline_raw_{timestamp}.csv"
                filepath_raw = os.path.join(self.user_folder, filename_raw)
                df_raw.to_csv(filepath_raw, index=False)
            
            # Show success message
            files_saved = []
            if self.data_log:
                files_saved.append(f"Filtered: {filename_filtered}")
            if self.raw_data_log:
                files_saved.append(f"Raw: {filename_raw}")
            
            QMessageBox.information(
                self,
                "Data Saved",
                f"Data has been saved successfully:\n" + "\n".join(files_saved)
            )
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Save Error",
                f"Error saving data:\n{str(e)}"
            )
    
    def closeEvent(self, event):
        """Handles window close event."""
        if self.is_logging:
            reply = QMessageBox.question(
                self,
                'Save Data?',
                'Data logging is active. Do you want to save before closing?',
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                QMessageBox.Yes
            )
            
            if reply == QMessageBox.Yes:
                self.on_save_data()
                event.accept()
            elif reply == QMessageBox.Cancel:
                event.ignore()
                return
            else:
                event.accept()
        else:
            event.accept()
        
        # Stop signal worker
        if self.signal_worker.isRunning():
            self.signal_worker.stop()
            self.signal_worker.wait()


def main():
    """Main function."""
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    window = BaselineApp()
    
    # Track previous task state
    window._previous_task_running = False
    
    # Auto-start logging when task starts
    def on_task_start():
        window.start_logging()
        window._previous_task_running = True
    
    def on_task_end():
        window.stop_logging()
        # Auto-save after task completion
        if window.data_log or window.raw_data_log:
            QTimer.singleShot(500, window.on_save_data)
    
    # Connect task start to logging
    def handle_task_button_click():
        if not window.baseline_task.is_running:
            print("[DEBUG] Task button clicked, starting logging...")
            on_task_start()  # Start logging immediately (no delay needed)
    
    window.baseline_task.start_btn.clicked.connect(handle_task_button_click)
    
    # Also connect to phase changes to ensure logging continues
    def on_phase_changed_handler(phase_name):
        print(f"[DEBUG] Phase changed to: {phase_name}")
        # Ensure logging is still active
        if not window.is_logging and window.baseline_task.is_running:
            print("[DEBUG] Logging was stopped, restarting...")
            window.start_logging()
    
    window.baseline_task.phase_changed_signal.connect(on_phase_changed_handler)
    
    # Monitor task completion
    def check_task_status():
        current_running = window.baseline_task.is_running
        if window._previous_task_running and not current_running and window.is_logging:
            print("[DEBUG] Task stopped, calling on_task_end...")
            on_task_end()
        window._previous_task_running = current_running
    
    status_timer = QTimer()
    status_timer.timeout.connect(check_task_status)
    status_timer.start(500)  # Check every 500ms
    
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
