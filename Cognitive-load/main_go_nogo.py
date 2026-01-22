"""
main_go_nogo.py
Standalone application for Go/No-Go task with EEG data recording.
Runs for 2 minutes and saves data to CSV.
"""

import sys
import pandas as pd
import numpy as np
import os
from datetime import datetime
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QMessageBox, QInputDialog
from PyQt5.QtCore import QTimer

from signal_worker import SignalWorker
from go_nogo_task import GoNoGoTask


class GoNoGoApp(QMainWindow):
    """
    Main application window for Go/No-Go task with EEG recording.
    """
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Go/No-Go Task - EEG Recording")
        self.setGeometry(100, 100, 1200, 800)
        
        # Main components
        self.signal_worker = SignalWorker(sample_rate=250, n_channels=8, buffer_duration=2.0)
        self.go_nogo_task = GoNoGoTask(duration_seconds=120)  # 2 minutes
        
        # Data logging
        self.data_log = []
        self.raw_data_log = []
        self.is_logging = False
        self.current_user = None
        self.user_folder = None
        self.current_trial_label = "idle"
        
        # Setup UI
        self.init_ui()
        
        # Connect signals
        self._connect_signals()
        
        # Timer to update plots (if needed)
        self.plot_timer = QTimer()
        self.plot_timer.timeout.connect(self.update_plots)
        self.plot_timer.start(200)  # Update every 200ms
        
        # Data logging is now handled directly in signal callbacks
    
    def init_ui(self):
        """Initializes the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout()
        central_widget.setLayout(layout)
        
        # Add Go/No-Go task widget
        layout.addWidget(self.go_nogo_task)
        
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
        
        # Go/No-Go task signals
        self.go_nogo_task.response_signal.connect(self.on_trial_response)
        self.go_nogo_task.trial_complete_signal.connect(self.on_trial_complete)
    
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
                'trial_label': self.current_trial_label,
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
    
    def on_raw_data_ready(self, data, timestamp):
        """Handles raw data from SignalWorker."""
        if not self.is_logging:
            return
        
        # Log all raw data (full 250 Hz)
        record = {
            'timestamp': timestamp,
            'trial_label': self.current_trial_label,
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
    
    def check_and_log_data(self):
        """Checks for new data and logs it with appropriate labels."""
        # This method is kept for compatibility but logging is now done
        # directly in on_data_ready and on_raw_data_ready
        pass
    
    def on_trial_response(self, stimulus_label, is_correct, reaction_time):
        """Handles a trial response from the Go/No-Go task."""
        # Update current trial label for data logging
        self.current_trial_label = stimulus_label
        
        # Log trial event
        if hasattr(self, '_trial_events'):
            self._trial_events.append({
                'timestamp': datetime.now().isoformat(),
                'stimulus_label': stimulus_label,
                'is_correct': is_correct,
                'reaction_time': reaction_time
            })
        else:
            self._trial_events = [{
                'timestamp': datetime.now().isoformat(),
                'stimulus_label': stimulus_label,
                'is_correct': is_correct,
                'reaction_time': reaction_time
            }]
    
    def on_trial_complete(self, go_correct, go_incorrect, no_go_correct, no_go_incorrect):
        """Handles trial completion statistics."""
        # This is called after each trial and at the end
        pass
    
    def start_logging(self):
        """Starts data logging."""
        self.is_logging = True
        self.current_trial_label = "go_nogo_task"
        self._trial_events = []
        self.statusBar().showMessage("Logging started")
    
    def stop_logging(self):
        """Stops data logging."""
        self.is_logging = False
        self.current_trial_label = "idle"
        self.statusBar().showMessage("Logging stopped")
    
    def update_plots(self):
        """Updates plots if needed."""
        # Can be used for real-time visualization if needed
        pass
    
    def on_save_data(self):
        """Saves logged data to CSV files."""
        if not self.data_log and not self.raw_data_log:
            QMessageBox.warning(self, "No Data", "No data to save.")
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
                filename_filtered = f"go_nogo_filtered_{timestamp}.csv"
                filepath_filtered = os.path.join(self.user_folder, filename_filtered)
                df_filtered.to_csv(filepath_filtered, index=False)
            
            # Save raw data (full rate)
            if self.raw_data_log:
                df_raw = pd.DataFrame(self.raw_data_log)
                filename_raw = f"go_nogo_raw_{timestamp}.csv"
                filepath_raw = os.path.join(self.user_folder, filename_raw)
                df_raw.to_csv(filepath_raw, index=False)
            
            # Save trial events
            if hasattr(self, '_trial_events') and self._trial_events:
                df_events = pd.DataFrame(self._trial_events)
                filename_events = f"go_nogo_events_{timestamp}.csv"
                filepath_events = os.path.join(self.user_folder, filename_events)
                df_events.to_csv(filepath_events, index=False)
            
            # Show success message
            files_saved = []
            if self.data_log:
                files_saved.append(f"Filtered: {filename_filtered}")
            if self.raw_data_log:
                files_saved.append(f"Raw: {filename_raw}")
            if hasattr(self, '_trial_events') and self._trial_events:
                files_saved.append(f"Events: {filename_events}")
            
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
    
    window = GoNoGoApp()
    
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
            QTimer.singleShot(500, window.on_save_data)  # Small delay to ensure all data is logged
    
    # Connect task start to logging
    def handle_task_button_click():
        if not window.go_nogo_task.is_running:
            QTimer.singleShot(1000, on_task_start)
    
    window.go_nogo_task.start_btn.clicked.connect(handle_task_button_click)
    
    # Monitor task completion (check every 500ms)
    def check_task_status():
        current_running = window.go_nogo_task.is_running
        if window._previous_task_running and not current_running and window.is_logging:
            on_task_end()
        window._previous_task_running = current_running
    
    status_timer = QTimer()
    status_timer.timeout.connect(check_task_status)
    status_timer.start(500)  # Check every 500ms
    
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
