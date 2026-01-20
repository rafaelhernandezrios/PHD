"""
ui_main.py
Main graphical interface for the EEG experimentation platform.
Uses PyQt5 with dark scientific dashboard design.
"""

import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QTextEdit, QFrame, QGridLayout, QMessageBox
)
from PyQt5.QtCore import Qt, pyqtSlot, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QKeyEvent
import time
from collections import deque


class StroopTask(QWidget):
    """
    Widget for the Stroop task (High Cognitive Load).
    The user must identify the COLOR of the ink, not the written word.
    """
    
    response_signal = pyqtSignal(bool)  # True if correct answer, False if incorrect
    
    def __init__(self):
        super().__init__()
        self.colors = {
            'RED': '#ff4444',
            'BLUE': '#4444ff',
            'GREEN': '#44ff44',
            'YELLOW': '#ffff44'
        }
        self.color_keys = {
            Qt.Key_R: 'RED',
            Qt.Key_B: 'BLUE',
            Qt.Key_G: 'GREEN',
            Qt.Key_Y: 'YELLOW'
        }
        self.color_names = {
            'RED': 'R',
            'BLUE': 'B',
            'GREEN': 'G',
            'YELLOW': 'Y'
        }
        self.trial_count = 0
        self.correct_count = 0
        self.total_responses = 0
        self.congruent_count = 0
        self.incongruent_count = 0
        self.stimulus_interval = 2000  # 2 segundos entre estímulos (en ms)
        self.stimulus_timer = QTimer()
        self.stimulus_timer.timeout.connect(self.on_stimulus_timeout)
        self.current_word = None
        self.current_color = None
        self.is_congruent = None
        self.waiting_for_response = False
        
        self.init_ui()
        self.generate_stimulus()
    
    def init_ui(self):
        """Initializes the Stroop task interface."""
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Stroop Task")
        title.setAlignment(Qt.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(22)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setStyleSheet("color: #00ff88; margin: 10px;")
        layout.addWidget(title)
        
        # Instructions
        instructions = QLabel(
            "Press the key for the COLOR of the ink, NOT the written word:\n"
            "R = Red | B = Blue | G = Green | Y = Yellow"
        )
        instructions.setAlignment(Qt.AlignCenter)
        instructions_font = QFont()
        instructions_font.setPointSize(18)
        instructions.setFont(instructions_font)
        instructions.setStyleSheet("color: #cccccc; margin: 10px; font-size: 18px;")
        layout.addWidget(instructions)
        
        # Stimulus area
        self.stimulus_label = QLabel("")
        self.stimulus_label.setAlignment(Qt.AlignCenter)
        stimulus_font = QFont()
        stimulus_font.setPointSize(72)
        stimulus_font.setBold(True)
        self.stimulus_label.setFont(stimulus_font)
        self.stimulus_label.setStyleSheet(
            "background-color: #1a1a1a; "
            "border: 2px solid #00ff88; "
            "border-radius: 10px; "
            "padding: 40px; "
            "min-height: 200px;"
        )
        layout.addWidget(self.stimulus_label)
        
        # Feedback
        self.feedback_label = QLabel("")
        self.feedback_label.setAlignment(Qt.AlignCenter)
        feedback_font = QFont()
        feedback_font.setPointSize(20)
        self.feedback_label.setFont(feedback_font)
        self.feedback_label.setStyleSheet("color: #ffaa00; margin: 10px; font-size: 20px;")
        layout.addWidget(self.feedback_label)
        
        # Statistics
        stats_layout = QVBoxLayout()
        self.stats_label = QLabel("Correct: 0 / Responses: 0")
        self.stats_label.setAlignment(Qt.AlignCenter)
        stats_font = QFont()
        stats_font.setPointSize(16)
        self.stats_label.setFont(stats_font)
        self.stats_label.setStyleSheet("color: #888888;")
        stats_layout.addWidget(self.stats_label)
        
        self.congruency_label = QLabel("Congruent: 0 | Incongruent: 0")
        self.congruency_label.setAlignment(Qt.AlignCenter)
        self.congruency_label.setFont(stats_font)
        self.congruency_label.setStyleSheet("color: #888888;")
        stats_layout.addWidget(self.congruency_label)
        
        layout.addLayout(stats_layout)
        layout.addStretch()
        self.setLayout(layout)
    
    def start_stimulus_timer(self):
        """Starts the timer for automatic stimulus changes."""
        self.stimulus_timer.start(self.stimulus_interval)
    
    def stop_stimulus_timer(self):
        """Stops the stimulus timer."""
        self.stimulus_timer.stop()
    
    def on_stimulus_timeout(self):
        """Callback when the stimulus timer expires."""
        if self.waiting_for_response:
            # Timeout - no response
            self.check_response(None)
    
    def generate_stimulus(self):
        """Generates a new Stroop stimulus."""
        # Stop timer while processing
        self.stop_stimulus_timer()
        self.waiting_for_response = False
        
        # Decide if it will be congruent or incongruent (70% incongruent for more load)
        self.is_congruent = np.random.random() < 0.3  # 30% congruent, 70% incongruent
        
        # Select word and color
        word_options = list(self.colors.keys())
        self.current_word = np.random.choice(word_options)
        
        if self.is_congruent:
            # Congruent: word and color match
            self.current_color = self.current_word
            self.congruent_count += 1
        else:
            # Incongruent: word and color do NOT match
            color_options = [c for c in word_options if c != self.current_word]
            self.current_color = np.random.choice(color_options)
            self.incongruent_count += 1
        
        # Show stimulus with corresponding color
        color_hex = self.colors[self.current_color]
        self.stimulus_label.setText(self.current_word)
        self.stimulus_label.setStyleSheet(
            f"background-color: #1a1a1a; "
            f"border: 2px solid #00ff88; "
            f"border-radius: 10px; "
            f"padding: 40px; "
            f"color: {color_hex}; "
            f"min-height: 200px;"
        )
        
        self.trial_count += 1
        self.waiting_for_response = True
        
        # Clear previous feedback
        self.feedback_label.setText("")
        
        # Restart timer
        self.start_stimulus_timer()
    
    def check_response(self, pressed_key):
        """
        Checks if the user's response was correct.
        
        Args:
            pressed_key: Qt.Key of the pressed key, or None if timeout
        """
        # Stop timer since a response was processed
        self.stop_stimulus_timer()
        self.waiting_for_response = False
        
        if pressed_key is None:
            # Timeout - no response
            self.feedback_label.setText("⏱ No response")
            self.feedback_label.setStyleSheet("color: #ffaa00; margin: 10px; font-size: 20px;")
            self.total_responses += 1
            self.stats_label.setText(f"Correct: {self.correct_count} / Responses: {self.total_responses}")
            # Generate new stimulus after timeout
            QTimer.singleShot(500, self.generate_stimulus)
            return
        
        # Check if the pressed key corresponds to the correct color
        if pressed_key in self.color_keys:
            selected_color = self.color_keys[pressed_key]
            is_correct = selected_color == self.current_color
            
            if is_correct:
                self.correct_count += 1
                self.feedback_label.setText("✓ Correct")
                self.feedback_label.setStyleSheet("color: #00ff88; margin: 10px; font-size: 20px;")
                self.response_signal.emit(True)
            else:
                self.feedback_label.setText(f"✗ Incorrect (Was {self.color_names[self.current_color]})")
                self.feedback_label.setStyleSheet("color: #ff4444; margin: 10px; font-size: 20px;")
                self.response_signal.emit(False)
            
            self.total_responses += 1
            self.stats_label.setText(f"Correct: {self.correct_count} / Responses: {self.total_responses}")
            self.congruency_label.setText(
                f"Congruent: {self.congruent_count} | Incongruent: {self.incongruent_count}"
            )
            
            # Generate new stimulus after showing feedback
            QTimer.singleShot(500, self.generate_stimulus)
        else:
            # Invalid key, ignore
            pass
    
    def keyPressEvent(self, event: QKeyEvent):
        """Handles keys pressed during the task."""
        if self.waiting_for_response:
            if event.key() in self.color_keys:
                # Stop timer and process response immediately
                self.stop_stimulus_timer()
                self.check_response(event.key())
        else:
            super().keyPressEvent(event)
    
    def focusInEvent(self, event):
        """Ensures the widget receives keyboard events."""
        self.setFocus()
        super().focusInEvent(event)


class NBackTask(QWidget):
    """
    Widget for the N-Back task (High Cognitive Load).
    Implements a visual version of N-Back.
    """
    
    response_signal = pyqtSignal(bool)  # True if correct answer, False if incorrect
    
    def __init__(self, n_level=2):
        super().__init__()
        self.n_level = n_level  # N-Back level (e.g., 2-Back)
        self.stimulus_history = deque(maxlen=n_level + 1)
        self.current_stimulus = None
        self.trial_count = 0
        self.correct_count = 0
        self.total_responses = 0
        self.stimulus_interval = 2000  # 2 segundos entre estímulos (en ms)
        self.stimulus_timer = QTimer()
        self.stimulus_timer.timeout.connect(self.on_stimulus_timeout)
        
        self.init_ui()
        self.generate_stimulus()
        self.start_stimulus_timer()
    
    def init_ui(self):
        """Initializes the N-Back task interface."""
        layout = QVBoxLayout()
        
        # Title
        title = QLabel(f"{self.n_level}-Back Task")
        title.setAlignment(Qt.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(22)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setStyleSheet("color: #00ff88; margin: 10px;")
        layout.addWidget(title)
        
        # Instructions
        instructions = QLabel(
            f"Press SPACE when the number matches the one {self.n_level} positions back"
        )
        instructions.setAlignment(Qt.AlignCenter)
        instructions.setStyleSheet("color: #cccccc; margin: 5px;")
        layout.addWidget(instructions)
        
        # Área de estímulo
        self.stimulus_label = QLabel("")
        self.stimulus_label.setAlignment(Qt.AlignCenter)
        stimulus_font = QFont()
        stimulus_font.setPointSize(72)
        stimulus_font.setBold(True)
        self.stimulus_label.setFont(stimulus_font)
        self.stimulus_label.setStyleSheet(
            "background-color: #1a1a1a; "
            "border: 2px solid #00ff88; "
            "border-radius: 10px; "
            "padding: 40px; "
            "color: #00ff88; "
            "min-height: 200px;"
        )
        layout.addWidget(self.stimulus_label)
        
        # Feedback
        self.feedback_label = QLabel("")
        self.feedback_label.setAlignment(Qt.AlignCenter)
        self.feedback_label.setStyleSheet("color: #ffaa00; margin: 10px;")
        layout.addWidget(self.feedback_label)
        
        # Statistics
        stats_layout = QHBoxLayout()
        self.stats_label = QLabel("Correct: 0 / Responses: 0")
        self.stats_label.setAlignment(Qt.AlignCenter)
        self.stats_label.setStyleSheet("color: #888888;")
        stats_layout.addWidget(self.stats_label)
        layout.addLayout(stats_layout)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def start_stimulus_timer(self):
        """Starts the timer for automatic stimulus changes."""
        self.stimulus_timer.start(self.stimulus_interval)
    
    def stop_stimulus_timer(self):
        """Stops the stimulus timer."""
        self.stimulus_timer.stop()
    
    def on_stimulus_timeout(self):
        """Callback when the stimulus timer expires."""
        # If space was not pressed, it's considered "no response"
        if len(self.stimulus_history) >= self.n_level:
            self.check_response(False)  # Space was not pressed
        else:
            # If there aren't enough stimuli yet, just generate the next one
            # without evaluating response
            self.generate_stimulus()
    
    def generate_stimulus(self):
        """Generates a new stimulus (number from 1 to 9)."""
        # Stop timer while processing
        self.stop_stimulus_timer()
        
        # Add current stimulus to history before generating a new one
        if self.current_stimulus is not None:
            self.stimulus_history.append(self.current_stimulus)
        
        # Generate new stimulus
        self.current_stimulus = np.random.randint(1, 10)
        self.stimulus_label.setText(str(self.current_stimulus))
        self.trial_count += 1
        
        # Clear previous feedback
        self.feedback_label.setText("")
        
        # IMPORTANT: Restart timer so next stimulus appears automatically
        self.start_stimulus_timer()
    
    def check_response(self, responded):
        """
        Checks if the user's response was correct.
        
        Args:
            responded: True if user pressed space, False if not
        """
        # Stop timer since a response was processed
        self.stop_stimulus_timer()
        
        # We need at least n_level stimuli in history to compare
        if len(self.stimulus_history) < self.n_level:
            # Not enough stimuli yet, just continue
            return
        
        # Check if current stimulus matches the one n_level positions back
        expected_response = False
        if len(self.stimulus_history) >= self.n_level:
            expected_response = self.stimulus_history[-self.n_level] == self.current_stimulus
        
        # Evaluate response
        if responded == expected_response:
            self.correct_count += 1
            self.feedback_label.setText("✓ Correct")
            self.feedback_label.setStyleSheet("color: #00ff88; margin: 10px;")
            self.response_signal.emit(True)
        else:
            if responded:
                # Only show "Incorrect" if space was pressed (not if timeout)
                self.feedback_label.setText("✗ Incorrect")
                self.feedback_label.setStyleSheet("color: #ff4444; margin: 10px;")
            else:
                # Timeout - no response
                self.feedback_label.setText("⏱ No response")
                self.feedback_label.setStyleSheet("color: #ffaa00; margin: 10px;")
            self.response_signal.emit(False)
        
        self.total_responses += 1
        self.stats_label.setText(f"Correct: {self.correct_count} / Responses: {self.total_responses}")
        
        # Generate new stimulus after showing feedback (500ms delay)
        # generate_stimulus already restarts timer automatically
        QTimer.singleShot(500, self.generate_stimulus)
    
    def keyPressEvent(self, event: QKeyEvent):
        """Handles keys pressed during the task."""
        if event.key() == Qt.Key_Space:
            # Stop timer and process response immediately
            self.stop_stimulus_timer()
            self.check_response(True)
        else:
            super().keyPressEvent(event)
    
    def focusInEvent(self, event):
        """Ensures the widget receives keyboard events."""
        self.setFocus()
        super().focusInEvent(event)


class MainWindow(QMainWindow):
    """
    Main application window.
    Scientific dashboard with real-time visualization.
    """
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EEG Experimentation Platform - Cognitive Load")
        self.setGeometry(100, 100, 1400, 900)
        
        # Dark style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0d1117;
                color: #c9d1d9;
            }
            QPushButton {
                background-color: #21262d;
                border: 1px solid #30363d;
                border-radius: 6px;
                padding: 8px 16px;
                color: #c9d1d9;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #30363d;
                border-color: #00ff88;
            }
            QPushButton:pressed {
                background-color: #161b22;
            }
            QLabel {
                color: #c9d1d9;
            }
            QTextEdit {
                background-color: #161b22;
                border: 1px solid #30363d;
                border-radius: 6px;
                color: #c9d1d9;
                padding: 10px;
            }
            QFrame {
                border: 1px solid #30363d;
                border-radius: 6px;
                background-color: #161b22;
            }
        """)
        
        # Buffers for plots (reduced for better performance)
        self.plot_buffer_size = 750  # ~3 seconds at 250 Hz (reduced for better performance)
        self.raw_data_buffer = {i: deque(maxlen=self.plot_buffer_size) for i in range(8)}
        self.timestamp_buffer = deque(maxlen=self.plot_buffer_size)
        self.ratio_buffer = deque(maxlen=300)  # Smaller buffer for ratio
        self.ratio_timestamps = deque(maxlen=300)
        
        # Plot update control
        self.plot_update_counter = 0
        self.plot_update_skip = 5  # Update every 5 received samples (reduced from 2)
        self.last_update_time = {}  # To limit update frequency per channel
        
        # Initialize list of plot widgets (will be filled in init_ui)
        self.raw_plot_widgets = []
        
        self.init_ui()
    
    def init_ui(self):
        """Initializes all interface components."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Top bar with controls
        controls_layout = QHBoxLayout()
        
        self.connect_btn = QPushButton("Connect AURA")
        self.connect_btn.clicked.connect(self.on_connect_clicked)
        controls_layout.addWidget(self.connect_btn)
        
        self.start_setup_btn = QPushButton("Start Setup")
        self.start_setup_btn.clicked.connect(self.on_start_setup)
        self.start_setup_btn.setEnabled(False)
        controls_layout.addWidget(self.start_setup_btn)
        
        self.start_baseline_btn = QPushButton("Start Baseline")
        self.start_baseline_btn.clicked.connect(self.on_start_baseline)
        self.start_baseline_btn.setEnabled(False)
        controls_layout.addWidget(self.start_baseline_btn)
        
        self.start_low_load_btn = QPushButton("Start Low Load")
        self.start_low_load_btn.clicked.connect(self.on_start_low_load)
        self.start_low_load_btn.setEnabled(False)
        controls_layout.addWidget(self.start_low_load_btn)
        
        self.start_high_load_btn = QPushButton("Start High Load")
        self.start_high_load_btn.clicked.connect(self.on_start_high_load)
        self.start_high_load_btn.setEnabled(False)
        controls_layout.addWidget(self.start_high_load_btn)
        
        self.save_data_btn = QPushButton("Save Data")
        self.save_data_btn.clicked.connect(self.on_save_data)
        self.save_data_btn.setEnabled(False)
        controls_layout.addWidget(self.save_data_btn)
        
        controls_layout.addStretch()
        
        # Status and timer
        self.status_label = QLabel("Status: Disconnected")
        self.status_label.setStyleSheet("color: #ff4444; font-weight: bold; padding: 5px;")
        controls_layout.addWidget(self.status_label)
        
        self.timer_label = QLabel("Time: --:--")
        self.timer_label.setStyleSheet("color: #00ff88; font-weight: bold; padding: 5px;")
        controls_layout.addWidget(self.timer_label)
        
        main_layout.addLayout(controls_layout)
        
        # Main horizontal layout
        content_layout = QHBoxLayout()
        
        # Left panel: Signal plots
        left_panel = QVBoxLayout()
        
        # Create scroll area for individual plots
        from PyQt5.QtWidgets import QScrollArea, QWidget as QW
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background-color: #0d1117; border: none;")
        scroll.setMinimumHeight(400)
        
        # Container widget for plots
        plots_container = QW()
        plots_layout = QVBoxLayout()
        plots_layout.setSpacing(5)  # Space between plots
        plots_layout.setContentsMargins(5, 5, 5, 5)
        plots_container.setLayout(plots_layout)
        
        # Create 8 individual plots, one per channel
        # Reinitialize lists if they already exist
        if hasattr(self, 'raw_plot_widgets'):
            self.raw_plot_widgets.clear()
        else:
            self.raw_plot_widgets = []
            
        if hasattr(self, 'raw_curves'):
            self.raw_curves.clear()
        else:
            self.raw_curves = []
        
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#f9ca24', 
                  '#6c5ce7', '#a29bfe', '#fd79a8', '#00b894']
        # Channel mapping: 0=Fp1, 1=Fp2, 2=F3, 3=Fz, 4=F4, 5=P3, 6=Pz, 7=P4
        channel_names = ['Channel 0 (Fp1)', 'Channel 1 (Fp2)', 'Channel 2 (F3)', 'Channel 3 (Fz)',
                        'Channel 4 (F4)', 'Channel 5 (P3)', 'Channel 6 (Pz)', 'Channel 7 (P4)']
        
        for i in range(8):
            plot_widget = pg.PlotWidget(title=f"{channel_names[i]}")
            plot_widget.setBackground('#0d1117')
            plot_widget.setLabel('left', 'Amplitude (μV)')
            plot_widget.setLabel('bottom', 'Time (s)')
            plot_widget.showGrid(x=True, y=True, alpha=0.3)
            # Limited auto-range for better performance
            # Don't use continuous enableAutoRange, better to update manually when needed
            plot_widget.setMinimumHeight(120)
            plot_widget.setMaximumHeight(150)
            # Reasonable initial range
            plot_widget.setYRange(-200, 200)
            
            # Initialize with empty data to avoid errors
            curve = plot_widget.plot([], [], pen=pg.mkPen(color=colors[i], width=2))
            self.raw_curves.append(curve)
            self.raw_plot_widgets.append(plot_widget)
            plots_layout.addWidget(plot_widget)
        
        scroll.setWidget(plots_container)
        left_panel.addWidget(scroll)
        
        # Cognitive load ratio plot
        self.ratio_plot_widget = pg.PlotWidget(title="Cognitive Load Ratio (Theta_Fz / Alpha_Pz)")
        self.ratio_plot_widget.setBackground('#0d1117')
        self.ratio_plot_widget.setLabel('left', 'Ratio')
        self.ratio_plot_widget.setLabel('bottom', 'Time (s)')
        self.ratio_plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.ratio_curve = self.ratio_plot_widget.plot(pen=pg.mkPen(color='#00ff88', width=2))
        
        left_panel.addWidget(self.ratio_plot_widget)
        
        # Right panel: Task/instructions area
        right_panel = QVBoxLayout()
        
        # Instructions
        self.instruction_label = QLabel("Connect the AURA device to begin")
        self.instruction_label.setAlignment(Qt.AlignCenter)
        instruction_font = QFont()
        instruction_font.setPointSize(14)
        instruction_font.setBold(True)
        self.instruction_label.setFont(instruction_font)
        self.instruction_label.setStyleSheet(
            "background-color: #161b22; "
            "border: 2px solid #30363d; "
            "border-radius: 6px; "
            "padding: 20px; "
            "color: #00ff88; "
            "min-height: 60px;"
        )
        right_panel.addWidget(self.instruction_label)
        
        # Central area (changes according to phase)
        self.central_area = QFrame()
        self.central_area.setStyleSheet("""
            QFrame {
                background-color: #161b22;
                border: 2px solid #30363d;
                border-radius: 6px;
                padding: 20px;
            }
        """)
        self.central_layout = QVBoxLayout()
        self.central_area.setLayout(self.central_layout)
        
        # Initial widget (placeholder)
        self.current_task_widget = QLabel("Waiting for experiment to start...")
        self.current_task_widget.setAlignment(Qt.AlignCenter)
        self.current_task_widget.setStyleSheet("color: #888888; padding: 40px;")
        self.central_layout.addWidget(self.current_task_widget)
        
        right_panel.addWidget(self.central_area)
        
        # Add panels to main layout
        content_layout.addLayout(left_panel, 2)  # 60% of space
        content_layout.addLayout(right_panel, 1)  # 40% of space
        
        main_layout.addLayout(content_layout)
    
    def on_connect_clicked(self):
        """Callback for connection button."""
        # This function will be connected from main.py
        pass
    
    def on_start_setup(self):
        """Callback to start Setup."""
        # This function will be connected from main.py
        pass
    
    def on_start_baseline(self):
        """Callback to start Baseline."""
        # This function will be connected from main.py
        pass
    
    def on_start_low_load(self):
        """Callback to start Low Load."""
        # This function will be connected from main.py
        pass
    
    def on_start_high_load(self):
        """Callback to start High Load."""
        # This function will be connected from main.py
        pass
    
    def on_save_data(self):
        """Callback to save data."""
        # This function will be connected from main.py
        pass
    
    @pyqtSlot(np.ndarray, float)
    def update_raw_plot(self, data, timestamp):
        """
        Updates the raw signal plot.
        With subsampling to avoid saturation.
        
        Args:
            data: Array with 8 values (one per channel)
            timestamp: Sample timestamp
        """
        # DEBUG: Imprimir datos recibidos en UI (solo las primeras 5 veces)
        if not hasattr(self, '_ui_debug_counter'):
            self._ui_debug_counter = 0
        if self._ui_debug_counter < 5:
            print(f"\n[UI - update_raw_plot {self._ui_debug_counter}]")
            print(f"  Tipo de data: {type(data)}")
            print(f"  Data shape: {data.shape if hasattr(data, 'shape') else 'N/A'}")
            print(f"  Data length: {len(data) if hasattr(data, '__len__') else 'N/A'}")
            print(f"  Data valores (8 canales): {data}")
            print(f"  Valores por canal:")
            for ch in range(min(8, len(data))):
                print(f"    Canal {ch}: {data[ch]:.2f}")
            print(f"  Timestamp: {timestamp}")
            self._ui_debug_counter += 1
        
        # Subsampling: update every N samples (increased for better performance)
        self.plot_update_counter += 1
        if self.plot_update_counter % self.plot_update_skip != 0:
            return
        
        # Add to buffers
        for i in range(min(8, len(data))):
            self.raw_data_buffer[i].append(data[i])
        self.timestamp_buffer.append(timestamp)
        
        if len(self.timestamp_buffer) < 2:
            return
        
        # Convert to numpy arrays and normalize times
        times = np.array(self.timestamp_buffer)
        if len(times) > 1:
            # Normalize to most recent time (last timestamp)
            times = times - times[-1]  # Now the most recent time is 0
            # Invert so time goes from negative (past) to 0 (present)
            times = -times  # Now goes from negative to 0, where 0 is the present
        
        # Update curves (only if there's enough new data)
        # Verify that curves are initialized
        if len(self.raw_curves) == 0:
            return
        
        # Limit update frequency per channel (maximum every 200ms)
        current_time = time.time()
        
        for i, curve in enumerate(self.raw_curves):
            if i >= len(self.raw_data_buffer) or len(self.raw_data_buffer[i]) == 0:
                continue
            
            # Throttling: update each channel maximum every 200ms
            if i in self.last_update_time:
                if current_time - self.last_update_time[i] < 0.2:
                    continue  # Skip this update for this channel
            
            try:
                # Convert to microvolts
                # AURA values come in nanovolts (nV), we need to divide by 1000 for microvolts (μV)
                raw_values = np.array(self.raw_data_buffer[i])
                
                # Scale from nanovolts to microvolts
                # AURA sends data in nanovolts, so we divide by 1000
                values_microvolts = raw_values / 1000.0
                
                # No offset - each plot has its own scale
                values = values_microvolts
                
                # Ensure times and values have the same length
                if len(times) != len(values):
                    min_len = min(len(times), len(values))
                    times_plot = times[:min_len]
                    values_plot = values[:min_len]
                else:
                    times_plot = times
                    values_plot = values
                
                # Update curve only if there's valid data
                if len(times_plot) > 0 and len(values_plot) > 0:
                    # Update data
                    curve.setData(times_plot, values_plot)
                    
                    # Update Y range only occasionally (every 1 second) for better performance
                    if i not in self.last_update_time or (current_time - self.last_update_time[i]) >= 1.0:
                        if len(values_plot) > 10:
                            y_min = np.min(values_plot)
                            y_max = np.max(values_plot)
                            y_range = y_max - y_min
                            if y_range > 0:
                                # Add 20% margin
                                margin = y_range * 0.2
                                self.raw_plot_widgets[i].setYRange(y_min - margin, y_max + margin)
                    
                    # Record update time
                    self.last_update_time[i] = current_time
                    
            except Exception as e:
                print(f"Error updating plot channel {i}: {str(e)}")
                continue
    
    @pyqtSlot(float, float, float)
    def update_ratio_plot(self, ratio, theta_power, alpha_power):
        """
        Updates the cognitive load ratio plot.
        
        Args:
            ratio: Theta_Fz / Alpha_Pz ratio
            theta_power: Power in Theta band
            alpha_power: Power in Alpha band
        """
        current_time = time.time()
        self.ratio_buffer.append(ratio)
        self.ratio_timestamps.append(current_time)
        
        if len(self.ratio_timestamps) < 2:
            return
        
        times = np.array(self.ratio_timestamps)
        if len(times) > 1:
            times = times - times[-1]  # Normalizar
        
        ratios = np.array(self.ratio_buffer)
        self.ratio_curve.setData(times, ratios)
    
    def show_instructions(self, text):
        """Shows instructions in the central area."""
        self.clear_central_area()
        label = QLabel(text)
        label.setAlignment(Qt.AlignCenter)
        label.setWordWrap(True)
        label.setStyleSheet("color: #c9d1d9; padding: 40px; font-size: 16px;")
        self.central_layout.addWidget(label)
        self.current_task_widget = label
    
    def show_text_reading(self, text):
        """Shows text for passive reading (Low Load)."""
        self.clear_central_area()
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setPlainText(text)
        text_edit.setStyleSheet("""
            QTextEdit {
                background-color: #0d1117;
                color: #c9d1d9;
                font-size: 28px;
                line-height: 2.2;
                padding: 40px;
            }
        """)
        # Increase font size programmatically as well
        font = QFont()
        font.setPointSize(28)
        text_edit.setFont(font)
        self.central_layout.addWidget(text_edit)
        self.current_task_widget = text_edit
    
    def show_nback_task(self):
        """Shows the N-Back task (High Load)."""
        self.clear_central_area()
        nback_widget = NBackTask(n_level=2)
        nback_widget.setFocusPolicy(Qt.StrongFocus)
        self.central_layout.addWidget(nback_widget)
        self.current_task_widget = nback_widget
        # Ensure widget receives focus
        QTimer.singleShot(100, lambda: nback_widget.setFocus())
        return nback_widget
    
    def show_stroop_task(self):
        """Shows the Stroop task (High Cognitive Load)."""
        self.clear_central_area()
        stroop_widget = StroopTask()
        stroop_widget.setFocusPolicy(Qt.StrongFocus)
        self.central_layout.addWidget(stroop_widget)
        self.current_task_widget = stroop_widget
        # Ensure widget receives focus
        QTimer.singleShot(100, lambda: stroop_widget.setFocus())
        return stroop_widget
    
    def clear_central_area(self):
        """Clears the central area."""
        while self.central_layout.count():
            child = self.central_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
    
    def update_status(self, message, color="#c9d1d9"):
        """Updates the status label."""
        self.status_label.setText(f"Status: {message}")
        self.status_label.setStyleSheet(f"color: {color}; font-weight: bold; padding: 5px;")
    
    def update_timer(self, seconds):
        """Updates the timer label."""
        minutes = seconds // 60
        secs = seconds % 60
        self.timer_label.setText(f"Time: {minutes:02d}:{secs:02d}")

