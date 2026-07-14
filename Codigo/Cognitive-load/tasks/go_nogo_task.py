"""
go_nogo_task.py
Widget for the Go/No-Go task.
The user must press SPACE only when a "Go" stimulus (green circle) appears.
"""

import random
import time
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QKeyEvent, QPainter, QColor, QPen, QBrush


class GoNoGoTask(QWidget):
    """
    Widget for the Go/No-Go task.
    Go stimuli (green circles): User must press SPACE
    No-Go stimuli (red squares): User must NOT press SPACE
    """
    
    response_signal = pyqtSignal(str, bool, float)  # stimulus_type, is_correct, reaction_time
    trial_complete_signal = pyqtSignal(int, int, int, int)  # go_correct, go_incorrect, no_go_correct, no_go_incorrect
    
    def __init__(self, duration_seconds=120):
        super().__init__()
        self.duration_seconds = duration_seconds
        self.start_time = None
        self.is_running = False
        
        # Task parameters
        self.current_stimulus = None  # 'go' or 'no_go'
        self.stimulus_start_time = None
        self.waiting_for_response = False
        self.max_response_time = 1500  # Maximum time to respond (ms)
        
        # Statistics
        self.go_correct = 0
        self.go_incorrect = 0
        self.no_go_correct = 0
        self.no_go_incorrect = 0
        self.total_trials = 0
        
        # Timers
        self.stimulus_timer = QTimer()
        self.stimulus_timer.timeout.connect(self.on_stimulus_timeout)
        self.response_timer = QTimer()
        self.response_timer.setSingleShot(True)
        self.response_timer.timeout.connect(self.on_response_timeout)
        self.experiment_timer = QTimer()
        self.experiment_timer.timeout.connect(self.check_experiment_duration)
        
        # Stimulus intervals (variable, 1-2 seconds)
        self.min_interval = 1000  # ms
        self.max_interval = 2000  # ms
        
        # Go/No-Go ratio (70% Go, 30% No-Go)
        self.go_probability = 0.7
        
        self.init_ui()
    
    def init_ui(self):
        """Initializes the Go/No-Go task interface."""
        layout = QVBoxLayout()
        layout.setSpacing(20)
        layout.setContentsMargins(40, 40, 40, 40)
        
        # Title
        title = QLabel("Go/No-Go Task")
        title.setAlignment(Qt.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(24)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setStyleSheet("color: #00ff88; margin: 20px;")
        layout.addWidget(title)
        
        # Instructions
        instructions = QLabel(
            "Press SPACE when you see a GREEN CIRCLE (Go)\n"
            "Do NOT press SPACE when you see a RED SQUARE (No-Go)"
        )
        instructions.setAlignment(Qt.AlignCenter)
        instructions_font = QFont()
        instructions_font.setPointSize(18)
        instructions.setFont(instructions_font)
        instructions.setStyleSheet("color: #cccccc; margin: 20px;")
        layout.addWidget(instructions)
        
        # Timer label
        self.timer_label = QLabel("Time: 02:00")
        self.timer_label.setAlignment(Qt.AlignCenter)
        timer_font = QFont()
        timer_font.setPointSize(20)
        timer_font.setBold(True)
        self.timer_label.setFont(timer_font)
        self.timer_label.setStyleSheet("color: #ffaa00; margin: 10px;")
        layout.addWidget(self.timer_label)
        
        # Stimulus area (custom painted)
        self.stimulus_widget = StimulusWidget()
        self.stimulus_widget.setMinimumHeight(300)
        self.stimulus_widget.setStyleSheet(
            "background-color: #0d1117; "
            "border: 3px solid #30363d; "
            "border-radius: 15px;"
        )
        layout.addWidget(self.stimulus_widget, stretch=1)
        
        # Feedback label
        self.feedback_label = QLabel("")
        self.feedback_label.setAlignment(Qt.AlignCenter)
        feedback_font = QFont()
        feedback_font.setPointSize(22)
        feedback_font.setBold(True)
        self.feedback_label.setFont(feedback_font)
        self.feedback_label.setStyleSheet("color: #ffaa00; margin: 10px; min-height: 40px;")
        layout.addWidget(self.feedback_label)
        
        # Statistics
        stats_layout = QHBoxLayout()
        stats_layout.setSpacing(30)
        
        self.stats_label = QLabel("Go Correct: 0 | Go Incorrect: 0\nNo-Go Correct: 0 | No-Go Incorrect: 0")
        self.stats_label.setAlignment(Qt.AlignCenter)
        stats_font = QFont()
        stats_font.setPointSize(14)
        self.stats_label.setFont(stats_font)
        self.stats_label.setStyleSheet("color: #8b949e; margin: 10px;")
        stats_layout.addWidget(self.stats_label)
        
        layout.addLayout(stats_layout)
        
        # Start button
        self.start_btn = QPushButton("Start Task")
        self.start_btn.setMinimumHeight(50)
        start_font = QFont()
        start_font.setPointSize(16)
        start_font.setBold(True)
        self.start_btn.setFont(start_font)
        self.start_btn.setStyleSheet(
            "QPushButton {"
            "background-color: #238636; "
            "color: white; "
            "border: none; "
            "border-radius: 5px; "
            "padding: 10px;"
            "}"
            "QPushButton:hover {"
            "background-color: #2ea043;"
            "}"
            "QPushButton:pressed {"
            "background-color: #1e6f2a;"
            "}"
        )
        self.start_btn.clicked.connect(self.start_task)
        layout.addWidget(self.start_btn)
        
        self.setLayout(layout)
        self.setFocusPolicy(Qt.StrongFocus)
    
    def start_task(self):
        """Starts the Go/No-Go task."""
        if self.is_running:
            return
        
        self.is_running = True
        self.start_time = None  # Will be set when first stimulus appears
        self.start_btn.setEnabled(False)
        self.start_btn.setText("Task Running...")
        self.feedback_label.setText("Get ready...")
        
        # Start experiment timer (check every 100ms)
        self.experiment_timer.start(100)
        
        # Generate first stimulus after a short delay
        QTimer.singleShot(1000, self.generate_stimulus)
    
    def check_experiment_duration(self):
        """Checks if the experiment duration has been reached."""
        if self.start_time is None:
            return
        
        elapsed_seconds = int(time.time() - self.start_time)
        
        # Update timer display
        remaining = max(0, self.duration_seconds - elapsed_seconds)
        minutes = remaining // 60
        seconds = remaining % 60
        self.timer_label.setText(f"Time: {minutes:02d}:{seconds:02d}")
        
        if remaining <= 0:
            self.end_task()
    
    def generate_stimulus(self):
        """Generates a new stimulus (Go or No-Go)."""
        if not self.is_running:
            return
        
        # Check if experiment time is up
        if hasattr(self, '_elapsed_seconds') and self._elapsed_seconds >= self.duration_seconds:
            self.end_task()
            return
        
        # Set start time on first stimulus
        if self.start_time is None:
            self.start_time = time.time()
        
        # Determine stimulus type
        if random.random() < self.go_probability:
            self.current_stimulus = 'go'
        else:
            self.current_stimulus = 'no_go'
        
        # Update stimulus widget
        self.stimulus_widget.set_stimulus(self.current_stimulus)
        self.stimulus_widget.update()
        
        # Record stimulus start time
        self.stimulus_start_time = time.time()
        self.waiting_for_response = True
        
        # Start response timeout timer
        self.response_timer.start(self.max_response_time)
        
        # Schedule next stimulus
        interval = random.randint(self.min_interval, self.max_interval)
        self.stimulus_timer.start(interval)
    
    def on_stimulus_timeout(self):
        """Called when the stimulus display time expires."""
        self.stimulus_timer.stop()
        # If no response was given, check if it was correct
        if self.waiting_for_response:
            if self.current_stimulus == 'no_go':
                # No response to No-Go is correct
                self.record_response(True, 0.0)
            else:
                # No response to Go is incorrect
                self.record_response(False, self.max_response_time / 1000.0)
        
        # Clear stimulus and generate next one
        self.stimulus_widget.clear_stimulus()
        self.stimulus_widget.update()
        self.waiting_for_response = False
        self.response_timer.stop()
        
        # Small delay before next stimulus
        QTimer.singleShot(300, self.generate_stimulus)
    
    def on_response_timeout(self):
        """Called when the maximum response time expires."""
        if self.waiting_for_response:
            if self.current_stimulus == 'no_go':
                # No response to No-Go is correct
                self.record_response(True, self.max_response_time / 1000.0)
            else:
                # No response to Go is incorrect
                self.record_response(False, self.max_response_time / 1000.0)
            
            self.stimulus_timer.stop()
            self.stimulus_widget.clear_stimulus()
            self.stimulus_widget.update()
            self.waiting_for_response = False
            QTimer.singleShot(300, self.generate_stimulus)
    
    def record_response(self, is_correct, reaction_time):
        """Records a response and updates statistics."""
        self.total_trials += 1
        
        if self.current_stimulus == 'go':
            if is_correct:
                self.go_correct += 1
                self.feedback_label.setText("✓ Correct!")
                self.feedback_label.setStyleSheet("color: #00ff88; margin: 10px; font-size: 22px; font-weight: bold;")
            else:
                self.go_incorrect += 1
                self.feedback_label.setText("✗ Missed!")
                self.feedback_label.setStyleSheet("color: #ff4444; margin: 10px; font-size: 22px; font-weight: bold;")
        else:  # no_go
            if is_correct:
                self.no_go_correct += 1
                self.feedback_label.setText("✓ Correct!")
                self.feedback_label.setStyleSheet("color: #00ff88; margin: 10px; font-size: 22px; font-weight: bold;")
            else:
                self.no_go_incorrect += 1
                self.feedback_label.setText("✗ False Alarm!")
                self.feedback_label.setStyleSheet("color: #ff4444; margin: 10px; font-size: 22px; font-weight: bold;")
        
        # Update statistics display
        self.stats_label.setText(
            f"Go Correct: {self.go_correct} | Go Incorrect: {self.go_incorrect}\n"
            f"No-Go Correct: {self.no_go_correct} | No-Go Incorrect: {self.no_go_incorrect}"
        )
        
        # Emit signals
        stimulus_label = f"{self.current_stimulus}_correct" if is_correct else f"{self.current_stimulus}_incorrect"
        self.response_signal.emit(stimulus_label, is_correct, reaction_time)
        self.trial_complete_signal.emit(
            self.go_correct, self.go_incorrect, 
            self.no_go_correct, self.no_go_incorrect
        )
    
    def keyPressEvent(self, event: QKeyEvent):
        """Handles key presses during the task."""
        if not self.is_running or not self.waiting_for_response:
            return
        
        if event.key() == Qt.Key_Space:
            # Stop timers
            self.stimulus_timer.stop()
            self.response_timer.stop()
            
            # Calculate reaction time
            reaction_time = time.time() - self.stimulus_start_time
            
            # Check if response is correct
            if self.current_stimulus == 'go':
                is_correct = True
            else:  # no_go
                is_correct = False
            
            # Record response
            self.record_response(is_correct, reaction_time)
            
            # Clear stimulus
            self.stimulus_widget.clear_stimulus()
            self.stimulus_widget.update()
            self.waiting_for_response = False
            
            # Generate next stimulus after delay
            QTimer.singleShot(300, self.generate_stimulus)
        else:
            super().keyPressEvent(event)
    
    def end_task(self):
        """Ends the Go/No-Go task."""
        self.is_running = False
        self.stimulus_timer.stop()
        self.response_timer.stop()
        self.experiment_timer.stop()
        
        self.stimulus_widget.clear_stimulus()
        self.stimulus_widget.update()
        
        self.feedback_label.setText("Task Completed!")
        self.feedback_label.setStyleSheet("color: #00ff88; margin: 10px; font-size: 22px; font-weight: bold;")
        
        self.start_btn.setEnabled(True)
        self.start_btn.setText("Task Completed - Restart?")
        
        # Emit final statistics
        self.trial_complete_signal.emit(
            self.go_correct, self.go_incorrect,
            self.no_go_correct, self.no_go_incorrect
        )
    
    def focusInEvent(self, event):
        """Ensures the widget receives keyboard events."""
        self.setFocus()
        super().focusInEvent(event)


class StimulusWidget(QWidget):
    """Custom widget for drawing Go/No-Go stimuli."""
    
    def __init__(self):
        super().__init__()
        self.stimulus_type = None  # 'go', 'no_go', or None
    
    def set_stimulus(self, stimulus_type):
        """Sets the current stimulus type."""
        self.stimulus_type = stimulus_type
        self.update()
    
    def clear_stimulus(self):
        """Clears the stimulus."""
        self.stimulus_type = None
        self.update()
    
    def paintEvent(self, event):
        """Paints the stimulus."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Get widget dimensions
        width = self.width()
        height = self.height()
        center_x = width // 2
        center_y = height // 2
        
        if self.stimulus_type == 'go':
            # Draw green circle
            radius = min(width, height) // 4
            painter.setBrush(QBrush(QColor(0, 255, 136)))  # Green
            painter.setPen(QPen(QColor(0, 200, 100), 3))
            painter.drawEllipse(center_x - radius, center_y - radius, 
                              radius * 2, radius * 2)
        
        elif self.stimulus_type == 'no_go':
            # Draw red square
            size = min(width, height) // 3
            painter.setBrush(QBrush(QColor(255, 68, 68)))  # Red
            painter.setPen(QPen(QColor(200, 0, 0), 3))
            painter.drawRect(center_x - size // 2, center_y - size // 2,
                           size, size)
        
        # If no stimulus, draw nothing (just background)
