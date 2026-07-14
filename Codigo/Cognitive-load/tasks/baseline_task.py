"""
baseline_task.py
Widget for baseline recording task.
Phase 1: Eyes open (1.5 minutes) - staring at a fixation point
Phase 2: Eyes closed (1.5 minutes) - eyes closed
"""

import time
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QPainter, QColor, QPen, QBrush


class BaselineTask(QWidget):
    """
    Widget for baseline recording task.
    Two phases: Eyes Open (1.5 min) and Eyes Closed (1.5 min)
    """
    
    phase_changed_signal = pyqtSignal(str)  # phase name: 'eyes_open' or 'eyes_closed'
    task_complete_signal = pyqtSignal()
    
    def __init__(self, duration_seconds=90):
        super().__init__()
        self.duration_seconds = duration_seconds  # 90 seconds = 1.5 minutes
        self.start_time = None
        self.is_running = False
        self.current_phase = None  # 'eyes_open' or 'eyes_closed'
        
        # Timers
        self.phase_timer = QTimer()
        self.phase_timer.timeout.connect(self.on_phase_timeout)
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_display)
        self.update_timer.start(100)  # Update every 100ms
        
        self.init_ui()
    
    def init_ui(self):
        """Initializes the baseline task interface."""
        layout = QVBoxLayout()
        layout.setSpacing(30)
        layout.setContentsMargins(40, 40, 40, 40)
        
        # Title
        title = QLabel("Baseline Recording")
        title.setAlignment(Qt.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(24)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setStyleSheet("color: #00ff88; margin: 20px;")
        layout.addWidget(title)
        
        # Phase label
        self.phase_label = QLabel("Ready to start")
        self.phase_label.setAlignment(Qt.AlignCenter)
        phase_font = QFont()
        phase_font.setPointSize(20)
        phase_font.setBold(True)
        self.phase_label.setFont(phase_font)
        self.phase_label.setStyleSheet("color: #ffaa00; margin: 20px;")
        layout.addWidget(self.phase_label)
        
        # Instructions
        self.instructions_label = QLabel("")
        self.instructions_label.setAlignment(Qt.AlignCenter)
        instructions_font = QFont()
        instructions_font.setPointSize(18)
        self.instructions_label.setFont(instructions_font)
        self.instructions_label.setStyleSheet("color: #cccccc; margin: 20px;")
        layout.addWidget(self.instructions_label)
        
        # Timer label
        self.timer_label = QLabel("Time: 01:30")
        self.timer_label.setAlignment(Qt.AlignCenter)
        timer_font = QFont()
        timer_font.setPointSize(28)
        timer_font.setBold(True)
        self.timer_label.setFont(timer_font)
        self.timer_label.setStyleSheet("color: #00ff88; margin: 20px;")
        layout.addWidget(self.timer_label)
        
        # Fixation point widget (for eyes open phase)
        self.fixation_widget = FixationPointWidget()
        self.fixation_widget.setMinimumHeight(400)
        self.fixation_widget.setStyleSheet(
            "background-color: #0d1117; "
            "border: 3px solid #30363d; "
            "border-radius: 15px;"
        )
        layout.addWidget(self.fixation_widget, stretch=1)
        
        # Status label
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignCenter)
        status_font = QFont()
        status_font.setPointSize(16)
        self.status_label.setFont(status_font)
        self.status_label.setStyleSheet("color: #8b949e; margin: 10px;")
        layout.addWidget(self.status_label)
        
        # Start button
        self.start_btn = QPushButton("Start Baseline Recording")
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
    
    def start_task(self):
        """Starts the baseline recording task."""
        if self.is_running:
            return
        
        self.is_running = True
        self.start_btn.setEnabled(False)
        self.start_btn.setText("Recording in progress...")
        
        # Start with eyes open phase
        self.start_eyes_open_phase()
    
    def start_eyes_open_phase(self):
        """Starts the eyes open phase."""
        self.current_phase = 'eyes_open'
        self.start_time = time.time()
        
        self.phase_label.setText("Phase 1: Eyes Open")
        self.phase_label.setStyleSheet("color: #00ff88; margin: 20px; font-size: 20px; font-weight: bold;")
        
        self.instructions_label.setText(
            "Keep your eyes open and stare at the fixation point.\n"
            "Try to remain relaxed and avoid blinking excessively."
        )
        
        self.fixation_widget.set_visible(True)
        self.fixation_widget.update()
        
        self.status_label.setText("Recording baseline with eyes open...")
        
        # Emit phase change signal
        self.phase_changed_signal.emit('eyes_open')
        
        # Start phase timer
        self.phase_timer.start(self.duration_seconds * 1000)  # Convert to milliseconds
    
    def start_eyes_closed_phase(self):
        """Starts the eyes closed phase."""
        self.current_phase = 'eyes_closed'
        self.start_time = time.time()
        
        self.phase_label.setText("Phase 2: Eyes Closed")
        self.phase_label.setStyleSheet("color: #ffaa00; margin: 20px; font-size: 20px; font-weight: bold;")
        
        self.instructions_label.setText(
            "Close your eyes and remain relaxed.\n"
            "Keep your eyes closed until the recording is complete."
        )
        
        self.fixation_widget.set_visible(False)
        self.fixation_widget.update()
        
        self.status_label.setText("Recording baseline with eyes closed...")
        
        # Emit phase change signal
        self.phase_changed_signal.emit('eyes_closed')
        
        # Start phase timer
        self.phase_timer.start(self.duration_seconds * 1000)
    
    def on_phase_timeout(self):
        """Called when a phase duration expires."""
        self.phase_timer.stop()
        
        if self.current_phase == 'eyes_open':
            # Transition to eyes closed phase
            self.status_label.setText("Transitioning to eyes closed phase...")
            # Keep task running during transition
            QTimer.singleShot(2000, self.start_eyes_closed_phase)  # 2 second transition
        
        elif self.current_phase == 'eyes_closed':
            # Task complete - small delay before ending to ensure all data is logged
            QTimer.singleShot(500, self.end_task)
    
    def update_display(self):
        """Updates the timer display."""
        if not self.is_running or self.start_time is None:
            return
        
        elapsed_seconds = int(time.time() - self.start_time)
        remaining = max(0, self.duration_seconds - elapsed_seconds)
        
        minutes = remaining // 60
        seconds = remaining % 60
        self.timer_label.setText(f"Time: {minutes:02d}:{seconds:02d}")
    
    def end_task(self):
        """Ends the baseline recording task."""
        self.is_running = False
        self.phase_timer.stop()
        self.current_phase = None
        
        self.phase_label.setText("Baseline Recording Complete")
        self.phase_label.setStyleSheet("color: #00ff88; margin: 20px; font-size: 20px; font-weight: bold;")
        
        self.instructions_label.setText("You can now open your eyes.")
        
        self.fixation_widget.set_visible(False)
        self.fixation_widget.update()
        
        self.status_label.setText("Recording completed successfully!")
        self.status_label.setStyleSheet("color: #00ff88; margin: 10px; font-size: 16px;")
        
        self.timer_label.setText("00:00")
        
        self.start_btn.setEnabled(True)
        self.start_btn.setText("Baseline Complete - Restart?")
        
        # Emit completion signal
        self.task_complete_signal.emit()
    
    def get_current_phase(self):
        """Returns the current phase name."""
        return self.current_phase


class FixationPointWidget(QWidget):
    """Custom widget for drawing a fixation point."""
    
    def __init__(self):
        super().__init__()
        self.visible = False
    
    def set_visible(self, visible):
        """Sets the visibility of the fixation point."""
        self.visible = visible
        self.update()
    
    def paintEvent(self, event):
        """Paints the fixation point."""
        if not self.visible:
            return
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Get widget dimensions
        width = self.width()
        height = self.height()
        center_x = width // 2
        center_y = height // 2
        
        # Draw outer circle (subtle)
        painter.setPen(QPen(QColor(100, 100, 100), 2))
        painter.setBrush(Qt.NoBrush)
        outer_radius = 30
        painter.drawEllipse(center_x - outer_radius, center_y - outer_radius,
                          outer_radius * 2, outer_radius * 2)
        
        # Draw inner circle (fixation point)
        painter.setPen(QPen(QColor(255, 255, 255), 3))
        painter.setBrush(QBrush(QColor(255, 255, 255)))
        inner_radius = 8
        painter.drawEllipse(center_x - inner_radius, center_y - inner_radius,
                          inner_radius * 2, inner_radius * 2)
        
        # Draw crosshair lines (subtle)
        painter.setPen(QPen(QColor(80, 80, 80), 1))
        line_length = 50
        # Horizontal line
        painter.drawLine(center_x - line_length, center_y,
                        center_x + line_length, center_y)
        # Vertical line
        painter.drawLine(center_x, center_y - line_length,
                        center_x, center_y + line_length)
