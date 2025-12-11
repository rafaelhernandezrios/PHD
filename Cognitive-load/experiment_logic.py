"""
experiment_logic.py
State machine for the cognitive load experimental protocol.
Handles phases: Setup, Baseline, Low Load, High Load.
"""

from enum import Enum
from PyQt5.QtCore import QObject, pyqtSignal, QTimer
import time


class ExperimentPhase(Enum):
    """Phases of the experimental protocol."""
    IDLE = "idle"
    SETUP = "setup"
    BASELINE_EYES_OPEN = "baseline_eyes_open"
    BASELINE_EYES_CLOSED = "baseline_eyes_closed"
    LOW_LOAD = "low_load"
    HIGH_LOAD = "high_load"
    ANALYSIS = "analysis"
    COMPLETED = "completed"


class ExperimentLogic(QObject):
    """
    Experiment logic. Controls timers and transitions between phases.
    """
    
    # Signals for communication with UI
    phase_changed = pyqtSignal(str, str)  # phase, message
    timer_update = pyqtSignal(int)  # remaining seconds
    instruction_update = pyqtSignal(str)  # current instruction
    
    def __init__(self):
        super().__init__()
        self.current_phase = ExperimentPhase.IDLE
        self.timer = QTimer()
        self.timer.timeout.connect(self._on_timer_tick)
        self.time_remaining = 0
        
        # Phase durations (in seconds)
        self.durations = {
            ExperimentPhase.BASELINE_EYES_OPEN: 90,  # 90 seconds
            ExperimentPhase.BASELINE_EYES_CLOSED: 90,  # 90 seconds
            ExperimentPhase.LOW_LOAD: 180,  # 3 minutes
            ExperimentPhase.HIGH_LOAD: 180,  # 3 minutes
        }
        
        # Instructions for each phase
        self.instructions = {
            ExperimentPhase.SETUP: "Check the EEG signal quality. Make sure all channels show activity.",
            ExperimentPhase.BASELINE_EYES_OPEN: "Keep your eyes open and relax. Stare at the center point.",
            ExperimentPhase.BASELINE_EYES_CLOSED: "Close your eyes and relax completely.",
            ExperimentPhase.LOW_LOAD: "Read the text on screen passively. You don't need to do anything else.",
            ExperimentPhase.HIGH_LOAD: "Perform the Stroop task. Press the key for the COLOR of the ink, not the written word.",
            ExperimentPhase.ANALYSIS: "Data analysis in progress...",
            ExperimentPhase.COMPLETED: "Experiment completed. You can close the application."
        }
    
    def start_setup(self):
        """Starts the Setup phase."""
        self.current_phase = ExperimentPhase.SETUP
        self.phase_changed.emit(self.current_phase.value, self.instructions[self.current_phase])
        self.instruction_update.emit(self.instructions[self.current_phase])
    
    def start_baseline(self):
        """Starts the Baseline phase (first eyes open)."""
        self.current_phase = ExperimentPhase.BASELINE_EYES_OPEN
        self.time_remaining = self.durations[self.current_phase]
        self.phase_changed.emit(self.current_phase.value, self.instructions[self.current_phase])
        self.instruction_update.emit(self.instructions[self.current_phase])
        self.timer.start(1000)  # Timer every 1 second
        self.timer_update.emit(self.time_remaining)
    
    def start_low_load(self):
        """Starts the Low Cognitive Load phase."""
        self.timer.stop()
        self.current_phase = ExperimentPhase.LOW_LOAD
        self.time_remaining = self.durations[self.current_phase]
        self.phase_changed.emit(self.current_phase.value, self.instructions[self.current_phase])
        self.instruction_update.emit(self.instructions[self.current_phase])
        self.timer.start(1000)
        self.timer_update.emit(self.time_remaining)
    
    def start_high_load(self):
        """Starts the High Cognitive Load phase (Stroop)."""
        self.timer.stop()
        self.current_phase = ExperimentPhase.HIGH_LOAD
        self.time_remaining = self.durations[self.current_phase]
        self.phase_changed.emit(self.current_phase.value, self.instructions[self.current_phase])
        self.instruction_update.emit(self.instructions[self.current_phase])
        self.timer.start(1000)
        self.timer_update.emit(self.time_remaining)
    
    def start_analysis(self):
        """Starts the Analysis phase."""
        self.timer.stop()
        self.current_phase = ExperimentPhase.ANALYSIS
        self.phase_changed.emit(self.current_phase.value, self.instructions[self.current_phase])
        self.instruction_update.emit(self.instructions[self.current_phase])
    
    def complete_experiment(self):
        """Marks the experiment as completed."""
        self.timer.stop()
        self.current_phase = ExperimentPhase.COMPLETED
        self.phase_changed.emit(self.current_phase.value, self.instructions[self.current_phase])
        self.instruction_update.emit(self.instructions[self.current_phase])
    
    def _on_timer_tick(self):
        """Timer callback. Executes every second."""
        if self.time_remaining > 0:
            self.time_remaining -= 1
            self.timer_update.emit(self.time_remaining)
            
            # Automatic transitions
            if self.current_phase == ExperimentPhase.BASELINE_EYES_OPEN and self.time_remaining == 0:
                # Transition to eyes closed
                self.current_phase = ExperimentPhase.BASELINE_EYES_CLOSED
                self.time_remaining = self.durations[self.current_phase]
                self.phase_changed.emit(self.current_phase.value, self.instructions[self.current_phase])
                self.instruction_update.emit(self.instructions[self.current_phase])
                self.timer_update.emit(self.time_remaining)
            
            elif self.current_phase == ExperimentPhase.BASELINE_EYES_CLOSED and self.time_remaining == 0:
                # Baseline completed, wait for manual start of next phase
                self.timer.stop()
                self.phase_changed.emit("baseline_completed", "Baseline completed. Press 'Start Low Load' to continue.")
            
            elif self.current_phase == ExperimentPhase.LOW_LOAD and self.time_remaining == 0:
                # Low load completed, wait for manual start of next phase
                self.timer.stop()
                self.phase_changed.emit("low_load_completed", "Low Load completed. Press 'Start High Load' to continue.")
            
            elif self.current_phase == ExperimentPhase.HIGH_LOAD and self.time_remaining == 0:
                # High load completed
                self.timer.stop()
                self.start_analysis()
    
    def get_current_phase(self):
        """Returns the current experiment phase."""
        return self.current_phase
    
    def pause(self):
        """Pauses the experiment timer."""
        self.timer.stop()
    
    def resume(self):
        """Resumes the experiment timer."""
        if self.time_remaining > 0:
            self.timer.start(1000)
    
    def reset(self):
        """Resets the experiment to IDLE phase."""
        self.timer.stop()
        self.current_phase = ExperimentPhase.IDLE
        self.time_remaining = 0
        self.phase_changed.emit(self.current_phase.value, "Experiment reset")

