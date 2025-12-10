"""
experiment_logic.py
Máquina de estados para el protocolo experimental de carga cognitiva.
Maneja las fases: Setup, Baseline, Baja Carga, Alta Carga.
"""

from enum import Enum
from PyQt5.QtCore import QObject, pyqtSignal, QTimer
import time


class ExperimentPhase(Enum):
    """Fases del protocolo experimental."""
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
    Lógica del experimento. Controla los timers y transiciones entre fases.
    """
    
    # Señales para comunicación con la UI
    phase_changed = pyqtSignal(str, str)  # fase, mensaje
    timer_update = pyqtSignal(int)  # segundos restantes
    instruction_update = pyqtSignal(str)  # instrucción actual
    
    def __init__(self):
        super().__init__()
        self.current_phase = ExperimentPhase.IDLE
        self.timer = QTimer()
        self.timer.timeout.connect(self._on_timer_tick)
        self.time_remaining = 0
        
        # Duraciones de las fases (en segundos)
        self.durations = {
            ExperimentPhase.BASELINE_EYES_OPEN: 90,  # 90 segundos
            ExperimentPhase.BASELINE_EYES_CLOSED: 90,  # 90 segundos
            ExperimentPhase.LOW_LOAD: 180,  # 3 minutos
            ExperimentPhase.HIGH_LOAD: 180,  # 3 minutos
        }
        
        # Instrucciones para cada fase
        self.instructions = {
            ExperimentPhase.SETUP: "Verifique la calidad de la señal EEG. Asegúrese de que todos los canales muestren actividad.",
            ExperimentPhase.BASELINE_EYES_OPEN: "Mantenga los ojos abiertos y relájese. Mire fijamente el punto central.",
            ExperimentPhase.BASELINE_EYES_CLOSED: "Cierre los ojos y relájese completamente.",
            ExperimentPhase.LOW_LOAD: "Lea el texto que aparece en pantalla de manera pasiva. No necesita hacer nada más.",
            ExperimentPhase.HIGH_LOAD: "Realice la tarea N-Back. Presione la tecla cuando el estímulo coincida con el de N posiciones atrás.",
            ExperimentPhase.ANALYSIS: "Análisis de datos en curso...",
            ExperimentPhase.COMPLETED: "Experimento completado. Puede cerrar la aplicación."
        }
    
    def start_setup(self):
        """Inicia la fase de Setup."""
        self.current_phase = ExperimentPhase.SETUP
        self.phase_changed.emit(self.current_phase.value, self.instructions[self.current_phase])
        self.instruction_update.emit(self.instructions[self.current_phase])
    
    def start_baseline(self):
        """Inicia la fase Baseline (primero ojos abiertos)."""
        self.current_phase = ExperimentPhase.BASELINE_EYES_OPEN
        self.time_remaining = self.durations[self.current_phase]
        self.phase_changed.emit(self.current_phase.value, self.instructions[self.current_phase])
        self.instruction_update.emit(self.instructions[self.current_phase])
        self.timer.start(1000)  # Timer cada 1 segundo
        self.timer_update.emit(self.time_remaining)
    
    def start_low_load(self):
        """Inicia la fase de Baja Carga Cognitiva."""
        self.timer.stop()
        self.current_phase = ExperimentPhase.LOW_LOAD
        self.time_remaining = self.durations[self.current_phase]
        self.phase_changed.emit(self.current_phase.value, self.instructions[self.current_phase])
        self.instruction_update.emit(self.instructions[self.current_phase])
        self.timer.start(1000)
        self.timer_update.emit(self.time_remaining)
    
    def start_high_load(self):
        """Inicia la fase de Alta Carga Cognitiva (N-Back)."""
        self.timer.stop()
        self.current_phase = ExperimentPhase.HIGH_LOAD
        self.time_remaining = self.durations[self.current_phase]
        self.phase_changed.emit(self.current_phase.value, self.instructions[self.current_phase])
        self.instruction_update.emit(self.instructions[self.current_phase])
        self.timer.start(1000)
        self.timer_update.emit(self.time_remaining)
    
    def start_analysis(self):
        """Inicia la fase de Análisis."""
        self.timer.stop()
        self.current_phase = ExperimentPhase.ANALYSIS
        self.phase_changed.emit(self.current_phase.value, self.instructions[self.current_phase])
        self.instruction_update.emit(self.instructions[self.current_phase])
    
    def complete_experiment(self):
        """Marca el experimento como completado."""
        self.timer.stop()
        self.current_phase = ExperimentPhase.COMPLETED
        self.phase_changed.emit(self.current_phase.value, self.instructions[self.current_phase])
        self.instruction_update.emit(self.instructions[self.current_phase])
    
    def _on_timer_tick(self):
        """Callback del timer. Se ejecuta cada segundo."""
        if self.time_remaining > 0:
            self.time_remaining -= 1
            self.timer_update.emit(self.time_remaining)
            
            # Transiciones automáticas
            if self.current_phase == ExperimentPhase.BASELINE_EYES_OPEN and self.time_remaining == 0:
                # Transición a ojos cerrados
                self.current_phase = ExperimentPhase.BASELINE_EYES_CLOSED
                self.time_remaining = self.durations[self.current_phase]
                self.phase_changed.emit(self.current_phase.value, self.instructions[self.current_phase])
                self.instruction_update.emit(self.instructions[self.current_phase])
                self.timer_update.emit(self.time_remaining)
            
            elif self.current_phase == ExperimentPhase.BASELINE_EYES_CLOSED and self.time_remaining == 0:
                # Baseline completado, esperar inicio manual de siguiente fase
                self.timer.stop()
                self.phase_changed.emit("baseline_completed", "Baseline completado. Presione 'Iniciar Baja Carga' para continuar.")
            
            elif self.current_phase == ExperimentPhase.LOW_LOAD and self.time_remaining == 0:
                # Baja carga completada, esperar inicio manual de siguiente fase
                self.timer.stop()
                self.phase_changed.emit("low_load_completed", "Baja Carga completada. Presione 'Iniciar Alta Carga' para continuar.")
            
            elif self.current_phase == ExperimentPhase.HIGH_LOAD and self.time_remaining == 0:
                # Alta carga completada
                self.timer.stop()
                self.start_analysis()
    
    def get_current_phase(self):
        """Retorna la fase actual del experimento."""
        return self.current_phase
    
    def pause(self):
        """Pausa el timer del experimento."""
        self.timer.stop()
    
    def resume(self):
        """Reanuda el timer del experimento."""
        if self.time_remaining > 0:
            self.timer.start(1000)
    
    def reset(self):
        """Reinicia el experimento a la fase IDLE."""
        self.timer.stop()
        self.current_phase = ExperimentPhase.IDLE
        self.time_remaining = 0
        self.phase_changed.emit(self.current_phase.value, "Experimento reiniciado")

