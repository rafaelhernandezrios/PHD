"""
main.py
Punto de entrada de la aplicación.
Conecta todos los módulos: SignalWorker, ExperimentLogic y MainWindow.
"""

import sys
import pandas as pd
import numpy as np
import os
from datetime import datetime
from PyQt5.QtWidgets import QApplication, QMessageBox, QInputDialog
from PyQt5.QtCore import QTimer

from signal_worker import SignalWorker
from experiment_logic import ExperimentLogic, ExperimentPhase
from ui_main import MainWindow


class EEGExperimentApp:
    """
    Clase principal que coordina todos los componentes de la aplicación.
    """
    
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.window = MainWindow()
        
        # Componentes principales
        self.signal_worker = SignalWorker(sample_rate=250, n_channels=8, buffer_duration=2.0)
        self.experiment_logic = ExperimentLogic()
        
        # Datos para logging
        self.data_log = []
        self.current_phase_name = "idle"
        self.is_logging = False
        self.current_user = None
        self.user_folder = None
        
        # Texto para la fase de baja carga
        self.low_load_text = """
        La neurotecnología es un campo interdisciplinario que combina neurociencia, 
        ingeniería y tecnología para desarrollar interfaces que conecten el cerebro 
        humano con sistemas computacionales. Las interfaces cerebro-computadora (BCI) 
        permiten la comunicación directa entre el cerebro y dispositivos externos, 
        abriendo nuevas posibilidades para la rehabilitación, el control de prótesis 
        y la mejora de capacidades cognitivas.
        
        Los sistemas BCI utilizan diversas modalidades de adquisición de señales 
        neuronales, incluyendo electroencefalografía (EEG), magnetoencefalografía (MEG), 
        y registros intracraneales. El EEG es particularmente atractivo debido a su 
        naturaleza no invasiva, bajo costo y alta resolución temporal, aunque presenta 
        limitaciones en la resolución espacial.
        
        El procesamiento de señales EEG requiere técnicas avanzadas de filtrado, 
        análisis espectral y clasificación de patrones. Los algoritmos de machine 
        learning, especialmente las redes neuronales profundas, han demostrado ser 
        efectivos para la decodificación de intenciones motoras y estados cognitivos 
        a partir de señales EEG.
        """
        
        # Conectar señales
        self._connect_signals()
        
        # Timer para actualizar gráficos (reducido para mejor rendimiento)
        self.plot_timer = QTimer()
        self.plot_timer.timeout.connect(self.update_plots)
        self.plot_timer.start(200)  # Actualizar cada 200ms (~5 FPS, reducido para evitar saturación)
        
        # Timer para calcular ratio
        self.ratio_timer = QTimer()
        self.ratio_timer.timeout.connect(self.calculate_and_update_ratio)
        self.ratio_timer.start(1000)  # Calcular ratio cada 1 segundo (reducido de 500ms)
    
    def _connect_signals(self):
        """Conecta todas las señales entre componentes."""
        
        # SignalWorker -> UI
        self.signal_worker.raw_data_ready.connect(self.window.update_raw_plot)
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
        """Maneja el clic en el botón de conexión."""
        if not self.signal_worker.isRunning():
            self.window.connect_btn.setText("Conectando...")
            self.window.connect_btn.setEnabled(False)
            self.signal_worker.connect_to_stream()
            if self.signal_worker.inlet:
                self.signal_worker.start()
                self.window.start_setup_btn.setEnabled(True)
        else:
            self.signal_worker.stop()
            self.signal_worker.wait()
            self.window.connect_btn.setText("Conectar AURA")
            self.window.start_setup_btn.setEnabled(False)
            self.window.start_baseline_btn.setEnabled(False)
            self.window.start_low_load_btn.setEnabled(False)
            self.window.start_high_load_btn.setEnabled(False)
    
    def on_connection_status(self, connected, message):
        """Maneja cambios en el estado de conexión."""
        if connected:
            self.window.connect_btn.setText("Desconectar")
            self.window.connect_btn.setEnabled(True)
            self.window.update_status("Conectado", "#00ff88")
        else:
            self.window.connect_btn.setText("Conectar AURA")
            self.window.connect_btn.setEnabled(True)
            self.window.update_status("Desconectado", "#ff4444")
            if message:
                QMessageBox.warning(self.window, "Error de Conexión", message)
    
    def on_start_setup(self):
        """Inicia la fase de Setup."""
        # Solicitar nombre de usuario si no se ha establecido
        if self.current_user is None:
            user_name, ok = QInputDialog.getText(
                self.window, 
                "Usuario del Experimento", 
                "Ingrese el nombre o ID del usuario:"
            )
            if not ok or not user_name.strip():
                QMessageBox.warning(self.window, "Usuario Requerido", 
                                  "Debe ingresar un nombre de usuario para continuar.")
                return
            
            self.current_user = user_name.strip()
            # Crear carpeta para el usuario
            self.user_folder = f"data_{self.current_user}"
            os.makedirs(self.user_folder, exist_ok=True)
            self.window.update_status(f"Usuario: {self.current_user}", "#00ff88")
        
        self.experiment_logic.start_setup()
        self.current_phase_name = "setup"
        self.window.show_instructions(
            "Verifique la calidad de la señal EEG en los gráficos. "
            "Asegúrese de que todos los canales muestren actividad sin artefactos excesivos. "
            "Cuando esté listo, presione 'Iniciar Baseline'."
        )
        self.window.start_baseline_btn.setEnabled(True)
        self.is_logging = True
    
    def on_start_baseline(self):
        """Inicia la fase Baseline."""
        self.experiment_logic.start_baseline()
        self.current_phase_name = "baseline_eyes_open"
        self.window.show_instructions(
            "Mantenga los ojos abiertos y relájese. "
            "Mire fijamente el punto central de la pantalla."
        )
        self.window.start_baseline_btn.setEnabled(False)
    
    def on_start_low_load(self):
        """Inicia la fase de Baja Carga Cognitiva."""
        self.experiment_logic.start_low_load()
        self.current_phase_name = "low_load"
        self.window.show_text_reading(self.low_load_text)
        self.window.start_low_load_btn.setEnabled(False)
        self.window.start_high_load_btn.setEnabled(True)
    
    def on_start_high_load(self):
        """Inicia la fase de Alta Carga Cognitiva (Stroop)."""
        self.experiment_logic.start_high_load()
        self.current_phase_name = "high_load"
        stroop_widget = self.window.show_stroop_task()
        # Conectar señal de respuesta del Stroop si es necesario
        self.window.start_high_load_btn.setEnabled(False)
        self.window.save_data_btn.setEnabled(True)
    
    def on_phase_changed(self, phase, message):
        """Maneja cambios de fase del experimento."""
        if phase == "baseline_eyes_closed":
            self.current_phase_name = "baseline_eyes_closed"
            self.window.show_instructions(
                "Cierre los ojos y relájese completamente. "
                "No se mueva y mantenga una respiración tranquila."
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
                "Análisis de datos en curso. "
                "Los resultados se están procesando."
            )
        elif phase == "completed":
            self.current_phase_name = "completed"
            self.window.show_instructions(
                "Experimento completado exitosamente. "
                "Puede guardar los datos y cerrar la aplicación."
            )
    
    def log_data_sample(self, data, timestamp):
        """
        Registra una muestra de datos para posterior guardado.
        Submuestreo para evitar saturación de memoria.
        
        Args:
            data: Array con datos filtrados de los 8 canales
            timestamp: Timestamp de la muestra
        """
        if not self.is_logging:
            return
        
        # Submuestreo: guardar cada 5 muestras (~50 Hz en lugar de 250 Hz)
        # Esto reduce significativamente el uso de memoria
        if not hasattr(self, '_log_counter'):
            self._log_counter = 0
        
        self._log_counter += 1
        if self._log_counter % 5 != 0:
            return
        
        # Mapeo de fases a labels más descriptivos
        phase_labels = {
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
        
        # Crear registro con todos los canales y metadatos
        record = {
            'timestamp': timestamp,
            'phase': self.current_phase_name,
            'label': phase_labels.get(self.current_phase_name, self.current_phase_name),
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
    
    def calculate_and_update_ratio(self):
        """Calcula y actualiza el ratio de carga cognitiva."""
        if not self.signal_worker.isRunning():
            return
        result = self.signal_worker.get_cognitive_load_ratio()
        if result is not None:
            ratio, theta_power, alpha_power = result
            self.window.update_ratio_plot(ratio, theta_power, alpha_power)
    
    def update_plots(self):
        """Actualiza los gráficos (llamado por el timer)."""
        # Los gráficos se actualizan automáticamente vía señales
        # Esta función puede usarse para actualizaciones adicionales si es necesario
        pass
    
    def on_save_data(self):
        """Guarda los datos registrados en un archivo CSV."""
        if not self.data_log:
            QMessageBox.warning(self.window, "Sin Datos", "No hay datos para guardar.")
            return
        
        if self.current_user is None:
            QMessageBox.warning(self.window, "Usuario Requerido", 
                              "No se ha establecido un usuario. Los datos no se guardarán.")
            return
        
        try:
            # Crear DataFrame
            df = pd.DataFrame(self.data_log)
            
            # Generar nombre de archivo con timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"eeg_data_{timestamp}.csv"
            
            # Guardar en la carpeta del usuario
            filepath = os.path.join(self.user_folder, filename)
            df.to_csv(filepath, index=False)
            
            QMessageBox.information(
                self.window, 
                "Datos Guardados", 
                f"Los datos se han guardado exitosamente en:\n{filepath}"
            )
            
        except Exception as e:
            QMessageBox.critical(
                self.window, 
                "Error al Guardar", 
                f"Error al guardar los datos:\n{str(e)}"
            )
    
    def run(self):
        """Ejecuta la aplicación."""
        self.window.show()
        return self.app.exec()


def main():
    """Función principal."""
    app = EEGExperimentApp()
    sys.exit(app.run())


if __name__ == "__main__":
    main()

