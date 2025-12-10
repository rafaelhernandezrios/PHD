"""
ui_main.py
Interfaz gráfica principal de la plataforma de experimentación EEG.
Utiliza PyQt6 con diseño oscuro tipo dashboard científico.
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
    Widget para la tarea Stroop (Alta Carga Cognitiva).
    El usuario debe identificar el COLOR de la tinta, no la palabra escrita.
    """
    
    response_signal = pyqtSignal(bool)  # True si respuesta correcta, False si incorrecta
    
    def __init__(self):
        super().__init__()
        self.colors = {
            'ROJO': '#ff4444',
            'AZUL': '#4444ff',
            'VERDE': '#44ff44',
            'AMARILLO': '#ffff44'
        }
        self.color_keys = {
            Qt.Key_R: 'ROJO',
            Qt.Key_A: 'AZUL',
            Qt.Key_V: 'VERDE',
            Qt.Key_Y: 'AMARILLO'
        }
        self.color_names_spanish = {
            'ROJO': 'R',
            'AZUL': 'A',
            'VERDE': 'V',
            'AMARILLO': 'Y'
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
        """Inicializa la interfaz de la tarea Stroop."""
        layout = QVBoxLayout()
        
        # Título
        title = QLabel("Tarea Stroop")
        title.setAlignment(Qt.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(22)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setStyleSheet("color: #00ff88; margin: 10px;")
        layout.addWidget(title)
        
        # Instrucciones
        instructions = QLabel(
            "Presione la tecla del COLOR de la tinta, NO la palabra escrita:\n"
            "R = Rojo | A = Azul | V = Verde | Y = Amarillo"
        )
        instructions.setAlignment(Qt.AlignCenter)
        instructions_font = QFont()
        instructions_font.setPointSize(18)
        instructions.setFont(instructions_font)
        instructions.setStyleSheet("color: #cccccc; margin: 10px; font-size: 18px;")
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
        
        # Estadísticas
        stats_layout = QVBoxLayout()
        self.stats_label = QLabel("Aciertos: 0 / Respuestas: 0")
        self.stats_label.setAlignment(Qt.AlignCenter)
        stats_font = QFont()
        stats_font.setPointSize(16)
        self.stats_label.setFont(stats_font)
        self.stats_label.setStyleSheet("color: #888888;")
        stats_layout.addWidget(self.stats_label)
        
        self.congruency_label = QLabel("Congruentes: 0 | Incongruentes: 0")
        self.congruency_label.setAlignment(Qt.AlignCenter)
        self.congruency_label.setFont(stats_font)
        self.congruency_label.setStyleSheet("color: #888888;")
        stats_layout.addWidget(self.congruency_label)
        
        layout.addLayout(stats_layout)
        layout.addStretch()
        self.setLayout(layout)
    
    def start_stimulus_timer(self):
        """Inicia el timer para cambio automático de estímulos."""
        self.stimulus_timer.start(self.stimulus_interval)
    
    def stop_stimulus_timer(self):
        """Detiene el timer de estímulos."""
        self.stimulus_timer.stop()
    
    def on_stimulus_timeout(self):
        """Callback cuando el timer de estímulo expira."""
        if self.waiting_for_response:
            # Timeout - no respuesta
            self.check_response(None)
    
    def generate_stimulus(self):
        """Genera un nuevo estímulo Stroop."""
        # Detener timer mientras se procesa
        self.stop_stimulus_timer()
        self.waiting_for_response = False
        
        # Decidir si será congruente o incongruente (70% incongruentes para más carga)
        self.is_congruent = np.random.random() < 0.3  # 30% congruentes, 70% incongruentes
        
        # Seleccionar palabra y color
        word_options = list(self.colors.keys())
        self.current_word = np.random.choice(word_options)
        
        if self.is_congruent:
            # Congruente: palabra y color coinciden
            self.current_color = self.current_word
            self.congruent_count += 1
        else:
            # Incongruente: palabra y color NO coinciden
            color_options = [c for c in word_options if c != self.current_word]
            self.current_color = np.random.choice(color_options)
            self.incongruent_count += 1
        
        # Mostrar estímulo con el color correspondiente
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
        
        # Limpiar feedback anterior
        self.feedback_label.setText("")
        
        # Reiniciar el timer
        self.start_stimulus_timer()
    
    def check_response(self, pressed_key):
        """
        Verifica si la respuesta del usuario fue correcta.
        
        Args:
            pressed_key: Qt.Key de la tecla presionada, o None si fue timeout
        """
        # Detener el timer ya que se procesó una respuesta
        self.stop_stimulus_timer()
        self.waiting_for_response = False
        
        if pressed_key is None:
            # Timeout - no respuesta
            self.feedback_label.setText("⏱ Sin respuesta")
            self.feedback_label.setStyleSheet("color: #ffaa00; margin: 10px; font-size: 20px;")
            self.total_responses += 1
            self.stats_label.setText(f"Aciertos: {self.correct_count} / Respuestas: {self.total_responses}")
            # Generar nuevo estímulo después de timeout
            QTimer.singleShot(500, self.generate_stimulus)
            return
        
        # Verificar si la tecla presionada corresponde al color correcto
        if pressed_key in self.color_keys:
            selected_color = self.color_keys[pressed_key]
            is_correct = selected_color == self.current_color
            
            if is_correct:
                self.correct_count += 1
                self.feedback_label.setText("✓ Correcto")
                self.feedback_label.setStyleSheet("color: #00ff88; margin: 10px; font-size: 20px;")
                self.response_signal.emit(True)
            else:
                self.feedback_label.setText(f"✗ Incorrecto (Era {self.color_names_spanish[self.current_color]})")
                self.feedback_label.setStyleSheet("color: #ff4444; margin: 10px; font-size: 20px;")
                self.response_signal.emit(False)
            
            self.total_responses += 1
            self.stats_label.setText(f"Aciertos: {self.correct_count} / Respuestas: {self.total_responses}")
            self.congruency_label.setText(
                f"Congruentes: {self.congruent_count} | Incongruentes: {self.incongruent_count}"
            )
            
            # Generar nuevo estímulo después de mostrar feedback
            QTimer.singleShot(500, self.generate_stimulus)
        else:
            # Tecla no válida, ignorar
            pass
    
    def keyPressEvent(self, event: QKeyEvent):
        """Maneja las teclas presionadas durante la tarea."""
        if self.waiting_for_response:
            if event.key() in self.color_keys:
                # Detener el timer y procesar respuesta inmediatamente
                self.stop_stimulus_timer()
                self.check_response(event.key())
        else:
            super().keyPressEvent(event)
    
    def focusInEvent(self, event):
        """Asegura que el widget reciba eventos de teclado."""
        self.setFocus()
        super().focusInEvent(event)


class NBackTask(QWidget):
    """
    Widget para la tarea N-Back (Alta Carga Cognitiva).
    Implementa una versión visual de N-Back.
    """
    
    response_signal = pyqtSignal(bool)  # True si respuesta correcta, False si incorrecta
    
    def __init__(self, n_level=2):
        super().__init__()
        self.n_level = n_level  # N-Back level (ej: 2-Back)
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
        """Inicializa la interfaz de la tarea N-Back."""
        layout = QVBoxLayout()
        
        # Título
        title = QLabel(f"Tarea {self.n_level}-Back")
        title.setAlignment(Qt.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(22)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setStyleSheet("color: #00ff88; margin: 10px;")
        layout.addWidget(title)
        
        # Instrucciones
        instructions = QLabel(
            f"Presione ESPACIO cuando el número coincida con el de {self.n_level} posiciones atrás"
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
        
        # Estadísticas
        stats_layout = QHBoxLayout()
        self.stats_label = QLabel("Aciertos: 0 / Respuestas: 0")
        self.stats_label.setAlignment(Qt.AlignCenter)
        self.stats_label.setStyleSheet("color: #888888;")
        stats_layout.addWidget(self.stats_label)
        layout.addLayout(stats_layout)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def start_stimulus_timer(self):
        """Inicia el timer para cambio automático de estímulos."""
        self.stimulus_timer.start(self.stimulus_interval)
    
    def stop_stimulus_timer(self):
        """Detiene el timer de estímulos."""
        self.stimulus_timer.stop()
    
    def on_stimulus_timeout(self):
        """Callback cuando el timer de estímulo expira."""
        # Si no se presionó espacio, se considera como "no respuesta"
        if len(self.stimulus_history) >= self.n_level:
            self.check_response(False)  # No se presionó espacio
        else:
            # Si aún no hay suficientes estímulos, solo generar el siguiente
            # sin evaluar respuesta
            self.generate_stimulus()
    
    def generate_stimulus(self):
        """Genera un nuevo estímulo (número del 1 al 9)."""
        # Detener timer mientras se procesa
        self.stop_stimulus_timer()
        
        # Añadir el estímulo actual al historial antes de generar uno nuevo
        if self.current_stimulus is not None:
            self.stimulus_history.append(self.current_stimulus)
        
        # Generar nuevo estímulo
        self.current_stimulus = np.random.randint(1, 10)
        self.stimulus_label.setText(str(self.current_stimulus))
        self.trial_count += 1
        
        # Limpiar feedback anterior
        self.feedback_label.setText("")
        
        # IMPORTANTE: Reiniciar el timer para que el siguiente estímulo aparezca automáticamente
        self.start_stimulus_timer()
    
    def check_response(self, responded):
        """
        Verifica si la respuesta del usuario fue correcta.
        
        Args:
            responded: True si el usuario presionó espacio, False si no
        """
        # Detener el timer ya que se procesó una respuesta
        self.stop_stimulus_timer()
        
        # Necesitamos al menos n_level estímulos en el historial para comparar
        if len(self.stimulus_history) < self.n_level:
            # Aún no hay suficientes estímulos, solo continuar
            return
        
        # Verificar si el estímulo actual coincide con el de n_level posiciones atrás
        expected_response = False
        if len(self.stimulus_history) >= self.n_level:
            expected_response = self.stimulus_history[-self.n_level] == self.current_stimulus
        
        # Evaluar respuesta
        if responded == expected_response:
            self.correct_count += 1
            self.feedback_label.setText("✓ Correcto")
            self.feedback_label.setStyleSheet("color: #00ff88; margin: 10px;")
            self.response_signal.emit(True)
        else:
            if responded:
                # Solo mostrar "Incorrecto" si presionó espacio (no si fue timeout)
                self.feedback_label.setText("✗ Incorrecto")
                self.feedback_label.setStyleSheet("color: #ff4444; margin: 10px;")
            else:
                # Timeout - no respuesta
                self.feedback_label.setText("⏱ Sin respuesta")
                self.feedback_label.setStyleSheet("color: #ffaa00; margin: 10px;")
            self.response_signal.emit(False)
        
        self.total_responses += 1
        self.stats_label.setText(f"Aciertos: {self.correct_count} / Respuestas: {self.total_responses}")
        
        # Generar nuevo estímulo después de mostrar feedback (500ms de delay)
        # generate_stimulus ya reinicia el timer automáticamente
        QTimer.singleShot(500, self.generate_stimulus)
    
    def keyPressEvent(self, event: QKeyEvent):
        """Maneja las teclas presionadas durante la tarea."""
        if event.key() == Qt.Key_Space:
            # Detener el timer y procesar respuesta inmediatamente
            self.stop_stimulus_timer()
            self.check_response(True)
        else:
            super().keyPressEvent(event)
    
    def focusInEvent(self, event):
        """Asegura que el widget reciba eventos de teclado."""
        self.setFocus()
        super().focusInEvent(event)


class MainWindow(QMainWindow):
    """
    Ventana principal de la aplicación.
    Dashboard científico con visualización en tiempo real.
    """
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Plataforma de Experimentación EEG - Carga Cognitiva")
        self.setGeometry(100, 100, 1400, 900)
        
        # Estilo oscuro
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
        
        # Buffers para gráficos (reducidos para mejor rendimiento)
        self.plot_buffer_size = 750  # ~3 segundos a 250 Hz (reducido para mejor rendimiento)
        self.raw_data_buffer = {i: deque(maxlen=self.plot_buffer_size) for i in range(8)}
        self.timestamp_buffer = deque(maxlen=self.plot_buffer_size)
        self.ratio_buffer = deque(maxlen=300)  # Buffer más pequeño para ratio
        self.ratio_timestamps = deque(maxlen=300)
        
        # Control de actualización de gráficos
        self.plot_update_counter = 0
        self.plot_update_skip = 5  # Actualizar cada 5 muestras recibidas (reducido de 2)
        self.last_update_time = {}  # Para limitar frecuencia de actualización por canal
        
        # Inicializar lista de widgets de gráficos (se llenará en init_ui)
        self.raw_plot_widgets = []
        
        self.init_ui()
    
    def init_ui(self):
        """Inicializa todos los componentes de la interfaz."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Barra superior con controles
        controls_layout = QHBoxLayout()
        
        self.connect_btn = QPushButton("Conectar AURA")
        self.connect_btn.clicked.connect(self.on_connect_clicked)
        controls_layout.addWidget(self.connect_btn)
        
        self.start_setup_btn = QPushButton("Iniciar Setup")
        self.start_setup_btn.clicked.connect(self.on_start_setup)
        self.start_setup_btn.setEnabled(False)
        controls_layout.addWidget(self.start_setup_btn)
        
        self.start_baseline_btn = QPushButton("Iniciar Baseline")
        self.start_baseline_btn.clicked.connect(self.on_start_baseline)
        self.start_baseline_btn.setEnabled(False)
        controls_layout.addWidget(self.start_baseline_btn)
        
        self.start_low_load_btn = QPushButton("Iniciar Baja Carga")
        self.start_low_load_btn.clicked.connect(self.on_start_low_load)
        self.start_low_load_btn.setEnabled(False)
        controls_layout.addWidget(self.start_low_load_btn)
        
        self.start_high_load_btn = QPushButton("Iniciar Alta Carga")
        self.start_high_load_btn.clicked.connect(self.on_start_high_load)
        self.start_high_load_btn.setEnabled(False)
        controls_layout.addWidget(self.start_high_load_btn)
        
        self.save_data_btn = QPushButton("Guardar Datos")
        self.save_data_btn.clicked.connect(self.on_save_data)
        self.save_data_btn.setEnabled(False)
        controls_layout.addWidget(self.save_data_btn)
        
        controls_layout.addStretch()
        
        # Estado y timer
        self.status_label = QLabel("Estado: Desconectado")
        self.status_label.setStyleSheet("color: #ff4444; font-weight: bold; padding: 5px;")
        controls_layout.addWidget(self.status_label)
        
        self.timer_label = QLabel("Tiempo: --:--")
        self.timer_label.setStyleSheet("color: #00ff88; font-weight: bold; padding: 5px;")
        controls_layout.addWidget(self.timer_label)
        
        main_layout.addLayout(controls_layout)
        
        # Layout horizontal principal
        content_layout = QHBoxLayout()
        
        # Panel izquierdo: Gráficos de señal
        left_panel = QVBoxLayout()
        
        # Crear scroll area para los gráficos individuales
        from PyQt5.QtWidgets import QScrollArea, QWidget as QW
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background-color: #0d1117; border: none;")
        scroll.setMinimumHeight(400)
        
        # Widget contenedor para los gráficos
        plots_container = QW()
        plots_layout = QVBoxLayout()
        plots_layout.setSpacing(5)  # Espacio entre gráficos
        plots_layout.setContentsMargins(5, 5, 5, 5)
        plots_container.setLayout(plots_layout)
        
        # Crear 8 gráficos individuales, uno por canal
        # Reinicializar listas si ya existen
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
        channel_names = ['Canal 0 (Fz)', 'Canal 1', 'Canal 2', 'Canal 3',
                        'Canal 4 (Pz)', 'Canal 5', 'Canal 6', 'Canal 7']
        
        for i in range(8):
            plot_widget = pg.PlotWidget(title=f"{channel_names[i]}")
            plot_widget.setBackground('#0d1117')
            plot_widget.setLabel('left', 'Amplitud (μV)')
            plot_widget.setLabel('bottom', 'Tiempo (s)')
            plot_widget.showGrid(x=True, y=True, alpha=0.3)
            # Auto-range limitado para mejor rendimiento
            # No usar enableAutoRange continuo, mejor actualizar manualmente cuando sea necesario
            plot_widget.setMinimumHeight(120)
            plot_widget.setMaximumHeight(150)
            # Rango inicial razonable
            plot_widget.setYRange(-200, 200)
            
            # Inicializar con datos vacíos para evitar errores
            curve = plot_widget.plot([], [], pen=pg.mkPen(color=colors[i], width=2))
            self.raw_curves.append(curve)
            self.raw_plot_widgets.append(plot_widget)
            plots_layout.addWidget(plot_widget)
        
        scroll.setWidget(plots_container)
        left_panel.addWidget(scroll)
        
        # Gráfico de ratio de carga cognitiva
        self.ratio_plot_widget = pg.PlotWidget(title="Ratio de Carga Cognitiva (Theta_Fz / Alpha_Pz)")
        self.ratio_plot_widget.setBackground('#0d1117')
        self.ratio_plot_widget.setLabel('left', 'Ratio')
        self.ratio_plot_widget.setLabel('bottom', 'Tiempo (s)')
        self.ratio_plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.ratio_curve = self.ratio_plot_widget.plot(pen=pg.mkPen(color='#00ff88', width=2))
        
        left_panel.addWidget(self.ratio_plot_widget)
        
        # Panel derecho: Área de tarea/instrucciones
        right_panel = QVBoxLayout()
        
        # Instrucciones
        self.instruction_label = QLabel("Conecte el dispositivo AURA para comenzar")
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
        
        # Área central (cambia según la fase)
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
        
        # Widget inicial (placeholder)
        self.current_task_widget = QLabel("Esperando inicio del experimento...")
        self.current_task_widget.setAlignment(Qt.AlignCenter)
        self.current_task_widget.setStyleSheet("color: #888888; padding: 40px;")
        self.central_layout.addWidget(self.current_task_widget)
        
        right_panel.addWidget(self.central_area)
        
        # Agregar paneles al layout principal
        content_layout.addLayout(left_panel, 2)  # 60% del espacio
        content_layout.addLayout(right_panel, 1)  # 40% del espacio
        
        main_layout.addLayout(content_layout)
    
    def on_connect_clicked(self):
        """Callback para el botón de conexión."""
        # Esta función será conectada desde main.py
        pass
    
    def on_start_setup(self):
        """Callback para iniciar Setup."""
        # Esta función será conectada desde main.py
        pass
    
    def on_start_baseline(self):
        """Callback para iniciar Baseline."""
        # Esta función será conectada desde main.py
        pass
    
    def on_start_low_load(self):
        """Callback para iniciar Baja Carga."""
        # Esta función será conectada desde main.py
        pass
    
    def on_start_high_load(self):
        """Callback para iniciar Alta Carga."""
        # Esta función será conectada desde main.py
        pass
    
    def on_save_data(self):
        """Callback para guardar datos."""
        # Esta función será conectada desde main.py
        pass
    
    @pyqtSlot(np.ndarray, float)
    def update_raw_plot(self, data, timestamp):
        """
        Actualiza el gráfico de señales raw.
        Con submuestreo para evitar saturación.
        
        Args:
            data: Array con 8 valores (uno por canal)
            timestamp: Timestamp de la muestra
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
        
        # Submuestreo: actualizar cada N muestras (aumentado para mejor rendimiento)
        self.plot_update_counter += 1
        if self.plot_update_counter % self.plot_update_skip != 0:
            return
        
        # Añadir a buffers
        for i in range(min(8, len(data))):
            self.raw_data_buffer[i].append(data[i])
        self.timestamp_buffer.append(timestamp)
        
        if len(self.timestamp_buffer) < 2:
            return
        
        # Convertir a arrays numpy y normalizar tiempos
        times = np.array(self.timestamp_buffer)
        if len(times) > 1:
            # Normalizar al tiempo más reciente (último timestamp)
            times = times - times[-1]  # Ahora el tiempo más reciente es 0
            # Invertir para que el tiempo vaya de negativo (pasado) a 0 (presente)
            times = -times  # Ahora va de negativo a 0, donde 0 es el presente
        
        # Actualizar curvas (solo si hay suficientes datos nuevos)
        # Verificar que las curvas estén inicializadas
        if len(self.raw_curves) == 0:
            return
        
        # Limitar frecuencia de actualización por canal (cada 200ms máximo)
        current_time = time.time()
        
        for i, curve in enumerate(self.raw_curves):
            if i >= len(self.raw_data_buffer) or len(self.raw_data_buffer[i]) == 0:
                continue
            
            # Throttling: actualizar cada canal máximo cada 200ms
            if i in self.last_update_time:
                if current_time - self.last_update_time[i] < 0.2:
                    continue  # Saltar esta actualización para este canal
            
            try:
                # Convertir a microvolts
                # Los valores de AURA vienen en nanovolts (nV), necesitamos dividir por 1000 para microvolts (μV)
                raw_values = np.array(self.raw_data_buffer[i])
                
                # Escalar de nanovolts a microvolts
                # AURA envía datos en nanovolts, así que dividimos por 1000
                values_microvolts = raw_values / 1000.0
                
                # Sin offset - cada gráfico tiene su propia escala
                values = values_microvolts
                
                # Asegurar que times y values tengan la misma longitud
                if len(times) != len(values):
                    min_len = min(len(times), len(values))
                    times_plot = times[:min_len]
                    values_plot = values[:min_len]
                else:
                    times_plot = times
                    values_plot = values
                
                # Actualizar curva solo si hay datos válidos
                if len(times_plot) > 0 and len(values_plot) > 0:
                    # Actualizar datos
                    curve.setData(times_plot, values_plot)
                    
                    # Actualizar rango Y solo ocasionalmente (cada 1 segundo) para mejor rendimiento
                    if i not in self.last_update_time or (current_time - self.last_update_time[i]) >= 1.0:
                        if len(values_plot) > 10:
                            y_min = np.min(values_plot)
                            y_max = np.max(values_plot)
                            y_range = y_max - y_min
                            if y_range > 0:
                                # Añadir margen del 20%
                                margin = y_range * 0.2
                                self.raw_plot_widgets[i].setYRange(y_min - margin, y_max + margin)
                    
                    # Registrar tiempo de actualización
                    self.last_update_time[i] = current_time
                    
            except Exception as e:
                print(f"Error actualizando gráfico canal {i}: {str(e)}")
                continue
    
    @pyqtSlot(float, float, float)
    def update_ratio_plot(self, ratio, theta_power, alpha_power):
        """
        Actualiza el gráfico del ratio de carga cognitiva.
        
        Args:
            ratio: Ratio Theta_Fz / Alpha_Pz
            theta_power: Potencia en banda Theta
            alpha_power: Potencia en banda Alpha
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
        """Muestra instrucciones en el área central."""
        self.clear_central_area()
        label = QLabel(text)
        label.setAlignment(Qt.AlignCenter)
        label.setWordWrap(True)
        label.setStyleSheet("color: #c9d1d9; padding: 40px; font-size: 16px;")
        self.central_layout.addWidget(label)
        self.current_task_widget = label
    
    def show_text_reading(self, text):
        """Muestra texto para lectura pasiva (Baja Carga)."""
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
        # Aumentar tamaño de fuente programáticamente también
        font = QFont()
        font.setPointSize(28)
        text_edit.setFont(font)
        self.central_layout.addWidget(text_edit)
        self.current_task_widget = text_edit
    
    def show_nback_task(self):
        """Muestra la tarea N-Back (Alta Carga)."""
        self.clear_central_area()
        nback_widget = NBackTask(n_level=2)
        nback_widget.setFocusPolicy(Qt.StrongFocus)
        self.central_layout.addWidget(nback_widget)
        self.current_task_widget = nback_widget
        # Asegurar que el widget reciba el foco
        QTimer.singleShot(100, lambda: nback_widget.setFocus())
        return nback_widget
    
    def show_stroop_task(self):
        """Muestra la tarea Stroop (Alta Carga Cognitiva)."""
        self.clear_central_area()
        stroop_widget = StroopTask()
        stroop_widget.setFocusPolicy(Qt.StrongFocus)
        self.central_layout.addWidget(stroop_widget)
        self.current_task_widget = stroop_widget
        # Asegurar que el widget reciba el foco
        QTimer.singleShot(100, lambda: stroop_widget.setFocus())
        return stroop_widget
    
    def clear_central_area(self):
        """Limpia el área central."""
        while self.central_layout.count():
            child = self.central_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
    
    def update_status(self, message, color="#c9d1d9"):
        """Actualiza el label de estado."""
        self.status_label.setText(f"Estado: {message}")
        self.status_label.setStyleSheet(f"color: {color}; font-weight: bold; padding: 5px;")
    
    def update_timer(self, seconds):
        """Actualiza el label del timer."""
        minutes = seconds // 60
        secs = seconds % 60
        self.timer_label.setText(f"Tiempo: {minutes:02d}:{secs:02d}")

