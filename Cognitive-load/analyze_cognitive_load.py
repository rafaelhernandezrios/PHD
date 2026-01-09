"""
Script para analizar datos EEG y calcular el ratio de carga cognitiva.
Demuestra que durante high_load hay más carga cognitiva que en otras etapas.
"""

import pandas as pd
import numpy as np
from scipy import signal
from scipy import stats
from pathlib import Path

# Importar librerías de visualización opcionalmente
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("Advertencia: matplotlib/seaborn no estan instalados. Las visualizaciones se omitiran.")
    print("Warning: matplotlib/seaborn not installed. Visualizations will be skipped.")

# Importar PyWavelets para denoising opcionalmente
try:
    import pywt
    HAS_WAVELETS = True
except ImportError:
    HAS_WAVELETS = False
    print("Advertencia: PyWavelets (pywt) no esta instalado. El denoising por wavelets se omitira.")
    print("Warning: PyWavelets (pywt) not installed. Wavelet denoising will be skipped.")
    print("Instalar con: pip install PyWavelets")

# Configuración
SAMPLE_RATE = 250  # Hz (frecuencia de muestreo de AURA)
WINDOW_DURATION = 2.0  # segundos para análisis espectral
WINDOW_SAMPLES = int(WINDOW_DURATION * SAMPLE_RATE)  # 500 muestras
# Configuración de canales (configurable)
FZ_CHANNEL = 0  # Canal Fz (frontal)
PZ_CHANNEL = 3  # Canal Pz (parietal)
BAD_CHANNEL = 4  # Canal defectuoso que debe ser excluido del CAR
N_CHANNELS = 8  # Total de canales
THETA_BAND = (4.0, 7.0)  # Hz
ALPHA_BAND = (8.0, 12.0)  # Hz

# Configurar estilo de gráficos (si están disponibles)
if HAS_PLOTTING:
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (14, 8)
    plt.rcParams['font.size'] = 10


def filter_signal_temporal(signal_data, sample_rate, low_freq=0.5, high_freq=30.0):
    """
    Filtra la señal con bandpass (0.5-30 Hz) y notch (60 Hz).
    Usa 30 Hz como corte superior para reducir ruido muscular.
    
    Args:
        signal_data: Array 1D con datos de señal
        sample_rate: Frecuencia de muestreo
        low_freq: Frecuencia de corte inferior (default: 0.5 Hz)
        high_freq: Frecuencia de corte superior (default: 30.0 Hz)
        
    Returns:
        Señal filtrada (1D array)
    """
    if len(signal_data) < 3:
        return signal_data
    
    # Bandpass filter 0.5-30 Hz (Butterworth, order 4)
    nyquist = sample_rate / 2
    low = low_freq / nyquist
    high = high_freq / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    
    # Notch filter 60 Hz
    notch_freq = 60.0
    quality_factor = 30.0
    b_notch, a_notch = signal.iirnotch(notch_freq, quality_factor, sample_rate)
    
    # Aplicar filtros: notch primero, luego bandpass
    filtered = signal.filtfilt(b_notch, a_notch, signal_data)
    filtered = signal.filtfilt(b, a, filtered)
    
    return filtered


def apply_car(eeg_data, bad_channel_idx=BAD_CHANNEL):
    """
    Aplica Common Average Reference (CAR) a los datos EEG.
    Calcula el promedio de los canales válidos (excluyendo el canal defectuoso)
    y resta ese promedio a todos los canales.
    
    Args:
        eeg_data: Array 2D con forma (n_samples, n_channels) o (n_channels, n_samples)
        bad_channel_idx: Índice del canal defectuoso a excluir (default: 4)
        
    Returns:
        Datos EEG con CAR aplicado (misma forma que la entrada)
    """
    # Determinar orientación de los datos
    if eeg_data.shape[0] < eeg_data.shape[1]:
        # Forma (n_channels, n_samples) - transponer a (n_samples, n_channels)
        eeg_data = eeg_data.T
        transpose_back = True
    else:
        # Forma (n_samples, n_channels) - correcta
        transpose_back = False
    
    n_samples, n_channels = eeg_data.shape
    
    # Crear máscara de canales válidos (excluir canal defectuoso)
    valid_channels = [i for i in range(n_channels) if i != bad_channel_idx]
    
    if len(valid_channels) == 0:
        print("Advertencia: No hay canales validos para CAR. Retornando datos sin modificar.")
        return eeg_data.T if transpose_back else eeg_data
    
    # Calcular promedio de canales válidos
    car_reference = np.mean(eeg_data[:, valid_channels], axis=1, keepdims=True)
    
    # Restar el promedio a todos los canales
    eeg_data_car = eeg_data - car_reference
    
    # Restaurar orientación original si es necesario
    if transpose_back:
        eeg_data_car = eeg_data_car.T
    
    return eeg_data_car


def wavelet_denoise(signal_data, wavelet='db4', mode='soft', threshold_mode='sure'):
    """
    Aplica denoising por wavelets usando Universal Threshold (WAAF implementation).
    
    Args:
        signal_data: Array 1D con datos de señal
        wavelet: Tipo de wavelet (default: 'db4' - Daubechies 4)
        mode: Modo de thresholding ('soft' o 'hard', default: 'soft')
        threshold_mode: Método de selección de umbral ('sure' o 'universal', default: 'sure')
        
    Returns:
        Señal denoised
    """
    if not HAS_WAVELETS:
        return signal_data  # Retornar sin modificar si no hay PyWavelets
    
    if len(signal_data) < 8:  # Mínimo para wavelets
        return signal_data
    
    try:
        # Descomposición en nivel 4
        coeffs = pywt.wavedec(signal_data, wavelet, level=4)
        
        # Universal Threshold: sqrt(2 * log(N)) * sigma
        # donde sigma se estima usando la mediana de los coeficientes de detalle del nivel más fino
        if len(coeffs) > 1:
            # Estimar ruido usando el nivel de detalle más fino (último)
            detail_coeff_fine = coeffs[-1]
            sigma = np.median(np.abs(detail_coeff_fine)) / 0.6745  # Estimación robusta de sigma
            N = len(signal_data)
            universal_threshold = sigma * np.sqrt(2 * np.log(N))
        else:
            # Fallback si no hay coeficientes de detalle
            universal_threshold = np.std(signal_data) * np.sqrt(2 * np.log(len(signal_data)))
        
        # Aplicar thresholding a todos los niveles de detalle (no a la aproximación)
        thresholded_coeffs = [coeffs[0]]  # Aproximación sin modificar
        
        for detail_coeff in coeffs[1:]:
            if mode == 'soft':
                thresholded = pywt.threshold(detail_coeff, universal_threshold, mode='soft')
            else:
                thresholded = pywt.threshold(detail_coeff, universal_threshold, mode='hard')
            thresholded_coeffs.append(thresholded)
        
        # Reconstruir señal
        denoised = pywt.waverec(thresholded_coeffs, wavelet)
        
        # Asegurar que tenga la misma longitud que la entrada
        if len(denoised) != len(signal_data):
            denoised = denoised[:len(signal_data)]
        
        return denoised
    
    except Exception as e:
        print(f"    Error en wavelet denoising: {e}. Retornando señal sin denoising.")
        return signal_data


def calculate_bandpower(signal_data, freq_band, sample_rate):
    """
    Calcula la potencia espectral en una banda de frecuencias usando el método de Welch.
    
    Args:
        signal_data: Array 1D con datos de señal
        freq_band: Tupla (fmin, fmax) con límites de la banda
        sample_rate: Frecuencia de muestreo
        
    Returns:
        Potencia promedio en la banda especificada
    """
    if len(signal_data) < sample_rate // 2:  # Necesitamos al menos 0.5 segundos
        return 0.0
    
    # Método de Welch para estimación espectral
    nperseg = min(len(signal_data), sample_rate)
    freqs, psd = signal.welch(signal_data, sample_rate, nperseg=nperseg, noverlap=nperseg//2)
    
    # Encontrar índices de la banda de frecuencia
    idx_band = np.logical_and(freqs >= freq_band[0], freqs <= freq_band[1])
    
    if np.sum(idx_band) == 0:
        return 0.0
    
    # Calcular potencia promedio en la banda usando trapezoid (nueva función)
    bandpower = np.trapezoid(psd[idx_band], freqs[idx_band])
    
    return bandpower


def reject_artifacts(signal_window, threshold=100):
    """
    Detecta artefactos (ej. parpadeos) en una ventana de señal.
    Los parpadeos se caracterizan por picos abruptos de alta amplitud.
    
    Args:
        signal_window: Array 1D con datos de señal en µV
        threshold: Umbral base en µV (default: 100 µV)
        
    Returns:
        True si se detectan artefactos claros, False en caso contrario
    """
    if len(signal_window) == 0:
        return True  # Rechazar ventanas vacías
    
    max_abs_value = np.max(np.abs(signal_window))
    
    # Rechazar solo si hay valores MUY extremos (> 500 µV) - claramente artefactos de parpadeo
    if max_abs_value > 500:
        return True
    
    # Para valores entre 300-500 µV, verificar si hay picos abruptos típicos de parpadeos
    if max_abs_value > 300:
        if len(signal_window) > 1:
            # Calcular la derivada (cambio entre muestras consecutivas)
            diff = np.abs(np.diff(signal_window))
            max_diff = np.max(diff)
            # Si hay un cambio muy abrupto (> 150 µV entre muestras consecutivas), es un parpadeo
            if max_diff > 150:
                return True
            # Si hay muchos puntos consecutivos (> 5 muestras = 20ms) con valores altos, rechazar
            abs_signal = np.abs(signal_window)
            n_consecutive = 0
            max_consecutive = 0
            for val in abs_signal:
                if val > 300:
                    n_consecutive += 1
                    max_consecutive = max(max_consecutive, n_consecutive)
                else:
                    n_consecutive = 0
            if max_consecutive > 5:  # Más de 20ms de señal alta = parpadeo
                return True
    
    return False


def calculate_cognitive_load_ratio(fz_signal, pz_signal, sample_rate):
    """
    Calcula el ratio de carga cognitiva: Theta_Fz / Alpha_Pz
    Asume que las señales ya han sido preprocesadas (filtradas, CAR, denoised).
    
    Args:
        fz_signal: Señal del canal Fz (frontal) ya preprocesada
        pz_signal: Señal del canal Pz (parietal) ya preprocesada
        sample_rate: Frecuencia de muestreo
        
    Returns:
        Tupla (ratio, theta_power, alpha_power) o None si no hay suficiente datos
    """
    # Mínimo necesario: al menos 0.5 segundos (125 muestras a 250 Hz)
    min_samples = sample_rate // 2
    
    if len(fz_signal) < min_samples or len(pz_signal) < min_samples:
        return None
    
    # Calcular potencia en bandas (las señales ya vienen preprocesadas)
    theta_power = calculate_bandpower(fz_signal, THETA_BAND, sample_rate)
    alpha_power = calculate_bandpower(pz_signal, ALPHA_BAND, sample_rate)
    
    if alpha_power > 0 and np.isfinite(theta_power) and np.isfinite(alpha_power):
        ratio = theta_power / alpha_power
        if np.isfinite(ratio) and ratio > 0:
            return ratio, theta_power, alpha_power
    
    return None


def robust_preprocessing_pipeline(data_matrix, bad_channel_index=BAD_CHANNEL, sample_rate=SAMPLE_RATE):
    """
    Pipeline de preprocesamiento de "Limpieza Agresiva" para eliminar ruido EMG severo.
    
    Flujo exacto: Acquisition -> Bandpass -> CAR Selectivo -> WAAF -> Z-Score
    
    Args:
        data_matrix: Array 2D con forma (n_samples, n_channels) - 8 canales
        bad_channel_index: Índice del canal defectuoso a excluir (default: 4)
        sample_rate: Frecuencia de muestreo (default: 250 Hz)
        
    Returns:
        Array 2D preprocesado con forma (n_samples, n_channels)
        El canal defectuoso se mantiene en el array pero rellenado con ceros
    """
    n_samples, n_channels = data_matrix.shape
    
    # Validar que tenemos 8 canales
    if n_channels != N_CHANNELS:
        raise ValueError(f"Se esperaban {N_CHANNELS} canales, se recibieron {n_channels}")
    
    eeg_processed = data_matrix.copy().astype(float)
    
    # ============================================================
    # Paso 1: Filtrado de Frecuencia (Temporal)
    # ============================================================
    print("    [Paso 1/4] Filtrado temporal: Bandpass 0.5-30 Hz + Notch 60 Hz...")
    
    # Bandpass: Butterworth orden 4, 0.5 Hz - 30 Hz
    nyquist = sample_rate / 2
    low = 0.5 / nyquist
    high = 30.0 / nyquist
    b_bandpass, a_bandpass = signal.butter(4, [low, high], btype='band')
    
    # Stopband (Notch): 60 Hz
    notch_freq = 60.0
    quality_factor = 30.0
    b_notch, a_notch = signal.iirnotch(notch_freq, quality_factor, sample_rate)
    
    # Aplicar filtros a todos los canales (incluyendo el defectuoso)
    for ch in range(n_channels):
        if ch == bad_channel_index:
            # Para el canal defectuoso, solo aplicar filtros básicos
            # pero no lo usaremos en cálculos posteriores
            eeg_processed[:, ch] = signal.filtfilt(b_notch, a_notch, eeg_processed[:, ch])
            eeg_processed[:, ch] = signal.filtfilt(b_bandpass, a_bandpass, eeg_processed[:, ch])
        else:
            # Aplicar filtros completos a canales válidos
            eeg_processed[:, ch] = signal.filtfilt(b_notch, a_notch, eeg_processed[:, ch])
            eeg_processed[:, ch] = signal.filtfilt(b_bandpass, a_bandpass, eeg_processed[:, ch])
    
    # ============================================================
    # Paso 2: Common Average Reference (CAR) Selectivo
    # ============================================================
    print(f"    [Paso 2/4] CAR Selectivo: Calculando promedio usando SOLO los 7 canales buenos (excluyendo canal {bad_channel_index})...")
    
    # Identificar canales válidos (excluir el defectuoso)
    valid_channels = [i for i in range(n_channels) if i != bad_channel_index]
    
    if len(valid_channels) == 0:
        raise ValueError("No hay canales validos para CAR. Todos los canales estan marcados como defectuosos.")
    
    # Calcular promedio en cada instante de tiempo (axis=1) usando SOLO canales válidos
    car_reference = np.mean(eeg_processed[:, valid_channels], axis=1, keepdims=True)
    
    # Restar este promedio a TODOS los canales (incluyendo Fz y Pz)
    eeg_processed = eeg_processed - car_reference
    
    # ============================================================
    # Paso 3: WAAF (Wavelet Artifact Removal)
    # ============================================================
    print("    [Paso 3/4] WAAF: Wavelet denoising (db4, nivel 4, soft thresholding, Universal Threshold)...")
    
    if HAS_WAVELETS:
        for ch in range(n_channels):
            if ch == bad_channel_index:
                # Para el canal defectuoso, aplicar denoising pero será rellenado después
                eeg_processed[:, ch] = wavelet_denoise(
                    eeg_processed[:, ch],
                    wavelet='db4',
                    mode='soft',
                    threshold_mode='sure'
                )
            else:
                # Aplicar WAAF a canales válidos para eliminar picos transitorios
                eeg_processed[:, ch] = wavelet_denoise(
                    eeg_processed[:, ch],
                    wavelet='db4',
                    mode='soft',
                    threshold_mode='sure'
                )
    else:
        print("    Advertencia: PyWavelets no disponible. Saltando WAAF.")
        print("    Instalar con: pip install PyWavelets")
    
    # ============================================================
    # Paso 4: Z-Score Normalization
    # ============================================================
    print("    [Paso 4/4] Z-Score Normalization: Estandarizando cada canal (media=0, std=1)...")
    
    for ch in range(n_channels):
        channel_data = eeg_processed[:, ch]
        mean_val = np.mean(channel_data)
        std_val = np.std(channel_data)
        
        # Estandarización: x' = (x - μ) / σ
        if std_val > 1e-10:
            eeg_processed[:, ch] = (channel_data - mean_val) / std_val
        else:
            # Si std es muy pequeña, solo centrar
            eeg_processed[:, ch] = channel_data - mean_val
    
    # ============================================================
    # Post-procesamiento: Rellenar canal defectuoso
    # ============================================================
    # Mantener el array con 8 columnas pero rellenar el canal 4 con ceros
    eeg_processed[:, bad_channel_index] = 0.0
    
    return eeg_processed


def preprocess_eeg_data(df):
    """
    Aplica el pipeline completo de preprocesamiento de "Limpieza Agresiva" a los datos EEG.
    
    Pipeline:
    1. Carga todos los 8 canales en matriz
    2. Aplica robust_preprocessing_pipeline (Bandpass -> CAR Selectivo -> WAAF -> Z-Score)
    3. Extrae canales Fz y Pz después del preprocesamiento completo
    
    Args:
        df: DataFrame con datos de la fase (debe tener columnas channel_0 a channel_7)
        
    Returns:
        Tupla (fz_signal_clean, pz_signal_clean) con señales preprocesadas
    """
    n_samples = len(df)
    
    # 1. Cargar todos los 8 canales en matriz (n_samples, n_channels)
    eeg_matrix = np.zeros((n_samples, N_CHANNELS))
    for ch in range(N_CHANNELS):
        eeg_matrix[:, ch] = df[f'channel_{ch}'].values
    
    # 2. Aplicar pipeline de limpieza agresiva
    eeg_processed = robust_preprocessing_pipeline(
        eeg_matrix, 
        bad_channel_index=BAD_CHANNEL, 
        sample_rate=SAMPLE_RATE
    )
    
    # 3. Extraer canales Fz y Pz después del preprocesamiento completo
    # Estos canales ya están limpios (filtrados, CAR, WAAF, Z-Score)
    fz_signal_clean = eeg_processed[:, FZ_CHANNEL]
    pz_signal_clean = eeg_processed[:, PZ_CHANNEL]
    
    return fz_signal_clean, pz_signal_clean


def analyze_phase_data(df, phase_name, window_samples=WINDOW_SAMPLES, step_samples=125):
    """
    Analiza los datos aplicando PRIMERO el pipeline robusto y LUEGO segmentando.
    Evita descartar datos prematuramente. Confía en el pipeline de limpieza agresiva.
    
    Args:
        df: DataFrame con datos de la fase
        phase_name: Nombre de la fase
        window_samples: Tamaño de ventana en muestras
        step_samples: Paso entre ventanas (overlap)
        
    Returns:
        Tupla (ratios, theta_powers, alpha_powers)
    """
    ratios = []
    theta_powers = []
    alpha_powers = []
    
    # 1. Extraer la matriz completa de datos (n_samples x n_channels)
    # Asumimos que las columnas son 'channel_0', 'channel_1', etc.
    channel_cols = [f'channel_{i}' for i in range(N_CHANNELS)]
    eeg_matrix = df[channel_cols].values
    
    # Verificar duración mínima (al menos 0.5 segundos para análisis espectral)
    min_samples = SAMPLE_RATE // 2  # 125 muestras = 0.5 segundos
    if len(eeg_matrix) < min_samples:
        return [], [], []
    
    # -----------------------------------------------------------
    # PASO CRÍTICO: Aplicar limpieza a TODO el bloque antes de cortar
    # Esto usa la función 'robust_preprocessing_pipeline'
    # -----------------------------------------------------------
    try:
        # Aplicar pipeline de limpieza agresiva a toda la matriz
        clean_matrix = robust_preprocessing_pipeline(
            eeg_matrix,
            bad_channel_index=BAD_CHANNEL,
            sample_rate=SAMPLE_RATE
        )
    except Exception as e:
        print(f"  Error en preprocesamiento de {phase_name}: {e}")
        return [], [], []
    
    # Extraer canales LIMPIOS (Fz y Pz)
    # Estos canales ya están filtrados, con CAR, WAAF y Z-Score aplicados
    fz_clean = clean_matrix[:, FZ_CHANNEL]
    pz_clean = clean_matrix[:, PZ_CHANNEL]
    
    # 2. Ventaneo sobre datos LIMPIOS
    n_samples = len(clean_matrix)
    
    # Ajustar tamaño de ventana según datos disponibles
    if n_samples < window_samples:
        # Usar toda la señal si es menor que window_samples pero >= min_samples
        if n_samples >= min_samples:
            actual_window = n_samples
            actual_step = n_samples  # Solo una ventana
        else:
            return [], [], []  # Muy pocos datos
    else:
        actual_window = window_samples
        actual_step = step_samples
    
    rejected_windows = 0
    
    for start_idx in range(0, n_samples - actual_window + 1, actual_step):
        end_idx = start_idx + actual_window
        
        fz_window = fz_clean[start_idx:end_idx]
        pz_window = pz_clean[start_idx:end_idx]
        
        # --- NUEVA LÓGICA DE RECHAZO ---
        # Ya no miramos amplitud absoluta (µV) porque Z-score normaliza.
        # Solo descartamos si hay desviaciones estadísticas extremas (ej. electrodo se desconectó).
        # Un Z-score > 5 o < -5 es extremadamente raro (0.00006% probabilidad).
        if np.max(np.abs(fz_window)) > 5 or np.max(np.abs(pz_window)) > 5:
            rejected_windows += 1
            continue  # Saltamos ventana solo si es catastrófica
        
        # Calcular potencias directamente (los datos ya están filtrados y limpios)
        # No necesitamos calculate_cognitive_load_ratio porque ya están preprocesados
        theta = calculate_bandpower(fz_window, THETA_BAND, SAMPLE_RATE)
        alpha = calculate_bandpower(pz_window, ALPHA_BAND, SAMPLE_RATE)
        
        if alpha > 0 and np.isfinite(theta) and np.isfinite(alpha):
            ratio = theta / alpha
            
            # Filtro de cordura para el ratio resultante
            if 0.1 < ratio < 50:
                ratios.append(ratio)
                theta_powers.append(theta)
                alpha_powers.append(alpha)
    
    if rejected_windows > 0:
        print(f"  - Ventanas rechazadas por valores extremos (Z-score > 5): {rejected_windows}")
    
    return ratios, theta_powers, alpha_powers


def load_and_analyze_data(csv_path):
    """
    Carga y analiza los datos del CSV aplicando el pipeline completo de preprocesamiento.
    
    Pipeline de preprocesamiento:
    1. Carga de 8 canales del CSV
    2. Filtrado temporal (Bandpass 0.5-30 Hz + Notch 60 Hz)
    3. CAR (Common Average Reference) excluyendo canal defectuoso
    4. Wavelet denoising en Fz y Pz
    5. Cálculo de ratios Theta/Alpha
    
    Args:
        csv_path: Ruta al archivo CSV
        
    Returns:
        Tupla (results, df) donde results es un diccionario con resultados por fase
    """
    print(f"Cargando datos de {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print(f"Total de muestras: {len(df)}")
    print(f"Fases encontradas: {df['phase'].unique()}")
    print(f"Labels encontrados: {df['label'].unique()}\n")
    
    # Verificar que todos los canales estén presentes
    required_channels = [f'channel_{i}' for i in range(N_CHANNELS)]
    missing_channels = [ch for ch in required_channels if ch not in df.columns]
    if missing_channels:
        print(f"Error: Faltan canales en el CSV: {missing_channels}")
        return {}, df
    
    print(f"Pipeline de preprocesamiento: 'Limpieza Agresiva' para eliminar ruido EMG:")
    print(f"  1. Filtrado temporal: Bandpass 0.5-30 Hz (Butterworth orden 4) + Notch 60 Hz")
    print(f"  2. CAR Selectivo: Common Average Reference usando SOLO 7 canales buenos (excluyendo canal {BAD_CHANNEL})")
    print(f"  3. WAAF: Wavelet Artifact Removal (db4, nivel 4, soft thresholding, Universal Threshold)")
    print(f"  4. Z-Score normalization: Estandarizacion por canal (media=0, std=1)")
    print(f"  Nota: Canal {BAD_CHANNEL} se mantiene en el array pero rellenado con ceros")
    print()
    
    # Filtrar fases de interés (excluir setup y baseline_completed)
    phases_of_interest = ['baseline_eyes_open', 'baseline_eyes_closed', 'low_load', 'high_load']
    
    results = {}
    
    for phase in phases_of_interest:
        phase_df = df[df['phase'] == phase].copy()
        
        if len(phase_df) == 0:
            print(f"Advertencia: No se encontraron datos para la fase '{phase}'")
            continue
        
        print(f"Analizando fase: {phase} ({len(phase_df)} muestras)...")
        print(f"  Usando canal {FZ_CHANNEL} para Fz y canal {PZ_CHANNEL} para Pz")
        
        ratios, theta_powers, alpha_powers = analyze_phase_data(phase_df, phase)
        
        if len(ratios) > 0:
            results[phase] = {
                'ratios': np.array(ratios),
                'theta_powers': np.array(theta_powers),
                'alpha_powers': np.array(alpha_powers),
                'n_samples': len(phase_df),
                'n_windows': len(ratios)
            }
            print(f"  - Ventanas analizadas: {len(ratios)}")
            print(f"  - Ratio promedio: {np.mean(ratios):.4f} ± {np.std(ratios):.4f}")
            print(f"  - Ratio mediano: {np.median(ratios):.4f}")
        else:
            print(f"  - No se pudieron calcular ratios para esta fase")
        
        print()
    
    return results, df


def statistical_comparison(results):
    """
    Realiza comparaciones estadísticas entre high_load y otras fases.
    
    Args:
        results: Diccionario con resultados por fase
        
    Returns:
        Diccionario con p-values para cada comparación: {phase: p_value}
    """
    p_values = {}
    
    if 'high_load' not in results:
        print("Error: No se encontraron datos de high_load")
        return p_values
    
    high_load_ratios = results['high_load']['ratios']
    
    print("=" * 70)
    print("COMPARACION ESTADISTICA: HIGH_LOAD vs OTRAS FASES")
    print("=" * 70)
    print()
    
    comparison_phases = ['baseline_eyes_open', 'baseline_eyes_closed', 'low_load']
    
    for phase in comparison_phases:
        if phase not in results:
            continue
        
        other_ratios = results[phase]['ratios']
        
        # Test de normalidad (Shapiro-Wilk) - requiere al menos 3 muestras
        if len(high_load_ratios) >= 3:
            _, p_high = stats.shapiro(high_load_ratios[:5000]) if len(high_load_ratios) > 5000 else stats.shapiro(high_load_ratios)
        else:
            p_high = 0.0  # Asumir no normal si hay muy pocas muestras
        
        if len(other_ratios) >= 3:
            _, p_other = stats.shapiro(other_ratios[:5000]) if len(other_ratios) > 5000 else stats.shapiro(other_ratios)
        else:
            p_other = 0.0  # Asumir no normal si hay muy pocas muestras
        
        # Elegir test según normalidad y número de muestras
        if len(high_load_ratios) >= 3 and len(other_ratios) >= 3:
            if p_high > 0.05 and p_other > 0.05:
                # Ambas distribuciones son normales -> t-test
                statistic, p_value = stats.ttest_ind(high_load_ratios, other_ratios)
                test_name = "t-test (independiente)"
            else:
                # Al menos una no es normal -> Mann-Whitney U
                statistic, p_value = stats.mannwhitneyu(high_load_ratios, other_ratios, alternative='two-sided')
                test_name = "Mann-Whitney U"
        elif len(high_load_ratios) == 1 and len(other_ratios) == 1:
            # Solo comparación de medias si hay una muestra de cada una
            statistic = np.mean(high_load_ratios) - np.mean(other_ratios)
            p_value = np.nan  # No se puede calcular p-value con una muestra
            test_name = "Comparacion directa (1 muestra cada una)"
        else:
            # Usar Mann-Whitney si hay al menos 2 muestras en total
            try:
                statistic, p_value = stats.mannwhitneyu(high_load_ratios, other_ratios, alternative='two-sided')
                test_name = "Mann-Whitney U"
            except:
                statistic = np.mean(high_load_ratios) - np.mean(other_ratios)
                p_value = np.nan
                test_name = "Comparacion directa"
        
        # Calcular tamaño del efecto (Cohen's d)
        pooled_std = np.sqrt((np.var(high_load_ratios) + np.var(other_ratios)) / 2)
        cohens_d = (np.mean(high_load_ratios) - np.mean(other_ratios)) / pooled_std if pooled_std > 0 else 0
        
        print(f"High Load vs {phase.replace('_', ' ').title()}:")
        print(f"  - High Load:  M={np.mean(high_load_ratios):.4f}, SD={np.std(high_load_ratios):.4f}, N={len(high_load_ratios)}")
        print(f"  - {phase.replace('_', ' ').title()}: M={np.mean(other_ratios):.4f}, SD={np.std(other_ratios):.4f}, N={len(other_ratios)}")
        print(f"  - Diferencia: {np.mean(high_load_ratios) - np.mean(other_ratios):.4f}")
        print(f"  - Test: {test_name}")
        print(f"  - Estadistico: {statistic:.4f}")
        if not np.isnan(p_value):
            print(f"  - p-value: {p_value:.6f}")
        else:
            print(f"  - p-value: N/A (insuficientes muestras para test estadistico)")
        print(f"  - Cohen's d: {cohens_d:.4f}")
        
        if p_value < 0.001:
            significance = "***"
        elif p_value < 0.01:
            significance = "**"
        elif p_value < 0.05:
            significance = "*"
        else:
            significance = "ns"
        
        print(f"  - Significancia: {significance}")
        print()
        
        # Guardar p-value para visualización
        p_values[phase] = p_value if not np.isnan(p_value) else None
    
    return p_values


def create_visualizations(results, p_values=None, output_dir='analysis_output'):
    """
    Crea visualizaciones de los resultados.
    
    Args:
        results: Diccionario con resultados por fase
        p_values: Diccionario con p-values para comparaciones (opcional)
        output_dir: Directorio para guardar las figuras
    """
    if not HAS_PLOTTING:
        print("Omitiendo visualizaciones (matplotlib/seaborn no disponibles)")
        print("Skipping visualizations (matplotlib/seaborn not available)")
        return
    
    Path(output_dir).mkdir(exist_ok=True)
    
    # 1. Boxplot comparativo
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Boxplot de ratios
    ax1 = axes[0, 0]
    phase_names = []
    ratio_data = []
    
    phase_order = ['baseline_eyes_open', 'baseline_eyes_closed', 'low_load', 'high_load']
    for phase in phase_order:
        if phase in results:
            phase_names.append(phase.replace('_', ' ').title())
            ratio_data.append(results[phase]['ratios'])
    
    bp = ax1.boxplot(ratio_data, tick_labels=phase_names, patch_artist=True)
    colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax1.set_ylabel('Ratio de Carga Cognitiva (Theta/Alpha)', fontsize=12)
    ax1.set_title('Distribución de Ratios por Fase', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Violin plot
    ax2 = axes[0, 1]
    data_for_violin = []
    labels_for_violin = []
    for phase in phase_order:
        if phase in results:
            data_for_violin.append(results[phase]['ratios'])
            labels_for_violin.append(phase.replace('_', ' ').title())
    
    parts = ax2.violinplot(data_for_violin, positions=range(len(data_for_violin)), 
                          showmeans=True, showmedians=True)
    ax2.set_xticks(range(len(labels_for_violin)))
    ax2.set_xticklabels(labels_for_violin)
    ax2.set_ylabel('Ratio de Carga Cognitiva (Theta/Alpha)', fontsize=12)
    ax2.set_title('Distribución Detallada de Ratios', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Histograma comparativo
    ax3 = axes[1, 0]
    for phase in phase_order:
        if phase in results:
            ax3.hist(results[phase]['ratios'], bins=50, alpha=0.6, 
                    label=phase.replace('_', ' ').title(), density=True)
    ax3.set_xlabel('Ratio de Carga Cognitiva', fontsize=12)
    ax3.set_ylabel('Densidad', fontsize=12)
    ax3.set_title('Distribución de Ratios (Histograma)', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Estadísticas resumidas
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    stats_text = "ESTADISTICAS RESUMIDAS\n" + "=" * 50 + "\n\n"
    for phase in phase_order:
        if phase in results:
            ratios = results[phase]['ratios']
            stats_text += f"{phase.replace('_', ' ').title()}:\n"
            stats_text += f"  Media: {np.mean(ratios):.4f}\n"
            stats_text += f"  Mediana: {np.median(ratios):.4f}\n"
            stats_text += f"  SD: {np.std(ratios):.4f}\n"
            stats_text += f"  N: {len(ratios)}\n\n"
    
    ax4.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
            verticalalignment='center', transform=ax4.transAxes)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/cognitive_load_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Grafico guardado en: {output_dir}/cognitive_load_comparison.png")
    
    # 2. Grafico de barras con barras de error
    fig, ax = plt.subplots(figsize=(12, 6))
    
    means = []
    stds = []
    labels = []
    
    for phase in phase_order:
        if phase in results:
            means.append(np.mean(results[phase]['ratios']))
            stds.append(np.std(results[phase]['ratios']))
            labels.append(phase.replace('_', ' ').title())
    
    x_pos = np.arange(len(labels))
    bars = ax.bar(x_pos, means, yerr=stds, capsize=10, alpha=0.7, 
                 color=['lightblue', 'lightgreen', 'lightyellow', 'lightcoral'])
    
    ax.set_ylabel('Ratio de Carga Cognitiva (Theta/Alpha)', fontsize=12)
    ax.set_xlabel('Fase del Experimento', fontsize=12)
    ax.set_title('Ratio Promedio de Carga Cognitiva por Fase', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Añadir valores en las barras
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(i, mean + std + 0.05, f'{mean:.3f}', ha='center', fontweight='bold')
    
    # Agregar líneas de significancia estadística si hay p-values
    if p_values is not None:
        # Encontrar índices de High Load y Low Load
        high_load_idx = None
        low_load_idx = None
        
        phase_list = []
        for phase in phase_order:
            if phase in results:
                phase_list.append(phase)
        
        if 'high_load' in phase_list:
            high_load_idx = phase_list.index('high_load')
        if 'low_load' in phase_list:
            low_load_idx = phase_list.index('low_load')
        
        # Dibujar significancia entre High Load y Low Load
        if high_load_idx is not None and low_load_idx is not None and 'low_load' in p_values:
            p_val = p_values['low_load']
            if p_val is not None:
                # Determinar símbolo de significancia
                if p_val < 0.001:
                    sig_symbol = '***'
                elif p_val < 0.01:
                    sig_symbol = '**'
                elif p_val < 0.05:
                    sig_symbol = '*'
                else:
                    sig_symbol = 'n.s.'
                
                # Calcular altura para la línea
                max_height = max(means[high_load_idx] + stds[high_load_idx], 
                               means[low_load_idx] + stds[low_load_idx])
                line_height = max_height + max(stds) * 0.3  # 30% más alto que el máximo
                
                # Dibujar línea horizontal
                x1 = min(high_load_idx, low_load_idx)
                x2 = max(high_load_idx, low_load_idx)
                ax.plot([x1, x2], [line_height, line_height], 'k-', linewidth=1.5)
                
                # Dibujar líneas verticales en los extremos
                ax.plot([x1, x1], [means[x1] + stds[x1], line_height], 'k-', linewidth=1.5)
                ax.plot([x2, x2], [means[x2] + stds[x2], line_height], 'k-', linewidth=1.5)
                
                # Añadir texto de significancia
                mid_x = (x1 + x2) / 2
                ax.text(mid_x, line_height + max(stds) * 0.1, sig_symbol, 
                       ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        # Opcional: Dibujar significancia entre High Load y Baseline Eyes Open
        baseline_open_idx = None
        if 'baseline_eyes_open' in phase_list:
            baseline_open_idx = phase_list.index('baseline_eyes_open')
        
        if high_load_idx is not None and baseline_open_idx is not None and 'baseline_eyes_open' in p_values:
            p_val = p_values['baseline_eyes_open']
            if p_val is not None:
                # Determinar símbolo de significancia
                if p_val < 0.001:
                    sig_symbol = '***'
                elif p_val < 0.01:
                    sig_symbol = '**'
                elif p_val < 0.05:
                    sig_symbol = '*'
                else:
                    sig_symbol = 'n.s.'
                
                # Calcular altura para la línea (más alta que la anterior)
                max_height = max(means[high_load_idx] + stds[high_load_idx], 
                               means[baseline_open_idx] + stds[baseline_open_idx])
                line_height = max_height + max(stds) * 0.6  # Más alto para no solaparse
                
                # Dibujar línea horizontal
                x1 = min(high_load_idx, baseline_open_idx)
                x2 = max(high_load_idx, baseline_open_idx)
                ax.plot([x1, x2], [line_height, line_height], 'k-', linewidth=1.5)
                
                # Dibujar líneas verticales en los extremos
                ax.plot([x1, x1], [means[x1] + stds[x1], line_height], 'k-', linewidth=1.5)
                ax.plot([x2, x2], [means[x2] + stds[x2], line_height], 'k-', linewidth=1.5)
                
                # Añadir texto de significancia
                mid_x = (x1 + x2) / 2
                ax.text(mid_x, line_height + max(stds) * 0.1, sig_symbol, 
                       ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        # Ajustar límites del eje Y para que las líneas de significancia sean visibles
        current_ylim = ax.get_ylim()
        max_line_height = max(means) + max(stds) * 1.2
        ax.set_ylim([current_ylim[0], max(current_ylim[1], max_line_height)])
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/cognitive_load_barplot.png', dpi=300, bbox_inches='tight')
    print(f"Grafico guardado en: {output_dir}/cognitive_load_barplot.png")
    
    plt.close('all')


def save_results_to_csv(results, output_dir='analysis_output'):
    """
    Guarda los resultados en un archivo CSV.
    
    Args:
        results: Diccionario con resultados por fase
        output_dir: Directorio para guardar el archivo
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    # Crear DataFrame con todos los ratios
    all_data = []
    for phase, data in results.items():
        for ratio in data['ratios']:
            all_data.append({
                'phase': phase,
                'ratio': ratio
            })
    
    df_results = pd.DataFrame(all_data)
    df_results.to_csv(f'{output_dir}/cognitive_load_ratios.csv', index=False)
    print(f"Resultados guardados en: {output_dir}/cognitive_load_ratios.csv")
    
    # Crear resumen estadístico
    summary_data = []
    for phase, data in results.items():
        ratios = data['ratios']
        summary_data.append({
            'phase': phase,
            'mean': np.mean(ratios),
            'median': np.median(ratios),
            'std': np.std(ratios),
            'min': np.min(ratios),
            'max': np.max(ratios),
            'n_windows': len(ratios),
            'n_samples': data['n_samples']
        })
    
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(f'{output_dir}/cognitive_load_summary.csv', index=False)
    print(f"Resumen guardado en: {output_dir}/cognitive_load_summary.csv")


def main():
    """Función principal."""
    csv_path = 'data_mai/eeg_data_20251211_152323.csv'
    
    if not Path(csv_path).exists():
        print(f"Error: No se encontró el archivo {csv_path}")
        return
    
    # Cargar y analizar datos
    results, df = load_and_analyze_data(csv_path)
    
    if not results:
        print("Error: No se pudieron analizar los datos")
        return
    
    # Comparación estadística
    p_values = statistical_comparison(results)
    
    # Visualizaciones
    print("Generando visualizaciones...")
    create_visualizations(results, p_values=p_values)
    
    # Guardar resultados
    print("\nGuardando resultados...")
    save_results_to_csv(results)
    
    print("\n" + "=" * 70)
    print("ANALISIS COMPLETADO")
    print("=" * 70)
    print("\nLos resultados se han guardado en el directorio 'analysis_output/'")


if __name__ == '__main__':
    main()

