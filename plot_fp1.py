import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from scipy import signal
import pywt

# Leer el CSV
# Saltamos la segunda línea (unidades) pero mantenemos los headers
df = pd.read_csv('data/eeg.csv', skiprows=[1])

# Limpiar los nombres de columnas (quitar comillas)
df.columns = df.columns.str.strip().str.replace("'", "")

# Convertir la columna de tiempo
def parse_time(time_str):
    """Convierte el formato '[hh:mm:ss.mmm dd/mm/yyyy]' a datetime"""
    try:
        # Quitar corchetes y comillas
        time_str = str(time_str).strip().replace("'", "").replace("[", "").replace("]", "")
        # Parsear el formato
        return datetime.strptime(time_str, '%H:%M:%S.%f %d/%m/%Y')
    except:
        return None

# Aplicar parseo de tiempo
df['Time and date'] = df['Time and date'].apply(parse_time)

# Función Wavelet-Assisted Adaptive Filter (WAAF)
def dcaro_WAAF(eeg, wavelet='db4', level=7, attn=None):
    """
    Wavelet-Assisted Adaptive Filter para remover artefactos oculares de señales EEG.
    
    Adaptado de MATLAB por Diego Caro López.
    
    Parameters:
    -----------
    eeg : numpy.ndarray
        Datos EEG con artefactos oculares [n_channels x n_samples]
    wavelet : str, optional
        Wavelet madre a usar. Default: 'db4'
    level : int, optional
        Nivel de descomposición wavelet. Default: 7
    attn : list, optional
        Niveles de coeficientes de detalle a atenuar. Default: [1, 2, 3, 4]
    
    Returns:
    --------
    eeg2 : numpy.ndarray
        Datos EEG denoised [n_channels x n_samples]
    refs : numpy.ndarray
        Señales de referencia removidas por ANC [n_channels x n_samples]
    weights : numpy.ndarray
        Pesos retornados por el algoritmo RLS [n_channels x n_samples]
    """
    if attn is None:
        attn = [1, 2, 3, 4]  # Default: primeros 4 niveles
    
    chans = eeg.shape[0]
    n_samples = eeg.shape[1]
    
    refs = np.zeros_like(eeg)
    eeg2 = np.zeros_like(eeg)
    weights = np.zeros((chans, n_samples))
    
    for j in range(chans):
        x = eeg[j, :]
        
        # Wavelet decomposition
        coeffs = pywt.wavedec(x, wavelet, level=level)
        cA = coeffs[0]  # Approximation coefficients
        cD_list = coeffs[1:]  # Detail coefficients (level 1 to level)
        
        # Soft thresholding
        D2_list = []
        T = np.zeros(level)
        
        for i in range(level):
            D = cD_list[i]
            if (i + 1) in attn:  # i+1 porque attn es 1-indexed
                # Using threshold value proposed in Coifman & Donoho (1995)
                sj = np.median(np.abs(D - np.median(D))) / 0.6745
                T[i] = sj * np.sqrt(2 * np.log(len(x)))
                # Soft thresholding
                D2 = np.sign(D) * np.maximum(np.abs(D) - T[i], 0)
            else:
                D2 = D
            D2_list.append(D2)
        
        # Signal reconstruction
        coeffs2 = [cA] + D2_list
        ref = pywt.waverec(coeffs2, wavelet)
        ref = ref[:n_samples]  # Asegurar mismo tamaño
        
        # ANC based on RLS (Order M=1; P is scalar)
        e = np.zeros(n_samples)
        P = 1e4  # Inverse Autocorrelation
        lambda_val = 0.98  # Forgetting factor
        w = np.ones(n_samples + 1)  # Initial weights
        
        for k in range(n_samples):
            # Cross-correlation
            Pi = ref[k] * P
            
            # Obtain gain & filter output
            g = Pi / (lambda_val + Pi * ref[k])
            y = w[k] * ref[k]
            
            # Update EEG
            e[k] = x[k] - y
            
            # Update weights and correlation matrix
            w[k + 1] = w[k] + g * e[k]
            P = (P - g * Pi) / lambda_val
        
        refs[j, :] = ref
        eeg2[j, :] = e
        weights[j, :] = w[:n_samples]
    
    return eeg2, refs, weights

# Convertir todos los canales EEG a numérico (quitar comillas si las hay)
canales_eeg = ['Fp1', 'Fp2', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2']
for canal in canales_eeg:
    df[canal] = df[canal].astype(str).str.replace("'", "").astype(float)

# Calcular frecuencia de muestreo
# Calcular diferencias de tiempo entre muestras consecutivas
df_sorted = df.sort_values('Time and date').reset_index(drop=True)
tiempos_diff = df_sorted['Time and date'].diff().dropna()
# Convertir a segundos y calcular frecuencia promedio
tiempos_segundos = tiempos_diff.dt.total_seconds()
frecuencia_muestreo = 1.0 / tiempos_segundos.mean()

# Filtrar 15 segundos a partir de la mitad del dataset
tiempo_min = df['Time and date'].min()
tiempo_max = df['Time and date'].max()
tiempo_medio = tiempo_min + (tiempo_max - tiempo_min) / 2

# Tomar 15 segundos a partir del punto medio
tiempo_inicio = tiempo_medio
tiempo_limite = tiempo_medio + pd.Timedelta(seconds=15)
df_15s = df[(df['Time and date'] >= tiempo_inicio) & (df['Time and date'] <= tiempo_limite)].copy()

# Ordenar por tiempo para asegurar orden correcto
df_15s = df_15s.sort_values('Time and date').reset_index(drop=True)

# Aplicar filtro bandpass 0.2-50 Hz
# Parámetros del filtro
lowcut = 0.2  # Hz
highcut = 50.0  # Hz
nyquist = frecuencia_muestreo / 2.0
low = lowcut / nyquist
high = highcut / nyquist

# Crear el filtro Butterworth
b, a = signal.butter(4, [low, high], btype='band')

# Aplicar el filtro a todos los canales EEG
canales_filtrados = {}
for canal in canales_eeg:
    canales_filtrados[canal] = signal.filtfilt(b, a, df_15s[canal].values)

# Aplicar Common Average Reference (CAR)
# Calcular el promedio de todos los canales filtrados
canales_array = np.array([canales_filtrados[canal] for canal in canales_eeg])
promedio_comun = np.mean(canales_array, axis=0)

# Restar el promedio común de cada canal
canales_car = {}
for canal in canales_eeg:
    canales_car[canal] = canales_filtrados[canal] - promedio_comun

# Aplicar filtro stopband de 60 Hz a los canales con CAR
# Parámetros del filtro stopband (notch filter)
freq_notch = 60.0  # Hz - frecuencia a eliminar
bandwidth = 2.0  # Hz - ancho de banda del notch
low_stop = (freq_notch - bandwidth/2) / nyquist
high_stop = (freq_notch + bandwidth/2) / nyquist

# Crear el filtro stopband (bandstop)
b_stop, a_stop = signal.butter(4, [low_stop, high_stop], btype='bandstop')

# Aplicar el filtro stopband a todos los canales con CAR
canales_car_stopband = {}
for canal in canales_eeg:
    canales_car_stopband[canal] = signal.filtfilt(b_stop, a_stop, canales_car[canal])

# Aplicar Wavelet-Assisted Adaptive Filter (WAAF)
# Preparar datos en formato [n_channels x n_samples]
eeg_input = np.array([canales_car_stopband[canal] for canal in canales_eeg])
eeg_waaf, refs_waaf, weights_waaf = dcaro_WAAF(eeg_input, wavelet='db4', level=7, attn=[1, 2, 3, 4])

# Convertir de vuelta a diccionario
canales_waaf = {}
for idx, canal in enumerate(canales_eeg):
    canales_waaf[canal] = eeg_waaf[idx, :]

# Extraer Fp1 en cada etapa
fp1_filtrado = canales_filtrados['Fp1']
fp1_car = canales_car['Fp1']
fp1_car_stopband = canales_car_stopband['Fp1']
fp1_waaf = canales_waaf['Fp1']

# Aplicar normalización z-score a la señal después del WAAF
fp1_waaf_mean = np.mean(fp1_waaf)
fp1_waaf_std = np.std(fp1_waaf)
fp1_waaf_zscore = (fp1_waaf - fp1_waaf_mean) / fp1_waaf_std

# Crear figura con seis subplots
fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1, figsize=(14, 18), sharex=True)

# Gráfico 1: Señal original
ax1.plot(df_15s['Time and date'], df_15s['Fp1'], linewidth=0.5, color='blue', alpha=0.7)
ax1.set_ylabel('Fp1 (μV)', fontsize=12)
ax1.set_title(f'Señal EEG Original - Canal Fp1 (15 segundos desde la mitad, {frecuencia_muestreo:.1f} Hz) / Original EEG Signal - Fp1 Channel (15 seconds from middle, {frecuencia_muestreo:.1f} Hz)', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Gráfico 2: Señal filtrada
ax2.plot(df_15s['Time and date'], fp1_filtrado, linewidth=0.5, color='red', alpha=0.7)
ax2.set_ylabel('Fp1 Filtrado (μV)', fontsize=12)
ax2.set_title(f'Señal EEG Filtrada - Canal Fp1 (Bandpass 0.2-50 Hz, {frecuencia_muestreo:.1f} Hz) / Filtered EEG Signal - Fp1 Channel (Bandpass 0.2-50 Hz, {frecuencia_muestreo:.1f} Hz)', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Gráfico 3: Señal filtrada + CAR
ax3.plot(df_15s['Time and date'], fp1_car, linewidth=0.5, color='green', alpha=0.7)
ax3.set_ylabel('Fp1 CAR (μV)', fontsize=12)
ax3.set_title(f'Señal EEG Filtrada + CAR - Canal Fp1 (Bandpass 0.2-50 Hz + Common Average Reference, {frecuencia_muestreo:.1f} Hz) / Filtered + CAR EEG Signal - Fp1 Channel (Bandpass 0.2-50 Hz + Common Average Reference, {frecuencia_muestreo:.1f} Hz)', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Gráfico 4: Señal filtrada + CAR + Stopband 60 Hz
ax4.plot(df_15s['Time and date'], fp1_car_stopband, linewidth=0.5, color='purple', alpha=0.7)
ax4.set_ylabel('Fp1 CAR+Stopband (μV)', fontsize=12)
ax4.set_title(f'Señal EEG Filtrada + CAR + Stopband 60 Hz - Canal Fp1 (Bandpass 0.2-50 Hz + CAR + Stopband 60 Hz, {frecuencia_muestreo:.1f} Hz) / Filtered + CAR + Stopband 60 Hz EEG Signal - Fp1 Channel (Bandpass 0.2-50 Hz + CAR + Stopband 60 Hz, {frecuencia_muestreo:.1f} Hz)', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)

# Gráfico 5: Señal filtrada + CAR + Stopband + WAAF
ax5.plot(df_15s['Time and date'], fp1_waaf, linewidth=0.5, color='orange', alpha=0.7)
ax5.set_ylabel('Fp1 WAAF (μV)', fontsize=12)
ax5.set_title(f'Señal EEG Filtrada + CAR + Stopband + WAAF - Canal Fp1 (Bandpass 0.2-50 Hz + CAR + Stopband 60 Hz + Wavelet-Assisted Adaptive Filter, {frecuencia_muestreo:.1f} Hz) / Filtered + CAR + Stopband + WAAF EEG Signal - Fp1 Channel (Bandpass 0.2-50 Hz + CAR + Stopband 60 Hz + Wavelet-Assisted Adaptive Filter, {frecuencia_muestreo:.1f} Hz)', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3)

# Gráfico 6: Señal filtrada + CAR + Stopband + WAAF + Z-score
ax6.plot(df_15s['Time and date'], fp1_waaf_zscore, linewidth=0.5, color='brown', alpha=0.7)
ax6.set_xlabel('Tiempo / Time', fontsize=12)
ax6.set_ylabel('Fp1 Z-score', fontsize=12)
ax6.set_title(f'Señal EEG Normalizada (Z-score) - Canal Fp1 (Bandpass 0.2-50 Hz + CAR + Stopband 60 Hz + WAAF + Z-score normalization, {frecuencia_muestreo:.1f} Hz) / Z-score Normalized EEG Signal - Fp1 Channel (Bandpass 0.2-50 Hz + CAR + Stopband 60 Hz + WAAF + Z-score normalization, {frecuencia_muestreo:.1f} Hz)', fontsize=12, fontweight='bold')
ax6.grid(True, alpha=0.3)
ax6.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)  # Línea en y=0

# Rotar las etiquetas del eje x para mejor visualización
plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45)

plt.tight_layout()

# Mostrar el gráfico
plt.show()

# ============================================================================
# ANÁLISIS ESPECTRAL - BANDA ALPHA (8-13 Hz)
# Comparativa: Datos Raw vs Datos Procesados
# ============================================================================

print("\n" + "="*70)
print("ANÁLISIS ESPECTRAL - BANDA ALPHA (8-13 Hz) / SPECTRAL ANALYSIS - ALPHA BAND (8-13 Hz)")
print("Comparativa: Raw Data vs Processed Data / Comparison: Raw Data vs Processed Data")
print("="*70)

# Definir banda Alpha
alpha_min = 8.0  # Hz
alpha_max = 13.0  # Hz

# Función para calcular densidad espectral de potencia (PSD)
def calcular_psd(señal, fs):
    """Calcula la densidad espectral de potencia usando Welch's method"""
    from scipy import signal as scipy_signal
    f, psd = scipy_signal.welch(señal, fs, nperseg=min(256, len(señal)//4))
    return f, psd

# Función para calcular potencia en banda Alpha
def calcular_potencia_alpha(señal, fs, fmin, fmax):
    """Calcula la potencia total en la banda Alpha"""
    f, psd = calcular_psd(señal, fs)
    # Encontrar índices de la banda Alpha
    idx_alpha = np.where((f >= fmin) & (f <= fmax))[0]
    # Calcular potencia total (área bajo la curva PSD)
    potencia = np.trapz(psd[idx_alpha], f[idx_alpha])
    return potencia, f[idx_alpha], psd[idx_alpha], f, psd

# Señal raw (sin procesar)
fp1_raw = df_15s['Fp1'].values

# Calcular Alpha para señal RAW
potencia_alpha_raw, f_alpha_raw, psd_alpha_raw, f_psd_raw, psd_raw = calcular_potencia_alpha(
    fp1_raw, frecuencia_muestreo, alpha_min, alpha_max)

# Calcular Alpha para señal PROCESADA (después de WAAF)
potencia_alpha_proc, f_alpha_proc, psd_alpha_proc, f_psd_proc, psd_proc = calcular_potencia_alpha(
    fp1_waaf, frecuencia_muestreo, alpha_min, alpha_max)

# Imprimir resultados
print(f"\nPotencia Alpha - Datos RAW / Alpha Power - Raw Data:")
print(f"  Potencia total: {potencia_alpha_raw:.2f} μV²/Hz")
print(f"  Rango: {alpha_min}-{alpha_max} Hz")

print(f"\nPotencia Alpha - Datos PROCESADOS / Alpha Power - Processed Data:")
print(f"  Potencia total: {potencia_alpha_proc:.2f} μV²/Hz")
print(f"  Rango: {alpha_min}-{alpha_max} Hz")

# Calcular diferencia
diferencia = potencia_alpha_proc - potencia_alpha_raw
porcentaje_cambio = (diferencia / potencia_alpha_raw) * 100 if potencia_alpha_raw > 0 else 0

print(f"\nDiferencia / Difference:")
print(f"  Absoluta: {diferencia:+.2f} μV²/Hz")
print(f"  Porcentaje: {porcentaje_cambio:+.2f}%")

# Crear gráfico comparativo
fig2, (ax_psd, ax_alpha, ax_comparacion) = plt.subplots(3, 1, figsize=(14, 12))

# Gráfico 1: PSD completo - Comparativa
ax_psd.semilogy(f_psd_raw, psd_raw, linewidth=1.5, color='red', alpha=0.7, label='Raw Data / Datos Raw')
ax_psd.semilogy(f_psd_proc, psd_proc, linewidth=1.5, color='blue', alpha=0.7, label='Processed Data / Datos Procesados')
ax_psd.set_xlabel('Frecuencia (Hz) / Frequency (Hz)', fontsize=12)
ax_psd.set_ylabel('Densidad Espectral de Potencia (μV²/Hz) / Power Spectral Density (μV²/Hz)', fontsize=12)
ax_psd.set_title('Espectro de Frecuencia Completo - Comparativa Raw vs Procesado / Full Frequency Spectrum - Raw vs Processed Comparison', fontsize=12, fontweight='bold')
ax_psd.set_xlim(0, 50)
ax_psd.grid(True, alpha=0.3, which='both')
ax_psd.legend(loc='upper right')

# Resaltar banda Alpha
idx_alpha_plot = np.where((f_psd_raw >= alpha_min) & (f_psd_raw <= alpha_max))[0]
if len(idx_alpha_plot) > 0:
    ax_psd.axvspan(alpha_min, alpha_max, alpha=0.2, color='yellow', label='Alpha Band (8-13 Hz)')
    ax_psd.legend(loc='upper right')

# Gráfico 2: Zoom en banda Alpha
ax_alpha.plot(f_alpha_raw, psd_alpha_raw, linewidth=2, color='red', alpha=0.8, marker='o', markersize=3, label='Raw Data / Datos Raw')
ax_alpha.plot(f_alpha_proc, psd_alpha_proc, linewidth=2, color='blue', alpha=0.8, marker='s', markersize=3, label='Processed Data / Datos Procesados')
ax_alpha.fill_between(f_alpha_raw, psd_alpha_raw, alpha=0.3, color='red')
ax_alpha.fill_between(f_alpha_proc, psd_alpha_proc, alpha=0.3, color='blue')
ax_alpha.set_xlabel('Frecuencia (Hz) / Frequency (Hz)', fontsize=12)
ax_alpha.set_ylabel('Densidad Espectral de Potencia (μV²/Hz) / Power Spectral Density (μV²/Hz)', fontsize=12)
ax_alpha.set_title(f'Banda Alpha (8-13 Hz) - Comparativa / Alpha Band (8-13 Hz) - Comparison', fontsize=12, fontweight='bold')
ax_alpha.set_xlim(alpha_min - 1, alpha_max + 1)
ax_alpha.grid(True, alpha=0.3, which='both')
ax_alpha.legend(loc='upper right')

# Gráfico 3: Comparación de potencia total Alpha (barras)
categorias = ['Raw Data\nDatos Raw', 'Processed Data\nDatos Procesados']
potencias = [potencia_alpha_raw, potencia_alpha_proc]
colores = ['red', 'blue']

bars = ax_comparacion.bar(categorias, potencias, color=colores, alpha=0.7, edgecolor='black', linewidth=2)
ax_comparacion.set_ylabel('Potencia Alpha (μV²/Hz) / Alpha Power (μV²/Hz)', fontsize=12)
ax_comparacion.set_title('Comparación de Potencia Alpha - Raw vs Procesado / Alpha Power Comparison - Raw vs Processed', fontsize=12, fontweight='bold')
ax_comparacion.grid(True, alpha=0.3, axis='y')

# Agregar valores en las barras
for bar, valor in zip(bars, potencias):
    height = bar.get_height()
    ax_comparacion.text(bar.get_x() + bar.get_width()/2., height,
                       f'{valor:.2f}\nμV²/Hz',
                       ha='center', va='bottom', fontsize=11, fontweight='bold')

# Agregar línea de diferencia
ax_comparacion.plot([0, 1], [potencia_alpha_raw, potencia_alpha_proc], 
                   'k--', linewidth=2, alpha=0.5, label=f'Diferencia: {diferencia:+.2f} μV²/Hz ({porcentaje_cambio:+.1f}%)')
ax_comparacion.legend(loc='upper left')

plt.tight_layout()
plt.show()

print("\n" + "="*70)
print("NOTA / NOTE:")
print("="*70)
print("La señal procesada sigue en DOMINIO DEL TIEMPO con amplitud en μV.")
print("Para visualizar bandas de frecuencia se requiere análisis espectral (PSD).")
print("\nThe processed signal is still in TIME DOMAIN with amplitude in μV.")
print("To visualize frequency bands, spectral analysis (PSD) is required.")
print("="*70)

