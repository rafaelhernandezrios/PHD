"""
WAAF: Wavelet-Assisted Adaptive Filter
Implementación en Python del método dcaro_WAAF de MATLAB.

Basado en:
Peng, H., et al. (2013). Removal of ocular artifacts in EEG—An improved approach 
combining DWT and ANC for portable applications. IEEE journal of biomedical and 
health informatics, 17(3), 600-607.

Coifman, R. R., & Donoho, D. L. (1995). Translation-invariant de-noising. 
In Wavelets and statistics (pp. 125-150).
"""

import numpy as np
try:
    import pywt
    HAS_WAVELETS = True
except ImportError:
    HAS_WAVELETS = False
    print("Advertencia: PyWavelets no esta instalado. Instalar con: pip install PyWavelets")


def waaf_filter(eeg, wavelet='db4', level=7, attn=None):
    """
    WAAF: Wavelet-Assisted Adaptive Filter
    
    Elimina artefactos oculares de señales EEG creando una señal de referencia
    que contiene solo estos artefactos usando soft thresholding en niveles de
    baja frecuencia de una descomposición wavelet. Esta señal de referencia se
    remueve de la señal EEG original a través de un algoritmo de Cancelación
    Adaptativa de Ruido (ANC) basado en Recursive Least Squares (RLS).
    
    Args:
        eeg: Array 2D con datos EEG [n_channels, n_samples] o [n_samples, n_channels]
        wavelet: Wavelet madre para descomposición (default: 'db4')
        level: Nivel de descomposición wavelet (default: 7)
        attn: Niveles de coeficientes de detalle a atenuar (default: [1,2,3,4])
        
    Returns:
        eeg_clean: Datos EEG denoised [misma forma que entrada]
        refs: Señales de referencia removidas [misma forma que entrada]
        weights: Pesos retornados por el algoritmo RLS [misma forma que entrada]
    """
    if not HAS_WAVELETS:
        raise ImportError("PyWavelets es requerido. Instalar con: pip install PyWavelets")
    
    # Determinar orientación de los datos
    if eeg.ndim == 1:
        eeg = eeg.reshape(1, -1)
        was_1d = True
    else:
        was_1d = False
    
    # Determinar si es [channels, samples] o [samples, channels]
    if eeg.shape[0] < eeg.shape[1]:
        # Asumir [channels, samples]
        transpose_back = False
        n_channels, n_samples = eeg.shape
    else:
        # Asumir [samples, channels]
        eeg = eeg.T
        transpose_back = True
        n_channels, n_samples = eeg.shape
    
    # Niveles a atenuar por defecto
    if attn is None:
        attn = list(range(1, min(5, level + 1)))  # Niveles 1-4 por defecto
    
    # Inicializar arrays de salida
    eeg_clean = np.zeros_like(eeg)
    refs = np.zeros_like(eeg)
    weights = np.zeros((n_channels, n_samples))
    
    # Procesar cada canal
    for j in range(n_channels):
        x = eeg[j, :].copy()
        
        # Ajustar nivel dinámicamente según longitud de señal
        max_level = pywt.dwt_max_level(n_samples, wavelet)
        actual_level = min(level, max_level)
        
        if actual_level < 1:
            # Señal muy corta, saltar WAAF
            eeg_clean[j, :] = x
            refs[j, :] = np.zeros_like(x)
            weights[j, :] = np.ones(n_samples)
            continue
        
        # Descomposición wavelet
        coeffs = pywt.wavedec(x, wavelet, level=actual_level)
        cA = coeffs[0]  # Coeficiente de aproximación
        cD_list = coeffs[1:]  # Lista de coeficientes de detalle
        
        # Ajustar attn al nivel real
        actual_attn = [a for a in attn if a <= actual_level]
        if not actual_attn:
            actual_attn = list(range(1, min(5, actual_level + 1)))
        
        # Calcular umbrales y aplicar soft thresholding
        T = np.zeros(actual_level)
        cD_thresholded = []
        
        for i in range(actual_level):
            if (i + 1) in actual_attn:  # i+1 porque attn es 1-indexed
                # Usar umbral propuesto en Coifman & Donoho (1995)
                detail = cD_list[i]
                sj = np.median(np.abs(detail - np.median(detail))) / 0.6745
                T[i] = sj * np.sqrt(2 * np.log(len(detail)))
                # Soft thresholding
                cD_thresholded.append(pywt.threshold(detail, T[i], mode='soft'))
            else:
                # No aplicar thresholding a estos niveles
                cD_thresholded.append(cD_list[i])
        
        # Reconstruir señal de referencia
        coeffs_ref = [cA] + cD_thresholded
        ref = pywt.waverec(coeffs_ref, wavelet)
        
        # Asegurar misma longitud (waverec puede cambiar la longitud ligeramente)
        if len(ref) != n_samples:
            if len(ref) > n_samples:
                ref = ref[:n_samples]
            else:
                # Interpolar si es más corta usando numpy
                indices_old = np.linspace(0, len(ref) - 1, len(ref))
                indices_new = np.linspace(0, len(ref) - 1, n_samples)
                ref = np.interp(indices_new, indices_old, ref)
        
        # Asegurar que ref tenga exactamente n_samples
        ref = np.array(ref[:n_samples]) if len(ref) >= n_samples else np.pad(ref, (0, n_samples - len(ref)), mode='edge')
        assert len(ref) == n_samples, f"ref tiene longitud {len(ref)}, esperada {n_samples}"
        
        # ANC basado en RLS (Recursive Least Squares)
        # Orden M=1; P es escalar (tomado directamente de Peng et al. 2013)
        e = np.zeros(n_samples)
        P = 1e4  # Inverse Autocorrelation
        lambda_forget = 0.98  # Forgetting factor
        w = np.ones(n_samples)  # Pesos iniciales (ajustado a n_samples)
        
        for k in range(n_samples):
            # Cross-correlation
            Pi = ref[k] * P
            
            # Obtener ganancia y salida del filtro
            g = Pi / (lambda_forget + Pi * ref[k])
            y = w[k] * ref[k]
            
            # Actualizar EEG
            e[k] = x[k] - y
            
            # Actualizar pesos y matriz de correlación
            if k < n_samples - 1:
                w[k + 1] = w[k] + g * e[k]
                P = (P - g * Pi) / lambda_forget
        
        # Asegurar que todos los arrays tengan exactamente n_samples
        assert len(ref) == n_samples, f"ref tiene longitud {len(ref)}, esperada {n_samples}"
        assert len(e) == n_samples, f"e tiene longitud {len(e)}, esperada {n_samples}"
        assert len(w) == n_samples, f"w tiene longitud {len(w)}, esperada {n_samples}"
        
        refs[j, :] = ref
        eeg_clean[j, :] = e
        weights[j, :] = w
    
    # Restaurar orientación original
    if transpose_back:
        eeg_clean = eeg_clean.T
        refs = refs.T
        weights = weights.T
    
    # Si era 1D, retornar 1D
    if was_1d:
        eeg_clean = eeg_clean.flatten()
        refs = refs.flatten()
        weights = weights.flatten()
    
    return eeg_clean, refs, weights


def waaf_filter_2d(eeg_matrix, wavelet='db4', level=7, attn=None):
    """
    Versión optimizada para matrices 2D [n_samples, n_channels].
    
    Args:
        eeg_matrix: Array 2D [n_samples, n_channels]
        wavelet: Wavelet madre (default: 'db4')
        level: Nivel de descomposición (default: 7)
        attn: Niveles a atenuar (default: [1,2,3,4])
        
    Returns:
        eeg_clean: Matriz denoised [n_samples, n_channels]
        refs: Referencias removidas [n_samples, n_channels]
        weights: Pesos RLS [n_samples, n_channels]
    """
    # Transponer para procesar como [channels, samples]
    eeg_T = eeg_matrix.T
    eeg_clean_T, refs_T, weights_T = waaf_filter(eeg_T, wavelet, level, attn)
    
    # Transponer de vuelta
    return eeg_clean_T.T, refs_T.T, weights_T.T
