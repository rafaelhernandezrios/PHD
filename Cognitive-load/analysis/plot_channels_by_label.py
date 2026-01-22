"""
Script para graficar los datos de los canales agrupados por label (fase del experimento).
Paso 2: Agregar cálculo y visualización de carga cognitiva.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from pathlib import Path
import os

# Configuración
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, 'data', 'DATA', 'Data-Experimento-Rafa', 'data_Joss', 'eeg_data_20260120_120639.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output', 'analysis_output')
SAMPLE_RATE = 250  # Hz
THETA_BAND = (4.0, 7.0)  # Hz
ALPHA_BAND = (8.0, 12.0)  # Hz
WINDOW_DURATION = 2.0  # segundos
WINDOW_SAMPLES = int(WINDOW_DURATION * SAMPLE_RATE)  # 500 muestras
FZ_CHANNEL = 3  # Canal Fz (frontal)
PZ_CHANNEL = 6  # Canal Pz (parietal)

# Crear directorio de salida si no existe
Path(OUTPUT_DIR).mkdir(exist_ok=True)

# Mapeo de canales a nombres de electrodos
CHANNEL_NAMES = {
    0: 'Fp1',
    1: 'Fp2',
    2: 'F3',
    3: 'Fz',
    4: 'F4',
    5: 'P3',
    6: 'Pz',
    7: 'P4'
}

# Labels de interés (excluir setup, completed, analysis)
LABELS_OF_INTEREST = [
    'baseline_eyes_open',
    'baseline_eyes_closed',
    'low_cognitive_load',
    'high_cognitive_load'
]

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
    
    # Calcular potencia promedio en la banda
    bandpower = np.trapz(psd[idx_band], freqs[idx_band])
    
    return bandpower

def calculate_cognitive_load_by_phase(df, labels_of_interest):
    """
    Calcula la carga cognitiva (ratio Theta/Alpha) para cada fase usando ventanas deslizantes.
    
    Args:
        df: DataFrame con los datos
        labels_of_interest: Lista de labels a analizar
        
    Returns:
        Diccionario con ratios por fase
    """
    results = {}
    min_samples = SAMPLE_RATE // 2  # Mínimo 0.5 segundos (125 muestras)
    
    for label in labels_of_interest:
        label_data = df[df['label'] == label].copy()
        
        if len(label_data) < min_samples:
            print(f"  Advertencia: {label} tiene muy pocos datos ({len(label_data)} muestras, mínimo {min_samples})")
            results[label] = []
            continue
        
        # Extraer señales Fz y Pz
        fz_signal = label_data[f'channel_{FZ_CHANNEL}'].values
        pz_signal = label_data[f'channel_{PZ_CHANNEL}'].values
        
        ratios = []
        
        # Ajustar tamaño de ventana según datos disponibles
        if len(label_data) >= WINDOW_SAMPLES:
            # Usar ventana completa de 2 segundos
            actual_window = WINDOW_SAMPLES
            step_samples = WINDOW_SAMPLES // 2  # 50% overlap
        else:
            # Usar toda la señal disponible si es menor que 2 segundos pero >= 0.5 segundos
            actual_window = len(label_data)
            step_samples = actual_window  # Solo una ventana
        
        # Ventaneo deslizante
        for start_idx in range(0, len(fz_signal) - actual_window + 1, step_samples):
            end_idx = start_idx + actual_window
            
            fz_window = fz_signal[start_idx:end_idx]
            pz_window = pz_signal[start_idx:end_idx]
            
            # Calcular potencias
            theta_power = calculate_bandpower(fz_window, THETA_BAND, SAMPLE_RATE)
            alpha_power = calculate_bandpower(pz_window, ALPHA_BAND, SAMPLE_RATE)
            
            # Calcular ratio
            if alpha_power > 0 and np.isfinite(theta_power) and np.isfinite(alpha_power):
                ratio = theta_power / alpha_power
                if np.isfinite(ratio) and 0.1 < ratio < 50:  # Filtro de cordura
                    ratios.append(ratio)
        
        results[label] = ratios
        if len(ratios) > 0:
            print(f"  {label}: {len(ratios)} ratios calculados (ventana: {actual_window} muestras = {actual_window/SAMPLE_RATE:.2f}s)")
        else:
            print(f"  {label}: No se pudieron calcular ratios")
    
    return results

def load_data(csv_path):
    """Carga los datos del CSV."""
    print(f"Cargando datos de {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Total de muestras: {len(df)}")
    print(f"Labels encontrados: {df['label'].unique()}")
    return df

def plot_channels_by_label(df, labels_of_interest=None, cognitive_load_ratios=None):
    """
    Grafica los datos de cada canal agrupados por label y la carga cognitiva.
    
    Args:
        df: DataFrame con los datos
        labels_of_interest: Lista de labels a graficar (None = todos)
        cognitive_load_ratios: Diccionario con ratios por fase
    """
    if labels_of_interest is None:
        labels_of_interest = df['label'].unique()
    
    # Filtrar datos de interés
    df_filtered = df[df['label'].isin(labels_of_interest)].copy()
    
    print(f"\nGraficando {len(labels_of_interest)} labels...")
    print(f"Muestras por label:")
    print(df_filtered['label'].value_counts())
    
    # Crear figura con subplots: Fz, Pz + 1 gráfico de carga cognitiva
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 1, hspace=0.3)
    fig.suptitle('Fz, Pz Channels and Cognitive Load by Experiment Phase', 
                 fontsize=16, fontweight='bold')
    
    axes = [fig.add_subplot(gs[i, 0]) for i in range(3)]
    
    # Colores para cada label
    label_colors = {
        'baseline_eyes_open': 'lightblue',
        'baseline_eyes_closed': 'lightgreen',
        'low_cognitive_load': 'lightyellow',
        'high_cognitive_load': 'lightcoral',
        'setup': 'gray',
        'baseline_completed': 'gray',
        'low_load_completed': 'gray',
        'analysis': 'gray'
    }
    
    # Canales a graficar: Fz y Pz
    channels_to_plot = [FZ_CHANNEL, PZ_CHANNEL]
    
    # Graficar Fz y Pz
    for i, channel_idx in enumerate(channels_to_plot):
        ax = axes[i]
        channel_name = CHANNEL_NAMES[channel_idx]
        channel_col = f'channel_{channel_idx}'
        
        # Graficar datos de cada label
        for label in labels_of_interest:
            label_data = df_filtered[df_filtered['label'] == label]
            if len(label_data) > 0:
                # Crear índice de tiempo relativo para cada fase
                time_idx = np.arange(len(label_data))
                ax.plot(
                    time_idx,
                    label_data[channel_col].values,
                    label=label,
                    color=label_colors.get(label, 'black'),
                    alpha=0.7,
                    linewidth=0.5
                )
        
        ax.set_ylabel(f'{channel_name}\n(μV)', fontsize=12, fontweight='bold')
        ax.set_title(f'Channel {channel_idx}: {channel_name} - {"Theta (4-7 Hz)" if channel_idx == FZ_CHANNEL else "Alpha (8-12 Hz)"}', 
                    fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)
        
        # Solo mostrar etiquetas de tiempo en el último subplot de canales
        if i < len(channels_to_plot) - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('Sample Index', fontsize=11)
    
    # Último subplot: Carga cognitiva
    ax_cognitive = axes[2]
    
    if cognitive_load_ratios:
        # Preparar datos para boxplot
        ratio_data = []
        phase_names = []
        
        for label in labels_of_interest:
            if label in cognitive_load_ratios and len(cognitive_load_ratios[label]) > 0:
                ratio_data.append(cognitive_load_ratios[label])
                phase_names.append(label.replace('_', ' ').title())
        
        if ratio_data:
            # Boxplot
            bp = ax_cognitive.boxplot(ratio_data, labels=phase_names, patch_artist=True)
            
            # Colorear boxes
            colors_list = [label_colors.get(label, 'gray') for label in labels_of_interest 
                          if label in cognitive_load_ratios and len(cognitive_load_ratios[label]) > 0]
            for patch, color in zip(bp['boxes'], colors_list):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax_cognitive.set_ylabel('Cognitive Load Ratio\n(Theta/Alpha)', fontsize=12, fontweight='bold')
            ax_cognitive.set_title('Cognitive Load by Phase', fontsize=13, fontweight='bold')
            ax_cognitive.grid(True, alpha=0.3, axis='y')
            ax_cognitive.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, 
                                label='Ratio = 1.0')
            ax_cognitive.legend(fontsize=9)
            
            # Agregar estadísticas (recuadro más visible)
            stats_text = "STATISTICS\n" + "="*30 + "\n"
            for i, label in enumerate(labels_of_interest):
                if label in cognitive_load_ratios and len(cognitive_load_ratios[label]) > 0:
                    ratios = cognitive_load_ratios[label]
                    stats_text += f"\n{label.replace('_', ' ').title()}:\n"
                    stats_text += f"  Mean: {np.mean(ratios):.3f}\n"
                    stats_text += f"  Median: {np.median(ratios):.3f}\n"
                    stats_text += f"  Std: {np.std(ratios):.3f}\n"
                    stats_text += f"  N: {len(ratios)}\n"
            
            # Recuadro más grande y visible
            ax_cognitive.text(0.98, 0.98, stats_text, transform=ax_cognitive.transAxes,
                            fontsize=10, verticalalignment='top', horizontalalignment='right',
                            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8, 
                                    edgecolor='black', linewidth=2))
    
    plt.tight_layout()
    
    # Guardar figura
    output_path = f'{OUTPUT_DIR}/channels_and_cognitive_load.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n[OK] Grafico guardado en: {output_path}")
    
    plt.show()

def analyze_cognitive_load_results(cognitive_load_ratios):
    """
    Analiza y presenta conclusiones sobre la carga cognitiva.
    
    Args:
        cognitive_load_ratios: Diccionario con ratios por fase
    """
    print("\n" + "="*60)
    print("ANÁLISIS DE CARGA COGNITIVA")
    print("="*60)
    
    # Calcular estadísticas por fase
    phase_stats = {}
    for label in LABELS_OF_INTEREST:
        if label in cognitive_load_ratios and len(cognitive_load_ratios[label]) > 0:
            ratios = cognitive_load_ratios[label]
            phase_stats[label] = {
                'mean': np.mean(ratios),
                'median': np.median(ratios),
                'std': np.std(ratios),
                'min': np.min(ratios),
                'max': np.max(ratios),
                'n': len(ratios)
            }
    
    # Mostrar estadísticas
    print("\nEstadísticas por fase:")
    print("-" * 60)
    for label, stats in phase_stats.items():
        print(f"\n{label.replace('_', ' ').title()}:")
        print(f"  Media: {stats['mean']:.3f}")
        print(f"  Mediana: {stats['median']:.3f}")
        print(f"  Desv. Est.: {stats['std']:.3f}")
        print(f"  Rango: [{stats['min']:.3f}, {stats['max']:.3f}]")
        print(f"  N: {stats['n']}")
    
    # Comparaciones
    print("\n" + "="*60)
    print("CONCLUSIONES:")
    print("="*60)
    
    if 'high_cognitive_load' in phase_stats and 'low_cognitive_load' in phase_stats:
        high_mean = phase_stats['high_cognitive_load']['mean']
        low_mean = phase_stats['low_cognitive_load']['mean']
        diff = high_mean - low_mean
        
        print(f"\n1. Comparación High vs Low Cognitive Load:")
        print(f"   - High Load: {high_mean:.3f}")
        print(f"   - Low Load: {low_mean:.3f}")
        print(f"   - Diferencia: {diff:.3f}")
        
        if diff > 0:
            print(f"   [OK] La carga cognitiva es {diff:.3f} unidades MAYOR en High Load")
            print(f"   [OK] Esto indica mayor esfuerzo mental durante la tarea Stroop")
        else:
            print(f"   [ADVERTENCIA] La carga cognitiva es {abs(diff):.3f} unidades MENOR en High Load")
            print(f"   [ADVERTENCIA] Resultado inesperado - revisar datos o metodologia")
    
    if 'baseline_eyes_open' in phase_stats and 'baseline_eyes_closed' in phase_stats:
        open_mean = phase_stats['baseline_eyes_open']['mean']
        closed_mean = phase_stats['baseline_eyes_closed']['mean']
        
        print(f"\n2. Comparación Baseline:")
        print(f"   - Eyes Open: {open_mean:.3f}")
        print(f"   - Eyes Closed: {closed_mean:.3f}")
        
        if closed_mean > open_mean:
            print(f"   [OK] Mayor ratio con ojos cerrados (esperado - mas relajacion)")
        else:
            print(f"   [INFO] Mayor ratio con ojos abiertos")
    
    # Interpretación general
    print(f"\n3. Interpretacion del Ratio Theta/Alpha:")
    print(f"   - Ratio > 1.0: Mayor carga cognitiva (mas Theta relativo a Alpha)")
    print(f"   - Ratio < 1.0: Menor carga cognitiva (mas Alpha relativo a Theta)")
    print(f"   - Ratio ~ 1.0: Carga cognitiva moderada")
    
    # Ordenar fases por carga cognitiva
    if len(phase_stats) > 0:
        sorted_phases = sorted(phase_stats.items(), key=lambda x: x[1]['mean'], reverse=True)
        print(f"\n4. Orden de carga cognitiva (de mayor a menor):")
        for i, (label, stats) in enumerate(sorted_phases, 1):
            print(f"   {i}. {label.replace('_', ' ').title()}: {stats['mean']:.3f}")
    
    print("\n" + "="*60)

def main():
    """Función principal."""
    # Cargar datos
    df = load_data(CSV_PATH)
    
    # Calcular carga cognitiva por fase
    print("\nCalculando carga cognitiva por fase...")
    cognitive_load_ratios = calculate_cognitive_load_by_phase(df, LABELS_OF_INTEREST)
    
    # Analizar resultados
    analyze_cognitive_load_results(cognitive_load_ratios)
    
    # Graficar canales por label y carga cognitiva
    plot_channels_by_label(df, labels_of_interest=LABELS_OF_INTEREST, 
                          cognitive_load_ratios=cognitive_load_ratios)
    
    print("\n" + "="*60)
    print("ANÁLISIS COMPLETADO")
    print("="*60)

if __name__ == '__main__':
    main()
