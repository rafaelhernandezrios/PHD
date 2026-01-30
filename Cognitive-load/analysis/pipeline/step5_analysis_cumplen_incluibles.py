"""
Paso 5: Análisis y gráficas solo de sujetos INCLUIBLES que CUMPLEN la hipótesis
- Lee los resultados de step4 (cognitive_load_cleaned_summary.csv)
- Filtra: incluibles (artifact%<=50, n_windows>=2) Y High Load > Low Load
- Genera gráficas y análisis solo para ese subconjunto
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path

# Configuración (misma salida que step4)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output', 'analysis_output')
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Criterios de exclusión (igual que step4)
ARTIFACT_PCT_THRESHOLD = 50.0
MIN_WINDOWS_AFTER = 2

CLEANED_SUMMARY_PATH = os.path.join(OUTPUT_DIR, 'cognitive_load_cleaned_summary.csv')
NORMALIZED_SUMMARY_PATH = os.path.join(OUTPUT_DIR, 'cognitive_load_normalized_summary.csv')


def get_included_subjects(df_summary):
    """Sujetos incluibles: artifact%<=50 y n_windows_after>=2 en low y high."""
    included = []
    excluded = {}
    for subject in df_summary['subject'].unique():
        subject_data = df_summary[df_summary['subject'] == subject]
        low_data = subject_data[subject_data['phase'] == 'low_cognitive_load']
        high_data = subject_data[subject_data['phase'] == 'high_cognitive_load']
        if len(low_data) == 0 or len(high_data) == 0:
            excluded[subject] = "falta fase low o high"
            continue
        low_row = low_data.iloc[0]
        high_row = high_data.iloc[0]
        reasons = []
        if low_row['fz_artifact_pct'] > ARTIFACT_PCT_THRESHOLD or low_row['pz_artifact_pct'] > ARTIFACT_PCT_THRESHOLD:
            reasons.append("low artifacts")
        if high_row['fz_artifact_pct'] > ARTIFACT_PCT_THRESHOLD or high_row['pz_artifact_pct'] > ARTIFACT_PCT_THRESHOLD:
            reasons.append("high artifacts")
        if low_row['n_windows_after'] < MIN_WINDOWS_AFTER:
            reasons.append("low ventanas<2")
        if high_row['n_windows_after'] < MIN_WINDOWS_AFTER:
            reasons.append("high ventanas<2")
        if reasons:
            excluded[subject] = "; ".join(reasons)
        else:
            included.append(subject)
    return included, excluded


def get_cumplen_incluibles(df_summary, included_list):
    """De los incluibles, los que cumplen hipótesis: High > Low (mean_ratio_after)."""
    cumplen = []
    for subject in included_list:
        subject_data = df_summary[df_summary['subject'] == subject]
        low_row = subject_data[subject_data['phase'] == 'low_cognitive_load']
        high_row = subject_data[subject_data['phase'] == 'high_cognitive_load']
        if len(low_row) == 0 or len(high_row) == 0:
            continue
        low_ratio = low_row.iloc[0]['mean_ratio_after']
        high_ratio = high_row.iloc[0]['mean_ratio_after']
        if np.isfinite(high_ratio) and np.isfinite(low_ratio) and high_ratio > low_ratio:
            cumplen.append(subject)
    return cumplen


def load_and_filter():
    """Carga CSVs de step4 y devuelve (df_cleaned_filtered, df_norm_filtered, cumplen_list)."""
    if not os.path.isfile(CLEANED_SUMMARY_PATH):
        raise FileNotFoundError(
            f"No se encontro {CLEANED_SUMMARY_PATH}. Ejecuta antes step4_cognitive_load_cleaned.py"
        )
    df_cleaned = pd.read_csv(CLEANED_SUMMARY_PATH)
    included_list, excluded_dict = get_included_subjects(df_cleaned)
    cumplen_list = get_cumplen_incluibles(df_cleaned, included_list)

    df_cleaned_f = df_cleaned[df_cleaned['subject'].isin(cumplen_list)].copy()
    df_norm_f = None
    if os.path.isfile(NORMALIZED_SUMMARY_PATH):
        df_norm = pd.read_csv(NORMALIZED_SUMMARY_PATH)
        df_norm_f = df_norm[df_norm['subject'].isin(cumplen_list)].copy()
    return df_cleaned_f, df_norm_f, cumplen_list, included_list, excluded_dict


def compute_summary_stats(df_cleaned, subjects):
    """Estadísticas agregadas (media ± std) de Low y High para los sujetos cumplen incluibles."""
    low_ratios = []
    high_ratios = []
    for s in subjects:
        low_row = df_cleaned[(df_cleaned['subject'] == s) & (df_cleaned['phase'] == 'low_cognitive_load')]
        high_row = df_cleaned[(df_cleaned['subject'] == s) & (df_cleaned['phase'] == 'high_cognitive_load')]
        if len(low_row) and len(high_row):
            low_ratios.append(low_row.iloc[0]['mean_ratio_after'])
            high_ratios.append(high_row.iloc[0]['mean_ratio_after'])
    low_ratios = np.array(low_ratios)
    high_ratios = np.array(high_ratios)
    n = len(low_ratios)
    if n == 0:
        return None
    # Cohen's d (paired): (mean_high - mean_low) / pooled_std
    mean_low = np.mean(low_ratios)
    mean_high = np.mean(high_ratios)
    std_low = np.std(low_ratios, ddof=1) if n > 1 else 0.0
    std_high = np.std(high_ratios, ddof=1) if n > 1 else 0.0
    diff = high_ratios - low_ratios
    std_diff = np.std(diff, ddof=1) if n > 1 else 0.0
    cohen_d = (mean_high - mean_low) / std_diff if std_diff > 0 else 0.0
    return {
        'n': n,
        'mean_low': mean_low, 'std_low': std_low,
        'mean_high': mean_high, 'std_high': std_high,
        'mean_diff': mean_high - mean_low,
        'cohen_d': cohen_d,
    }


def plot_absolutes(df_cleaned, subjects, output_dir):
    """Gráfica: ratios absolutos Low vs High por sujeto (solo cumplen incluibles)."""
    phases = ['low_cognitive_load', 'high_cognitive_load']
    phase_labels = ['Low Load', 'High Load']
    data_low = []
    data_high = []
    for s in subjects:
        low_row = df_cleaned[(df_cleaned['subject'] == s) & (df_cleaned['phase'] == 'low_cognitive_load')]
        high_row = df_cleaned[(df_cleaned['subject'] == s) & (df_cleaned['phase'] == 'high_cognitive_load')]
        data_low.append(low_row.iloc[0]['mean_ratio_after'] if len(low_row) else np.nan)
        data_high.append(high_row.iloc[0]['mean_ratio_after'] if len(high_row) else np.nan)

    x = np.arange(len(subjects))
    width = 0.35
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(x - width/2, data_low, width, label='Low Load', color='#81c784', alpha=0.9, edgecolor='white')
    ax.bar(x + width/2, data_high, width, label='High Load', color='#e57373', alpha=0.9, edgecolor='white')
    ax.set_xlabel('Sujeto', fontsize=12, color='white')
    ax.set_ylabel('Cognitive Load Ratio (Theta/Alpha)', fontsize=12, color='white')
    ax.set_title('Solo sujetos incluibles que cumplen hipotesis (High > Low)', fontsize=14, fontweight='bold', color='white')
    ax.set_xticks(x)
    ax.set_xticklabels(subjects, rotation=45, ha='right', color='white')
    ax.legend(facecolor='#21262d', edgecolor='#00ff88', labelcolor='white', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y', color='gray')
    ax.set_facecolor('#0d1117')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')
    fig.patch.set_facecolor('#0d1117')
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'cumplen_incluibles_ratios_absolutos.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f"Grafico guardado: {out_path}")


def plot_normalized(df_norm, subjects, output_dir):
    """Gráfica: ratios normalizados por baseline (solo cumplen incluibles)."""
    phases = ['baseline_eyes_open', 'baseline_eyes_closed', 'low_cognitive_load', 'high_cognitive_load']
    phase_labels = ['Baseline Open', 'Baseline Closed', 'Low Load', 'High Load']
    colors = ['#4fc3f7', '#81c784', '#ffb74d', '#e57373']
    data = {pl: [] for pl in phase_labels}
    for s in subjects:
        for ph, pl in zip(phases, phase_labels):
            row = df_norm[(df_norm['subject'] == s) & (df_norm['phase'] == ph)]
            val = row.iloc[0]['ratio_normalizado'] if len(row) else np.nan
            data[pl].append(val if np.isfinite(val) else np.nan)

    x = np.arange(len(subjects))
    width = 0.2
    fig, ax = plt.subplots(figsize=(14, 7))
    for i, (pl, color) in enumerate(zip(phase_labels, colors)):
        offset = (i - 1.5) * width
        vals = [v if not np.isnan(v) else 0 for v in data[pl]]
        ax.bar(x + offset, vals, width, label=pl, color=color, alpha=0.85, edgecolor='white')
    ax.axhline(y=1.0, color='yellow', linestyle='--', linewidth=1.5, alpha=0.7, label='Baseline=1')
    ax.set_xlabel('Sujeto', fontsize=12, color='white')
    ax.set_ylabel('Ratio normalizado (Baseline = 1.0)', fontsize=12, color='white')
    ax.set_title('Solo cumplen incluibles - Ratios normalizados por baseline', fontsize=14, fontweight='bold', color='white')
    ax.set_xticks(x)
    ax.set_xticklabels(subjects, rotation=45, ha='right', color='white')
    ax.legend(facecolor='#21262d', edgecolor='#00ff88', labelcolor='white', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y', color='gray')
    ax.set_facecolor('#0d1117')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')
    fig.patch.set_facecolor('#0d1117')
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'cumplen_incluibles_ratios_normalizados.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f"Grafico guardado: {out_path}")


def plot_relative_change(df_cleaned, subjects, output_dir):
    """Cambio relativo desde baseline (%) por fase (solo cumplen incluibles)."""
    # Baseline = baseline_eyes_closed (o open si no hay)
    baseline_ratios = {}
    for s in subjects:
        row_c = df_cleaned[(df_cleaned['subject'] == s) & (df_cleaned['phase'] == 'baseline_eyes_closed')]
        if len(row_c) == 0:
            row_c = df_cleaned[(df_cleaned['subject'] == s) & (df_cleaned['phase'] == 'baseline_eyes_open')]
        baseline_ratios[s] = row_c.iloc[0]['mean_ratio_after'] if len(row_c) else np.nan

    phases = ['low_cognitive_load', 'high_cognitive_load']
    phase_labels = ['Low Load', 'High Load']
    data_low_pct = []
    data_high_pct = []
    for s in subjects:
        base = baseline_ratios.get(s)
        low_row = df_cleaned[(df_cleaned['subject'] == s) & (df_cleaned['phase'] == 'low_cognitive_load')]
        high_row = df_cleaned[(df_cleaned['subject'] == s) & (df_cleaned['phase'] == 'high_cognitive_load')]
        low_r = low_row.iloc[0]['mean_ratio_after'] if len(low_row) else np.nan
        high_r = high_row.iloc[0]['mean_ratio_after'] if len(high_row) else np.nan
        if base is not None and np.isfinite(base) and base > 0:
            data_low_pct.append(100 * (low_r - base) / base if np.isfinite(low_r) else np.nan)
            data_high_pct.append(100 * (high_r - base) / base if np.isfinite(high_r) else np.nan)
        else:
            data_low_pct.append(np.nan)
            data_high_pct.append(np.nan)

    x = np.arange(len(subjects))
    width = 0.35
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(x - width/2, data_low_pct, width, label='Low Load', color='#81c784', alpha=0.9, edgecolor='white')
    ax.bar(x + width/2, data_high_pct, width, label='High Load', color='#e57373', alpha=0.9, edgecolor='white')
    ax.axhline(y=0, color='yellow', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('Sujeto', fontsize=12, color='white')
    ax.set_ylabel('Cambio relativo desde baseline (%)', fontsize=12, color='white')
    ax.set_title('Solo cumplen incluibles - Cambio relativo desde baseline', fontsize=14, fontweight='bold', color='white')
    ax.set_xticks(x)
    ax.set_xticklabels(subjects, rotation=45, ha='right', color='white')
    ax.legend(facecolor='#21262d', edgecolor='#00ff88', labelcolor='white', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y', color='gray')
    ax.set_facecolor('#0d1117')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')
    fig.patch.set_facecolor('#0d1117')
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'cumplen_incluibles_cambio_relativo.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f"Grafico guardado: {out_path}")


def save_cumplen_summary_csv(df_cleaned, df_norm, subjects, stats, output_dir):
    """Guarda CSV con resumen solo de cumplen incluibles (una fila por sujeto con low/high, etc.)."""
    rows = []
    for s in subjects:
        low_row = df_cleaned[(df_cleaned['subject'] == s) & (df_cleaned['phase'] == 'low_cognitive_load')].iloc[0]
        high_row = df_cleaned[(df_cleaned['subject'] == s) & (df_cleaned['phase'] == 'high_cognitive_load')].iloc[0]
        norm_low = norm_high = np.nan
        if df_norm is not None:
            nlow = df_norm[(df_norm['subject'] == s) & (df_norm['phase'] == 'low_cognitive_load')]
            nhigh = df_norm[(df_norm['subject'] == s) & (df_norm['phase'] == 'high_cognitive_load')]
            if len(nlow):
                norm_low = nlow.iloc[0]['ratio_normalizado']
            if len(nhigh):
                norm_high = nhigh.iloc[0]['ratio_normalizado']
        rows.append({
            'subject': s,
            'mean_ratio_low': low_row['mean_ratio_after'],
            'mean_ratio_high': high_row['mean_ratio_after'],
            'ratio_normalizado_low': norm_low,
            'ratio_normalizado_high': norm_high,
            'n_windows_low': int(low_row['n_windows_after']),
            'n_windows_high': int(high_row['n_windows_after']),
            'fz_artifact_pct_low': low_row['fz_artifact_pct'],
            'pz_artifact_pct_low': low_row['pz_artifact_pct'],
            'fz_artifact_pct_high': high_row['fz_artifact_pct'],
            'pz_artifact_pct_high': high_row['pz_artifact_pct'],
        })
    df_out = pd.DataFrame(rows)
    out_path = os.path.join(output_dir, 'cumplen_incluibles_summary.csv')
    df_out.to_csv(out_path, index=False)
    print(f"CSV guardado: {out_path}")

    if stats:
        stats_path = os.path.join(output_dir, 'cumplen_incluibles_estadisticas.txt')
        with open(stats_path, 'w', encoding='utf-8') as f:
            f.write("ESTADISTICAS AGREGADAS (solo cumplen incluibles)\n")
            f.write("="*50 + "\n")
            f.write(f"N = {stats['n']}\n")
            f.write(f"Low  Load:  mean = {stats['mean_low']:.4f}  std = {stats['std_low']:.4f}\n")
            f.write(f"High Load:  mean = {stats['mean_high']:.4f}  std = {stats['std_high']:.4f}\n")
            f.write(f"Diferencia (High - Low): mean = {stats['mean_diff']:.4f}\n")
            f.write(f"Cohen's d (paired): {stats['cohen_d']:.4f}\n")
        print(f"Estadisticas guardadas: {stats_path}")


def main():
    print("="*80)
    print("PASO 5: ANALISIS SOLO SUJETOS INCLUIBLES QUE CUMPLEN HIPOTESIS (High > Low)")
    print("="*80)
    print(f"Lectura desde: {CLEANED_SUMMARY_PATH}")
    print(f"Salida: {OUTPUT_DIR}")
    print(f"Criterios: artifact%<={ARTIFACT_PCT_THRESHOLD:.0f}%, n_windows>={MIN_WINDOWS_AFTER}")

    df_cleaned_f, df_norm_f, cumplen_list, included_list, excluded_dict = load_and_filter()

    print(f"\nSujetos incluibles (calidad OK): {len(included_list)} -> {included_list}")
    print(f"Sujetos que cumplen hipotesis (High > Low) entre incluibles: {len(cumplen_list)} -> {cumplen_list}")

    if not cumplen_list:
        print("\nNo hay ningun sujeto cumplen incluible. No se generan graficas.")
        return

    stats = compute_summary_stats(df_cleaned_f, cumplen_list)
    if stats:
        print("\nESTADISTICAS AGREGADAS (solo cumplen incluibles):")
        print(f"  N = {stats['n']}")
        print(f"  Low  Load:  mean = {stats['mean_low']:.4f}  std = {stats['std_low']:.4f}")
        print(f"  High Load:  mean = {stats['mean_high']:.4f}  std = {stats['std_high']:.4f}")
        print(f"  Diferencia (High - Low): mean = {stats['mean_diff']:.4f}")
        print(f"  Cohen's d (paired): {stats['cohen_d']:.4f}")

    plot_absolutes(df_cleaned_f, cumplen_list, OUTPUT_DIR)
    plot_relative_change(df_cleaned_f, cumplen_list, OUTPUT_DIR)
    if df_norm_f is not None:
        plot_normalized(df_norm_f, cumplen_list, OUTPUT_DIR)
    save_cumplen_summary_csv(df_cleaned_f, df_norm_f, cumplen_list, stats, OUTPUT_DIR)

    print("\n" + "="*80)
    print("PASO 5 COMPLETADO")
    print("="*80)


if __name__ == '__main__':
    main()
