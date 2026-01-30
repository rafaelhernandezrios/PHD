"""
Análisis y gráficas: sesiones que CUMPLEN la hipótesis (actividad > baseline)
+ TODAS las sesiones de Rafa y Joss aunque no cumplan.

- Lee jeronimo_cognitive_load_summary.csv
- Incluye: sesiones con al menos una fase normalized_ratio > 1, y todas las de Rafa y Joss
- Genera gráficas para ese conjunto de sesiones
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_CSV = os.path.join(BASE_DIR, 'output', 'jeronimo_analysis', 'jeronimo_cognitive_load_summary.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output', 'jeronimo_analysis')
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


def get_sessions_to_include(df_summary):
    """
    Sesiones a incluir: (1) las que cumplen hipótesis (al menos una fase > baseline),
    (2) TODAS las sesiones de Rafa y Joss aunque no cumplan.
    Returns: list of session names
    """
    df_with_norm = df_summary.dropna(subset=['normalized_ratio'])
    cumplen = []
    if len(df_with_norm) > 0:
        cumplen = df_with_norm.groupby('session')['normalized_ratio'].apply(
            lambda x: (x > 1).any()
        )
        cumplen = cumplen[cumplen].index.tolist()
    # Todas las sesiones de Rafa y Joss (aunque no cumplan)
    all_sessions = df_summary['session'].unique()
    rafa_joss = [s for s in all_sessions if s.startswith('Rafa-') or s.startswith('Joss-')]
    return sorted(set(cumplen) | set(rafa_joss))


def filter_df(df_summary, sessions_to_include):
    """Filtra el DataFrame a las sesiones a incluir."""
    return df_summary[df_summary['session'].isin(sessions_to_include)].copy()


def plot_by_session(df, output_dir):
    """Gráfica: por sesión (cumplen), barras mean_ratio por fase y línea baseline."""
    sessions = df['session'].unique()
    sessions = sorted(sessions)
    n = len(sessions)
    if n == 0:
        return
    n_cols = 3
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    for idx, session in enumerate(sessions):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]
        data = df[df['session'] == session]
        phase_order = sorted(data['phase'].unique(), key=lambda x: (0 if x == 'pre' else 1, x))
        data = data.set_index('phase').loc[phase_order].reset_index()
        phases = data['phase'].tolist()
        means = data['mean_ratio'].tolist()
        baseline_val = data['baseline_ratio'].iloc[0] if 'baseline_ratio' in data.columns else np.nan
        x = np.arange(len(phases))
        colors = ['#81c784' if p == 'pre' else '#ffb74d' if p.startswith('segment') else '#4fc3f7' for p in phases]
        ax.bar(x, means, color=colors, alpha=0.9, edgecolor='white')
        if np.isfinite(baseline_val):
            ax.axhline(y=baseline_val, color='red', linestyle='--', linewidth=1.5, label='Baseline')
        ax.set_xticks(x)
        ax.set_xticklabels(phases, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Theta/Alpha ratio')
        ax.set_title(session[:28] + ('...' if len(session) > 28 else ''), fontsize=9)
        if np.isfinite(baseline_val):
            ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_facecolor('#fafafa')
    for idx in range(len(sessions), n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].set_visible(False)
    plt.suptitle('Cognitive load por tarea (Jeronimo)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'jeronimo_cumplen_by_session.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Grafico guardado: {out_path}")


def plot_normalized_cumplen(df, output_dir):
    """Gráfica: ratio normalizado (baseline=1) por fase, solo sesiones cumplen."""
    df_n = df.dropna(subset=['normalized_ratio'])
    if len(df_n) == 0:
        return
    sessions = sorted(df_n['session'].unique())
    phases_all = sorted(df_n['phase'].unique(), key=lambda x: (0 if x == 'pre' else 1, x))
    x = np.arange(len(phases_all))
    width = 0.8 / max(len(sessions), 1)
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, session in enumerate(sessions):
        data = df_n[df_n['session'] == session]
        vals = []
        for p in phases_all:
            row = data[data['phase'] == p]
            vals.append(row['normalized_ratio'].iloc[0] if len(row) else np.nan)
        offset = (i - len(sessions) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=session[:24], alpha=0.85, edgecolor='white')
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Baseline = 1')
    ax.set_xticks(x)
    ax.set_xticklabels(phases_all, rotation=45, ha='right')
    ax.set_ylabel('Ratio normalizado (Baseline = 1)')
    ax.set_title('Cognitive load por tarea (Jeronimo)')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_facecolor('#fafafa')
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'jeronimo_cumplen_normalized.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Grafico guardado: {out_path}")


def plot_by_subject(df, output_dir):
    """Gráfica: agrupada por sujeto (Dani, Edwin, Joss, Rafa), barras por sesión/fase."""
    df_n = df.dropna(subset=['normalized_ratio'])
    if len(df_n) == 0:
        return
    # Extraer sujeto: primera parte del session (Dani-CanineQuest -> Dani)
    df_n = df_n.copy()
    df_n['subject'] = df_n['session'].str.split('-').str[0]
    subjects = ['Dani', 'Edwin', 'Joss', 'Rafa']
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for idx, subject in enumerate(subjects):
        ax = axes[idx]
        data = df_n[df_n['subject'] == subject]
        if len(data) == 0:
            ax.set_visible(False)
            continue
        # Por sesión y fase: valor normalizado
        sessions_subj = sorted(data['session'].unique())
        phases_all = sorted(data['phase'].unique(), key=lambda x: (0 if x == 'pre' else 1, x))
        x_labels = []
        vals = []
        for sess in sessions_subj:
            for ph in phases_all:
                row = data[(data['session'] == sess) & (data['phase'] == ph)]
                if len(row):
                    x_labels.append(f"{sess.replace(subject + '-', '')}\n{ph}")
                    vals.append(row['normalized_ratio'].iloc[0])
        x = np.arange(len(vals))
        colors = ['#81c784' if 'pre' in str(l) else '#ffb74d' for l in x_labels]
        ax.bar(x, vals, color=colors, alpha=0.9, edgecolor='white')
        ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, label='Baseline=1')
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=7)
        ax.set_ylabel('Ratio normalizado')
        ax.set_title(f'{subject} (todas sus sesiones)')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_facecolor('#fafafa')
    plt.suptitle('Cognitive load por tarea vs baseline por sujeto', fontsize=12, fontweight='bold')
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'jeronimo_cumplen_by_subject.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Grafico guardado: {out_path}")


def main():
    print("="*80)
    print("ANALISIS: CUMPLEN HIPOTESIS + TODAS SESIONES RAFA Y JOSS")
    print("="*80)
    print(f"Entrada: {INPUT_CSV}")
    print(f"Salida: {OUTPUT_DIR}")
    print()

    if not os.path.isfile(INPUT_CSV):
        print(f"ERROR: No existe {INPUT_CSV}. Ejecuta antes step_jeronimo_cognitive_load.py")
        return

    df = pd.read_csv(INPUT_CSV)
    sessions_to_include = get_sessions_to_include(df)
    print(f"Sesiones incluidas (cumplen + todas Rafa/Joss): {len(sessions_to_include)}")
    for s in sessions_to_include:
        print(f"  - {s}")

    df_out = filter_df(df, sessions_to_include)
    if len(df_out) == 0:
        print("No hay sesiones para incluir. Nada que graficar.")
        return

    csv_out = os.path.join(OUTPUT_DIR, 'jeronimo_cumplen_hipotesis_summary.csv')
    df_out.to_csv(csv_out, index=False)
    print(f"\nResumen guardado: {csv_out}")

    plot_by_session(df_out, OUTPUT_DIR)
    plot_normalized_cumplen(df_out, OUTPUT_DIR)
    plot_by_subject(df_out, OUTPUT_DIR)

    print("\n" + "="*80)
    print("COMPLETADO")
    print("="*80)


if __name__ == '__main__':
    main()
