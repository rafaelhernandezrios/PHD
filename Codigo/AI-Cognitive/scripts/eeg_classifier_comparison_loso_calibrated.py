"""Compara RF/HistGB/SVM/LogReg/MLP bajo LOSO + calibracion por sujeto,
sobre el feature set limpio y correctamente etiquetado (218 features,
montaje real Fp1,Fp2,F7,F3,Fz,F4,F8,C4) -- eeg_window_features_aura.csv.

La comparacion original de 5 clasificadores (eeg_train_window_classifier.py)
se hizo con split por archivo (no LOSO) sobre el CSV crudo de 656 columnas
(incluye acelerometro/bateria/status). Esta version repite la comparacion
bajo el protocolo que realmente se publica (LOSO + calibracion), para saber
si la eleccion de Random Forest se sostiene ahi tambien.

Target: load_2 (normal vs alta). Guarda resultados en
csv/classifier_comparison_loso_calibrated.json
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "csv" / "eeg_window_features_aura.csv"
OUT = ROOT / "csv" / "classifier_comparison_loso_calibrated.json"

META = {"file_name", "subject_id", "task_type", "condition_4", "load_3", "load_2",
        "window_start_idx", "window_end_idx"}

ALL_CH = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "C4"]
REGION = ["theta_Fz_over_alpha_C4", "frontal_mean_theta", "central_mean_theta",
          "front_minus_central_theta", "frontal_mean_alpha", "central_mean_alpha",
          "front_minus_central_alpha", "frontal_mean_beta", "central_mean_beta",
          "front_minus_central_beta"]


def feature_cols(df):
    cols = []
    for c in df.columns:
        if c in META:
            continue
        if any(c.startswith(ch + "_") for ch in ALL_CH) and c.startswith(tuple(ch + "_" for ch in ALL_CH)):
            cols.append(c)
        elif c in REGION:
            cols.append(c)
    return [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]


def fmat(df, cols):
    X = df[cols].to_numpy(dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
    return np.clip(X, -1e10, 1e10)


def per_subject_calibrate(df, cols):
    out = df.copy()
    for s, g in df.groupby("subject_id"):
        b = g[g.condition_4.isin(["natural", "normal"])]
        base = b if len(b) >= 5 else g
        mu = base[cols].mean().to_numpy(float)
        sd = base[cols].std().to_numpy(float)
        sd = np.where(np.isfinite(sd) & (sd > 1e-9), sd, 1.0)
        Z = (fmat(g, cols) - mu) / sd
        out.loc[g.index, cols] = np.clip(np.nan_to_num(Z), -50, 50)
    return out


MODELS = {
    "RandomForest": lambda: RandomForestClassifier(
        n_estimators=400, class_weight="balanced", n_jobs=-1, random_state=42),
    "HistGradientBoosting": lambda: HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.1, random_state=42, class_weight="balanced"),
    "SVM_RBF": lambda: SVC(kernel="rbf", C=5.0, gamma="scale", class_weight="balanced"),
    "LogisticRegression": lambda: LogisticRegression(
        max_iter=1000, class_weight="balanced"),
    "MLP": lambda: MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu",
                                  max_iter=300, random_state=42),
}


def loso_one_model(df_c, cols, subjects, model_fn):
    ya, pa, sacc = [], [], []
    for s in subjects:
        tr, te = df_c[df_c.subject_id != s], df_c[df_c.subject_id == s]
        ytr = (tr["load_2"] == "alta").astype(int).to_numpy()
        yte = (te["load_2"] == "alta").astype(int).to_numpy()
        Xtr, Xte = fmat(tr, cols), fmat(te, cols)
        sc = StandardScaler().fit(Xtr)
        Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        clf = model_fn()
        clf.fit(Xtr, ytr)
        pr = clf.predict(Xte)
        ya.extend(yte); pa.extend(pr); sacc.append(accuracy_score(yte, pr))
    ya, pa = np.array(ya), np.array(pa)
    return {
        "acc": accuracy_score(ya, pa),
        "macro_f1": f1_score(ya, pa, average="macro"),
        "recall_normal": recall_score(ya, pa, pos_label=0),
        "recall_alta": recall_score(ya, pa, pos_label=1),
        "subj_acc_mean": float(np.mean(sacc)),
        "subj_acc_std": float(np.std(sacc)),
    }


def main():
    df = pd.read_csv(CSV)
    df = df[df.load_2.isin(["normal", "alta"])].copy()
    cols = feature_cols(df)
    subjects = sorted(df.subject_id.unique())
    print(f"n_features={len(cols)} n_subjects={len(subjects)} n_windows={len(df)}")
    df_c = per_subject_calibrate(df, cols)

    results = {}
    for name, model_fn in MODELS.items():
        print(f"\n--- {name} (LOSO + per-subject calibration) ---")
        r = loso_one_model(df_c, cols, subjects, model_fn)
        print(json.dumps(r, indent=2))
        results[name] = r

    OUT.write_text(json.dumps({"n_features": len(cols), "protocol": "LOSO + per-subject calibration, load_2, aura CSV (real montage)",
                                "models": results}, indent=2))
    print("\nSaved ->", OUT)


if __name__ == "__main__":
    main()
