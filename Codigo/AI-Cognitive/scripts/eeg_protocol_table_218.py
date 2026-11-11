"""Tabla de protocolo (tesis, Cap. 5) sobre UN SOLO conjunto de features.

Todas las filas usan eeg_window_features_aura.csv (218 features, montaje real
Fp1,Fp2,F7,F3,Fz,F4,F8,C4, sin FAA), RF de 400 arboles balanceado, load_2:

  none      LOSO sin calibracion
  full      LOSO + calibracion por sujeto contra TODO su reposo
            (= eeg_classifier_comparison_loso_calibrated.py, fila RandomForest)
  reserved  LOSO + calibracion; del sujeto excluido se reserva el 50% de su
            reposo para calibrar y se retira del stream de evaluacion
  online    reserved + recalibracion EMA en linea (eeg_online_recalib.py),
            con los parametros conservadores y con los relajados

Por que existe: la fila "LOSO, sin calibracion" de main.tex (Table I) y la
recalibracion en linea se midieron sobre el CSV heredado de 648 columnas, asi
que su contraste con la fila calibrada (218 features) no era una ablacion pura
de protocolo. Este script mide las cuatro filas sobre el mismo conjunto.

Nota sobre 'reserved': al retirar la mitad del reposo del stream, la clase
minoritaria pesa menos en la evaluacion y la exactitud sube sin que el modelo
mejore; F1-macro y recall de reposo son las metricas comparables entre filas.

Guarda csv/protocol_table_218.json. Ejecutar desde AI-Cognitive/.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eeg_classifier_comparison_loso_calibrated as C  # noqa: E402
import eeg_online_recalib as O  # noqa: E402

OUT = C.ROOT / "csv" / "protocol_table_218.json"


def rf():
    return RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                  n_jobs=-1, random_state=42)


def base_stats(g, cols):
    mu = g[cols].mean().to_numpy(float)
    sd = g[cols].std().to_numpy(float)
    return mu, np.where(np.isfinite(sd) & (sd > 1e-9), sd, 1.0)


def zscore(g, cols, mu, sd):
    return np.clip(np.nan_to_num((C.fmat(g, cols) - mu) / sd), -50, 50)


def reserved(df, cols, subjects):
    ya, pa, sacc, sf1 = [], [], [], []
    for s in subjects:
        tr, te = df[df.subject_id != s], df[df.subject_id == s]
        parts, ytr = [], []
        for _, g in tr.groupby("subject_id"):
            b = g[g.condition_4.isin(["natural", "normal"])]
            parts.append(zscore(g, cols, *base_stats(b if len(b) >= 5 else g, cols)))
            ytr.append((g.load_2 == "alta").astype(int).to_numpy())
        b = te[te.condition_4.isin(["natural", "normal"])]
        cal = b.head(max(5, len(b) // 2))
        stream = te.drop(cal.index)
        Xtr, ytr = np.vstack(parts), np.concatenate(ytr)
        Xte = zscore(stream, cols, *base_stats(cal, cols))
        yte = (stream.load_2 == "alta").astype(int).to_numpy()
        sc = StandardScaler().fit(Xtr)
        clf = rf().fit(sc.transform(Xtr), ytr)
        pr = clf.predict(sc.transform(Xte))
        ya.extend(yte); pa.extend(pr)
        sacc.append(accuracy_score(yte, pr)); sf1.append(f1_score(yte, pr, average="macro"))
    ya, pa = np.array(ya), np.array(pa)
    return {
        "acc": accuracy_score(ya, pa),
        "macro_f1": f1_score(ya, pa, average="macro"),
        "recall_normal": recall_score(ya, pa, pos_label=0),
        "recall_alta": recall_score(ya, pa, pos_label=1),
        "subj_acc_mean": float(np.mean(sacc)), "subj_acc_std": float(np.std(sacc)),
        "subj_f1_mean": float(np.mean(sf1)), "subj_f1_std": float(np.std(sf1)),
    }


def online(alpha, conf):
    """eeg_online_recalib.run_dataset apuntado al CSV de 218 features."""
    O._ALPHA, O._CONF = alpha, conf
    O.cols_of = C.feature_cols
    O._RESULTS.clear()
    O.run_dataset(str(C.CSV), "AS-218")
    return O._RESULTS["AS-218"]


def main():
    df = pd.read_csv(C.CSV)
    df = df[df.load_2.isin(["normal", "alta"])].copy()
    cols = C.feature_cols(df)
    subjects = sorted(df.subject_id.unique())
    print(f"n_features={len(cols)} n_subjects={len(subjects)} n_windows={len(df)}")

    res = {"csv": str(C.CSV.relative_to(C.ROOT)), "n_features": len(cols), "rows": {}}
    res["rows"]["none"] = C.loso_one_model(df, cols, subjects, rf)
    res["rows"]["full"] = C.loso_one_model(C.per_subject_calibrate(df, cols), cols, subjects, rf)
    res["rows"]["reserved"] = reserved(df, cols, subjects)
    for k, v in res["rows"].items():
        print(k, json.dumps({m: round(x, 4) for m, x in v.items()}))

    res["online"] = {"conservative (0.001, 0.95)": online(0.001, 0.95),
                     "relaxed (0.02, 0.75)": online(0.02, 0.75)}
    OUT.write_text(json.dumps(res, indent=2))
    print("\nSaved ->", OUT)


if __name__ == "__main__":
    main()
