"""Metrike za binarnu klasifikaciju stvarno (0) / lazno (1).

Zasto ROC-AUC kao glavna metrika: ne zavisi od izbora praga odluke.
Tacnost i F1 zavise od praga (ovdje 0.5), pa ih prijavljujemo uz AUC.
"""
import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             precision_score, recall_score, confusion_matrix)


def izracunaj(vjerovatnoce: np.ndarray, oznake: np.ndarray, prag: float = 0.5) -> dict:
    pred = (vjerovatnoce >= prag).astype(int)
    rez = {
        "n": int(len(oznake)),
        "tacnost": float(accuracy_score(oznake, pred)),
        "f1": float(f1_score(oznake, pred, zero_division=0)),
        "preciznost": float(precision_score(oznake, pred, zero_division=0)),
        "odziv": float(recall_score(oznake, pred, zero_division=0)),
    }
    # AUC je definisan samo ako postoje obje klase
    if len(np.unique(oznake)) == 2:
        rez["roc_auc"] = float(roc_auc_score(oznake, vjerovatnoce))
    tn, fp, fn, tp = confusion_matrix(oznake, pred, labels=[0, 1]).ravel()
    rez.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    return rez
