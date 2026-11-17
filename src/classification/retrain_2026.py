"""Retrain the TF-IDF + LR classifiers under the Nov 2026 protocol.

Follows config/retrain_protocol_2026-11.md: two fixed training sets (A = the 210 gold training
decisions; B = A plus the supplementary decisions coded by the logged model pass), C and the
training set chosen on the validation set only. Test labels are never loaded here; the chosen
models are scored once by src/analysis/validation_analysis.py.

Replaces retrain_expanded.py, whose shipped variant was chosen by test-set score.

Usage (from project root, after data_preparation and the supplementary model pass):
    .venv-ml/bin/python -m src.classification.retrain_2026
"""

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SPLITS_DIR = DATA_DIR / "annotations" / "splits"
SUP_DIR = DATA_DIR / "annotations" / "supplementary_2026-11"
CORPUS_PATH = DATA_DIR / "extracted" / "all_decisions.jsonl"
PROTOCOL = PROJECT_ROOT / "config" / "retrain_protocol_2026-11.md"
MODEL_DIR = PROJECT_ROOT / "models" / "sklearn_misconduct"
RESULTS_PATH = PROJECT_ROOT / "outputs" / "statistics" / "retrain_2026_selection.json"
PREDICTIONS_PATH = DATA_DIR / "extracted" / "misconduct_predictions_sklearn.jsonl"

MISCONDUCT_L1_CODES = ["CLIN", "PRESC", "BOUND", "IMPR", "FRAUD", "COMM", "PROF", "N/A"]
FINDING_CLASSES = ["NONE", "NO_FINDING", "PM", "UPC_GEN", "UPC_SAT"]
C_GRID = [0.5, 1.0, 5.0, 10.0, 50.0, 100.0]


def sha1(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def load_split(name: str) -> pd.DataFrame:
    with open(SPLITS_DIR / f"{name}.jsonl") as f:
        return pd.DataFrame([json.loads(line) for line in f])


def load_supplementary() -> tuple[pd.DataFrame, dict]:
    """Supplementary decisions in the split schema, from the logged model pass."""
    ids = json.loads((SUP_DIR / "item_ids.json").read_text())
    codes = json.loads((SUP_DIR / "model_pass" / "model_codes.json").read_text())
    summary = json.loads((SUP_DIR / "model_pass" / "run_summary.json").read_text())
    want = {ids[c["item"]] for c in codes}
    texts = {}
    with open(CORPUS_PATH) as f:
        for line in f:
            r = json.loads(line)
            if r["id"] in want:
                texts[r["id"]] = r["full_text"]
    rows = []
    for c in codes:
        disc = c["is_disciplinary"] == "Y"
        rows.append({
            "id": ids[c["item"]],
            "full_text": texts[ids[c["item"]]],
            "is_disciplinary": disc,
            "misconduct_L1": [k for k in MISCONDUCT_L1_CODES[:-1] if c[k] == "Y"] if disc else ["N/A"],
            "finding": None if c["finding"] == "NA" else c["finding"],
        })
    info = {"batch_id": summary["batch_id"], "n_coded": len(rows), "n_left_out": 300 - len(rows),
            "problems": summary["problems"]}
    return pd.DataFrame(rows), info


def binarise(series: pd.Series) -> np.ndarray:
    m = np.zeros((len(series), len(MISCONDUCT_L1_CODES)), dtype=int)
    for i, labels in enumerate(series):
        for label in labels or []:
            if label in MISCONDUCT_L1_CODES:
                m[i, MISCONDUCT_L1_CODES.index(label)] = 1
    return m


def grid(make, X_tr, y_tr, X_va, y_va, average):
    """Fit each C; keep the highest validation score (ties go to the smaller C)."""
    scores, best = {}, None
    for C in C_GRID:
        clf = make(C).fit(X_tr, y_tr)
        s = f1_score(y_va, clf.predict(X_va), average=average, zero_division=0)
        scores[str(C)] = round(float(s), 4)
        if best is None or s > best[1]:
            best = (C, s, clf)
    return {"C": best[0], "val": round(float(best[1]), 4), "val_by_C": scores, "_clf": best[2]}


def lr(C):
    return LogisticRegression(C=C, max_iter=2000, solver="lbfgs", random_state=42,
                              class_weight="balanced")


def fit_set(train: pd.DataFrame, val: pd.DataFrame, le: LabelEncoder) -> dict:
    vec = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True,
                          min_df=2, max_df=0.95, strip_accents="unicode")
    X_tr, X_va = vec.fit_transform(train["full_text"]), vec.transform(val["full_text"])
    enc = lambda s: le.transform(s.fillna("NONE"))
    return {
        "n_train": len(train),
        "_vec": vec,
        "misconduct": grid(lambda C: OneVsRestClassifier(lr(C)), X_tr, binarise(train["misconduct_L1"]),
                           X_va, binarise(val["misconduct_L1"]), "macro"),
        "finding": grid(lr, X_tr, enc(train["finding"]), X_va, enc(val["finding"]), "macro"),
        "disciplinary": grid(lr, X_tr, train["is_disciplinary"].astype(int),
                             X_va, val["is_disciplinary"].astype(int), "binary"),
    }


def predict_corpus(models: dict) -> None:
    """Apply the chosen models to the full corpus, each with its own vectoriser."""
    with open(CORPUS_PATH) as f:
        records = [json.loads(line) for line in f]
    texts = [r["full_text"] for r in records]
    X, X_f = models["vectorizer.pkl"].transform(texts), models["vectorizer_finding.pkl"].transform(texts)
    misc = models["misconduct_clf.pkl"].predict(X)
    disc = models["disciplinary_clf.pkl"].predict(X)
    finding = models["finding_le.pkl"].inverse_transform(models["finding_clf.pkl"].predict(X_f))
    with open(PREDICTIONS_PATH, "w") as f:
        for i, r in enumerate(records):
            f.write(json.dumps({
                "id": r["id"],
                "predicted_misconduct_L1": [c for j, c in enumerate(MISCONDUCT_L1_CODES) if misc[i, j]],
                "predicted_finding": str(finding[i]),
                "predicted_is_disciplinary": bool(disc[i]),
            }) + "\n")
    print(f"Predictions for {len(records)} decisions saved to {PREDICTIONS_PATH}")


def main():
    train, val = load_split("train"), load_split("val")
    sup, sup_info = load_supplementary()
    if set(sup["id"]) & set(train["id"]) | set(sup["id"]) & set(val["id"]):
        raise SystemExit("supplementary decisions overlap the gold splits")
    le = LabelEncoder().fit(FINDING_CLASSES)

    cols = ["id", "full_text", "is_disciplinary", "misconduct_L1", "finding"]
    sets = {"A": fit_set(train, val, le),
            "B": fit_set(pd.concat([train[cols], sup[cols]], ignore_index=True), val, le)}
    for name, s in sets.items():
        print(f"{name} (n={s['n_train']}): misconduct val macro-F1 {s['misconduct']['val']} "
              f"(C={s['misconduct']['C']}); finding {s['finding']['val']} (C={s['finding']['C']}); "
              f"disciplinary {s['disciplinary']['val']} (C={s['disciplinary']['C']})")

    pick = lambda task: "B" if sets["B"][task]["val"] > sets["A"][task]["val"] else "A"
    chosen = {"misconduct": pick("misconduct"), "finding": pick("finding")}
    chosen["disciplinary"] = chosen["misconduct"]  # shares the misconduct vectoriser
    print(f"Chosen on validation: {chosen}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    models = dict([
        ("vectorizer.pkl", sets[chosen["misconduct"]]["_vec"]),
        ("misconduct_clf.pkl", sets[chosen["misconduct"]]["misconduct"]["_clf"]),
        ("disciplinary_clf.pkl", sets[chosen["disciplinary"]]["disciplinary"]["_clf"]),
        ("vectorizer_finding.pkl", sets[chosen["finding"]]["_vec"]),
        ("finding_clf.pkl", sets[chosen["finding"]]["finding"]["_clf"]),
        ("finding_le.pkl", le),
    ])
    for fname, obj in models.items():
        with open(MODEL_DIR / fname, "wb") as f:
            pickle.dump(obj, f)

    strip = lambda d: {k: v for k, v in d.items() if not k.startswith("_")}
    RESULTS_PATH.write_text(json.dumps({
        "protocol": {"file": str(PROTOCOL.relative_to(PROJECT_ROOT)), "sha1": sha1(PROTOCOL)},
        "gold_split_ids_sha1": sha1(SPLITS_DIR / "split_ids.json"),
        "supplementary": sup_info,
        "training_sets": {n: {"n_train": s["n_train"],
                              **{t: strip(s[t]) for t in ("misconduct", "finding", "disciplinary")}}
                          for n, s in sets.items()},
        "chosen": chosen,
        "test_labels_loaded": False,
    }, indent=2))
    print(f"Models saved to {MODEL_DIR}; selection record {RESULTS_PATH}")
    predict_corpus(models)


if __name__ == "__main__":
    main()
