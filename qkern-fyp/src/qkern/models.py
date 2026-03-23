from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


@dataclass
class EvalResult:
    acc: float
    f1: float
    auc: Optional[float]


def evaluate_svc_precomputed(G_train: np.ndarray, y_train: np.ndarray,
                             G_test: np.ndarray, y_test: np.ndarray,
                             C: float = 1.0) -> EvalResult:
    clf = SVC(kernel="precomputed", C=C, probability=True)
    clf.fit(G_train, y_train)
    y_pred = clf.predict(G_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    try:
        y_proba = clf.predict_proba(G_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
    except Exception:
        auc = None
    return EvalResult(acc=acc, f1=f1, auc=auc)


def evaluate_svc_rbf(X_train: np.ndarray, y_train: np.ndarray,
                     X_test: np.ndarray, y_test: np.ndarray,
                     C: float = 1.0, gamma: str | float = "scale") -> EvalResult:
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf", C=C, gamma=gamma, probability=True)),
    ])
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    try:
        y_proba = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
    except Exception:
        auc = None
    return EvalResult(acc=acc, f1=f1, auc=auc)


def evaluate_svc_linear(X_train: np.ndarray, y_train: np.ndarray,
                       X_test: np.ndarray, y_test: np.ndarray,
                       C: float = 1.0) -> EvalResult:
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="linear", C=C, probability=True)),
    ])
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    try:
        y_proba = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
    except Exception:
        auc = None
    return EvalResult(acc=acc, f1=f1, auc=auc)


def evaluate_svc_poly(X_train: np.ndarray, y_train: np.ndarray,
                      X_test: np.ndarray, y_test: np.ndarray,
                      C: float = 1.0, degree: int = 3, gamma: str | float = "scale",
                      coef0: float = 1.0) -> EvalResult:
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="poly", degree=degree, C=C, gamma=gamma, coef0=coef0, probability=True)),
    ])
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    try:
        y_proba = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
    except Exception:
        auc = None
    return EvalResult(acc=acc, f1=f1, auc=auc)
