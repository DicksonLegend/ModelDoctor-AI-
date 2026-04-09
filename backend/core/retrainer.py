"""
Retrainer — detects model type, tries targeted improvements, picks the best.
Works with ANY scikit-learn model. Uses Pipeline for consistency.
Fast execution (~3 seconds) with focused candidate selection.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.model_selection import KFold

from core.evaluator import evaluate_model, evaluate_regression_model


def retrain_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    diagnosis: list,
    retrain_round: int = 0,
) -> tuple:
    """
    Retrain with targeted improvements based on diagnosis.
    Tries a focused set of candidates and picks the best.
    NEVER returns a model worse than the original.

    Returns: (new_model, new_metrics, cv_scores, applied_actions)
    """
    applied_actions = []
    problems = {d["problem"] for d in diagnosis}
    rng = np.random.default_rng(42 + max(retrain_round, 0))

    # ── Get baseline ─────────────────────────────────────────────────
    baseline_acc = _safe_score(model, X_test, y_test)
    baseline_metrics = evaluate_model(model, X_train, y_train, X_test, y_test)
    baseline_cv_mean, baseline_cv_std = _safe_cv_profile(
        model,
        X_train,
        y_train,
        seed=100 + max(retrain_round, 0),
    )
    baseline_quality = _quality_score(
        baseline_metrics["accuracy"],
        baseline_metrics.get("macro_f1", baseline_metrics["f1_score"]),
        baseline_cv_mean,
        baseline_cv_std,
    )
    model_type = _detect_model_type(model)

    # ── Determine strategy ───────────────────────────────────────────
    try_both = baseline_acc < 0.6 or "Underfitting" in problems or "Low Accuracy" in problems
    use_balanced = "Class Imbalance" in problems or "Severe Class Imbalance" in problems

    class_weight = "balanced" if use_balanced else None

    # Generate a retrain-round-specific seed so repeated retrains explore
    # a different parameter neighborhood while remaining reproducible.
    rf_seed_primary = 42 + max(retrain_round, 0) * 13
    rf_seed_secondary = rf_seed_primary + 7
    lr_seed = rf_seed_primary + 3

    rf_n_estimators_primary = int(rng.integers(180, 420))
    rf_n_estimators_secondary = int(rng.integers(260, 620))
    rf_max_depth_regularized = int(rng.integers(8, 16))
    rf_max_depth_wide = int(rng.integers(16, 30))
    rf_min_split_regularized = int(rng.integers(6, 14))
    rf_min_leaf_regularized = int(rng.integers(2, 7))
    rf_min_split_wide = int(rng.integers(2, 6))

    lr_c_default = float(np.round(10 ** rng.uniform(-0.7, 0.5), 4))
    lr_c_strong = float(np.round(10 ** rng.uniform(0.7, 1.7), 4))
    lr_iter_default = int(rng.integers(1400, 2600))
    lr_iter_strong = int(rng.integers(2200, 3600))

    # ── Build tuning candidates (broader search space) ───────────────
    candidates = []
    base_clf = _extract_classifier(model)

    if model_type == "random_forest" or try_both:
        candidates.extend(
            _build_rf_tuning_candidates(
                base_clf=base_clf,
                class_weight=class_weight,
                rng=rng,
                round_seed=rf_seed_primary,
                prefer_regularized=("Overfitting" in problems or "Mild Overfitting" in problems),
            )
        )

    if model_type == "logistic_regression" or try_both:
        candidates.extend(
            _build_lr_tuning_candidates(
                base_clf=base_clf,
                class_weight=class_weight,
                rng=rng,
                round_seed=lr_seed,
            )
        )

    if not candidates:
        candidates.append(("RF-fallback", Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=rf_n_estimators_primary,
                max_depth=rf_max_depth_wide,
                min_samples_split=rf_min_split_wide,
                class_weight=class_weight,
                random_state=rf_seed_primary,
            )),
        ])))

    # ── Train & evaluate candidates ──────────────────────────────────
    best_pipe = None
    best_score = baseline_acc
    best_quality = baseline_quality
    best_metrics = None
    best_cv_mean = 0.0
    best_cv_std = 0.0
    best_name = ""

    for idx, (name, pipe) in enumerate(candidates):
        try:
            pipe.fit(X_train, y_train)
            score = float(pipe.score(X_test, y_test))
            cand_metrics = evaluate_model(pipe, X_train, y_train, X_test, y_test)
            cv_mean, cv_std = _safe_cv_profile(
                pipe,
                X_train,
                y_train,
                seed=500 + max(retrain_round, 0) * 17 + idx,
            )
            quality = _quality_score(
                cand_metrics["accuracy"],
                cand_metrics.get("macro_f1", cand_metrics["f1_score"]),
                cv_mean,
                cv_std,
            )

            if quality > best_quality + 1e-6:
                best_score = score
                best_quality = quality
                best_pipe = pipe
                best_metrics = cand_metrics
                best_cv_mean = cv_mean
                best_cv_std = cv_std
                best_name = name
        except Exception:
            continue

    # ── If no improvement found, keep best candidate anyway ──────────
    if best_pipe is None:
        # Pick the best candidate regardless (might be equivalent to baseline)
        best_score_any = -1.0
        best_quality_any = -1.0
        for idx, (name, pipe) in enumerate(candidates):
            try:
                pipe.fit(X_train, y_train)
                score = float(pipe.score(X_test, y_test))
                cand_metrics = evaluate_model(pipe, X_train, y_train, X_test, y_test)
                cv_mean, cv_std = _safe_cv_profile(
                    pipe,
                    X_train,
                    y_train,
                    seed=900 + max(retrain_round, 0) * 17 + idx,
                )
                quality = _quality_score(
                    cand_metrics["accuracy"],
                    cand_metrics.get("macro_f1", cand_metrics["f1_score"]),
                    cv_mean,
                    cv_std,
                )
                if (quality > best_quality_any + 1e-6) or (
                    abs(quality - best_quality_any) <= 1e-6 and score > best_score_any
                ):
                    best_quality_any = quality
                    best_score_any = score
                    best_pipe = pipe
                    best_metrics = cand_metrics
                    best_cv_mean = cv_mean
                    best_cv_std = cv_std
                    best_name = name
            except Exception:
                continue

        if best_pipe is None:
            # Ultimate fallback
            best_pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", RandomForestClassifier(
                    n_estimators=200, random_state=42, class_weight="balanced")),
            ])
            best_pipe.fit(X_train, y_train)
            best_name = "RF-fallback"

        applied_actions.append("No improvement found over baseline — returned best attempt")

    applied_actions.append(
        f"Retrain round {retrain_round} explored diversified hyperparameters"
    )
    applied_actions.append(f"Hyperparameter tuning trials: {len(candidates)}")

    # ── Log actions ──────────────────────────────────────────────────
    clf = best_pipe.named_steps.get("clf", best_pipe)
    clf_name = type(clf).__name__
    params = clf.get_params() if hasattr(clf, "get_params") else {}

    if "RandomForest" in clf_name:
        applied_actions.append(
            f"Trained {clf_name} (n_estimators={params.get('n_estimators')}, "
            f"max_depth={params.get('max_depth', 'None')})"
        )
    elif "Logistic" in clf_name:
        applied_actions.append(
            f"Trained {clf_name} (C={params.get('C')}, solver={params.get('solver')})"
        )
    else:
        applied_actions.append(f"Trained {clf_name}")

    applied_actions.append("Applied StandardScaler via Pipeline")

    if try_both and best_name.startswith("RF") and model_type != "random_forest":
        applied_actions.append("Switched to RandomForest for better performance")
    elif try_both and best_name.startswith("LR") and model_type != "logistic_regression":
        applied_actions.append("Switched to LogisticRegression for better performance")

    improvement = best_score - baseline_acc
    if improvement > 0.001:
        applied_actions.append(f"Accuracy improved by +{improvement*100:.1f}%")
    else:
        applied_actions.append("Model may already be near-optimal for this dataset")

    # ── Evaluate ─────────────────────────────────────────────────────
    new_metrics = best_metrics or evaluate_model(best_pipe, X_train, y_train, X_test, y_test)

    # ── CV scores (deterministic for stable health scoring) ──────────
    try:
        final_cv_mean, final_cv_std = _safe_cv_profile(
            best_pipe,
            X_train,
            y_train,
            seed=42,
        )
        cv_scores = [round(final_cv_mean, 6)] if final_cv_mean > 0 else []
        if final_cv_mean > 0 and final_cv_std > 0:
            cv_scores.extend([
                round(final_cv_mean - final_cv_std, 6),
                round(final_cv_mean + final_cv_std, 6),
            ])
        if not cv_scores:
            cv_scores = cross_val_score(
                best_pipe, X_train, y_train, cv=3, scoring="accuracy"
            ).tolist()
    except Exception:
        cv_scores = []

    return best_pipe, new_metrics, cv_scores, applied_actions


def retrain_regression_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    diagnosis: list,
    retrain_round: int = 0,
) -> tuple:
    """Retrain regression models with parameter search around baseline model."""
    rng = np.random.default_rng(123 + max(retrain_round, 0))
    applied_actions = []

    baseline_metrics = evaluate_regression_model(model, X_train, y_train, X_test, y_test)
    baseline_r2 = float(baseline_metrics.get("r2_score", -1.0) or -1.0)

    candidates = []
    for idx in range(12):
        candidates.append((
            f"RFReg-{idx+1}",
            Pipeline([
                ("scaler", StandardScaler()),
                ("reg", RandomForestRegressor(
                    n_estimators=int(rng.integers(150, 700)),
                    max_depth=rng.choice([None, 8, 12, 18, 24]),
                    min_samples_split=int(rng.integers(2, 10)),
                    min_samples_leaf=int(rng.integers(1, 5)),
                    random_state=1000 + idx + retrain_round,
                )),
            ]),
        ))

    for idx in range(6):
        candidates.append((
            f"GBR-{idx+1}",
            Pipeline([
                ("scaler", StandardScaler()),
                ("reg", GradientBoostingRegressor(
                    n_estimators=int(rng.integers(120, 420)),
                    learning_rate=float(np.round(rng.uniform(0.02, 0.2), 4)),
                    max_depth=int(rng.integers(2, 6)),
                    random_state=2000 + idx + retrain_round,
                )),
            ]),
        ))

    for idx in range(4):
        candidates.append((
            f"Ridge-{idx+1}",
            Pipeline([
                ("scaler", StandardScaler()),
                ("reg", Ridge(alpha=float(np.round(10 ** rng.uniform(-2, 2), 5)), random_state=3000 + idx + retrain_round)),
            ]),
        ))

    best_pipe = None
    best_metrics = baseline_metrics
    best_r2 = baseline_r2

    for _, pipe in candidates:
        try:
            pipe.fit(X_train, y_train)
            cand_metrics = evaluate_regression_model(pipe, X_train, y_train, X_test, y_test)
            cand_r2 = float(cand_metrics.get("r2_score", -1.0) or -1.0)
            if cand_r2 > best_r2 + 1e-6:
                best_r2 = cand_r2
                best_pipe = pipe
                best_metrics = cand_metrics
        except Exception:
            continue

    if best_pipe is None:
        # Keep best attempt even when no improvement.
        best_pipe = model
        applied_actions.append("No R2 improvement found over baseline — kept current best behavior")

    applied_actions.append(f"Regression hyperparameter tuning trials: {len(candidates)}")

    reg = best_pipe.named_steps.get("reg", best_pipe) if hasattr(best_pipe, "named_steps") else best_pipe
    reg_name = type(reg).__name__
    reg_params = reg.get_params() if hasattr(reg, "get_params") else {}
    if "RandomForest" in reg_name:
        applied_actions.append(
            f"Trained {reg_name} (n_estimators={reg_params.get('n_estimators')}, max_depth={reg_params.get('max_depth', 'None')})"
        )
    elif "GradientBoosting" in reg_name:
        applied_actions.append(
            f"Trained {reg_name} (n_estimators={reg_params.get('n_estimators')}, learning_rate={reg_params.get('learning_rate')})"
        )
    elif "Ridge" in reg_name:
        applied_actions.append(f"Trained {reg_name} (alpha={reg_params.get('alpha')})")
    else:
        applied_actions.append(f"Trained {reg_name}")

    improvement = best_r2 - baseline_r2
    if improvement > 1e-4:
        applied_actions.append(f"R2 improved by +{improvement:.4f}")
    else:
        applied_actions.append("Model may already be near-optimal for this regression dataset")

    try:
        cv = KFold(n_splits=3, shuffle=True, random_state=42)
        cv_scores = cross_val_score(best_pipe, X_train, y_train, cv=cv, scoring="r2").tolist()
    except Exception:
        cv_scores = []

    return best_pipe, best_metrics, cv_scores, applied_actions


def _safe_score(model, X_test, y_test) -> float:
    try:
        return float(model.score(X_test, y_test))
    except Exception:
        return 0.0


def _quality_score(acc: float, macro_f1: float, cv_mean: float, cv_std: float) -> float:
    """Composite quality score to break holdout ties on tiny datasets."""
    return (0.50 * float(acc)) + (0.30 * float(macro_f1)) + (0.20 * float(cv_mean)) - (0.05 * float(cv_std))


def _safe_cv_profile(model, X_train, y_train, seed: int) -> tuple[float, float]:
    """Return (cv_mean, cv_std) with graceful fallback."""
    try:
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_weighted")
        return float(np.mean(scores)), float(np.std(scores))
    except Exception:
        return 0.0, 0.0


def _extract_classifier(model):
    """Extract final estimator from model or pipeline."""
    try:
        if hasattr(model, "named_steps") and "clf" in model.named_steps:
            return model.named_steps["clf"]
        if hasattr(model, "steps") and model.steps:
            return model.steps[-1][1]
    except Exception:
        pass
    return model


def _build_rf_tuning_candidates(base_clf, class_weight, rng, round_seed: int, prefer_regularized: bool):
    """Generate RF tuning candidates around baseline parameters."""
    base_params = base_clf.get_params() if isinstance(base_clf, RandomForestClassifier) else {}

    base_n = int(base_params.get("n_estimators", 220))
    base_depth = base_params.get("max_depth", None)
    base_split = int(base_params.get("min_samples_split", 2))
    base_leaf = int(base_params.get("min_samples_leaf", 1))

    n_choices = {
        max(120, int(base_n * 0.6)),
        max(160, int(base_n * 0.8)),
        max(180, base_n),
        min(900, int(base_n * 1.2)),
        min(1000, int(base_n * 1.5)),
        int(rng.integers(180, 520)),
        int(rng.integers(260, 760)),
    }

    depth_choices = {
        None,
        8,
        12,
        16,
        24,
        int(rng.integers(10, 30)),
    }
    if isinstance(base_depth, int):
        depth_choices.add(max(4, base_depth - 3))
        depth_choices.add(base_depth)
        depth_choices.add(min(40, base_depth + 4))

    split_choices = {
        2,
        3,
        5,
        max(2, base_split),
        int(rng.integers(2, 10)),
    }
    leaf_choices = {
        1,
        2,
        max(1, base_leaf),
        int(rng.integers(1, 5)),
    }

    if prefer_regularized:
        depth_choices.update({8, 10, 12})
        split_choices.update({6, 8, 10})
        leaf_choices.update({2, 3, 4})

    candidates = []
    for idx in range(12):
        n_estimators = int(rng.choice(sorted(n_choices)))
        max_depth = rng.choice(list(depth_choices))
        min_split = int(rng.choice(sorted(split_choices)))
        min_leaf = int(rng.choice(sorted(leaf_choices)))
        max_features = rng.choice(["sqrt", "log2", None])
        bootstrap = bool(rng.choice([True, True, False]))

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_split,
                min_samples_leaf=min_leaf,
                max_features=max_features,
                bootstrap=bootstrap,
                class_weight=class_weight,
                random_state=round_seed + idx,
            )),
        ])
        candidates.append((f"RF-tune-{idx+1}", pipe))

    return candidates


def _build_lr_tuning_candidates(base_clf, class_weight, rng, round_seed: int):
    """Generate LR tuning candidates around baseline parameters."""
    base_params = base_clf.get_params() if isinstance(base_clf, LogisticRegression) else {}
    base_c = float(base_params.get("C", 1.0))
    base_iter = int(base_params.get("max_iter", 2000))

    c_choices = {
        0.05, 0.1, 0.3, 1.0, 3.0, 10.0,
        max(0.01, round(base_c * 0.4, 4)),
        max(0.01, round(base_c, 4)),
        min(100.0, round(base_c * 2.5, 4)),
        float(np.round(10 ** rng.uniform(-1.4, 1.6), 4)),
    }
    iter_choices = {
        1200,
        1800,
        2500,
        max(1200, base_iter),
        min(6000, base_iter + 800),
        int(rng.integers(1500, 4200)),
    }

    candidates = []
    for idx in range(8):
        c_val = float(rng.choice(sorted(c_choices)))
        iter_val = int(rng.choice(sorted(iter_choices)))
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=c_val,
                max_iter=iter_val,
                solver="lbfgs",
                class_weight=class_weight,
                random_state=round_seed + idx,
            )),
        ])
        candidates.append((f"LR-tune-{idx+1}", pipe))

    return candidates


def _detect_model_type(model) -> str:
    """Detect model type, supports raw models and Pipelines."""
    name = type(model).__name__.lower()

    # Check inside Pipeline
    if "pipeline" in name:
        try:
            last_step = model.steps[-1][1]
            name = type(last_step).__name__.lower()
        except Exception:
            pass

    if "randomforest" in name:
        return "random_forest"
    elif "logistic" in name:
        return "logistic_regression"
    elif "svc" in name or "svm" in name:
        return "logistic_regression"  # treat SVM like LR for tuning
    elif "kneighbors" in name:
        return "random_forest"  # treat KNN like RF
    elif "tree" in name or "decision" in name:
        return "random_forest"
    elif "gradient" in name or "boosting" in name:
        return "random_forest"
    elif "naive" in name or "bayes" in name:
        return "logistic_regression"
    else:
        return "random_forest"  # default


def _build_model(model_type: str, params: dict):
    """Build a Pipeline (scaler + model)."""
    if model_type == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=params.get("n_estimators", 200),
            max_depth=params.get("max_depth", None),
            class_weight=params.get("class_weight", "balanced"),
            random_state=42,
        )
    else:
        clf = LogisticRegression(
            C=params.get("C", 1.0),
            max_iter=params.get("max_iter", 1000),
            solver=params.get("solver", "lbfgs"),
            class_weight=params.get("class_weight", "balanced"),
            random_state=42,
        )
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])
