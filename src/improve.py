from itertools import product

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from src.data import TARGET, load_data
from src.evaluate import plot_sweep, split_train_validation_test, threshold_sweep
from src.model import (
    BASELINE_FEATURES,
    FEATURES,
    TEST_SIZE,
    build_baseline_pipeline,
    build_xgb_pipeline,
    evaluate,
)

GRID = {
    "max_depth": [2, 3, 4],
    "learning_rate": [0.05, 0.1],
    "scale_pos_weight": [1, 3.6],
}
FIXED = {
    "n_estimators": 300,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}


def extreme_share(scores):
    return float(((scores < 0.05) | (scores > 0.95)).mean())


def run_grid(X_fit, X_val, y_fit, y_val):
    rows = []
    for depth, learning_rate, pos_weight in product(
        GRID["max_depth"],
        GRID["learning_rate"],
        GRID["scale_pos_weight"],
    ):
        params = {
            "max_depth": depth,
            "learning_rate": learning_rate,
            "scale_pos_weight": pos_weight,
        }
        model = build_xgb_pipeline(**FIXED, **params)
        model.fit(X_fit, y_fit)
        train_results = evaluate(model, X_fit, y_fit)
        validation_results = evaluate(model, X_val, y_val)
        scores = model.predict_proba(X_val)[:, 1]
        rows.append(
            {
                **params,
                "train_auc": train_results["roc_auc"],
                "validation_auc": validation_results["roc_auc"],
                "gap": train_results["roc_auc"] - validation_results["roc_auc"],
                "validation_recall": validation_results["recall_1"],
                "validation_precision": validation_results["precision_1"],
                "extreme_share": extreme_share(scores),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "validation_auc", ascending=False
    ).reset_index(drop=True)


def choose_winner(grid_results):
    top_auc = grid_results.iloc[0]["validation_auc"]
    close = grid_results[grid_results["validation_auc"] >= top_auc - 0.01]
    return close.sort_values(
        ["gap", "validation_auc"], ascending=[True, False]
    ).iloc[0]


def cv_compare(df, winner_params):
    X = df[FEATURES].copy()
    y = df[TARGET]
    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=42,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = (
        ("Starter LR", build_baseline_pipeline(), BASELINE_FEATURES),
        ("Default XGBoost", build_xgb_pipeline(), FEATURES),
        (
            "Regularised XGBoost",
            build_xgb_pipeline(**FIXED, **winner_params),
            FEATURES,
        ),
    )
    rows = []
    for name, model, columns in models:
        scores = cross_val_score(
            model,
            X_train[columns],
            y_train,
            cv=cv,
            scoring="roc_auc",
        )
        rows.append(
            {
                "model": name,
                "auc_mean": scores.mean(),
                "auc_std": scores.std(),
            }
        )
    return pd.DataFrame(rows)


def main():
    df = load_data()
    X_fit, X_val, y_fit, y_val = split_train_validation_test(df)
    grid_results = run_grid(X_fit, X_val, y_fit, y_val)
    print("Grid results (sorted by validation AUC):")
    print(grid_results.to_string(index=False))

    winner = choose_winner(grid_results)
    winner_params = {
        "max_depth": int(winner["max_depth"]),
        "learning_rate": float(winner["learning_rate"]),
        "scale_pos_weight": float(winner["scale_pos_weight"]),
    }
    if winner["validation_auc"] == grid_results.iloc[0]["validation_auc"]:
        reason = "highest validation AUC"
    else:
        reason = "within 0.01 of the best AUC, with a smaller overfitting gap"
    print(f"\nChosen settings ({reason}): {winner_params}")

    print("\n5-fold CV on the 640 training rows:")
    print(cv_compare(df, winner_params).to_string(index=False))

    model = build_xgb_pipeline(**FIXED, **winner_params)
    model.fit(X_fit, y_fit)
    scores = model.predict_proba(X_val)[:, 1]
    sweep = threshold_sweep(y_val, scores)
    print("\nImproved model threshold sweep:")
    print(sweep.round(3).to_string(index=False))
    plot_sweep(sweep, "Improved XGBoost threshold sweep", "sweep_improved")


if __name__ == "__main__":
    main()
