"""
Wine Quality Classification - 6-Class Pipeline

Pipeline per la classificazione della qualita del vino rosso sul task originario
a 6 classi, con target WineQuality compreso tra 3 e 8.
"""

import os
import warnings

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


warnings.filterwarnings("ignore")

RANDOM_SEED = 8
np.random.seed(RANDOM_SEED)

TRAIN_PATH = os.getenv("WINE_TRAIN_PATH", "/content/wine_train.xlsx")
TEST_PATH = os.getenv("WINE_TEST_PATH", "/content/wine_test.xlsx")
PLOTS_DIR = os.getenv("WINE_PLOTS_DIR", "/content/plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

CV_FOLDS = 3

FEATURE_COLS_BASE = [
    "fixed_acidity",
    "volatile_acidity",
    "citric_acid",
    "residual_sugar",
    "chlorides",
    "free_sulfur_dioxide",
    "total_sulfur_dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]

FEATURE_COLS_DERIVED = [
    "Acidity_Sulfur_Interaction",
    "Alcohol_Density_Ratio",
    "Volatile_Enhanced",
]

FEATURE_COLS_FULL = FEATURE_COLS_BASE + FEATURE_COLS_DERIVED

FEATURE_SETS = {
    "FULL_with_correlation": FEATURE_COLS_FULL,
    "NOCORR_without_derived": FEATURE_COLS_BASE,
}


OUTPUT_FILES = {
    "cv_results": "/content/cv_results_6class_full_vs_nocorr.csv",
    "test_results": "/content/test_set_results_6class.csv",
    "sensitivity": "/content/hyperparameter_sensitivity_6class_full_vs_nocorr.csv",
}


def resolve_output_path(path):
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        return os.path.join(PLOTS_DIR, os.path.basename(path))
    return path


def load_data(train_path=TRAIN_PATH, test_path=TEST_PATH):
    df_train = pd.read_excel(train_path)
    df_test = pd.read_excel(test_path)

    df_train = df_train.apply(pd.to_numeric, errors="coerce")
    df_test = df_test.apply(pd.to_numeric, errors="coerce")

    return df_train, df_test


def filter_synthetic_noise(df, remove_noise=True):
    if remove_noise and "Synthetic_Noise_Flag" in df.columns:
        return df[df["Synthetic_Noise_Flag"] == 0].copy()
    return df.copy()


def fill_missing_derived_features_only(df):
    df = df.copy()

    if {"Alcohol_Density_Ratio", "alcohol", "density"}.issubset(df.columns):
        mask = df["Alcohol_Density_Ratio"].isna()
        n_missing = int(mask.sum())
        if n_missing > 0:
            df.loc[mask, "Alcohol_Density_Ratio"] = (
                df.loc[mask, "alcohol"] / df.loc[mask, "density"]
            )
        print(f"  Alcohol_Density_Ratio: recomputed {n_missing} missing values")

    if {"Acidity_Sulfur_Interaction", "fixed_acidity", "sulphates"}.issubset(df.columns):
        mask = df["Acidity_Sulfur_Interaction"].isna()
        n_missing = int(mask.sum())
        if n_missing > 0:
            df.loc[mask, "Acidity_Sulfur_Interaction"] = (
                df.loc[mask, "fixed_acidity"] * df.loc[mask, "sulphates"]
            )
        print(f"  Acidity_Sulfur_Interaction: recomputed {n_missing} missing values")

    return df


def preprocess_data(remove_noise=True):
    df_train_raw, df_test_raw = load_data()

    print("\nInitial size:")
    print(f"  Train raw: {len(df_train_raw)}")
    print(f"  Test raw:  {len(df_test_raw)}")

    df_train = filter_synthetic_noise(df_train_raw, remove_noise=remove_noise)
    df_test = df_test_raw.copy()

    if remove_noise:
        print("\nAfter Synthetic_Noise_Flag == 0 filter:")
        print(f"  Train filtered: {len(df_train)}")
    else:
        print("\nSynthetic_Noise_Flag filter not applied.")

    print("\nRecomputing derived features only where missing - train:")
    df_train = fill_missing_derived_features_only(df_train)

    print("\nRecomputing derived features only where missing - test:")
    df_test = fill_missing_derived_features_only(df_test)

    return df_train, df_test


def evaluate_with_cv(model, X, y, cv_folds=CV_FOLDS, model_name="Model"):
    skf = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    accuracies = []
    f1_scores = []
    all_y_true = []
    all_y_pred = []

    for train_idx, val_idx in skf.split(X, y):
        X_train_fold = X[train_idx]
        X_val_fold = X[val_idx]
        y_train_fold = y[train_idx]
        y_val_fold = y[val_idx]

        model_fold = clone(model)
        model_fold.fit(X_train_fold, y_train_fold)
        y_pred_fold = model_fold.predict(X_val_fold)

        accuracies.append(accuracy_score(y_val_fold, y_pred_fold))
        f1_scores.append(
            f1_score(y_val_fold, y_pred_fold, average="macro", zero_division=0)
        )

        all_y_true.extend(y_val_fold)
        all_y_pred.extend(y_pred_fold)

    return {
        "model_name": model_name,
        "accuracy_mean": np.mean(accuracies) * 100,
        "accuracy_std": np.std(accuracies) * 100,
        "f1_mean": np.mean(f1_scores) * 100,
        "f1_std": np.std(f1_scores) * 100,
        "y_true": np.array(all_y_true),
        "y_pred": np.array(all_y_pred),
    }


def baseline_majority_cv(y, cv_folds=CV_FOLDS, name="Majority class baseline"):
    skf = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    accuracies = []
    f1_scores = []

    for train_idx, val_idx in skf.split(np.zeros(len(y)), y):
        y_train_fold = y[train_idx]
        y_val_fold = y[val_idx]

        majority_class = pd.Series(y_train_fold).mode()[0]
        y_pred_fold = np.full_like(y_val_fold, fill_value=majority_class)

        accuracies.append(accuracy_score(y_val_fold, y_pred_fold))
        f1_scores.append(
            f1_score(y_val_fold, y_pred_fold, average="macro", zero_division=0)
        )

    return {
        "model_name": name,
        "accuracy_mean": np.mean(accuracies) * 100,
        "accuracy_std": np.std(accuracies) * 100,
        "f1_mean": np.mean(f1_scores) * 100,
        "f1_std": np.std(f1_scores) * 100,
        "y_true": None,
        "y_pred": None,
    }


def get_models():
    return {
        "Naive Bayes": Pipeline([
            ("scaler", StandardScaler()),
            ("model", GaussianNB()),
        ]),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=10,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "SVM Linear": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(
                kernel="linear",
                C=1,
                class_weight="balanced",
                random_state=RANDOM_SEED,
            )),
        ]),
        "SVM RBF": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(
                kernel="rbf",
                C=1,
                gamma="scale",
                class_weight="balanced",
                random_state=RANDOM_SEED,
            )),
        ]),
    }


def analyze_hyperparameter_sensitivity(X, y, feature_set_name, cv_folds=CV_FOLDS):
    print(f"\nRandom Forest hyperparameter sensitivity - {feature_set_name}")

    param_grid = {
        "n_estimators": [100, 300, 500],
        "max_depth": [5, 10, 20, None],
        "min_samples_leaf": [1, 4, 10],
    }

    results = []
    skf = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    for params in ParameterGrid(param_grid):
        accuracies = []
        f1_scores = []

        for train_idx, val_idx in skf.split(X, y):
            X_train_fold = X[train_idx]
            X_val_fold = X[val_idx]
            y_train_fold = y[train_idx]
            y_val_fold = y[val_idx]

            rf = RandomForestClassifier(
                **params,
                class_weight="balanced",
                random_state=RANDOM_SEED,
                n_jobs=-1,
            )

            rf.fit(X_train_fold, y_train_fold)
            y_pred = rf.predict(X_val_fold)

            accuracies.append(accuracy_score(y_val_fold, y_pred))
            f1_scores.append(
                f1_score(y_val_fold, y_pred, average="macro", zero_division=0)
            )

        results.append({
            "Feature_set": feature_set_name,
            "n_estimators": params["n_estimators"],
            "max_depth": params["max_depth"],
            "min_samples_leaf": params["min_samples_leaf"],
            "Accuracy_mean": np.mean(accuracies) * 100,
            "Accuracy_std": np.std(accuracies) * 100,
            "Macro_F1_mean": np.mean(f1_scores) * 100,
            "Macro_F1_std": np.std(f1_scores) * 100,
        })

    df_results = pd.DataFrame(results).sort_values(
        "Macro_F1_mean",
        ascending=False,
    )

    print("\nTop 10 configurations by Macro F1:")
    print(df_results.head(10).to_string(index=False))

    return df_results


def plot_cv_comparison(df_results):
    plot_df = df_results.copy()
    plot_df["Label"] = plot_df["Feature_set"] + " | " + plot_df["Model"]
    plot_df = plot_df.sort_values("Macro_F1_mean", ascending=False).head(20)

    y_pos = np.arange(len(plot_df))
    height = 0.35

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(
        y_pos - height / 2,
        plot_df["Accuracy_mean"],
        height,
        label="Accuracy",
        alpha=0.7,
    )
    ax.barh(
        y_pos + height / 2,
        plot_df["Macro_F1_mean"],
        height,
        label="Macro F1",
        alpha=0.7,
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["Label"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Mean CV score (%)")
    ax.set_title("6-class model comparison: FULL vs NOCORR")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "cv_model_comparison_6class_full_vs_nocorr.pdf")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> {out_path}")


def plot_hyperparameter_sensitivity(df_sensitivity):
    feature_sets = df_sensitivity["Feature_set"].unique()
    fig, axes = plt.subplots(
        1,
        len(feature_sets),
        figsize=(7 * len(feature_sets), 5),
        sharey=True,
    )

    if len(feature_sets) == 1:
        axes = [axes]

    for ax, feature_set_name in zip(axes, feature_sets):
        subset = df_sensitivity[df_sensitivity["Feature_set"] == feature_set_name].copy()

        for max_depth in subset["max_depth"].unique():
            sub = subset[subset["max_depth"] == max_depth]
            grouped = sub.groupby("n_estimators")["Macro_F1_mean"].mean()
            label = str(max_depth) if pd.notna(max_depth) else "None"
            ax.plot(grouped.index, grouped.values, marker="o", label=f"depth={label}")

        ax.set_title(feature_set_name)
        ax.set_xlabel("n_estimators")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    axes[0].set_ylabel("Mean CV Macro F1 (%)")
    plt.suptitle("Random Forest sensitivity - 6 classes")
    plt.tight_layout()

    out_path = os.path.join(PLOTS_DIR, "hyperparameter_sensitivity_6class_full_vs_nocorr.pdf")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> {out_path}")


def plot_correlation_heatmap(df_train):
    existing_cols = [col for col in FEATURE_COLS_FULL if col in df_train.columns]
    corr = df_train[existing_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values, vmin=-1, vmax=1)

    ax.set_xticks(range(len(existing_cols)))
    ax.set_yticks(range(len(existing_cols)))
    ax.set_xticklabels(existing_cols, rotation=90, fontsize=6)
    ax.set_yticklabels(existing_cols, fontsize=6)

    for i in range(len(existing_cols)):
        for j in range(len(existing_cols)):
            val = corr.values[i, j]
            if abs(val) >= 0.65 or i == j:
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=5)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Pearson Correlation Matrix")
    plt.tight_layout()

    out_path = os.path.join(PLOTS_DIR, "correlation_heatmap_full.pdf")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> {out_path}")


def plot_target_distribution(df_train_pre, filename="target_distribution_6class"):
    y = df_train_pre["WineQuality"].astype(int).values
    counts = pd.Series(y).value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(counts.index.astype(str), counts.values, alpha=0.8)

    for i, value in enumerate(counts.values):
        ax.text(i, value + 5, str(value), ha="center", fontsize=8)

    ax.set_title("Distribuzione del target - 6 classi")
    ax.set_xlabel("WineQuality")
    ax.set_ylabel("Numero di osservazioni")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    pdf_path = os.path.join(PLOTS_DIR, f"{filename}.pdf")
    png_path = os.path.join(PLOTS_DIR, f"{filename}.png")
    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"  -> {pdf_path}")
    print(f"  -> {png_path}")

    return pdf_path, png_path


def plot_baseline_comparison(df_all_results, filename="best_model_vs_baseline_6class"):
    baseline_rows = df_all_results[df_all_results["Model"].str.contains("Baseline")]
    model_rows = df_all_results[~df_all_results["Model"].str.contains("Baseline")]

    best_row = model_rows.loc[model_rows["Macro_F1_mean"].idxmax()]
    baseline_value = baseline_rows["Macro_F1_mean"].mean()

    bar_labels = [
        "Baseline",
        f"{best_row['Model']} {best_row['Feature_set'].split('_')[0]}",
    ]
    bar_values = [baseline_value, best_row["Macro_F1_mean"]]

    y_pos = np.arange(len(bar_labels))

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.barh(y_pos, bar_values, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(bar_labels)
    ax.invert_yaxis()

    for i, value in enumerate(bar_values):
        ax.text(value + 0.5, i, f"{value:.2f}", va="center", fontsize=8)

    ax.set_xlabel("Macro F1 media in CV (%)")
    ax.set_title("Best model vs baseline - 6 classes")
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()

    pdf_path = os.path.join(PLOTS_DIR, f"{filename}.pdf")
    png_path = os.path.join(PLOTS_DIR, f"{filename}.png")
    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"  -> {pdf_path}")
    print(f"  -> {png_path}")

    return pdf_path, png_path

def plot_confusion_matrix_rf(X_train, y_train, X_test, y_test,
                             filename="confusion_matrix_rf_full_test_6class"):
    rf = clone(get_models()["Random Forest"])
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    labels = sorted(set(y_test) | set(y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="viridis")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=9, color="black")

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel("Classe predetta")
    ax.set_ylabel("Classe reale")
    ax.set_title("Random Forest FULL - Test set")
    plt.tight_layout()

    pdf_path = os.path.join(PLOTS_DIR, f"{filename}.pdf")
    png_path = os.path.join(PLOTS_DIR, f"{filename}.png")
    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"  -> {pdf_path}")
    print(f"  -> {png_path}")

    return pdf_path, png_path


def print_cv_results(df_results, title):
    print(f"\n{title}")

    display_cols = [
        "Feature_set",
        "Model",
        "Accuracy_mean",
        "Accuracy_std",
        "Macro_F1_mean",
        "Macro_F1_std",
    ]

    print(
        df_results.sort_values("Macro_F1_mean", ascending=False)[display_cols]
        .to_string(index=False)
    )


def baseline_majority_test(y_train, y_test, feature_set_name, name="Majority class baseline"):
    majority_class = pd.Series(y_train).mode()[0]
    y_pred = np.full_like(y_test, fill_value=majority_class)

    return {
        "Feature_set": feature_set_name,
        "Model": name,
        "Test_Accuracy": accuracy_score(y_test, y_pred) * 100,
        "Test_Macro_F1": f1_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0,
        ) * 100,
        "Test_Macro_Prec": precision_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0,
        ) * 100,
        "Test_Macro_Rec": recall_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0,
        ) * 100,
    }


def evaluate_on_test_set(models, X_train, y_train, X_test, y_test, feature_set_name):
    rows = []

    for name, model in models.items():
        mdl = clone(model)
        mdl.fit(X_train, y_train)
        y_pred = mdl.predict(X_test)

        classes_present = sorted(set(y_test) | set(y_pred))
        f1_per_class = f1_score(
            y_test,
            y_pred,
            average=None,
            labels=classes_present,
            zero_division=0,
        ) * 100

        clean_f1 = {int(k): round(float(v), 1) for k, v in zip(classes_present, f1_per_class)}
        print(f"    {name} - per-class F1: {clean_f1}")

        rows.append({
            "Feature_set": feature_set_name,
            "Model": name,
            "Test_Accuracy": accuracy_score(y_test, y_pred) * 100,
            "Test_Macro_F1": f1_score(
                y_test,
                y_pred,
                average="macro",
                zero_division=0,
            ) * 100,
            "Test_Macro_Prec": precision_score(
                y_test,
                y_pred,
                average="macro",
                zero_division=0,
            ) * 100,
            "Test_Macro_Rec": recall_score(
                y_test,
                y_pred,
                average="macro",
                zero_division=0,
            ) * 100,
        })

    return rows


def add_results(all_results, results, feature_set_name):
    for result in results:
        all_results.append({
            "Feature_set": feature_set_name,
            "Model": result["model_name"],
            "Accuracy_mean": result["accuracy_mean"],
            "Accuracy_std": result["accuracy_std"],
            "Macro_F1_mean": result["f1_mean"],
            "Macro_F1_std": result["f1_std"],
        })


def main():
    print("WINE QUALITY CLASSIFICATION - 6-CLASS FINAL PIPELINE")

    df_train, df_test = preprocess_data(remove_noise=True)

    print("\nGenerating correlation heatmap:")
    plot_correlation_heatmap(df_train)

    print("\nGenerating target distribution plot:")
    plot_target_distribution(df_train)

    all_results = []
    all_sensitivity = []
    all_test_results = []
    all_clean_test_results = []

    for feature_set_name, feature_cols in FEATURE_SETS.items():
        print(f"\nFEATURE SET SCENARIO: {feature_set_name}")

        existing_features = [
            feature for feature in feature_cols
            if feature in df_train.columns and feature in df_test.columns
        ]

        X = df_train[existing_features].values
        y = df_train["WineQuality"].values.astype(int)

        X_test = df_test[existing_features].values
        y_test = df_test["WineQuality"].values.astype(int)

        print(f"\nFeatures used ({len(existing_features)}):")
        for feature in existing_features:
            print(f"  - {feature}")

        print(f"\nX shape: {X.shape}")
        class_dist = {int(k): int(v) for k, v in pd.Series(y).value_counts().items()}
        print(f"6-class distribution: {class_dist}")
        scenario_results = []
        scenario_results.append(
            baseline_majority_cv(
                y,
                cv_folds=CV_FOLDS,
                name="Majority class baseline",
            )
        )

        for name, model in get_models().items():
            print(f"  Evaluating 6-class | {feature_set_name} | {name}...")
            result = evaluate_with_cv(
                model,
                X,
                y,
                cv_folds=CV_FOLDS,
                model_name=name,
            )
            scenario_results.append(result)

        add_results(all_results, scenario_results, feature_set_name)

        all_test_results.append(
            baseline_majority_test(
                y,
                y_test,
                feature_set_name=feature_set_name,
            )
        )
        all_test_results += evaluate_on_test_set(
            get_models(),
            X,
            y,
            X_test,
            y_test,
            feature_set_name=feature_set_name,
        )

        if "Synthetic_Noise_Flag" in df_test.columns:
            clean_mask = (df_test["Synthetic_Noise_Flag"] == 0).values
            all_clean_test_results.append(
                baseline_majority_test(
                    y,
                    y_test[clean_mask],
                    feature_set_name=feature_set_name,
                )
            )
            all_clean_test_results += evaluate_on_test_set(
                get_models(),
                X,
                y,
                X_test[clean_mask],
                y_test[clean_mask],
                feature_set_name=feature_set_name,
            )
        if feature_set_name == "FULL_with_correlation":
            plot_confusion_matrix_rf(X, y, X_test, y_test)
        df_sens = analyze_hyperparameter_sensitivity(
            X,
            y,
            feature_set_name=feature_set_name,
            cv_folds=CV_FOLDS,
        )
        all_sensitivity.append(df_sens)

    df_all_results = pd.DataFrame(all_results).sort_values(
        "Macro_F1_mean",
        ascending=False,
    )
    cv_results_path = resolve_output_path(OUTPUT_FILES["cv_results"])
    df_all_results.to_csv(cv_results_path, index=False)
    print_cv_results(df_all_results, "6-CLASS CV RESULTS - FULL vs NOCORR")
    print(f"\nCV results saved in: {cv_results_path}")

    df_test_results = pd.DataFrame(all_test_results).sort_values(
        "Test_Macro_F1",
        ascending=False,
    )
    test_results_path = resolve_output_path(OUTPUT_FILES["test_results"])
    df_test_results.to_csv(test_results_path, index=False)
    print("\nFINAL RESULTS ON THE TEST SET")
    print(df_test_results.to_string(index=False))
    print(f"\nTest results saved in: {test_results_path}")

    if all_clean_test_results:
        df_clean_test_results = pd.DataFrame(all_clean_test_results).sort_values(
            "Test_Macro_F1",
            ascending=False,
        )
        clean_test_results_path = os.path.join(PLOTS_DIR, "test_set_results_6class_clean.csv")
        df_clean_test_results.to_csv(clean_test_results_path, index=False)
        print("\nFINAL RESULTS ON THE CLEAN TEST SUBSET")
        print(df_clean_test_results.to_string(index=False))
        print(f"\nClean test results saved in: {clean_test_results_path}")
    else:
        df_clean_test_results = None

    df_all_sensitivity = pd.concat(all_sensitivity, ignore_index=True).sort_values(
        "Macro_F1_mean",
        ascending=False,
    )
    sensitivity_path = resolve_output_path(OUTPUT_FILES["sensitivity"])
    df_all_sensitivity.to_csv(sensitivity_path, index=False)
    print(f"\nSensitivity saved in: {sensitivity_path}")

    print("\nGenerating final plots:")
    plot_cv_comparison(df_all_results)
    plot_hyperparameter_sensitivity(df_all_sensitivity)
    plot_baseline_comparison(df_all_results)

    print("\nFINAL SUMMARY")
    best_model = df_all_results.iloc[0]

    print("\nBest 6-class model by Macro F1:")
    print(best_model.to_string())

    print(f"\nPlots saved in: {PLOTS_DIR}")
    print("\nPipeline completed successfully.")

    return {
        "cv_results": df_all_results,
        "test_results": df_test_results,
        "clean_test_results": df_clean_test_results,
        "sensitivity": df_all_sensitivity,
    }


if __name__ == "__main__":
    results = main()