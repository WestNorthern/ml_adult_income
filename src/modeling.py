import os
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)

# Define feature groups based on the Adult dataset
NUMERIC_FEATURES = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]

CATEGORICAL_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

TARGET_NAME = "income"


def load_data_splits(processed_dir: str = "data/processed") -> tuple:
    """
    Load train/val/test splits from CSV files and return X_train, X_val, X_test, y_train, y_val, y_test.
    """
    X_train = pd.read_csv(os.path.join(processed_dir, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(processed_dir, "y_train.csv")).squeeze()
    X_val = pd.read_csv(os.path.join(processed_dir, "X_val.csv"))
    y_val = pd.read_csv(os.path.join(processed_dir, "y_val.csv")).squeeze()
    X_test = pd.read_csv(os.path.join(processed_dir, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(processed_dir, "y_test.csv")).squeeze()
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_preprocessor(numeric_features=NUMERIC_FEATURES, categorical_features=CATEGORICAL_FEATURES) -> ColumnTransformer:
    """
    Create a ColumnTransformer for numeric and categorical preprocessing.
    Numeric: median imputation + standard scaling.
    Categorical: most-frequent imputation + one-hot encoding.
    """
    num_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    cat_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("num", num_pipeline, numeric_features),
        ("cat", cat_pipeline, categorical_features),
    ], remainder="drop")
    return preprocessor


def build_model_pipeline(model, numeric_features=NUMERIC_FEATURES, categorical_features=CATEGORICAL_FEATURES) -> Pipeline:
    """
    Wrap a scikit-learn estimator with the preprocessing pipeline.
    """
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("classifier", model),
    ])
    return pipeline


def evaluate_model(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, pos_label: str = ">50K") -> dict:
    """
    Evaluate the pipeline on X, y.
    Returns a dict of accuracy, precision, recall, f1.
    """
    y_pred = pipeline.predict(X)
    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, pos_label=pos_label),
        "recall": recall_score(y, y_pred, pos_label=pos_label),
        "f1": f1_score(y, y_pred, pos_label=pos_label),
    }
    return metrics


def evaluate_by_group(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, group_col: str, pos_label: str = ">50K") -> pd.DataFrame:
    """
    Compute evaluation metrics by subgroup defined in X[group_col].
    Returns a DataFrame indexed by subgroup with columns accuracy, precision, recall, f1.
    """
    results = {}
    for grp in X[group_col].unique():
        mask = X[group_col] == grp
        metrics = evaluate_model(pipeline, X[mask], y[mask], pos_label)
        results[grp] = metrics
    return pd.DataFrame(results).T


def plot_group_metric(df_metrics: pd.DataFrame, metric: str, output_dir: str = "reports/figures"):
    """
    Plot bar chart for a specified metric across subgroups.
    Saves figure to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    ax = df_metrics[metric].plot(kind="bar", title=f"{metric.title()} by Group")
    ax.set_ylabel(metric.title())
    fig = ax.get_figure()
    fig.savefig(os.path.join(output_dir, f"{metric}_by_group.png"))
    plt.close()


def main():
    X_train, X_val, X_test, y_train, y_val, y_test = load_data_splits()

    # Build pipelines with stronger models
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "DecisionTree": DecisionTreeClassifier(max_depth=5),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "HistGB": HistGradientBoostingClassifier(random_state=42),
    }

    pipelines = {name: build_model_pipeline(clf) for name, clf in models.items()}

    # Train all
    for name, pipe in pipelines.items():
        print(f"Training {name}...")
        pipe.fit(X_train, y_train)

    # Evaluate overall on validation
    print("\nOverall validation metrics:")
    for name, pipe in pipelines.items():
        print(f"{name}:", evaluate_model(pipe, X_val, y_val))

    # Evaluate by subgroup (race and sex)
    for name, pipe in pipelines.items():
        print(f"\n{name} metrics by race:")
        rm = evaluate_by_group(pipe, X_val, y_val, group_col="race")
        print(rm)
        plot_group_metric(rm, "recall")
        plot_group_metric(rm, "precision")

        print(f"\n{name} metrics by sex:")
        sm = evaluate_by_group(pipe, X_val, y_val, group_col="sex")
        print(sm)
        plot_group_metric(sm, "recall")
        plot_group_metric(sm, "precision")

if __name__ == "__main__":
    main()
