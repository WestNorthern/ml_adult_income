import os
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# Paths for raw and processed data
DATA_RAW_PATH = os.path.join("data", "raw", "adult_openml.csv")
DATA_PROCESSED_DIR = os.path.join("data", "processed")


def load_adult_from_openml(cache_path: str = DATA_RAW_PATH) -> pd.DataFrame:
    """
    Fetch the Adult dataset from OpenML, cache locally as CSV, and return a DataFrame with target column 'income'.
    """
    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path)
    else:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        adult = fetch_openml("adult", version=2, as_frame=True)
        df = adult.frame
        # Rename the target column (often named 'class') to 'income'
        target_col = adult.target.name if hasattr(adult.target, 'name') else 'class'
        df = df.rename(columns={target_col: 'income'})
        df.to_csv(cache_path, index=False)
    # Always ensure target column is named 'income'
    if 'class' in df.columns:
        df = df.rename(columns={'class': 'income'})
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw DataFrame by replacing '?' with NaN and dropping missing rows.
    """
    df = df.replace("?", pd.NA)
    df = df.dropna().reset_index(drop=True)
    return df


def split_data(
    df: pd.DataFrame,
    target: str = "income",
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    test_frac: float = 0.2,
    random_state: int = 42,
    stratify: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Split df into train, validation, and test sets with given fractions.
    Returns X_train, X_val, X_test, y_train, y_val, y_test.
    """
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in DataFrame columns: {df.columns.tolist()}")

    X = df.drop(columns=target)
    y = df[target]
    stratify_arg = y if stratify else None

    # First split off test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_frac, random_state=random_state, stratify=stratify_arg
    )
    # Compute fraction for validation from remaining portion
    val_fraction_of_temp = val_frac / (train_frac + val_frac)
    stratify_temp = y_temp if stratify else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_fraction_of_temp,
        random_state=random_state,
        stratify=stratify_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def save_splits(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    processed_dir: str = DATA_PROCESSED_DIR
):
    """
    Save the train/val/test splits as CSVs under the processed directory.
    """
    os.makedirs(processed_dir, exist_ok=True)
    X_train.to_csv(os.path.join(processed_dir, "X_train.csv"), index=False)
    y_train.to_csv(os.path.join(processed_dir, "y_train.csv"), index=False)
    X_val.to_csv(os.path.join(processed_dir, "X_val.csv"), index=False)
    y_val.to_csv(os.path.join(processed_dir, "y_val.csv"), index=False)
    X_test.to_csv(os.path.join(processed_dir, "X_test.csv"), index=False)
    y_test.to_csv(os.path.join(processed_dir, "y_test.csv"), index=False)


def main():
    # 1. Load
    df_raw = load_adult_from_openml()
    print(f"Raw data shape: {df_raw.shape}")

    # 2. Clean
    df_clean = clean_data(df_raw)
    print(f"Cleaned data shape: {df_clean.shape}")

    # 3. Split
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_clean)
    print(
        f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}"
    )

    # 4. Save
    save_splits(X_train, X_val, X_test, y_train, y_val, y_test)
    print("Data splits saved to data/processed/")


if __name__ == "__main__":
    main()
