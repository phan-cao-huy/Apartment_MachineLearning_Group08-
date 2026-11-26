import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def split_temporal_test(df, time_col="collected_at", test_days=7):
    df_sorted = df.dropna(subset=[time_col]).copy()
    df_sorted[time_col] = pd.to_datetime(df_sorted[time_col])
    max_date = df_sorted[time_col].max()
    cutoff_date = max_date - pd.Timedelta(days=test_days)

    test_mask = df_sorted[time_col] > cutoff_date
    train_val_df = df_sorted[~test_mask].reset_index(drop=True)
    test_df = df_sorted[test_mask].reset_index(drop=True)
    return train_val_df, test_df


class SpatialBlockCV:
    def __init__(self, n_splits=5, group_col="ward_2025", fallback_col="District"):
        self.n_splits = n_splits
        self.group_col = group_col
        self.fallback_col = fallback_col
        self.gkf = GroupKFold(n_splits=n_splits)

    def build_groups(self, df):
        groups = df[self.group_col].astype("object").copy()
        missing_mask = groups.isna() | (groups.astype(str).str.strip() == "") | (groups.astype(str).str.lower() == "nan")
        if self.fallback_col and self.fallback_col in df.columns:
            fallback_vals = "DISTRICT_" + df[self.fallback_col].fillna("UNKNOWN").astype(str)
            groups[missing_mask] = fallback_vals[missing_mask]
        else:
            groups[missing_mask] = "UNKNOWN_WARD"
        return groups.astype(str).values

    def split(self, X, y=None, groups=None):
        if groups is None:
            if isinstance(X, pd.DataFrame) and self.group_col in X.columns:
                groups = self.build_groups(X)
            else:
                raise ValueError(f"Cần truyền mảng groups hoặc DataFrame có cột '{self.group_col}'.")
        return self.gkf.split(X, y, groups=groups)

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


def prepare_hanoi_splits(df, time_col="collected_at", group_col="ward_2025", fallback_col="District", test_days=7, n_splits=5):
    train_val_df, test_df = split_temporal_test(df, time_col=time_col, test_days=test_days)
    cv = SpatialBlockCV(n_splits=n_splits, group_col=group_col, fallback_col=fallback_col)
    groups = cv.build_groups(train_val_df)
    folds = list(cv.split(train_val_df, groups=groups))
    return train_val_df, test_df, folds