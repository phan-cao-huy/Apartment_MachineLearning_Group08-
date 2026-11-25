import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def split_temporal_test(df, time_col="collected_at", test_days=7):
    """
    Tách tập kiểm tra theo thời gian (tuần thu thập cuối cùng).
    Tham số:
        df: DataFrame dữ liệu căn hộ Hà Nội.
        time_col: Tên cột lưu thời điểm thu thập dữ liệu.
        test_days: Số ngày cuối cùng dùng làm tập Test (mặc định 7 ngày = 1 tuần cuối).
    Trả về:
        train_val_df, test_df
    """
    df_sorted = df.copy()
    df_sorted[time_col] = pd.to_datetime(df_sorted[time_col])
    max_date = df_sorted[time_col].max()
    cutoff_date = max_date - pd.Timedelta(days=test_days)

    test_mask = df_sorted[time_col] > cutoff_date
    train_val_df = df_sorted[~test_mask].reset_index(drop=True)
    test_df = df_sorted[test_mask].reset_index(drop=True)
    return train_val_df, test_df


class SpatialBlockCV:
    """
    Kiểm định chéo theo khối không gian (Spatial Block Cross-Validation),
    nhóm các căn hộ theo đơn vị hành chính cấp xã/phường để tránh rò rỉ không gian.
    """
    def __init__(self, n_splits=5, group_col="ward"):
        self.n_splits = n_splits
        self.group_col = group_col
        self.gkf = GroupKFold(n_splits=n_splits)

    def split(self, X, y=None, groups=None):
        if groups is None:
            if isinstance(X, pd.DataFrame) and self.group_col in X.columns:
                groups = X[self.group_col].values
            else:
                raise ValueError(f"Cần truyền mảng groups hoặc DataFrame có cột '{self.group_col}'.")
        return self.gkf.split(X, y, groups=groups)

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


def prepare_hanoi_splits(df, time_col="collected_at", group_col="ward", test_days=7, n_splits=5):
    """
    Hàm tiện ích dùng chung cho cả nhóm (Vinh và Huy):
    1. Tách tập Test là tuần thu thập cuối cùng.
    2. Tạo danh sách các cặp chỉ số (train_idx, val_idx) theo Spatial Block CV trên phần dữ liệu còn lại.
    """
    train_val_df, test_df = split_temporal_test(df, time_col=time_col, test_days=test_days)
    cv = SpatialBlockCV(n_splits=n_splits, group_col=group_col)
    folds = list(cv.split(train_val_df, groups=train_val_df[group_col].values))
    return train_val_df, test_df, folds
