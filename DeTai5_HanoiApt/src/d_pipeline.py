# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

"""
Phần D – Ứng dụng thực tế: Định giá căn hộ Hà Nội
====================================================
Chạy từng bước:
    python d_pipeline.py --step prep       # D1: làm sạch + feature engineering
    python d_pipeline.py --step model      # D2: huấn luyện hedonic / RF / GB
    python d_pipeline.py --step tune       # D3: tối ưu Optuna
    python d_pipeline.py --step eval       # D4: đánh giá CV + SHAP
    python d_pipeline.py --step all        # chạy toàn bộ
"""

import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--step", default="all",
                    choices=["prep", "model", "tune", "eval", "all"])
args = parser.parse_args()

STEPS = ["prep", "model", "tune", "eval"] if args.step == "all" else [args.step]

# ── Import chung ──────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from pathlib import Path

DATA_RAW  = Path("hanoi_apt_clean_5k.csv")
DATA_PREP = Path("hanoi_apt_prepared.parquet")
RESULTS   = Path("d_results.txt")

# =============================================================================
# BƯỚC 1 – LÀM SẠCH & FEATURE ENGINEERING
# =============================================================================
def step_prep():
    print("\n" + "="*60)
    print("BƯỚC 1: Làm sạch & Feature Engineering")
    print("="*60)

    df = pd.read_csv(DATA_RAW)
    print(f"Đọc xong: {len(df):,} dòng, {df.shape[1]} cột")

    # 1a. Loại bản ghi thiếu giá hoặc diện tích
    before = len(df)
    df = df.dropna(subset=["log_price_m2", "area_m2"])
    df = df[df["area_m2"] > 0]
    df = df[df["price_per_m2"] > 0]
    print(f"Sau loại null giá/diện tích: {len(df):,} (bỏ {before - len(df):,})")

    # 1b. Lọc outlier giá (ngoài khoảng 1%–99%)
    lo, hi = df["log_price_m2"].quantile([0.01, 0.99])
    before = len(df)
    df = df[df["log_price_m2"].between(lo, hi)]
    print(f"Sau lọc outlier giá (1%–99%): {len(df):,} (bỏ {before - len(df):,})")

    # 1c. Điền null bedrooms / bathrooms bằng median theo quận
    for col in ["bedrooms", "bathrooms"]:
        med = df.groupby("District")[col].transform("median")
        global_med = df[col].median()
        df[col] = df[col].fillna(med).fillna(global_med)
    print(f"Đã điền null bedrooms/bathrooms theo median quận")

    # 1d. Tạo cột tuần từ collected_at (dùng cho test theo thời gian)
    df["collected_at"] = pd.to_datetime(df["collected_at"])
    df["week"] = df["collected_at"].dt.isocalendar().week.astype(int)
    df["year"] = df["collected_at"].dt.year
    # week_rank: 0 = tuần đầu, N = tuần cuối (dùng để split)
    df["week_id"] = df["year"] * 100 + df["week"]
    print(f"Phạm vi tuần: {df['week_id'].min()} – {df['week_id'].max()}")
    print(f"Số tuần khác nhau: {df['week_id'].nunique()}")

    # 1e. Mã hóa District (label encoding theo tần suất)
    freq = df["District"].value_counts()
    df["district_enc"] = df["District"].map(freq).astype(float)

    # 1f. Mã hóa interior_level (đã là int, giữ nguyên)
    # has_red_book: đã là 0/1

    # 1g. Feature diện tích gộp
    df["area_log"]   = np.log1p(df["area_m2"])
    df["room_ratio"] = df["bedrooms"] / df["bathrooms"].clip(lower=1)

    # 1h. Source encoding (one-hot)
    df = pd.get_dummies(df, columns=["source"], drop_first=False, dtype=int)

    # 1i. Tóm tắt feature cuối
    FEATURES = [
        "area_m2", "area_log",
        "bedrooms", "bathrooms", "room_ratio",
        "has_red_book", "interior_level",
        "district_enc",
    ] + [c for c in df.columns if c.startswith("source_")]

    TARGET = "log_price_m2"

    print(f"\nFeatures ({len(FEATURES)}): {FEATURES}")
    print(f"Target: {TARGET}")
    print(f"\nMẫu cuối: {len(df):,} dòng")

    # Lưu
    df["__features__"] = str(FEATURES)   # lưu tên feature vào meta
    df.to_parquet(DATA_PREP, index=False)
    print(f"Đã lưu → {DATA_PREP}")
    return df, FEATURES, TARGET


# =============================================================================
# BƯỚC 2 – MÔ HÌNH HÓA
# =============================================================================
def step_model():
    print("\n" + "="*60)
    print("BƯỚC 2: Mô hình hóa – Hedonic / RF / GB (thư viện + tự cài)")
    print("="*60)

    import sys, os
    sys.path.insert(0, str(Path(".").resolve()))

    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor as SKLearnRF, GradientBoostingRegressor as SKLearnGB
    from sklearn.metrics import mean_squared_error
    import xgboost as xgb
    import lightgbm as lgb

    df = pd.read_parquet(DATA_PREP)

    # Lấy lại danh sách feature
    FEATURES = [
        "area_m2", "area_log",
        "bedrooms", "bathrooms", "room_ratio",
        "has_red_book", "interior_level",
        "district_enc",
    ] + [c for c in df.columns if c.startswith("source_")]
    TARGET = "log_price_m2"

    # ── Split: test = tuần cuối cùng ──────────────────────────────────────────
    last_week = df["week_id"].max()
    mask_test = df["week_id"] == last_week
    mask_train = ~mask_test

    X_train = df.loc[mask_train, FEATURES].values
    y_train = df.loc[mask_train, TARGET].values
    X_test  = df.loc[mask_test,  FEATURES].values
    y_test  = df.loc[mask_test,  TARGET].values

    print(f"Train: {len(X_train):,}  |  Test (tuần {last_week}): {len(X_test):,}")

    results = {}

    # ── 2.1 Hồi quy Hedonic (Ridge) ──────────────────────────────────────────
    print("\n[2.1] Hedonic (Ridge)...")
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    pred = ridge.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mape = np.mean(np.abs(np.expm1(pred) - np.expm1(y_test)) / np.expm1(y_test)) * 100
    results["Hedonic (Ridge)"] = {"RMSE_log": rmse, "MAPE_%": mape}
    print(f"  RMSE(log) = {rmse:.4f}  |  MAPE = {mape:.2f}%")

    # ── 2.2 Random Forest (sklearn) ───────────────────────────────────────────
    print("\n[2.2] Random Forest (sklearn)...")
    rf_sk = SKLearnRF(n_estimators=200, max_features=1/3,
                      min_samples_leaf=5, n_jobs=-1, random_state=42)
    rf_sk.fit(X_train, y_train)
    pred = rf_sk.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mape = np.mean(np.abs(np.expm1(pred) - np.expm1(y_test)) / np.expm1(y_test)) * 100
    results["RF (sklearn)"] = {"RMSE_log": rmse, "MAPE_%": mape}
    print(f"  RMSE(log) = {rmse:.4f}  |  MAPE = {mape:.2f}%")

    # ── 2.3 Gradient Boosting (sklearn) ───────────────────────────────────────
    print("\n[2.3] Gradient Boosting (sklearn)...")
    gb_sk = SKLearnGB(n_estimators=300, learning_rate=0.05,
                      max_depth=5, subsample=0.8,
                      min_samples_leaf=5, random_state=42)
    gb_sk.fit(X_train, y_train)
    pred = gb_sk.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mape = np.mean(np.abs(np.expm1(pred) - np.expm1(y_test)) / np.expm1(y_test)) * 100
    results["GB (sklearn)"] = {"RMSE_log": rmse, "MAPE_%": mape}
    print(f"  RMSE(log) = {rmse:.4f}  |  MAPE = {mape:.2f}%")

    # ── 2.4 XGBoost (thư viện) ────────────────────────────────────────────────
    print("\n[2.4] XGBoost (thư viện)...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=300, learning_rate=0.05,
        max_depth=6, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, min_child_weight=5,
        early_stopping_rounds=20, eval_metric="rmse",
        random_state=42, verbosity=0
    )
    xgb_model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)], verbose=False)
    pred = xgb_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mape = np.mean(np.abs(np.expm1(pred) - np.expm1(y_test)) / np.expm1(y_test)) * 100
    results["XGBoost"] = {"RMSE_log": rmse, "MAPE_%": mape}
    print(f"  RMSE(log) = {rmse:.4f}  |  MAPE = {mape:.2f}%")

    # ── 2.5 LightGBM (thư viện) ───────────────────────────────────────────────
    print("\n[2.5] LightGBM (thư viện)...")
    lgb_model = lgb.LGBMRegressor(
        n_estimators=300, learning_rate=0.05,
        num_leaves=63, subsample=0.8, colsample_bytree=0.8,
        min_child_samples=10, reg_lambda=1.0,
        random_state=42, verbose=-1
    )
    lgb_model.fit(X_train, y_train,
                  eval_X=X_test, eval_y=y_test,
                  callbacks=[lgb.early_stopping(20, verbose=False),
                              lgb.log_evaluation(-1)])
    pred = lgb_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mape = np.mean(np.abs(np.expm1(pred) - np.expm1(y_test)) / np.expm1(y_test)) * 100
    results["LightGBM"] = {"RMSE_log": rmse, "MAPE_%": mape}
    print(f"  RMSE(log) = {rmse:.4f}  |  MAPE = {mape:.2f}%")

    # ── 2.6 RF + GB tự cài (phần A) ───────────────────────────────────────────
    print("\n[2.6] RF tự cài (forest.py)...")
    try:
        from forest import RandomForestRegressor as MyRF
        my_rf = MyRF(n_estimators=100, max_features="auto",
                     min_samples_leaf=5, n_jobs=1, random_state=42)
        my_rf.fit(X_train, y_train)
        pred = my_rf.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        mape = np.mean(np.abs(np.expm1(pred) - np.expm1(y_test)) / np.expm1(y_test)) * 100
        results["RF (tự cài)"] = {"RMSE_log": rmse, "MAPE_%": mape}
        print(f"  RMSE(log) = {rmse:.4f}  |  MAPE = {mape:.2f}%")
    except Exception as e:
        print(f"  Lỗi RF tự cài: {e}")
        results["RF (tự cài)"] = {"RMSE_log": float("nan"), "MAPE_%": float("nan")}

    print("\n[2.7] GradientBoosting tự cài (boosting.py)...")
    try:
        from boosting import GradientBoostingRegressor as MyGB
        my_gb = MyGB(n_estimators=100, learning_rate=0.1,
                     max_depth=4, subsample=0.8,
                     min_samples_leaf=5, random_state=42)
        my_gb.fit(X_train, y_train)
        pred = my_gb.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        mape = np.mean(np.abs(np.expm1(pred) - np.expm1(y_test)) / np.expm1(y_test)) * 100
        results["GB (tự cài)"] = {"RMSE_log": rmse, "MAPE_%": mape}
        print(f"  RMSE(log) = {rmse:.4f}  |  MAPE = {mape:.2f}%")
    except Exception as e:
        print(f"  Lỗi GB tự cài: {e}")
        results["GB (tự cài)"] = {"RMSE_log": float("nan"), "MAPE_%": float("nan")}

    # ── Tóm tắt ───────────────────────────────────────────────────────────────
    print("\n" + "-"*50)
    print(f"{'Mô hình':<25} {'RMSE(log)':>10} {'MAPE%':>8}")
    print("-"*50)
    for name, r in results.items():
        print(f"{name:<25} {r['RMSE_log']:>10.4f} {r['MAPE_%']:>7.2f}%")

    # Lưu kết quả để bước eval dùng
    import joblib
    joblib.dump({
        "models": {
            "ridge": ridge,
            "rf_sk": rf_sk,
            "gb_sk": gb_sk,
            "xgb": xgb_model,
            "lgb": lgb_model,
        },
        "features": FEATURES,
        "results_baseline": results,
        "last_week": last_week,
    }, "d_models.pkl")
    print("\nĐã lưu models → d_models.pkl")
    return results


# =============================================================================
# BƯỚC 3 – TỐI ƯU SIÊU THAM SỐ (Optuna TPE)
# =============================================================================
def step_tune():
    print("\n" + "="*60)
    print("BƯỚC 3: Tối ưu siêu tham số – Grid / Random / Optuna TPE")
    print("="*60)

    import optuna
    import joblib
    import xgboost as xgb
    from sklearn.model_selection import cross_val_score, GroupKFold
    from sklearn.metrics import mean_squared_error
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    df = pd.read_parquet(DATA_PREP)
    saved = joblib.load("d_models.pkl")
    FEATURES = saved["features"]
    TARGET = "log_price_m2"
    last_week = saved["last_week"]

    mask_train = df["week_id"] != last_week
    X_tr = df.loc[mask_train, FEATURES].values
    y_tr = df.loc[mask_train, TARGET].values
    groups = df.loc[mask_train, "district_enc"].values  # CV theo quận

    gkf = GroupKFold(n_splits=5)

    def cv_rmse(model):
        scores = cross_val_score(model, X_tr, y_tr, groups=groups,
                                 cv=gkf, scoring="neg_root_mean_squared_error",
                                 n_jobs=-1)
        return -scores.mean(), scores.std()

    # ── 3.1 Grid Search (nhỏ, minh họa) ──────────────────────────────────────
    print("\n[3.1] Grid Search (LightGBM, 9 tổ hợp)...")
    import lightgbm as lgb
    from itertools import product

    grid = {"learning_rate": [0.05, 0.1], "num_leaves": [31, 63], "n_estimators": [200]}
    best_grid = {"rmse": np.inf, "params": {}}
    for lr, nl in product(grid["learning_rate"], grid["num_leaves"]):
        m = lgb.LGBMRegressor(learning_rate=lr, num_leaves=nl,
                              n_estimators=200, random_state=42, verbose=-1)
        rmse, std = cv_rmse(m)
        print(f"  lr={lr} nl={nl}  →  CV RMSE={rmse:.4f} ±{std:.4f}")
        if rmse < best_grid["rmse"]:
            best_grid = {"rmse": rmse, "std": std, "params": {"lr": lr, "nl": nl}}
    print(f"  → Tốt nhất Grid: {best_grid}")

    # ── 3.2 Random Search (20 tổ hợp) ────────────────────────────────────────
    print("\n[3.2] Random Search (LightGBM, 20 tổ hợp)...")
    rng = np.random.default_rng(42)
    best_rand = {"rmse": np.inf, "params": {}}
    for _ in range(20):
        lr  = float(rng.choice([0.01, 0.03, 0.05, 0.08, 0.1, 0.15]))
        nl  = int(rng.choice([15, 31, 63, 127]))
        n   = int(rng.choice([100, 200, 300]))
        sub = float(rng.uniform(0.6, 1.0))
        m = lgb.LGBMRegressor(learning_rate=lr, num_leaves=nl,
                              n_estimators=n, subsample=sub,
                              random_state=42, verbose=-1)
        rmse, std = cv_rmse(m)
        if rmse < best_rand["rmse"]:
            best_rand = {"rmse": rmse, "std": std,
                         "params": {"lr": lr, "nl": nl, "n": n, "sub": sub}}
    print(f"  → Tốt nhất Random: {best_rand}")

    # ── 3.3 Optuna TPE (50 trials, XGBoost) ───────────────────────────────────
    print("\n[3.3] Optuna TPE (XGBoost, 50 trials)...")

    def objective(trial):
        params = {
            "n_estimators":    trial.suggest_int("n_estimators", 100, 500),
            "learning_rate":   trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth":       trial.suggest_int("max_depth", 3, 8),
            "subsample":       trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda":      trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
            "min_child_weight":trial.suggest_int("min_child_weight", 1, 20),
        }
        m = xgb.XGBRegressor(**params, random_state=42, verbosity=0)
        rmse, _ = cv_rmse(m)
        return rmse

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=50, show_progress_bar=False)

    best_tpe = study.best_trial
    print(f"  → Tốt nhất TPE: RMSE={best_tpe.value:.4f}")
    print(f"    Params: {best_tpe.params}")

    # ── So sánh 3 phương pháp ─────────────────────────────────────────────────
    print("\n" + "-"*55)
    print(f"{'Phương pháp':<20} {'CV RMSE(log)':>14} {'Tham số tốt nhất'}")
    print("-"*55)
    print(f"{'Grid Search':<20} {best_grid['rmse']:>14.4f}  {best_grid['params']}")
    print(f"{'Random Search':<20} {best_rand['rmse']:>14.4f}  {best_rand['params']}")
    print(f"{'Optuna TPE':<20} {best_tpe.value:>14.4f}  {best_tpe.params}")

    # Lưu best model (XGB với params TPE) để bước eval dùng
    best_xgb = xgb.XGBRegressor(**best_tpe.params, random_state=42, verbosity=0)
    best_xgb.fit(X_tr, y_tr)

    saved["best_xgb_tuned"] = best_xgb
    saved["tune_summary"] = {
        "grid":   best_grid,
        "random": best_rand,
        "tpe":    {"rmse": best_tpe.value, "params": best_tpe.params},
    }
    joblib.dump(saved, "d_models.pkl")
    print("\nĐã lưu model tối ưu → d_models.pkl")


# =============================================================================
# BƯỚC 4 – ĐÁNH GIÁ & DIỄN GIẢI
# =============================================================================
def step_eval():
    print("\n" + "="*60)
    print("BƯỚC 4: Đánh giá CV không gian + SHAP + Thảo luận thiên lệch")
    print("="*60)

    import joblib, shap
    import xgboost as xgb
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import mean_squared_error

    df = pd.read_parquet(DATA_PREP)
    saved = joblib.load("d_models.pkl")
    FEATURES = saved["features"]
    TARGET = "log_price_m2"
    last_week = saved["last_week"]

    mask_test  = df["week_id"] == last_week
    mask_train = ~mask_test
    X_tr  = df.loc[mask_train, FEATURES].values
    y_tr  = df.loc[mask_train, TARGET].values
    X_te  = df.loc[mask_test,  FEATURES].values
    y_te  = df.loc[mask_test,  TARGET].values
    groups_tr = df.loc[mask_train, "District"].values

    best_model = saved.get("best_xgb_tuned") or saved["models"]["xgb"]

    # ── 4.1 CV theo khối không gian (GroupKFold theo quận) ────────────────────
    print("\n[4.1] CV theo khối không gian (5 fold, nhóm theo quận)...")
    gkf = GroupKFold(n_splits=5)
    fold_rmses, fold_mapes = [], []
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_tr, y_tr, groups_tr)):
        best_model.fit(X_tr[tr_idx], y_tr[tr_idx])
        pred = best_model.predict(X_tr[va_idx])
        rmse = np.sqrt(mean_squared_error(y_tr[va_idx], pred))
        mape = np.mean(np.abs(np.expm1(pred) - np.expm1(y_tr[va_idx]))
                       / np.expm1(y_tr[va_idx])) * 100
        fold_rmses.append(rmse)
        fold_mapes.append(mape)
        print(f"  Fold {fold+1}: RMSE={rmse:.4f}  MAPE={mape:.2f}%")
    print(f"  → CV TB: RMSE={np.mean(fold_rmses):.4f} ±{np.std(fold_rmses):.4f}"
          f"   MAPE={np.mean(fold_mapes):.2f}% ±{np.std(fold_mapes):.2f}%")

    # ── 4.2 Test theo thời gian (tuần cuối) ───────────────────────────────────
    print(f"\n[4.2] Test theo thời gian (tuần {last_week})...")
    # Fit lại trên toàn bộ train
    best_model.fit(X_tr, y_tr)
    pred_te = best_model.predict(X_te)
    rmse_te = np.sqrt(mean_squared_error(y_te, pred_te))
    mape_te = np.mean(np.abs(np.expm1(pred_te) - np.expm1(y_te))
                      / np.expm1(y_te)) * 100
    print(f"  Test RMSE(log) = {rmse_te:.4f}")
    print(f"  Test MAPE      = {mape_te:.2f}%")

    # ── 4.3 SHAP (TreeExplainer) ──────────────────────────────────────────────
    print("\n[4.3] SHAP – tầm quan trọng feature...")
    sample_idx = np.random.default_rng(42).integers(0, len(X_te),
                                                     min(500, len(X_te)))
    explainer  = shap.TreeExplainer(best_model)
    shap_vals  = explainer.shap_values(X_te[sample_idx])
    mean_abs   = np.abs(shap_vals).mean(axis=0)
    order      = np.argsort(mean_abs)[::-1]
    print(f"\n  {'Feature':<25} {'|SHAP| trung bình':>18}")
    print(f"  {'-'*45}")
    for i in order:
        print(f"  {FEATURES[i]:<25} {mean_abs[i]:>18.4f}")

    # ── 4.4 Partial Dependence (thủ công, 2 feature quan trọng nhất) ──────────
    print("\n[4.4] Partial Dependence (top-2 feature)...")
    top2 = [FEATURES[order[0]], FEATURES[order[1]]]
    for feat in top2:
        fi = FEATURES.index(feat)
        grid_vals = np.percentile(X_te[:, fi],
                                  np.linspace(5, 95, 10))
        X_pdp = X_te[:min(500, len(X_te))].copy()
        preds = []
        for v in grid_vals:
            X_pdp[:, fi] = v
            preds.append(best_model.predict(X_pdp).mean())
        print(f"\n  PDP – {feat}:")
        for v, p in zip(grid_vals, preds):
            bar = "█" * int((p - min(preds)) / (max(preds) - min(preds) + 1e-9) * 20)
            print(f"    {v:8.2f} → log_giá={p:.4f}  {bar}")

    # ── 4.5 Thảo luận thiên lệch giá rao vs giá giao dịch ────────────────────
    print("\n[4.5] Thảo luận thiên lệch giá rao bán ≠ giá giao dịch")
    print("""
  Dữ liệu này là GIÁ RAO BÁN (asking price), không phải giá giao dịch thực.
  Các nguồn thiên lệch chính:

  1. Selection bias: chỉ rao những căn muốn bán, không có căn đã bán
     → phân phối giá có thể lệch phải (người bán kỳ vọng cao hơn thực tế)

  2. Negotiation discount: thực tế thương lượng thường giảm 5–15%
     → mô hình ước lượng giá "kỳ vọng người bán", không phải giá chốt

  3. Survivorship bias: tin rao lâu ngày (nhà khó bán) vẫn còn trên sàn
     → có thể làm tăng phương sai và nhiễu mô hình

  4. Giá "thỏa thuận": đã lọc / điền median, nhưng vẫn gây nhiễu gốc

  Khuyến nghị: để ước tính giá giao dịch thực, cần nhân hệ số chiết khấu
  (discount factor) ~0.90–0.95 hoặc dùng dữ liệu giao dịch từ Bộ Xây dựng.
    """)

    # ── Tổng kết kết quả ──────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("TỔNG KẾT KẾT QUẢ PHẦN D")
    print("="*60)
    if "results_baseline" in saved:
        print(f"\n{'Mô hình':<25} {'RMSE(log)':>10} {'MAPE%':>8}")
        print("-"*50)
        for name, r in saved["results_baseline"].items():
            print(f"{name:<25} {r['RMSE_log']:>10.4f} {r['MAPE_%']:>7.2f}%")
    if "tune_summary" in saved:
        ts = saved["tune_summary"]
        print(f"\nTối ưu siêu tham số:")
        print(f"  Grid Search  CV RMSE = {ts['grid']['rmse']:.4f}")
        print(f"  Random Search CV RMSE = {ts['random']['rmse']:.4f}")
        print(f"  Optuna TPE   CV RMSE = {ts['tpe']['rmse']:.4f}")
    print(f"\nModel tốt nhất (XGB tuned):")
    print(f"  Test RMSE(log) = {rmse_te:.4f}")
    print(f"  Test MAPE      = {mape_te:.2f}%")
    print(f"\n  → Trên thang giá/m²: sai số trung bình ≈ {mape_te:.1f}%")
    print("="*60)


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    if "prep" in STEPS:
        step_prep()
    if "model" in STEPS:
        step_model()
    if "tune" in STEPS:
        step_tune()
    if "eval" in STEPS:
        step_eval()
