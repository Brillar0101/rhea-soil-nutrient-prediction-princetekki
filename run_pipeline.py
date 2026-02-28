"""
=============================================================================
RHEA SOIL NUTRIENT PREDICTION CHALLENGE - AWARD WINNING PIPELINE v2
=============================================================================

KEY INSIGHT: 80% of test points are within 1km of training points!
This is fundamentally a SPATIAL INTERPOLATION problem.

Strategy:
1. Aggressive nearest-neighbor IDW features at multiple K values
2. Cross-nutrient correlations exploited via neighbor features
3. Auxiliary train features (C_organic, C_total, ph) as neighbor aggregates
4. Multi-model ensemble: LightGBM, XGBoost, CatBoost, ExtraTrees
5. Optimal weight blending per target
6. TargetPred_To_Keep masking for final submission

Critical observations:
- Depth is binary (0-20 or 20-50) - simple feature
- All train dates are identical - temporal features mostly useless
- Test has 2 date groups: 2008-2018 (81%) and 2022-2025 (19%)
- Sparse targets (B, Na, P, S, Zn): only ~1900 train samples each
- Strong cross-correlations: Ca-Mg=0.72, Cu-Mg=0.66, Ca-K=0.61
"""

import os
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
import sys

# Force unbuffered output
class Unbuffered:
    def __init__(self, stream):
        self.stream = stream
    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
    def writelines(self, datas):
        self.stream.writelines(datas)
        self.stream.flush()
    def __getattr__(self, attr):
        return getattr(self.stream, attr)

sys.stdout = Unbuffered(sys.stdout)
sys.stderr = Unbuffered(sys.stderr)
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import BallTree
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge
from scipy.optimize import minimize
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================
SEED = 42
N_FOLDS = 10
USE_GPU = False

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
SUBMISSION_DIR = os.path.join(os.path.dirname(__file__), "submissions")
os.makedirs(SUBMISSION_DIR, exist_ok=True)

TARGETS = ["Al", "B", "Ca", "Cu", "Fe", "K", "Mg", "Mn", "N", "Na", "P", "S", "Zn"]
TRAIN_EXTRA = ["C_organic", "C_total", "electrical_conductivity", "ph"]
SPARSE_TARGETS = {"B", "Na", "P", "S", "Zn"}


# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
def load_data():
    print("=" * 80)
    print("STEP 1: LOADING DATA")
    print("=" * 80)

    train = pd.read_csv(os.path.join(DATA_DIR, "Train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "TestSet.csv"))
    dates = pd.read_csv(os.path.join(DATA_DIR, "Sample_Collection_Dates.csv"))
    tk = pd.read_csv(os.path.join(DATA_DIR, "TargetPred_To_Keep.csv"))
    ss = pd.read_csv(os.path.join(DATA_DIR, "SampleSubmission.csv"))

    print(f"  Train: {train.shape}, Test: {test.shape}")
    return train, test, dates, tk, ss


# ============================================================================
# STEP 2: FEATURE ENGINEERING
# ============================================================================
def build_features(train, test, dates):
    print("\n" + "=" * 80)
    print("STEP 2: FEATURE ENGINEERING")
    print("=" * 80)

    # Merge dates into test
    test_dates = dates[dates["set"] == "Test"][["ID", "start_date", "end_date"]]
    test = test.merge(test_dates, on="ID", how="left")

    # Parse depth -> binary feature
    def parse_depth_mid(d):
        d = str(d).replace(" ", "").replace("cm", "")
        parts = d.split("-")
        if len(parts) == 2:
            try:
                return (float(parts[0]) + float(parts[1])) / 2
            except ValueError:
                pass
        return 15.0

    # Date group feature for test
    test["date_group"] = 0
    if "start_date" in test.columns:
        test["date_group"] = test["start_date"].apply(
            lambda x: 1 if "2022" in str(x) else 0
        )

    train["date_group"] = 0

    # Parse depth
    for df in [train, test]:
        df["depth_mid"] = df["Depth_cm"].apply(parse_depth_mid)
        df["is_topsoil"] = (df["depth_mid"] <= 20).astype(int)

    # Unify coordinate column order
    lat_col, lon_col = "Latitude", "Longitude"

    # =====================================================================
    # Build combined for feature engineering
    # =====================================================================
    keep_train = ["ID", lat_col, lon_col, "depth_mid", "is_topsoil", "date_group"] + \
                 TARGETS + [c for c in TRAIN_EXTRA if c in train.columns]
    keep_test = ["ID", lat_col, lon_col, "depth_mid", "is_topsoil", "date_group"]

    train_sub = train[[c for c in keep_train if c in train.columns]].copy()
    test_sub = test[[c for c in keep_test if c in test.columns]].copy()

    train_sub["_is_train"] = 1
    test_sub["_is_train"] = 0

    combined = pd.concat([train_sub, test_sub], axis=0, ignore_index=True)
    n_train = len(train_sub)
    n_test = len(test_sub)

    lat = combined[lat_col].values
    lon = combined[lon_col].values
    depth = combined["depth_mid"].values

    print(f"  Combined: {combined.shape} (train={n_train}, test={n_test})")

    # --- GEOGRAPHIC FEATURES ---
    print("  [1/6] Geographic features...")
    combined["abs_lat"] = np.abs(lat)
    combined["abs_lon"] = np.abs(lon)
    combined["lat_lon_product"] = lat * lon
    combined["lat_lon_ratio"] = lat / (lon + 1e-8)
    combined["geo_radius"] = np.sqrt(lat**2 + lon**2)
    combined["geo_angle"] = np.arctan2(lat, lon)
    combined["lat_sq"] = lat**2
    combined["lon_sq"] = lon**2
    combined["lat_cubed"] = lat**3
    combined["lon_cubed"] = lon**3
    combined["lat_x_lon_sq"] = lat * lon**2
    combined["lon_x_lat_sq"] = lon * lat**2

    # Grid bins
    for res in [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]:
        combined[f"lat_bin_{res}"] = np.floor(lat / res) * res
        combined[f"lon_bin_{res}"] = np.floor(lon / res) * res

    # Hemisphere/zone
    combined["is_southern"] = (lat < 0).astype(int)
    combined["is_equatorial"] = (np.abs(lat) < 5).astype(int)

    # Landmark distances (Haversine)
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

    landmarks = {
        "nairobi": (-1.29, 36.82), "dar": (-6.79, 39.28),
        "victoria": (-1.0, 33.0), "kilimanjaro": (-3.07, 37.35),
        "addis": (9.02, 38.75), "cape": (-33.93, 18.42),
        "kinshasa": (-4.32, 15.31), "kampala": (0.35, 32.58),
    }
    for name, (rlat, rlon) in landmarks.items():
        combined[f"hdist_{name}"] = haversine(lat, lon, rlat, rlon)

    # --- DEPTH FEATURES ---
    print("  [2/6] Depth features...")
    combined["log_depth"] = np.log1p(depth)
    combined["depth_x_lat"] = depth * lat
    combined["depth_x_lon"] = depth * lon
    combined["depth_x_abs_lat"] = depth * np.abs(lat)

    # --- CLIMATE PROXIES ---
    print("  [3/6] Climate proxy features...")
    abs_lat = np.abs(lat)
    combined["temp_proxy"] = 30.0 - 0.5 * abs_lat
    combined["precip_proxy"] = 2000 * np.exp(-0.05 * (abs_lat - 5)**2)
    combined["aridity_proxy"] = combined["temp_proxy"] / (combined["precip_proxy"] + 1)
    combined["moisture_proxy"] = combined["precip_proxy"] - 16 * (10 * np.clip(combined["temp_proxy"], 0, None) / 50)**1.5

    # Africa climate zones
    combined["zone_tropical_wet"] = ((abs_lat < 10) & (lon > 25)).astype(int)
    combined["zone_sahel"] = ((lat > 10) & (lat < 20)).astype(int)
    combined["zone_east_africa"] = ((lon > 30) & (lon < 42) & (lat > -12) & (lat < 5)).astype(int)
    combined["zone_southern"] = (lat < -15).astype(int)
    combined["zone_kenya_central"] = ((lat > -2) & (lat < 1) & (lon > 36) & (lon < 38)).astype(int)
    combined["zone_kenya_rift"] = ((lat > -2) & (lat < 1) & (lon > 35) & (lon < 36.5)).astype(int)

    # --- SPATIAL CLUSTERING ---
    print("  [4/6] Spatial clustering...")
    coords = np.column_stack([lat, lon])

    for n_cl in [3, 5, 10, 20, 50, 100, 200, 500]:
        km = MiniBatchKMeans(n_clusters=n_cl, random_state=SEED, n_init=3, batch_size=5000)
        labels = km.fit_predict(coords)
        combined[f"cluster_{n_cl}"] = labels
        centers = km.cluster_centers_[labels]
        combined[f"dist_cl_{n_cl}"] = np.sqrt(((coords - centers)**2).sum(axis=1))

    # 3D clustering (lat, lon, depth_normalized)
    coords_3d = np.column_stack([lat, lon, depth / 1000])
    for n_cl in [10, 30, 50, 100]:
        km = MiniBatchKMeans(n_clusters=n_cl, random_state=SEED, n_init=3, batch_size=5000)
        combined[f"cluster3d_{n_cl}"] = km.fit_predict(coords_3d)

    # --- LOCATION DENSITY ---
    print("  [5/6] Location density features...")
    for res in [0.05, 0.1, 0.5, 1.0]:
        cell = (combined[f"lat_bin_{res}"].astype(str) + "_" +
                combined[f"lon_bin_{res}"].astype(str))
        freq = cell.value_counts()
        combined[f"density_{res}"] = cell.map(freq).values

    for n_cl in [10, 50, 100, 200]:
        freq = combined[f"cluster_{n_cl}"].value_counts()
        combined[f"cl_freq_{n_cl}"] = combined[f"cluster_{n_cl}"].map(freq).values

    # --- NEAREST NEIGHBOR FEATURES (THE GAME WINNER) ---
    print("  [6/6] Nearest neighbor features...")

    train_mask = combined["_is_train"].values == 1
    train_coords_rad = np.radians(coords[train_mask])
    all_coords_rad = np.radians(coords)

    # BallTree for Haversine distances
    tree = BallTree(train_coords_rad, metric="haversine")

    # Also build per-depth trees for depth-aware neighbors
    depth_vals = combined["is_topsoil"].values
    for depth_val in [0, 1]:
        depth_mask = train_mask & (depth_vals == depth_val)
        if depth_mask.sum() > 10:
            tree_d = BallTree(np.radians(coords[depth_mask]), metric="haversine")
            # Query for samples with same depth
            query_mask = depth_vals == depth_val
            K_depth = min(20, depth_mask.sum() - 1)
            if K_depth > 0:
                dd, di = tree_d.query(np.radians(coords[query_mask]), k=K_depth)
                dd_km = dd * 6371

                train_idx_depth = np.where(depth_mask)[0]
                for target in TARGETS:
                    if target not in combined.columns:
                        continue
                    tv = combined[target].values
                    nn_tv = tv[train_idx_depth[di]]
                    w = 1.0 / (dd_km + 0.001)
                    w_norm = w / w.sum(axis=1, keepdims=True)
                    col = f"nnd{depth_val}_{target}_idw"
                    combined[col] = np.nan
                    combined.loc[query_mask, col] = np.nansum(nn_tv * w_norm, axis=1)

    # Main neighbor features at multiple K values
    for K in [1, 3, 5, 10, 20, 50, 100]:
        print(f"    K={K}...")
        K_actual = min(K, train_mask.sum() - 1)
        dists, indices = tree.query(all_coords_rad, k=K_actual)
        dists_km = dists * 6371

        # Distance stats
        combined[f"nn{K}_mean_dist"] = np.mean(dists_km, axis=1)
        combined[f"nn{K}_min_dist"] = dists_km[:, 0] if K_actual >= 1 else np.nan
        combined[f"nn{K}_max_dist"] = dists_km[:, -1] if K_actual >= 1 else np.nan
        combined[f"nn{K}_std_dist"] = np.std(dists_km, axis=1)

        # IDW weights
        weights = 1.0 / (dists_km + 0.001)
        weights_norm = weights / weights.sum(axis=1, keepdims=True)

        # Exponential decay weights (more aggressive for nearby points)
        exp_weights = np.exp(-dists_km / 1.0)  # 1km scale
        exp_w_norm = exp_weights / (exp_weights.sum(axis=1, keepdims=True) + 1e-10)

        train_indices = np.where(train_mask)[0]

        # Target neighbor features
        for target in TARGETS:
            if target not in combined.columns:
                continue
            tv = combined[target].values
            nn_vals = tv[train_indices[indices]]

            # IDW interpolation
            combined[f"nn{K}_{target}_idw"] = np.nansum(nn_vals * weights_norm, axis=1)
            # Exponential decay interpolation
            combined[f"nn{K}_{target}_exp"] = np.nansum(nn_vals * exp_w_norm, axis=1)
            # Simple stats
            combined[f"nn{K}_{target}_mean"] = np.nanmean(nn_vals, axis=1)
            combined[f"nn{K}_{target}_median"] = np.nanmedian(nn_vals, axis=1)
            combined[f"nn{K}_{target}_std"] = np.nanstd(nn_vals, axis=1)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                combined[f"nn{K}_{target}_min"] = np.nanmin(nn_vals, axis=1)
                combined[f"nn{K}_{target}_max"] = np.nanmax(nn_vals, axis=1)

        # Auxiliary feature neighbor aggregates
        for extra in TRAIN_EXTRA:
            if extra not in combined.columns:
                continue
            ev = combined[extra].values
            nn_ev = ev[train_indices[indices]]
            combined[f"nn{K}_{extra}_idw"] = np.nansum(nn_ev * weights_norm, axis=1)
            combined[f"nn{K}_{extra}_mean"] = np.nanmean(nn_ev, axis=1)
            combined[f"nn{K}_{extra}_std"] = np.nanstd(nn_ev, axis=1)

        # Cross-nutrient ratio features (use correlations!)
        # Ca/Mg ratio from neighbors, etc.
        if K <= 20:
            for t1, t2 in [("Ca", "Mg"), ("Ca", "K"), ("Al", "Fe"),
                            ("Cu", "Mn"), ("Al", "N"), ("Fe", "Mn")]:
                if t1 in combined.columns and t2 in combined.columns:
                    v1 = combined[t1].values[train_indices[indices]]
                    v2 = combined[t2].values[train_indices[indices]]
                    ratio = np.nanmean(v1, axis=1) / (np.nanmean(v2, axis=1) + 1e-8)
                    combined[f"nn{K}_{t1}_{t2}_ratio"] = ratio

        # Nearest neighbor depth feature
        nn_depths = combined["depth_mid"].values[train_indices[indices]]
        combined[f"nn{K}_depth_match"] = (nn_depths == depth.reshape(-1, 1)).mean(axis=1)

    # =====================================================================
    # SPLIT BACK
    # =====================================================================
    print("\n  Building final feature matrices...")
    train_fe = combined[combined["_is_train"] == 1].reset_index(drop=True)
    test_fe = combined[combined["_is_train"] == 0].reset_index(drop=True)

    exclude = set(["ID", "_is_train", "Depth_cm"] + TARGETS + TRAIN_EXTRA)
    feature_cols = [c for c in train_fe.columns
                    if c not in exclude
                    and train_fe[c].dtype in ["float64", "float32", "int64", "int32"]]

    print(f"  Total features: {len(feature_cols)}")
    print(f"  Train: {train_fe.shape}, Test: {test_fe.shape}")

    return train_fe, test_fe, feature_cols


# ============================================================================
# STEP 3: TRAIN MODELS
# ============================================================================
def train_and_predict(train_fe, test_fe, feature_cols):
    print("\n" + "=" * 80)
    print("STEP 3: TRAINING MODELS")
    print("=" * 80)

    X_train_raw = train_fe[feature_cols].values.astype(np.float32)
    X_test_raw = test_fe[feature_cols].values.astype(np.float32)
    X_train = np.nan_to_num(X_train_raw, nan=0, posinf=0, neginf=0)
    X_test = np.nan_to_num(X_test_raw, nan=0, posinf=0, neginf=0)

    print(f"  Features: {X_train.shape[1]}, Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    # Model configs - AGGRESSIVE settings with unlimited compute
    models = {
        "lgb1": ("lgb", {
            "objective": "regression", "metric": "rmse", "boosting_type": "gbdt",
            "n_estimators": 15000, "learning_rate": 0.005,
            "num_leaves": 511, "max_depth": -1, "min_child_samples": 3,
            "subsample": 0.7, "colsample_bytree": 0.5,
            "reg_alpha": 0.05, "reg_lambda": 0.5,
            "random_state": SEED, "verbose": -1, "n_jobs": -1,
        }),
        "lgb2": ("lgb", {
            "objective": "regression", "metric": "rmse", "boosting_type": "gbdt",
            "n_estimators": 12000, "learning_rate": 0.008,
            "num_leaves": 255, "max_depth": 12, "min_child_samples": 5,
            "subsample": 0.75, "colsample_bytree": 0.6,
            "reg_alpha": 0.1, "reg_lambda": 1.0,
            "random_state": SEED + 1, "verbose": -1, "n_jobs": -1,
        }),
        "lgb3": ("lgb", {
            "objective": "regression", "metric": "rmse", "boosting_type": "gbdt",
            "n_estimators": 10000, "learning_rate": 0.01,
            "num_leaves": 127, "max_depth": 8, "min_child_samples": 10,
            "subsample": 0.8, "colsample_bytree": 0.7,
            "reg_alpha": 0.5, "reg_lambda": 2.0,
            "random_state": SEED + 2, "verbose": -1, "n_jobs": -1,
        }),
        "lgb_dart": ("lgb", {
            "objective": "regression", "metric": "rmse", "boosting_type": "dart",
            "n_estimators": 4000, "learning_rate": 0.015,
            "num_leaves": 255, "max_depth": -1, "min_child_samples": 5,
            "subsample": 0.7, "colsample_bytree": 0.6,
            "reg_alpha": 0.1, "reg_lambda": 1.0, "drop_rate": 0.1,
            "random_state": SEED + 3, "verbose": -1, "n_jobs": -1,
        }),
        "xgb1": ("xgb", {
            "objective": "reg:squarederror", "eval_metric": "rmse",
            "n_estimators": 15000, "learning_rate": 0.005,
            "max_depth": 10, "min_child_weight": 3,
            "subsample": 0.7, "colsample_bytree": 0.5, "colsample_bylevel": 0.7,
            "reg_alpha": 0.05, "reg_lambda": 0.5,
            "random_state": SEED, "n_jobs": -1, "verbosity": 0,
            "early_stopping_rounds": 300,
        }),
        "xgb2": ("xgb", {
            "objective": "reg:squarederror", "eval_metric": "rmse",
            "n_estimators": 10000, "learning_rate": 0.01,
            "max_depth": 8, "min_child_weight": 5,
            "subsample": 0.8, "colsample_bytree": 0.6,
            "reg_alpha": 0.5, "reg_lambda": 2.0, "gamma": 0.1,
            "random_state": SEED + 10, "n_jobs": -1, "verbosity": 0,
            "early_stopping_rounds": 300,
        }),
        "cb1": ("cb", {
            "iterations": 15000, "learning_rate": 0.01, "depth": 10,
            "l2_leaf_reg": 3, "random_seed": SEED, "verbose": 0,
            "loss_function": "RMSE", "eval_metric": "RMSE",
            "min_data_in_leaf": 3, "subsample": 0.7, "colsample_bylevel": 0.5,
            "bootstrap_type": "Bernoulli",
        }),
        "cb2": ("cb", {
            "iterations": 10000, "learning_rate": 0.015, "depth": 12,
            "l2_leaf_reg": 5, "random_seed": SEED + 5, "verbose": 0,
            "loss_function": "RMSE", "eval_metric": "RMSE",
            "min_data_in_leaf": 5, "subsample": 0.8, "colsample_bylevel": 0.6,
            "bootstrap_type": "Bernoulli",
            "random_strength": 1.5, "bagging_temperature": 0.5,
        }),
        "et": ("sk", (ExtraTreesRegressor, {
            "n_estimators": 3000, "max_depth": None,
            "min_samples_split": 3, "min_samples_leaf": 1,
            "max_features": 0.5, "random_state": SEED, "n_jobs": -1,
        })),
        "rf": ("sk", (RandomForestRegressor, {
            "n_estimators": 3000, "max_depth": None,
            "min_samples_split": 5, "min_samples_leaf": 2,
            "max_features": 0.3, "random_state": SEED + 3, "n_jobs": -1,
        })),
    }

    if USE_GPU:
        for key in ["xgb1", "xgb2"]:
            models[key][1]["tree_method"] = "hist"
            models[key][1]["device"] = "cuda"
        for key in ["cb1", "cb2"]:
            models[key][1]["task_type"] = "GPU"

    all_predictions = {}
    all_cv_scores = {}
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    for t_idx, target in enumerate(TARGETS):
        print(f"\n{'='*60}")
        print(f"  TARGET {t_idx+1}/{len(TARGETS)}: {target}")
        print(f"{'='*60}")

        if target not in train_fe.columns:
            print(f"  {target} not in train, using zeros.")
            all_predictions[target] = np.zeros(len(X_test))
            continue

        y = train_fe[target].values
        valid = ~np.isnan(y)
        n_valid = valid.sum()
        print(f"  Valid samples: {n_valid}/{len(y)} ({n_valid/len(y)*100:.1f}%)")

        if n_valid < 50:
            # Fallback to IDW
            idw_col = f"nn5_{target}_idw"
            if idw_col in feature_cols:
                fidx = feature_cols.index(idw_col)
                all_predictions[target] = np.clip(X_test[:, fidx], 0, None)
            else:
                all_predictions[target] = np.zeros(len(X_test))
            print(f"  Too few samples, using IDW fallback.")
            continue

        # Train each model
        oof_dict = {}
        test_dict = {}
        score_dict = {}

        for mname, (mtype, mparams) in models.items():
            oof = np.zeros(len(X_train))
            tpred = np.zeros(len(X_test))
            fscores = []

            for fi, (tr_idx, va_idx) in enumerate(kf.split(X_train)):
                Xtr, Xva = X_train[tr_idx], X_train[va_idx]
                ytr, yva = y[tr_idx], y[va_idx]

                tr_ok = ~np.isnan(ytr)
                va_ok = ~np.isnan(yva)
                if tr_ok.sum() < 10 or va_ok.sum() < 3:
                    continue

                Xtr_c, ytr_c = Xtr[tr_ok], ytr[tr_ok]
                Xva_c, yva_c = Xva[va_ok], yva[va_ok]

                try:
                    if mtype == "lgb":
                        m = lgb.LGBMRegressor(**mparams)
                        m.fit(Xtr_c, ytr_c, eval_set=[(Xva_c, yva_c)],
                              callbacks=[lgb.early_stopping(300, verbose=False),
                                         lgb.log_evaluation(0)])
                        vp = m.predict(Xva_c)
                        tp = m.predict(X_test)
                    elif mtype == "xgb":
                        m = xgb.XGBRegressor(**mparams)
                        m.fit(Xtr_c, ytr_c, eval_set=[(Xva_c, yva_c)], verbose=False)
                        vp = m.predict(Xva_c)
                        tp = m.predict(X_test)
                    elif mtype == "cb":
                        m = cb.CatBoostRegressor(**mparams)
                        m.fit(Xtr_c, ytr_c, eval_set=(Xva_c, yva_c),
                              early_stopping_rounds=300, verbose=0)
                        vp = m.predict(Xva_c)
                        tp = m.predict(X_test)
                    elif mtype == "sk":
                        cls, prms = mparams
                        m = cls(**prms)
                        m.fit(Xtr_c, ytr_c)
                        vp = m.predict(Xva_c)
                        tp = m.predict(X_test)
                    else:
                        continue

                    vp = np.clip(vp, 0, None)
                    tp = np.clip(tp, 0, None)

                    oof[va_idx[va_ok]] = vp
                    fscores.append(np.sqrt(mean_squared_error(yva_c, vp)))
                    tpred += tp / N_FOLDS

                except Exception as e:
                    print(f"    {mname} fold {fi} error: {e}")

            if fscores:
                avg = np.mean(fscores)
                print(f"    {mname:12s}: CV RMSE = {avg:.4f} (std={np.std(fscores):.4f})")
                oof_dict[mname] = oof
                test_dict[mname] = tpred
                score_dict[mname] = avg

        # ==================================================================
        # MULTI-SEED ENSEMBLING for top models
        # ==================================================================
        if score_dict:
            best_model = min(score_dict, key=score_dict.get)
            print(f"  Best single model: {best_model} ({score_dict[best_model]:.4f})")

            # Add multi-seed variants of the best LGB model
            for seed_offset in [100, 200, 300]:
                seed_name = f"lgb_s{seed_offset}"
                lgb_params = {
                    "objective": "regression", "metric": "rmse", "boosting_type": "gbdt",
                    "n_estimators": 12000, "learning_rate": 0.008,
                    "num_leaves": 255, "max_depth": -1, "min_child_samples": 5,
                    "subsample": 0.7, "colsample_bytree": 0.5 + seed_offset * 0.0005,
                    "reg_alpha": 0.1, "reg_lambda": 1.0,
                    "random_state": SEED + seed_offset, "verbose": -1, "n_jobs": -1,
                }
                oof_s = np.zeros(len(X_train))
                tp_s = np.zeros(len(X_test))
                fs_s = []

                kf_s = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED + seed_offset)
                for fi, (tr_idx, va_idx) in enumerate(kf_s.split(X_train)):
                    Xtr, Xva = X_train[tr_idx], X_train[va_idx]
                    ytr, yva = y[tr_idx], y[va_idx]
                    tr_ok, va_ok = ~np.isnan(ytr), ~np.isnan(yva)
                    if tr_ok.sum() < 10 or va_ok.sum() < 3:
                        continue
                    try:
                        m = lgb.LGBMRegressor(**lgb_params)
                        m.fit(Xtr[tr_ok], ytr[tr_ok],
                              eval_set=[(Xva[va_ok], yva[va_ok])],
                              callbacks=[lgb.early_stopping(300, verbose=False),
                                         lgb.log_evaluation(0)])
                        vp = np.clip(m.predict(Xva[va_ok]), 0, None)
                        tp = np.clip(m.predict(X_test), 0, None)
                        oof_s[va_idx[va_ok]] = vp
                        fs_s.append(np.sqrt(mean_squared_error(yva[va_ok], vp)))
                        tp_s += tp / N_FOLDS
                    except Exception:
                        pass

                if fs_s:
                    avg_s = np.mean(fs_s)
                    print(f"    {seed_name:12s}: CV RMSE = {avg_s:.4f}")
                    oof_dict[seed_name] = oof_s
                    test_dict[seed_name] = tp_s
                    score_dict[seed_name] = avg_s

        # ==================================================================
        # ENSEMBLE: Optimal weight blending + Ridge stacking
        # ==================================================================
        if len(oof_dict) > 1:
            print(f"\n  Ensembling {len(oof_dict)} models for {target}...")
            mnames = sorted(oof_dict.keys())

            oof_mat = np.column_stack([oof_dict[m] for m in mnames])
            test_mat = np.column_stack([test_dict[m] for m in mnames])

            # Optimal weights
            def rmse_fn(w):
                w = np.abs(w); w /= w.sum()
                return np.sqrt(mean_squared_error(y[valid], (oof_mat[valid] * w).sum(axis=1)))

            best_rmse = float("inf")
            best_w = np.ones(len(mnames)) / len(mnames)
            best_pred = None

            for method in ["Nelder-Mead", "Powell"]:
                try:
                    res = minimize(rmse_fn, best_w, method=method, options={"maxiter": 50000})
                    if res.fun < best_rmse:
                        best_rmse = res.fun
                        best_w = np.abs(res.x); best_w /= best_w.sum()
                        best_pred = np.clip((test_mat * best_w).sum(axis=1), 0, None)
                except Exception:
                    pass

            # Ridge meta-learner
            try:
                scaler = RobustScaler()
                oof_sc = scaler.fit_transform(oof_mat[valid])
                test_sc = scaler.transform(test_mat)

                meta_oof = np.zeros(valid.sum())
                meta_test = np.zeros(len(X_test))
                meta_kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

                for tr_i, va_i in meta_kf.split(oof_sc):
                    ridge = Ridge(alpha=1.0)
                    ridge.fit(oof_sc[tr_i], y[valid][tr_i])
                    meta_oof[va_i] = ridge.predict(oof_sc[va_i])
                    meta_test += ridge.predict(test_sc) / 5

                meta_rmse = np.sqrt(mean_squared_error(y[valid], meta_oof))
                if meta_rmse < best_rmse:
                    best_rmse = meta_rmse
                    best_pred = np.clip(meta_test, 0, None)
                    print(f"  Using Ridge meta-learner: {meta_rmse:.4f}")
                else:
                    print(f"  Using weight blend: {best_rmse:.4f}")
                    print(f"  Weights: {dict(zip(mnames, [f'{w:.3f}' for w in best_w]))}")
            except Exception:
                if best_pred is None:
                    best_pred = np.clip(test_mat.mean(axis=1), 0, None)

            all_predictions[target] = best_pred
            all_cv_scores[target] = {"ensemble": best_rmse, **score_dict}

        elif len(test_dict) == 1:
            mn = list(test_dict.keys())[0]
            all_predictions[target] = test_dict[mn]
            all_cv_scores[target] = score_dict
        else:
            all_predictions[target] = np.zeros(len(X_test))

    return all_predictions, all_cv_scores


# ============================================================================
# STEP 4: SUBMISSION
# ============================================================================
def create_submission(test_fe, predictions, tk, ss):
    print("\n" + "=" * 80)
    print("STEP 4: CREATING SUBMISSION")
    print("=" * 80)

    sub = ss[["ID"]].copy()
    test_ids = test_fe["ID"].values

    for target in TARGETS:
        col = f"Target_{target}"
        if target in predictions:
            pred_map = dict(zip(test_ids, predictions[target]))
            sub[col] = sub["ID"].map(pred_map).fillna(0).clip(lower=0)
        else:
            sub[col] = 0.0

    # Apply TargetPred_To_Keep mask
    print("  Applying mask...")
    for target in TARGETS:
        col = f"Target_{target}"
        if target in tk.columns and col in sub.columns:
            mask = sub["ID"].map(tk.set_index("ID")[target]).fillna(0)
            before = (sub[col] > 0).sum()
            sub[col] = sub[col] * mask
            after = (sub[col] > 0).sum()
            print(f"    {target}: {before} -> {after} non-zero")

    # Verify columns match
    assert list(sub.columns) == list(ss.columns), "Column mismatch!"
    assert len(sub) == len(ss), "Row count mismatch!"

    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(SUBMISSION_DIR, f"submission_{ts}.csv")
    sub.to_csv(path, index=False)

    latest = os.path.join(SUBMISSION_DIR, "submission_latest.csv")
    sub.to_csv(latest, index=False)

    print(f"\n  Saved: {path}")
    print(f"  Shape: {sub.shape}")
    return sub


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    start = datetime.now()
    print(f"Started: {start}")
    print()

    train, test, dates, tk, ss = load_data()
    train_fe, test_fe, feature_cols = build_features(train, test, dates)
    predictions, cv_scores = train_and_predict(train_fe, test_fe, feature_cols)
    submission = create_submission(test_fe, predictions, tk, ss)

    # Summary
    print("\n" + "=" * 80)
    print("CV SCORE SUMMARY")
    print("=" * 80)
    scores = []
    for t in TARGETS:
        if t in cv_scores:
            ens = cv_scores[t].get("ensemble", min(cv_scores[t].values()))
            scores.append(ens)
            print(f"  {t:5s}: {ens:.4f}")
    if scores:
        print(f"\n  MEAN RMSE: {np.mean(scores):.4f}")

    print(f"\n  Completed: {datetime.now()} (elapsed: {datetime.now() - start})")
