#!/usr/bin/env python3
"""
=============================================================================
RHEA SOIL NUTRIENT PREDICTION - H200 GPU PIPELINE
=============================================================================
pip install lightgbm xgboost catboost scikit-learn pandas numpy scipy tqdm
python -u run_gpu.py 2>&1 | tee run_output.log
"""
import os, warnings, numpy as np, pandas as pd, sys, gc, subprocess
from datetime import datetime
from functools import partial
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import BallTree
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
from scipy.optimize import minimize
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from tqdm import tqdm as _tqdm_orig

warnings.filterwarnings("ignore")

# Force tqdm to show even when piped (| tee)
tqdm = partial(_tqdm_orig, dynamic_ncols=True, file=sys.stderr, mininterval=0.5)
tqdm_write = partial(_tqdm_orig.write, file=sys.stderr)

# Also make print flush immediately
_print = print
def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _print(*args, **kwargs)

SEED = 42
N_FOLDS = 10
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "submissions")
os.makedirs(SDIR, exist_ok=True)
TARGETS = ["Al","B","Ca","Cu","Fe","K","Mg","Mn","N","Na","P","S","Zn"]
EXTRA = ["C_organic","C_total","ph"]

# ============================================================================
# GPU DETECTION
# ============================================================================
def detect_gpu():
    """Detect available GPUs and return config."""
    print("\n" + "="*70)
    print("  GPU DETECTION")
    print("="*70)

    # Check nvidia-smi
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            # Extract GPU name
            for line in result.stdout.split("\n"):
                if "NVIDIA" in line and ("H200" in line or "H100" in line or "A100" in line
                                          or "V100" in line or "RTX" in line or "Tesla" in line):
                    gpu_name = line.strip()
                    print(f"  GPU found: {gpu_name}")
                    break
            else:
                print(f"  GPU found via nvidia-smi")
            # Show memory
            for line in result.stdout.split("\n"):
                if "MiB" in line and "|" in line:
                    print(f"  {line.strip()}")
                    break
        else:
            print("  nvidia-smi failed - no NVIDIA GPU?")
    except Exception as e:
        print(f"  nvidia-smi not found: {e}")

    # Test XGBoost GPU
    xgb_gpu = False
    try:
        m = xgb.XGBRegressor(tree_method="hist", device="cuda", n_estimators=5, verbosity=0)
        m.fit(np.random.rand(100, 5), np.random.rand(100))
        m.predict(np.random.rand(10, 5))
        xgb_gpu = True
        print("  XGBoost GPU:  AVAILABLE")
    except Exception as e:
        print(f"  XGBoost GPU:  NOT AVAILABLE ({e})")

    # Test CatBoost GPU
    cb_gpu = False
    try:
        m = cb.CatBoostRegressor(iterations=5, task_type="GPU", devices="0", verbose=0)
        m.fit(np.random.rand(100, 5), np.random.rand(100))
        m.predict(np.random.rand(10, 5))
        cb_gpu = True
        print("  CatBoost GPU: AVAILABLE")
    except Exception as e:
        print(f"  CatBoost GPU: NOT AVAILABLE ({e})")

    print(f"  LightGBM:     CPU (GPU requires special build)")
    print(f"  ExtraTrees:   CPU")
    print()

    return xgb_gpu, cb_gpu


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================
def build_features():
    print(f"\n{'='*70}")
    print(f"  STEP 1: FEATURE ENGINEERING")
    print(f"{'='*70}\n")

    # --- Load ---
    load_files = ["Train.csv", "TestSet.csv", "Sample_Collection_Dates.csv",
                   "TargetPred_To_Keep.csv", "SampleSubmission.csv"]
    loaded = {}
    for f in tqdm(load_files, desc="Loading data files"):
        loaded[f] = pd.read_csv(os.path.join(DATA, f))

    train = loaded["Train.csv"]
    test = loaded["TestSet.csv"]
    dates = loaded["Sample_Collection_Dates.csv"]
    tk = loaded["TargetPred_To_Keep.csv"]
    ss = loaded["SampleSubmission.csv"]
    print(f"  Train={train.shape}, Test={test.shape}")

    # --- Merge dates ---
    td = dates[dates["set"] == "Test"][["ID", "start_date"]]
    test = test.merge(td, on="ID", how="left")
    test["dg"] = test["start_date"].apply(lambda x: 1 if "2022" in str(x) else 0)
    train["dg"] = 0

    # --- Depth ---
    def dmid(d):
        p = str(d).replace(" ", "").replace("cm", "").split("-")
        try: return (float(p[0]) + float(p[1])) / 2
        except: return 15.0

    for d in [train, test]:
        d["dm"] = d["Depth_cm"].apply(dmid)
        d["ts"] = (d["dm"] <= 20).astype(int)

    # --- Combine ---
    kc = ["ID", "Latitude", "Longitude", "dm", "ts", "dg"]
    tc = kc + TARGETS + [c for c in EXTRA if c in train.columns]
    trs = train[[c for c in tc if c in train.columns]].copy()
    tes = test[[c for c in kc if c in test.columns]].copy()
    trs["_t"] = 1; tes["_t"] = 0
    df = pd.concat([trs, tes], ignore_index=True)

    lat, lon, dep = df["Latitude"].values, df["Longitude"].values, df["dm"].values

    # === GEO FEATURES ===
    geo_steps = [
        ("abs_lat", lambda: np.abs(lat)),
        ("lat*lon", lambda: lat * lon),
        ("lat/lon", lambda: lat / (lon + 1e-8)),
        ("geo_radius", lambda: np.sqrt(lat**2 + lon**2)),
        ("geo_angle", lambda: np.arctan2(lat, lon)),
        ("lat^2", lambda: lat**2),
        ("lon^2", lambda: lon**2),
        ("lat^3", lambda: lat**3),
        ("lon^3", lambda: lon**3),
        ("lat*lon^2", lambda: lat * lon**2),
        ("lon*lat^2", lambda: lon * lat**2),
    ]
    col_names = ["alat","llp","llr","gr","ga","lsq","lnsq","lcb","lncb","l_ln2","ln_l2"]
    for (desc, fn), cn in tqdm(list(zip(geo_steps, col_names)), desc="Geo features"):
        df[cn] = fn()

    # Grid bins
    resolutions = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
    for r in tqdm(resolutions, desc="Grid bins"):
        df[f"lb{r}"] = np.floor(lat / r) * r
        df[f"nb{r}"] = np.floor(lon / r) * r

    df["is_s"] = (lat < 0).astype(int)
    df["is_eq"] = (np.abs(lat) < 5).astype(int)

    # Haversine distances to landmarks
    def hav(a1, o1, a2, o2):
        a1, o1, a2, o2 = map(np.radians, [a1, o1, a2, o2])
        a = np.sin((a2-a1)/2)**2 + np.cos(a1)*np.cos(a2)*np.sin((o2-o1)/2)**2
        return 6371 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

    landmarks = {"nai": (-1.29, 36.82), "dar": (-6.79, 39.28),
                  "vic": (-1, 33), "kil": (-3.07, 37.35),
                  "kam": (0.35, 32.58), "add": (9.02, 38.75),
                  "cap": (-33.93, 18.42), "kin": (-4.32, 15.31)}
    for nm, (rl, rn) in tqdm(landmarks.items(), desc="Landmark distances"):
        df[f"h_{nm}"] = hav(lat, lon, rl, rn)

    # === DEPTH FEATURES ===
    depth_feats = {"ld": np.log1p(dep), "sd": np.sqrt(dep), "dsq": dep**2,
                    "dxl": dep*lat, "dxn": dep*lon, "dxal": dep*np.abs(lat),
                    "dxgr": dep*df["gr"].values, "ldxl": np.log1p(dep)*lat,
                    "ldxn": np.log1p(dep)*lon}
    for cn, vals in tqdm(depth_feats.items(), desc="Depth features"):
        df[cn] = vals

    # === CLIMATE PROXIES ===
    al = np.abs(lat)
    climate_feats = {
        "tp": 30 - 0.5 * al,
        "tr": 5.0 + 0.3 * al,
        "pp": 2000 * np.exp(-0.05 * (al - 5)**2),
    }
    for cn, vals in tqdm(climate_feats.items(), desc="Climate proxies"):
        df[cn] = vals
    df["ap"] = df["tp"] / (df["pp"] + 1)
    df["pet"] = 16 * (10 * np.clip(df["tp"], 0, None) / 50)**1.5
    df["mp"] = df["pp"] - df["pet"]

    zones = {"z_tw": (al < 10) & (lon > 25),
             "z_sh": (lat > 10) & (lat < 20),
             "z_ea": (lon > 30) & (lon < 42) & (lat > -12) & (lat < 5),
             "z_sa": lat < -15,
             "z_kc": (lat > -2) & (lat < 1) & (lon > 36) & (lon < 38),
             "z_kr": (lat > -2) & (lat < 1) & (lon > 35) & (lon < 36.5),
             "z_kco": (lat > -5) & (lat < -2) & (lon > 38) & (lon < 41)}
    for cn, mask in tqdm(zones.items(), desc="Climate zones"):
        df[cn] = mask.astype(int)

    # === CLUSTERS ===
    coords = np.column_stack([lat, lon])
    cluster_sizes = [3, 5, 10, 20, 50, 100, 200, 500]
    for nc in tqdm(cluster_sizes, desc="2D Clusters"):
        km = MiniBatchKMeans(n_clusters=nc, random_state=SEED, n_init=3, batch_size=5000)
        lb = km.fit_predict(coords)
        df[f"c{nc}"] = lb
        df[f"dc{nc}"] = np.sqrt(((coords - km.cluster_centers_[lb])**2).sum(1))

    c3d = np.column_stack([lat, lon, dep / 1000])
    for nc in tqdm([10, 30, 50, 100], desc="3D Clusters"):
        km = MiniBatchKMeans(n_clusters=nc, random_state=SEED, n_init=3, batch_size=5000)
        df[f"c3d{nc}"] = km.fit_predict(c3d)

    # === DENSITY ===
    for r in tqdm([0.05, 0.1, 0.5, 1.0], desc="Density features"):
        cell = df[f"lb{r}"].astype(str) + "_" + df[f"nb{r}"].astype(str)
        df[f"dns{r}"] = cell.map(cell.value_counts()).values

    for nc in tqdm([10, 50, 100, 200], desc="Cluster frequency"):
        df[f"cf{nc}"] = df[f"c{nc}"].map(df[f"c{nc}"].value_counts()).values

    # === NEAREST NEIGHBORS ===
    print(f"\n  Building BallTree on {(df['_t']==1).sum()} training points...")
    tmask = df["_t"].values == 1
    tree = BallTree(np.radians(coords[tmask]), metric="haversine")
    ti = np.where(tmask)[0]
    ac = np.radians(coords)

    K_values = [1, 3, 5, 10, 20, 50, 100]
    for K in tqdm(K_values, desc="NN queries (K-values)"):
        d_, idx_ = tree.query(ac, k=K)
        dk = d_ * 6371
        w = 1.0 / (dk + 0.001)
        wn = w / w.sum(1, keepdims=True)
        ew = np.exp(-dk / 1.0)
        ewn = ew / (ew.sum(1, keepdims=True) + 1e-10)

        df[f"n{K}md"] = np.mean(dk, 1)
        df[f"n{K}nd"] = dk[:, 0]
        df[f"n{K}xd"] = dk[:, -1]
        if K > 1: df[f"n{K}sd"] = np.std(dk, 1)

        for t in TARGETS:
            if t not in df.columns: continue
            tv = df[t].values[ti[idx_]]
            df[f"n{K}{t}i"] = np.nansum(tv * wn, 1)
            df[f"n{K}{t}e"] = np.nansum(tv * ewn, 1)
            df[f"n{K}{t}m"] = np.nanmean(tv, 1)
            if K >= 5:
                df[f"n{K}{t}s"] = np.nanstd(tv, 1)
                df[f"n{K}{t}d"] = np.nanmedian(tv, 1)
            if K >= 10:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    df[f"n{K}{t}mn"] = np.nanmin(tv, 1)
                    df[f"n{K}{t}mx"] = np.nanmax(tv, 1)

        for ex in EXTRA:
            if ex not in df.columns: continue
            ev = df[ex].values[ti[idx_]]
            df[f"n{K}{ex}i"] = np.nansum(ev * wn, 1)
            df[f"n{K}{ex}m"] = np.nanmean(ev, 1)
            if K >= 5:
                df[f"n{K}{ex}s"] = np.nanstd(ev, 1)

        if K <= 20:
            for t1, t2 in [("Ca","Mg"),("Ca","K"),("Al","Fe"),("Cu","Mn"),
                            ("Al","N"),("Fe","Mn"),("K","Mg"),("Cu","Mg")]:
                if t1 in df.columns and t2 in df.columns:
                    v1 = np.nanmean(df[t1].values[ti[idx_]], 1)
                    v2 = np.nanmean(df[t2].values[ti[idx_]], 1)
                    df[f"n{K}{t1}{t2}r"] = v1 / (v2 + 1e-8)

        nd = df["dm"].values[ti[idx_]]
        df[f"n{K}dm"] = (nd == dep.reshape(-1, 1)).mean(1)

    # Depth-aware neighbors
    for dv in tqdm([0, 1], desc="Depth-aware NN"):
        dm = tmask & (df["ts"].values == dv)
        if dm.sum() < 10: continue
        trd = BallTree(np.radians(coords[dm]), metric="haversine")
        qm = df["ts"].values == dv
        Kd = min(15, dm.sum() - 1)
        dd, di = trd.query(np.radians(coords[qm]), k=Kd)
        ddk = dd * 6371
        dw = 1.0 / (ddk + 0.001)
        dwn = dw / dw.sum(1, keepdims=True)
        dew = np.exp(-ddk / 1.0)
        dewn = dew / (dew.sum(1, keepdims=True) + 1e-10)
        dti = np.where(dm)[0]
        for t in TARGETS:
            if t not in df.columns: continue
            tv = df[t].values[dti[di]]
            df.loc[qm, f"d{dv}{t}i"] = np.nansum(tv * dwn, 1)
            df.loc[qm, f"d{dv}{t}e"] = np.nansum(tv * dewn, 1)
            df.loc[qm, f"d{dv}{t}m"] = np.nanmean(tv, 1)

    # === SPLIT ===
    exc = set(["ID", "_t", "Depth_cm", "start_date"] + TARGETS + EXTRA)
    trf = df[df["_t"] == 1].reset_index(drop=True)
    tef = df[df["_t"] == 0].reset_index(drop=True)
    fc = [c for c in trf.columns if c not in exc
          and trf[c].dtype in ["float64","float32","int64","int32","uint8"]]

    print(f"\n  Total features: {len(fc)}")
    print(f"  Train={trf.shape}, Test={tef.shape}")

    return trf, tef, fc, tk, ss


# ============================================================================
# GPU MODEL TRAINING
# ============================================================================
def train_all(trf, tef, fc, tk, ss, xgb_gpu=False, cb_gpu=False):
    print(f"\n{'='*70}")
    print(f"  STEP 2: MODEL TRAINING")
    print(f"  XGBoost GPU: {'YES' if xgb_gpu else 'NO (CPU)'}")
    print(f"  CatBoost GPU: {'YES' if cb_gpu else 'NO (CPU)'}")
    print(f"{'='*70}\n")

    X = np.nan_to_num(trf[fc].values.astype(np.float32))
    Xt = np.nan_to_num(tef[fc].values.astype(np.float32))
    print(f"  X={X.shape}, Xt={Xt.shape}")

    cfgs = {
        "lgb1": ("lgb", {
            "objective": "regression", "metric": "rmse", "boosting_type": "gbdt",
            "n_estimators": 10000, "learning_rate": 0.01,
            "num_leaves": 511, "max_depth": -1, "min_child_samples": 3,
            "subsample": 0.7, "colsample_bytree": 0.4,
            "reg_alpha": 0.05, "reg_lambda": 0.5,
            "random_state": SEED, "verbose": -1, "n_jobs": -1,
        }),
        "lgb2": ("lgb", {
            "objective": "regression", "metric": "rmse", "boosting_type": "gbdt",
            "n_estimators": 8000, "learning_rate": 0.015,
            "num_leaves": 255, "max_depth": 12, "min_child_samples": 5,
            "subsample": 0.75, "colsample_bytree": 0.5,
            "reg_alpha": 0.1, "reg_lambda": 1.0,
            "random_state": SEED + 1, "verbose": -1, "n_jobs": -1,
        }),
        "lgb3": ("lgb", {
            "objective": "regression", "metric": "rmse", "boosting_type": "gbdt",
            "n_estimators": 6000, "learning_rate": 0.02,
            "num_leaves": 127, "max_depth": 8, "min_child_samples": 10,
            "subsample": 0.8, "colsample_bytree": 0.6,
            "reg_alpha": 0.5, "reg_lambda": 2.0,
            "random_state": SEED + 2, "verbose": -1, "n_jobs": -1,
        }),
        "lgb4": ("lgb", {
            "objective": "regression", "metric": "rmse", "boosting_type": "gbdt",
            "n_estimators": 5000, "learning_rate": 0.025,
            "num_leaves": 63, "max_depth": 6, "min_child_samples": 20,
            "subsample": 0.85, "colsample_bytree": 0.7,
            "reg_alpha": 1.0, "reg_lambda": 3.0,
            "random_state": SEED + 3, "verbose": -1, "n_jobs": -1,
        }),
        "lgb_dart": ("lgb", {
            "objective": "regression", "metric": "rmse", "boosting_type": "dart",
            "n_estimators": 3000, "learning_rate": 0.02,
            "num_leaves": 255, "max_depth": -1, "min_child_samples": 5,
            "subsample": 0.7, "colsample_bytree": 0.5,
            "reg_alpha": 0.1, "reg_lambda": 1.0, "drop_rate": 0.1,
            "random_state": SEED + 4, "verbose": -1, "n_jobs": -1,
        }),
        "xgb1": ("xgb", {
            "objective": "reg:squarederror", "eval_metric": "rmse",
            "n_estimators": 10000, "learning_rate": 0.01,
            "max_depth": 10, "min_child_weight": 3,
            "subsample": 0.7, "colsample_bytree": 0.4, "colsample_bylevel": 0.7,
            "reg_alpha": 0.05, "reg_lambda": 0.5,
            **( {"tree_method": "hist", "device": "cuda"} if xgb_gpu else {"tree_method": "hist"} ),
            "random_state": SEED, "verbosity": 0, "early_stopping_rounds": 300,
        }),
        "xgb2": ("xgb", {
            "objective": "reg:squarederror", "eval_metric": "rmse",
            "n_estimators": 8000, "learning_rate": 0.015,
            "max_depth": 8, "min_child_weight": 5,
            "subsample": 0.8, "colsample_bytree": 0.5,
            "reg_alpha": 0.5, "reg_lambda": 2.0, "gamma": 0.1,
            **( {"tree_method": "hist", "device": "cuda"} if xgb_gpu else {"tree_method": "hist"} ),
            "random_state": SEED + 10, "verbosity": 0, "early_stopping_rounds": 300,
        }),
        "cb1": ("cb", {
            "iterations": 8000, "learning_rate": 0.02, "depth": 8,
            "l2_leaf_reg": 3, "random_seed": SEED, "verbose": 0,
            "loss_function": "RMSE", "eval_metric": "RMSE",
            "min_data_in_leaf": 5,
            **( {"task_type": "GPU", "devices": "0"} if cb_gpu else {} ),
        }),
        "cb2": ("cb", {
            "iterations": 6000, "learning_rate": 0.03, "depth": 6,
            "l2_leaf_reg": 5, "random_seed": SEED + 5, "verbose": 0,
            "loss_function": "RMSE", "eval_metric": "RMSE",
            "min_data_in_leaf": 10,
            **( {"task_type": "GPU", "devices": "0"} if cb_gpu else {} ),
        }),
        "cb3": ("cb", {
            "iterations": 5000, "learning_rate": 0.04, "depth": 10,
            "l2_leaf_reg": 1, "random_seed": SEED + 7, "verbose": 0,
            "loss_function": "RMSE", "eval_metric": "RMSE",
            "min_data_in_leaf": 3,
            **( {"task_type": "GPU", "devices": "0"} if cb_gpu else {} ),
        }),
        "et": ("et", {
            "n_estimators": 2000, "max_depth": None,
            "min_samples_split": 3, "min_samples_leaf": 1,
            "max_features": 0.5, "random_state": SEED, "n_jobs": -1,
        }),
    }

    preds = {}
    cvs = {}
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    target_pbar = tqdm(TARGETS, desc="Targets", position=0)
    for tidx, tgt in enumerate(target_pbar):
        t0 = datetime.now()
        target_pbar.set_description(f"Target {tgt}")

        if tgt not in trf.columns:
            preds[tgt] = np.zeros(len(Xt)); continue

        y = trf[tgt].values
        v = ~np.isnan(y)
        nv = v.sum()

        if nv < 50:
            c = f"n5{tgt}i"
            preds[tgt] = np.clip(Xt[:, fc.index(c)], 0, None) if c in fc else np.zeros(len(Xt))
            tqdm_write(f"  {tgt}: IDW fallback (n={nv})")
            continue

        od, td_, sd = {}, {}, {}

        model_pbar = tqdm(cfgs.items(), desc=f"  {tgt} models", position=1, leave=False)
        for mn, (mt, mp) in model_pbar:
            model_pbar.set_description(f"  {tgt} > {mn}")
            oof = np.zeros(len(X))
            tp = np.zeros(len(Xt))
            fs = []

            fold_pbar = tqdm(enumerate(kf.split(X)), total=N_FOLDS,
                              desc=f"    {mn} folds", position=2, leave=False)
            for fi, (tri, vai) in fold_pbar:
                fold_pbar.set_description(f"    {mn} fold {fi+1}/{N_FOLDS}")
                ytr, yva = y[tri], y[vai]
                tok, vok = ~np.isnan(ytr), ~np.isnan(yva)
                if tok.sum() < 10 or vok.sum() < 3: continue

                Xtr_c, ytr_c = X[tri][tok], ytr[tok]
                Xva_c, yva_c = X[vai][vok], yva[vok]

                try:
                    if mt == "lgb":
                        m = lgb.LGBMRegressor(**mp)
                        m.fit(Xtr_c, ytr_c, eval_set=[(Xva_c, yva_c)],
                              callbacks=[lgb.early_stopping(300, verbose=False),
                                         lgb.log_evaluation(0)])
                    elif mt == "xgb":
                        m = xgb.XGBRegressor(**mp)
                        m.fit(Xtr_c, ytr_c, eval_set=[(Xva_c, yva_c)], verbose=False)
                    elif mt == "cb":
                        m = cb.CatBoostRegressor(**mp)
                        m.fit(Xtr_c, ytr_c, eval_set=(Xva_c, yva_c),
                              early_stopping_rounds=300, verbose=0)
                    elif mt == "et":
                        m = ExtraTreesRegressor(**mp)
                        m.fit(Xtr_c, ytr_c)

                    vp = np.clip(m.predict(Xva_c), 0, None)
                    oof[vai[vok]] = vp
                    fold_rmse = np.sqrt(mean_squared_error(yva_c, vp))
                    fs.append(fold_rmse)
                    tp += np.clip(m.predict(Xt), 0, None) / N_FOLDS
                    fold_pbar.set_postfix(rmse=f"{fold_rmse:.1f}")

                except Exception as e:
                    tqdm_write(f"    ERROR {mn} fold {fi}: {e}")

            fold_pbar.close()

            if fs:
                avg = np.mean(fs)
                tqdm_write(f"    {mn:10s}: RMSE={avg:.4f} (std={np.std(fs):.4f})")
                od[mn] = oof; td_[mn] = tp; sd[mn] = avg

        model_pbar.close()

        # === MULTI-SEED for top 2 models ===
        if len(sd) >= 2:
            top2 = sorted(sd, key=sd.get)[:2]
            seed_combos = [(base_mn, si) for base_mn in top2
                           for si in range(3)
                           if cfgs[base_mn][0] in ("lgb", "xgb", "cb")]

            for base_mn, si in tqdm(seed_combos, desc=f"  {tgt} multi-seed", position=1, leave=False):
                base_mt, base_mp = cfgs[base_mn]
                smn = f"{base_mn}_s{si}"
                smp = base_mp.copy()
                if "random_state" in smp: smp["random_state"] = SEED + 100 + si * 50
                if "random_seed" in smp: smp["random_seed"] = SEED + 100 + si * 50

                oof_s = np.zeros(len(X)); tp_s = np.zeros(len(Xt)); fs_s = []
                kf_s = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED + 100 + si * 50)

                for fi, (tri, vai) in enumerate(kf_s.split(X)):
                    ytr, yva = y[tri], y[vai]
                    tok, vok = ~np.isnan(ytr), ~np.isnan(yva)
                    if tok.sum() < 10 or vok.sum() < 3: continue
                    try:
                        if base_mt == "lgb":
                            m = lgb.LGBMRegressor(**smp)
                            m.fit(X[tri][tok], ytr[tok],
                                  eval_set=[(X[vai][vok], yva[vok])],
                                  callbacks=[lgb.early_stopping(300, verbose=False),
                                             lgb.log_evaluation(0)])
                        elif base_mt == "xgb":
                            m = xgb.XGBRegressor(**smp)
                            m.fit(X[tri][tok], ytr[tok],
                                  eval_set=[(X[vai][vok], yva[vok])], verbose=False)
                        elif base_mt == "cb":
                            m = cb.CatBoostRegressor(**smp)
                            m.fit(X[tri][tok], ytr[tok],
                                  eval_set=(X[vai][vok], yva[vok]),
                                  early_stopping_rounds=300, verbose=0)

                        vp = np.clip(m.predict(X[vai][vok]), 0, None)
                        oof_s[vai[vok]] = vp
                        fs_s.append(np.sqrt(mean_squared_error(yva[vok], vp)))
                        tp_s += np.clip(m.predict(Xt), 0, None) / N_FOLDS
                    except Exception:
                        pass

                if fs_s:
                    avg_s = np.mean(fs_s)
                    tqdm_write(f"    {smn:10s}: RMSE={avg_s:.4f}")
                    od[smn] = oof_s; td_[smn] = tp_s; sd[smn] = avg_s

        # === ENSEMBLE ===
        if len(od) > 1:
            tqdm_write(f"\n  Ensembling {len(od)} models for {tgt}...")
            mns = sorted(od.keys())
            om = np.column_stack([od[m] for m in mns])
            tm = np.column_stack([td_[m] for m in mns])

            def fn(w):
                w = np.abs(w); w /= w.sum()
                return np.sqrt(mean_squared_error(y[v], (om[v] * w).sum(1)))

            bw = np.ones(len(mns)) / len(mns)
            br = fn(bw); bp = None

            for meth in tqdm(["Nelder-Mead", "Powell", "COBYLA"],
                              desc=f"  {tgt} weight opt", position=1, leave=False):
                try:
                    res = minimize(fn, bw, method=meth, options={"maxiter": 50000})
                    if res.fun < br:
                        br = res.fun
                        w_ = np.abs(res.x); w_ /= w_.sum()
                        bp = np.clip((tm * w_).sum(1), 0, None)
                        bw = w_
                except: pass

            # Ridge meta-learner
            try:
                sc = RobustScaler()
                os_ = sc.fit_transform(om[v])
                ts_ = sc.transform(tm)
                mo = np.zeros(v.sum()); mt_ = np.zeros(len(Xt))
                for tri, vai in KFold(5, shuffle=True, random_state=SEED).split(os_):
                    r = Ridge(alpha=1.0)
                    r.fit(os_[tri], y[v][tri])
                    mo[vai] = r.predict(os_[vai])
                    mt_ += r.predict(ts_) / 5
                mr = np.sqrt(mean_squared_error(y[v], mo))
                if mr < br:
                    br = mr; bp = np.clip(mt_, 0, None)
                    tqdm_write(f"  -> Ridge stacking wins: {mr:.4f}")
            except: pass

            if bp is None:
                bp = np.clip(tm.mean(1), 0, None)

            preds[tgt] = bp; cvs[tgt] = br
            best_single = min(sd, key=sd.get)
            tqdm_write(f"  ENSEMBLE: {br:.4f} | Best single: {best_single} ({sd[best_single]:.4f})")
            tqdm_write(f"  Weights: {dict(zip(mns, [f'{w:.3f}' for w in bw]))}")

        elif len(td_) == 1:
            mn_ = list(td_.keys())[0]
            preds[tgt] = td_[mn_]; cvs[tgt] = sd[mn_]
        else:
            preds[tgt] = np.zeros(len(Xt)); cvs[tgt] = 999

        elapsed = (datetime.now() - t0).total_seconds()
        tqdm_write(f"  {tgt} done in {elapsed:.0f}s\n")
        gc.collect()

    target_pbar.close()
    return preds, cvs, tef, tk, ss


# ============================================================================
# SUBMISSION
# ============================================================================
def make_submission(preds, tef, tk, ss):
    print(f"\n{'='*70}")
    print(f"  STEP 3: SUBMISSION")
    print(f"{'='*70}\n")

    sub = ss[["ID"]].copy()
    ids = tef["ID"].values

    for t in tqdm(TARGETS, desc="Mapping predictions"):
        c = f"Target_{t}"
        sub[c] = sub["ID"].map(dict(zip(ids, preds.get(t, np.zeros(len(ids)))))).fillna(0).clip(lower=0)

    for t in tqdm(TARGETS, desc="Applying mask"):
        c = f"Target_{t}"
        if t in tk.columns and c in sub.columns:
            mask = sub["ID"].map(tk.set_index("ID")[t]).fillna(0)
            b = (sub[c] > 0).sum()
            sub[c] *= mask
            a = (sub[c] > 0).sum()
            tqdm_write(f"  {t}: {b} -> {a} non-zero")

    assert list(sub.columns) == list(ss.columns) and len(sub) == len(ss)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    p = os.path.join(SDIR, f"sub_gpu_{ts}.csv")
    sub.to_csv(p, index=False)
    sub.to_csv(os.path.join(SDIR, "submission_latest.csv"), index=False)
    print(f"\n  Saved: {p} ({sub.shape})")
    return sub


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    t0 = datetime.now()
    print(f"\n{'='*70}")
    print(f"  RHEA SOIL NUTRIENT PREDICTION - H200 GPU PIPELINE")
    print(f"  Started: {t0}")
    print(f"{'='*70}")

    xgb_gpu, cb_gpu = detect_gpu()

    trf, tef, fc, tk, ss = build_features()
    preds, cvs, tef, tk, ss = train_all(trf, tef, fc, tk, ss, xgb_gpu=xgb_gpu, cb_gpu=cb_gpu)
    sub = make_submission(preds, tef, tk, ss)

    print(f"\n{'='*70}")
    print(f"  CV SCORES")
    print(f"{'='*70}")
    all_s = []
    for t in TARGETS:
        if t in cvs:
            print(f"  {t:5s}: {cvs[t]:.4f}")
            all_s.append(cvs[t])
    if all_s:
        print(f"\n  MEAN RMSE: {np.mean(all_s):.4f}")

    print(f"\n  Total time: {datetime.now() - t0}")
    print(f"  Done: {datetime.now()}")
