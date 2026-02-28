"""
RHEA SOIL NUTRIENT PREDICTION - FAST PIPELINE (v3)
Optimized for speed: 5-fold CV, 5 core models, moderate estimators.
Get a strong baseline submission fast, then iterate.
"""

import os, sys, warnings, numpy as np, pandas as pd
from datetime import datetime
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
from sklearn.ensemble import ExtraTreesRegressor

warnings.filterwarnings("ignore")

SEED = 42
N_FOLDS = 5
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
SUB_DIR = os.path.join(os.path.dirname(__file__), "submissions")
os.makedirs(SUB_DIR, exist_ok=True)

TARGETS = ["Al", "B", "Ca", "Cu", "Fe", "K", "Mg", "Mn", "N", "Na", "P", "S", "Zn"]
TRAIN_EXTRA = ["C_organic", "C_total", "ph"]

def load():
    print("LOADING...", flush=True)
    train = pd.read_csv(os.path.join(DATA_DIR, "Train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "TestSet.csv"))
    dates = pd.read_csv(os.path.join(DATA_DIR, "Sample_Collection_Dates.csv"))
    tk = pd.read_csv(os.path.join(DATA_DIR, "TargetPred_To_Keep.csv"))
    ss = pd.read_csv(os.path.join(DATA_DIR, "SampleSubmission.csv"))
    print(f"  Train={train.shape}, Test={test.shape}", flush=True)
    return train, test, dates, tk, ss

def features(train, test, dates):
    print("FEATURE ENGINEERING...", flush=True)

    # Merge dates
    td = dates[dates["set"] == "Test"][["ID", "start_date"]]
    test = test.merge(td, on="ID", how="left")
    test["date_group"] = test["start_date"].apply(lambda x: 1 if "2022" in str(x) else 0)
    train["date_group"] = 0

    # Depth
    def dmid(d):
        p = str(d).replace(" ","").replace("cm","").split("-")
        try: return (float(p[0])+float(p[1]))/2
        except: return 15.0
    for df in [train, test]:
        df["depth_mid"] = df["Depth_cm"].apply(dmid)
        df["is_topsoil"] = (df["depth_mid"] <= 20).astype(int)

    # Build combined
    keep = ["ID", "Latitude", "Longitude", "depth_mid", "is_topsoil", "date_group"]
    t_keep = keep + TARGETS + [c for c in TRAIN_EXTRA if c in train.columns]
    train_s = train[[c for c in t_keep if c in train.columns]].copy()
    test_s = test[[c for c in keep if c in test.columns]].copy()
    train_s["_tr"] = 1; test_s["_tr"] = 0
    df = pd.concat([train_s, test_s], ignore_index=True)
    nt = len(train_s)

    lat, lon, dep = df["Latitude"].values, df["Longitude"].values, df["depth_mid"].values

    # --- GEO ---
    print("  geo...", flush=True)
    df["abs_lat"] = np.abs(lat)
    df["lat_lon_prod"] = lat * lon
    df["geo_radius"] = np.sqrt(lat**2 + lon**2)
    df["geo_angle"] = np.arctan2(lat, lon)
    df["lat_sq"] = lat**2
    df["lon_sq"] = lon**2

    for r in [0.1, 0.5, 1.0, 2.0]:
        df[f"latb_{r}"] = np.floor(lat/r)*r
        df[f"lonb_{r}"] = np.floor(lon/r)*r

    df["is_south"] = (lat < 0).astype(int)

    def hav(la1,lo1,la2,lo2):
        R=6371; la1,lo1,la2,lo2=map(np.radians,[la1,lo1,la2,lo2])
        a=np.sin((la2-la1)/2)**2+np.cos(la1)*np.cos(la2)*np.sin((lo2-lo1)/2)**2
        return 2*R*np.arcsin(np.sqrt(np.clip(a,0,1)))

    for nm,(rl,rn) in {"nai":(-1.29,36.82),"dar":(-6.79,39.28),"vic":(-1,33),"kili":(-3.07,37.35),"kamp":(0.35,32.58)}.items():
        df[f"hd_{nm}"] = hav(lat,lon,rl,rn)

    # --- DEPTH ---
    print("  depth...", flush=True)
    df["log_dep"] = np.log1p(dep)
    df["dep_x_lat"] = dep * lat
    df["dep_x_lon"] = dep * lon

    # --- CLIMATE ---
    print("  climate...", flush=True)
    al = np.abs(lat)
    df["temp_p"] = 30 - 0.5*al
    df["prec_p"] = 2000*np.exp(-0.05*(al-5)**2)
    df["arid_p"] = df["temp_p"]/(df["prec_p"]+1)

    df["z_ea"] = ((lon>30)&(lon<42)&(lat>-12)&(lat<5)).astype(int)
    df["z_kc"] = ((lat>-2)&(lat<1)&(lon>36)&(lon<38)).astype(int)
    df["z_kr"] = ((lat>-2)&(lat<1)&(lon>35)&(lon<36.5)).astype(int)
    df["z_south"] = (lat<-15).astype(int)

    # --- CLUSTERS ---
    print("  clusters...", flush=True)
    coords = np.column_stack([lat, lon])
    for nc in [5, 10, 20, 50, 100, 200]:
        km = MiniBatchKMeans(n_clusters=nc, random_state=SEED, n_init=3, batch_size=5000)
        lb = km.fit_predict(coords)
        df[f"cl_{nc}"] = lb
        df[f"dcl_{nc}"] = np.sqrt(((coords - km.cluster_centers_[lb])**2).sum(1))

    # --- DENSITY ---
    print("  density...", flush=True)
    for r in [0.1, 0.5, 1.0]:
        cell = df[f"latb_{r}"].astype(str)+"_"+df[f"lonb_{r}"].astype(str)
        df[f"dens_{r}"] = cell.map(cell.value_counts()).values

    # --- NEAREST NEIGHBORS (THE KEY) ---
    print("  neighbors...", flush=True)
    tmask = df["_tr"].values == 1
    tree = BallTree(np.radians(coords[tmask]), metric="haversine")
    ti = np.where(tmask)[0]
    ac = np.radians(coords)

    for K in [1, 3, 5, 10, 20, 50]:
        print(f"    K={K}...", flush=True)
        d, idx = tree.query(ac, k=K)
        dk = d * 6371
        w = 1.0/(dk+0.001)
        wn = w / w.sum(1, keepdims=True)
        ew = np.exp(-dk/1.0)
        ewn = ew / (ew.sum(1, keepdims=True)+1e-10)

        df[f"n{K}_md"] = np.mean(dk,1)
        df[f"n{K}_xd"] = dk[:,-1]
        if K > 1:
            df[f"n{K}_sd"] = np.std(dk,1)

        for t in TARGETS:
            if t not in df.columns: continue
            tv = df[t].values[ti[idx]]
            df[f"n{K}_{t}_i"] = np.nansum(tv*wn, 1)
            df[f"n{K}_{t}_e"] = np.nansum(tv*ewn, 1)
            df[f"n{K}_{t}_m"] = np.nanmean(tv, 1)
            if K >= 5:
                df[f"n{K}_{t}_s"] = np.nanstd(tv, 1)
                df[f"n{K}_{t}_md"] = np.nanmedian(tv, 1)

        for ex in TRAIN_EXTRA:
            if ex not in df.columns: continue
            ev = df[ex].values[ti[idx]]
            df[f"n{K}_{ex}_i"] = np.nansum(ev*wn, 1)
            df[f"n{K}_{ex}_m"] = np.nanmean(ev, 1)

        # Cross-nutrient ratios
        if K <= 10:
            for t1,t2 in [("Ca","Mg"),("Ca","K"),("Al","Fe"),("Cu","Mn"),("Al","N")]:
                if t1 in df.columns and t2 in df.columns:
                    v1 = np.nanmean(df[t1].values[ti[idx]],1)
                    v2 = np.nanmean(df[t2].values[ti[idx]],1)
                    df[f"n{K}_{t1}_{t2}_r"] = v1/(v2+1e-8)

    # Depth-aware neighbors
    print("  depth-aware neighbors...", flush=True)
    for dv in [0, 1]:
        dm = tmask & (df["is_topsoil"].values == dv)
        if dm.sum() < 10: continue
        trd = BallTree(np.radians(coords[dm]), metric="haversine")
        qm = df["is_topsoil"].values == dv
        dd, di = trd.query(np.radians(coords[qm]), k=min(10, dm.sum()-1))
        ddk = dd * 6371
        dw = 1.0/(ddk+0.001)
        dwn = dw / dw.sum(1, keepdims=True)
        dti = np.where(dm)[0]
        for t in TARGETS:
            if t not in df.columns: continue
            tv = df[t].values[dti[di]]
            col = f"nd{dv}_{t}_i"
            df[col] = np.nan
            df.loc[qm, col] = np.nansum(tv*dwn, 1)

    # --- SPLIT ---
    print("  splitting...", flush=True)
    exc = set(["ID","_tr","Depth_cm","start_date"] + TARGETS + TRAIN_EXTRA)
    train_f = df[df["_tr"]==1].reset_index(drop=True)
    test_f = df[df["_tr"]==0].reset_index(drop=True)
    fcols = [c for c in train_f.columns if c not in exc and train_f[c].dtype in ["float64","float32","int64","int32"]]
    print(f"  Features: {len(fcols)}", flush=True)
    return train_f, test_f, fcols

def train_predict(train_f, test_f, fcols):
    print(f"\nTRAINING ({len(fcols)} features)...", flush=True)
    X = np.nan_to_num(train_f[fcols].values.astype(np.float32))
    Xt = np.nan_to_num(test_f[fcols].values.astype(np.float32))

    models = {
        "lgb1": ("lgb", {"objective":"regression","metric":"rmse","boosting_type":"gbdt",
            "n_estimators":5000,"learning_rate":0.02,"num_leaves":255,"min_child_samples":5,
            "subsample":0.7,"colsample_bytree":0.5,"reg_alpha":0.1,"reg_lambda":1.0,
            "random_state":SEED,"verbose":-1,"n_jobs":-1}),
        "lgb2": ("lgb", {"objective":"regression","metric":"rmse","boosting_type":"gbdt",
            "n_estimators":4000,"learning_rate":0.025,"num_leaves":127,"max_depth":10,
            "min_child_samples":10,"subsample":0.8,"colsample_bytree":0.6,
            "reg_alpha":0.5,"reg_lambda":2.0,
            "random_state":SEED+1,"verbose":-1,"n_jobs":-1}),
        "xgb": ("xgb", {"objective":"reg:squarederror","eval_metric":"rmse",
            "n_estimators":5000,"learning_rate":0.02,"max_depth":10,"min_child_weight":3,
            "subsample":0.7,"colsample_bytree":0.5,"reg_alpha":0.1,"reg_lambda":1.0,
            "random_state":SEED,"n_jobs":-1,"verbosity":0,"early_stopping_rounds":200}),
        "cb": ("cb", {"iterations":5000,"learning_rate":0.03,"depth":10,
            "l2_leaf_reg":3,"random_seed":SEED,"verbose":0,
            "loss_function":"RMSE","eval_metric":"RMSE",
            "min_data_in_leaf":5,"subsample":0.7,"colsample_bylevel":0.5,
            "bootstrap_type":"Bernoulli"}),
        "et": ("sk", {"n_estimators":1500,"max_depth":None,"min_samples_split":3,
            "min_samples_leaf":1,"max_features":0.5,"random_state":SEED,"n_jobs":-1}),
    }

    preds = {}; cvs = {}
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    for ti, tgt in enumerate(TARGETS):
        t0 = datetime.now()
        print(f"\n  [{ti+1}/13] {tgt}", end="", flush=True)
        if tgt not in train_f.columns:
            preds[tgt] = np.zeros(len(Xt)); continue
        y = train_f[tgt].values
        valid = ~np.isnan(y)
        nv = valid.sum()
        print(f" (n={nv})", end="", flush=True)
        if nv < 50:
            c = f"n5_{tgt}_i"
            if c in fcols:
                fi = fcols.index(c)
                preds[tgt] = np.clip(Xt[:,fi], 0, None)
            else:
                preds[tgt] = np.zeros(len(Xt))
            print(" -> IDW fallback", flush=True); continue

        od, td, sd = {}, {}, {}
        for mn, (mt, mp) in models.items():
            oof = np.zeros(len(X)); tp = np.zeros(len(Xt)); fs = []
            for fi_, (tri, vai) in enumerate(kf.split(X)):
                Xtr, Xva = X[tri], X[vai]
                ytr, yva = y[tri], y[vai]
                tok, vok = ~np.isnan(ytr), ~np.isnan(yva)
                if tok.sum()<10 or vok.sum()<3: continue
                try:
                    if mt=="lgb":
                        m = lgb.LGBMRegressor(**mp)
                        m.fit(Xtr[tok],ytr[tok],eval_set=[(Xva[vok],yva[vok])],
                              callbacks=[lgb.early_stopping(200,verbose=False),lgb.log_evaluation(0)])
                    elif mt=="xgb":
                        m = xgb.XGBRegressor(**mp)
                        m.fit(Xtr[tok],ytr[tok],eval_set=[(Xva[vok],yva[vok])],verbose=False)
                    elif mt=="cb":
                        m = cb.CatBoostRegressor(**mp)
                        m.fit(Xtr[tok],ytr[tok],eval_set=(Xva[vok],yva[vok]),early_stopping_rounds=200,verbose=0)
                    elif mt=="sk":
                        m = ExtraTreesRegressor(**mp); m.fit(Xtr[tok],ytr[tok])
                    vp = np.clip(m.predict(Xva[vok]),0,None)
                    oof[vai[vok]] = vp
                    fs.append(np.sqrt(mean_squared_error(yva[vok],vp)))
                    tp += np.clip(m.predict(Xt),0,None) / N_FOLDS
                except Exception as e:
                    print(f" ERR:{mn}:{e}", end="", flush=True)
            if fs:
                avg = np.mean(fs)
                print(f" {mn}={avg:.1f}", end="", flush=True)
                od[mn]=oof; td[mn]=tp; sd[mn]=avg

        # Ensemble
        if len(od) > 1:
            mns = sorted(od.keys())
            om = np.column_stack([od[m] for m in mns])
            tm = np.column_stack([td[m] for m in mns])
            def fn(w):
                w=np.abs(w); w/=w.sum()
                return np.sqrt(mean_squared_error(y[valid],(om[valid]*w).sum(1)))
            bw = np.ones(len(mns))/len(mns); br = fn(bw); bp = None
            for meth in ["Nelder-Mead","Powell"]:
                try:
                    res = minimize(fn,bw,method=meth,options={"maxiter":20000})
                    if res.fun < br:
                        br=res.fun; w=np.abs(res.x); w/=w.sum()
                        bp=np.clip((tm*w).sum(1),0,None); bw=w
                except: pass
            # Ridge meta
            try:
                sc = RobustScaler()
                os_=sc.fit_transform(om[valid]); ts_=sc.transform(tm)
                mo,mt_=np.zeros(valid.sum()),np.zeros(len(Xt))
                for tri,vai in KFold(5,shuffle=True,random_state=SEED).split(os_):
                    r=Ridge(alpha=1.0); r.fit(os_[tri],y[valid][tri])
                    mo[vai]=r.predict(os_[vai]); mt_+=r.predict(ts_)/5
                mr = np.sqrt(mean_squared_error(y[valid],mo))
                if mr < br: br=mr; bp=np.clip(mt_,0,None)
            except: pass
            if bp is None: bp = np.clip(tm.mean(1),0,None)
            preds[tgt]=bp; cvs[tgt]=br
            print(f" ENS={br:.1f}", end="", flush=True)
        elif len(td)==1:
            mn_=list(td.keys())[0]; preds[tgt]=td[mn_]; cvs[tgt]=sd[mn_]
        else:
            preds[tgt]=np.zeros(len(Xt)); cvs[tgt]=999
        print(f" ({(datetime.now()-t0).seconds}s)", flush=True)

    return preds, cvs

def submit(test_f, preds, tk, ss):
    print("\nCREATING SUBMISSION...", flush=True)
    sub = ss[["ID"]].copy()
    ids = test_f["ID"].values
    for t in TARGETS:
        c = f"Target_{t}"
        sub[c] = sub["ID"].map(dict(zip(ids,preds.get(t,np.zeros(len(ids)))))).fillna(0).clip(lower=0)
    # Mask
    for t in TARGETS:
        c = f"Target_{t}"
        if t in tk.columns and c in sub.columns:
            mask = sub["ID"].map(tk.set_index("ID")[t]).fillna(0)
            sub[c] *= mask
    assert list(sub.columns)==list(ss.columns) and len(sub)==len(ss)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    p = os.path.join(SUB_DIR, f"sub_{ts}.csv")
    sub.to_csv(p, index=False)
    sub.to_csv(os.path.join(SUB_DIR, "submission_latest.csv"), index=False)
    print(f"  Saved: {p} ({sub.shape})", flush=True)
    for t in TARGETS:
        c=f"Target_{t}"; nz=(sub[c]>0).sum()
        print(f"    {t}: {nz} non-zero", flush=True)
    return sub

if __name__=="__main__":
    t0 = datetime.now()
    print(f"Started: {t0}\n", flush=True)
    train, test, dates, tk, ss = load()
    train_f, test_f, fcols = features(train, test, dates)
    preds, cvs = train_predict(train_f, test_f, fcols)
    sub = submit(test_f, preds, tk, ss)
    print(f"\nCV SCORES:", flush=True)
    ss_ = []
    for t in TARGETS:
        if t in cvs:
            print(f"  {t}: {cvs[t]:.2f}", flush=True); ss_.append(cvs[t])
    if ss_: print(f"  MEAN: {np.mean(ss_):.2f}", flush=True)
    print(f"\nDone: {datetime.now()} (elapsed: {datetime.now()-t0})", flush=True)
