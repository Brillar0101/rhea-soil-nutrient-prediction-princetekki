"""
Multi-Model Arsenal for Rhea Soil Nutrient Prediction.
We're going ALL IN: LightGBM, XGBoost, CatBoost, ExtraTrees, Ridge,
KNN-regressor, and a 2-level stacking ensemble.
"""
import numpy as np
import pandas as pd
import os
import pickle
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, ElasticNet, BayesianRidge
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from config import TARGETS, SEED, N_FOLDS, USE_GPU, MODEL_DIR


# ============================================================================
# MODEL CONFIGURATIONS - Aggressive hyperparameters for maximum performance
# ============================================================================

def get_lgb_params(target_name):
    """LightGBM params - different configs per nutrient for diversity."""
    base = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "n_estimators": 10000,
        "learning_rate": 0.01,
        "num_leaves": 255,
        "max_depth": -1,
        "min_child_samples": 5,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": SEED,
        "verbose": -1,
        "n_jobs": -1,
    }
    return base


def get_lgb_dart_params(target_name):
    """LightGBM DART - dropout regularization, different model signature."""
    return {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "dart",
        "n_estimators": 3000,
        "learning_rate": 0.02,
        "num_leaves": 127,
        "max_depth": -1,
        "min_child_samples": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.6,
        "reg_alpha": 0.5,
        "reg_lambda": 2.0,
        "drop_rate": 0.1,
        "random_state": SEED + 1,
        "verbose": -1,
        "n_jobs": -1,
    }


def get_lgb_goss_params(target_name):
    """LightGBM GOSS - gradient-based one-side sampling."""
    return {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "goss",
        "n_estimators": 8000,
        "learning_rate": 0.015,
        "num_leaves": 200,
        "max_depth": 12,
        "min_child_samples": 7,
        "colsample_bytree": 0.65,
        "reg_alpha": 0.2,
        "reg_lambda": 1.5,
        "random_state": SEED + 2,
        "verbose": -1,
        "n_jobs": -1,
    }


def get_xgb_params(target_name):
    """XGBoost params - complementary to LightGBM."""
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "n_estimators": 10000,
        "learning_rate": 0.01,
        "max_depth": 10,
        "min_child_weight": 3,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "colsample_bylevel": 0.7,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": SEED,
        "n_jobs": -1,
        "verbosity": 0,
    }
    if USE_GPU:
        params["tree_method"] = "hist"
        params["device"] = "cuda"
    return params


def get_xgb_v2_params(target_name):
    """XGBoost v2 - different hyperparameters for diversity."""
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "n_estimators": 8000,
        "learning_rate": 0.015,
        "max_depth": 8,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.6,
        "colsample_bylevel": 0.8,
        "reg_alpha": 0.5,
        "reg_lambda": 2.0,
        "gamma": 0.1,
        "random_state": SEED + 10,
        "n_jobs": -1,
        "verbosity": 0,
    }
    if USE_GPU:
        params["tree_method"] = "hist"
        params["device"] = "cuda"
    return params


def get_catboost_params(target_name):
    """CatBoost params - ordered boosting, handles categoricals natively."""
    params = {
        "iterations": 10000,
        "learning_rate": 0.02,
        "depth": 10,
        "l2_leaf_reg": 3,
        "random_seed": SEED,
        "verbose": 0,
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "min_data_in_leaf": 5,
        "subsample": 0.7,
        "colsample_bylevel": 0.7,
        "bootstrap_type": "Bernoulli",
    }
    if USE_GPU:
        params["task_type"] = "GPU"
    return params


def get_catboost_v2_params(target_name):
    """CatBoost v2 - deeper trees, more regularization."""
    params = {
        "iterations": 8000,
        "learning_rate": 0.015,
        "depth": 12,
        "l2_leaf_reg": 5,
        "random_seed": SEED + 5,
        "verbose": 0,
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "min_data_in_leaf": 10,
        "subsample": 0.8,
        "colsample_bylevel": 0.6,
        "bootstrap_type": "Bernoulli",
        "random_strength": 1.5,
        "bagging_temperature": 0.5,
    }
    if USE_GPU:
        params["task_type"] = "GPU"
    return params


def get_extratrees_params(target_name):
    """ExtraTrees - extremely randomized trees, great for ensembles."""
    return {
        "n_estimators": 2000,
        "max_depth": None,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": 0.7,
        "random_state": SEED,
        "n_jobs": -1,
    }


def get_rf_params(target_name):
    """Random Forest - classic ensemble member."""
    return {
        "n_estimators": 2000,
        "max_depth": None,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": 0.5,
        "random_state": SEED + 3,
        "n_jobs": -1,
    }


# ============================================================================
# TRAINING ENGINE
# ============================================================================

class ModelTrainer:
    """Trains multiple models with K-fold CV for a single target."""

    def __init__(self, n_folds=N_FOLDS, seed=SEED):
        self.n_folds = n_folds
        self.seed = seed

    def get_feature_cols(self, df, id_col, target_cols):
        """Get feature columns, excluding IDs and targets."""
        exclude = set([id_col, "_is_train"] + list(target_cols))
        feature_cols = []
        for col in df.columns:
            if col in exclude:
                continue
            if df[col].dtype in ["float64", "float32", "int64", "int32", "uint8"]:
                feature_cols.append(col)
        return feature_cols

    def train_lgb_fold(self, X_train, y_train, X_val, y_val, params):
        """Train LightGBM with early stopping."""
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)],
        )
        return model

    def train_xgb_fold(self, X_train, y_train, X_val, y_val, params):
        """Train XGBoost with early stopping."""
        model = xgb.XGBRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        # XGBoost handles early stopping via callbacks internally
        return model

    def train_catboost_fold(self, X_train, y_train, X_val, y_val, params):
        """Train CatBoost with early stopping."""
        model = cb.CatBoostRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=200,
            verbose=0,
        )
        return model

    def train_sklearn_fold(self, X_train, y_train, X_val, y_val, model_class, params):
        """Train sklearn model."""
        model = model_class(**params)
        model.fit(X_train, y_train)
        return model

    def train_knn_fold(self, X_train, y_train, X_val, y_val, n_neighbors=20):
        """Train KNN with scaled features."""
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        model = KNeighborsRegressor(n_neighbors=n_neighbors, weights="distance", n_jobs=-1)
        model.fit(X_train_scaled, y_train)
        return model, scaler

    def train_ridge_fold(self, X_train, y_train, X_val, y_val):
        """Train Ridge regression."""
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        model = BayesianRidge()
        model.fit(X_train_scaled, y_train)
        return model, scaler

    def kfold_train(self, X, y, X_test, model_name, target_name, group_col=None):
        """
        Full K-Fold training pipeline for a single model type and target.
        Returns OOF predictions and test predictions.
        """
        oof_preds = np.zeros(len(X))
        test_preds = np.zeros(len(X_test))
        fold_scores = []

        if group_col is not None and group_col in X.columns:
            kf = GroupKFold(n_splits=self.n_folds)
            groups = X[group_col]
            split_iter = kf.split(X, y, groups)
        else:
            kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
            split_iter = kf.split(X)

        X_np = X.values if isinstance(X, pd.DataFrame) else X
        X_test_np = X_test.values if isinstance(X_test, pd.DataFrame) else X_test

        for fold_idx, (train_idx, val_idx) in enumerate(split_iter):
            X_tr, X_val = X_np[train_idx], X_np[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            # Remove NaN targets
            valid_train = ~np.isnan(y_tr)
            valid_val = ~np.isnan(y_val)

            if valid_train.sum() < 10:
                continue

            X_tr_clean = X_tr[valid_train]
            y_tr_clean = y_tr[valid_train]
            X_val_clean = X_val[valid_val] if valid_val.sum() > 0 else X_val
            y_val_clean = y_val[valid_val] if valid_val.sum() > 0 else y_val

            # Handle infinite values
            X_tr_clean = np.nan_to_num(X_tr_clean, nan=0, posinf=0, neginf=0)
            X_val_clean = np.nan_to_num(X_val_clean, nan=0, posinf=0, neginf=0)
            X_test_clean = np.nan_to_num(X_test_np.copy(), nan=0, posinf=0, neginf=0)

            if model_name == "lgb":
                params = get_lgb_params(target_name)
                model = self.train_lgb_fold(X_tr_clean, y_tr_clean, X_val_clean, y_val_clean, params)
                val_pred = model.predict(X_val_clean)
                test_pred = model.predict(X_test_clean)

            elif model_name == "lgb_dart":
                params = get_lgb_dart_params(target_name)
                model = self.train_lgb_fold(X_tr_clean, y_tr_clean, X_val_clean, y_val_clean, params)
                val_pred = model.predict(X_val_clean)
                test_pred = model.predict(X_test_clean)

            elif model_name == "lgb_goss":
                params = get_lgb_goss_params(target_name)
                model = self.train_lgb_fold(X_tr_clean, y_tr_clean, X_val_clean, y_val_clean, params)
                val_pred = model.predict(X_val_clean)
                test_pred = model.predict(X_test_clean)

            elif model_name == "xgb":
                params = get_xgb_params(target_name)
                params["early_stopping_rounds"] = 200
                model = self.train_xgb_fold(X_tr_clean, y_tr_clean, X_val_clean, y_val_clean, params)
                val_pred = model.predict(X_val_clean)
                test_pred = model.predict(X_test_clean)

            elif model_name == "xgb_v2":
                params = get_xgb_v2_params(target_name)
                params["early_stopping_rounds"] = 200
                model = self.train_xgb_fold(X_tr_clean, y_tr_clean, X_val_clean, y_val_clean, params)
                val_pred = model.predict(X_val_clean)
                test_pred = model.predict(X_test_clean)

            elif model_name == "catboost":
                params = get_catboost_params(target_name)
                model = self.train_catboost_fold(X_tr_clean, y_tr_clean, X_val_clean, y_val_clean, params)
                val_pred = model.predict(X_val_clean)
                test_pred = model.predict(X_test_clean)

            elif model_name == "catboost_v2":
                params = get_catboost_v2_params(target_name)
                model = self.train_catboost_fold(X_tr_clean, y_tr_clean, X_val_clean, y_val_clean, params)
                val_pred = model.predict(X_val_clean)
                test_pred = model.predict(X_test_clean)

            elif model_name == "extratrees":
                params = get_extratrees_params(target_name)
                model = self.train_sklearn_fold(X_tr_clean, y_tr_clean, X_val_clean, y_val_clean,
                                                 ExtraTreesRegressor, params)
                val_pred = model.predict(X_val_clean)
                test_pred = model.predict(X_test_clean)

            elif model_name == "rf":
                params = get_rf_params(target_name)
                model = self.train_sklearn_fold(X_tr_clean, y_tr_clean, X_val_clean, y_val_clean,
                                                 RandomForestRegressor, params)
                val_pred = model.predict(X_val_clean)
                test_pred = model.predict(X_test_clean)

            elif model_name == "knn":
                model, scaler = self.train_knn_fold(X_tr_clean, y_tr_clean, X_val_clean, y_val_clean)
                val_pred = model.predict(scaler.transform(X_val_clean))
                test_pred = model.predict(scaler.transform(X_test_clean))

            elif model_name == "ridge":
                model, scaler = self.train_ridge_fold(X_tr_clean, y_tr_clean, X_val_clean, y_val_clean)
                val_pred = model.predict(scaler.transform(X_val_clean))
                test_pred = model.predict(scaler.transform(X_test_clean))

            else:
                raise ValueError(f"Unknown model: {model_name}")

            # Clip predictions to be non-negative (soil nutrients can't be negative)
            val_pred = np.clip(val_pred, 0, None)
            test_pred = np.clip(test_pred, 0, None)

            if valid_val.sum() > 0:
                oof_preds[val_idx[valid_val]] = val_pred[:valid_val.sum()]
                fold_rmse = np.sqrt(mean_squared_error(y_val_clean[:valid_val.sum()],
                                                        val_pred[:valid_val.sum()]))
                fold_scores.append(fold_rmse)

            test_preds += test_pred / self.n_folds

        avg_rmse = np.mean(fold_scores) if fold_scores else float("inf")
        print(f"    {model_name} | {target_name} | CV RMSE: {avg_rmse:.6f} "
              f"(folds: {[f'{s:.4f}' for s in fold_scores]})")

        return oof_preds, test_preds, avg_rmse


# ============================================================================
# STACKING ENSEMBLE - Level 2 meta-learner
# ============================================================================

class StackingEnsemble:
    """
    2-Level stacking ensemble:
    Level 1: Individual model OOF predictions
    Level 2: Meta-learner (Ridge/LightGBM) combines level 1 predictions
    """

    def __init__(self, n_folds=5, seed=SEED):
        self.n_folds = n_folds
        self.seed = seed

    def fit_predict(self, oof_dict, test_dict, y_true, target_name):
        """
        oof_dict: {model_name: oof_predictions}
        test_dict: {model_name: test_predictions}
        y_true: true target values
        """
        model_names = sorted(oof_dict.keys())

        # Build meta-features
        oof_meta = np.column_stack([oof_dict[m] for m in model_names])
        test_meta = np.column_stack([test_dict[m] for m in model_names])

        # Valid indices (non-NaN in target)
        valid = ~np.isnan(y_true)
        oof_valid = oof_meta[valid]
        y_valid = y_true[valid]

        # Meta-learner: weighted blend using Ridge
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
        meta_oof = np.zeros(len(oof_valid))
        meta_test = np.zeros(len(test_meta))

        scaler = RobustScaler()
        oof_scaled = scaler.fit_transform(oof_valid)
        test_scaled = scaler.transform(test_meta)

        for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(oof_scaled)):
            X_tr, X_val = oof_scaled[tr_idx], oof_scaled[val_idx]
            y_tr, y_val = y_valid[tr_idx], y_valid[val_idx]

            meta_model = Ridge(alpha=1.0)
            meta_model.fit(X_tr, y_tr)

            meta_oof[val_idx] = meta_model.predict(X_val)
            meta_test += meta_model.predict(test_scaled) / self.n_folds

        meta_rmse = np.sqrt(mean_squared_error(y_valid, meta_oof))
        print(f"    STACKING | {target_name} | Meta-learner CV RMSE: {meta_rmse:.6f}")

        # Also compute optimal weights via simple optimization
        from scipy.optimize import minimize

        def rmse_blend(weights):
            weights = np.abs(weights)
            weights /= weights.sum()
            blend = oof_valid @ weights
            return np.sqrt(mean_squared_error(y_valid, blend))

        n_models = len(model_names)
        init_weights = np.ones(n_models) / n_models
        result = minimize(rmse_blend, init_weights, method="Nelder-Mead",
                          options={"maxiter": 10000})
        opt_weights = np.abs(result.x)
        opt_weights /= opt_weights.sum()

        blend_pred = test_meta @ opt_weights
        blend_rmse = result.fun

        print(f"    BLEND    | {target_name} | Optimized weights RMSE: {blend_rmse:.6f}")
        print(f"             | Weights: {dict(zip(model_names, [f'{w:.3f}' for w in opt_weights]))}")

        # Return the better of stacking vs blending
        if meta_rmse < blend_rmse:
            print(f"    >>> Using STACKING (RMSE: {meta_rmse:.6f})")
            final_pred = np.clip(meta_test, 0, None)
            return final_pred, meta_rmse
        else:
            print(f"    >>> Using BLEND (RMSE: {blend_rmse:.6f})")
            final_pred = np.clip(blend_pred, 0, None)
            return final_pred, blend_rmse


# ============================================================================
# MULTI-SEED ENSEMBLE - Train same model with different seeds
# ============================================================================

def multi_seed_train(trainer, X, y, X_test, model_name, target_name, n_seeds=5):
    """Train the same model with multiple seeds and average predictions."""
    all_oof = []
    all_test = []
    all_scores = []

    original_seed = trainer.seed
    for i in range(n_seeds):
        trainer.seed = original_seed + i * 100
        oof, test, score = trainer.kfold_train(X, y, X_test, model_name, target_name)
        all_oof.append(oof)
        all_test.append(test)
        all_scores.append(score)

    trainer.seed = original_seed  # Reset

    avg_oof = np.mean(all_oof, axis=0)
    avg_test = np.mean(all_test, axis=0)
    avg_score = np.mean(all_scores)

    print(f"    MULTI-SEED {model_name} | {target_name} | Avg RMSE: {avg_score:.6f}")
    return avg_oof, avg_test, avg_score
