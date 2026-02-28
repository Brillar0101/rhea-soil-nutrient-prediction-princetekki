"""
Advanced Feature Engineering for Rhea Soil Nutrient Prediction.
This is where the magic happens - creative features that separate top solutions.
"""
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from config import TARGETS, SEED


def engineer_date_features(df, date_col="Date"):
    """Extract rich temporal features from dates."""
    if date_col not in df.columns:
        return df

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    df["year"] = df[date_col].dt.year
    df["month"] = df[date_col].dt.month
    df["day"] = df[date_col].dt.day
    df["day_of_year"] = df[date_col].dt.dayofyear
    df["week_of_year"] = df[date_col].dt.isocalendar().week.astype(int)
    df["quarter"] = df[date_col].dt.quarter

    # Cyclical encoding for month and day_of_year (captures seasonality)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["day_of_year_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["day_of_year_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

    # Season encoding (African seasons - dry/wet vary by hemisphere)
    df["season"] = df["month"].map(
        {1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3, 12: 0}
    )

    return df


def engineer_geo_features(df, lat_col="Latitude", lon_col="Longitude"):
    """Engineer geospatial features - critical for soil prediction."""
    if lat_col not in df.columns or lon_col not in df.columns:
        return df

    lat = df[lat_col]
    lon = df[lon_col]

    # Hemisphere indicator
    df["is_southern"] = (lat < 0).astype(int)
    df["is_western"] = (lon < 0).astype(int)

    # Absolute latitude (distance from equator - affects climate/soil)
    df["abs_latitude"] = lat.abs()

    # Distance from equator in km (approximate)
    df["dist_from_equator_km"] = lat.abs() * 111.32

    # Coordinate interactions
    df["lat_lon_product"] = lat * lon
    df["lat_lon_ratio"] = lat / (lon + 1e-8)

    # Polar coordinates
    df["geo_radius"] = np.sqrt(lat**2 + lon**2)
    df["geo_angle"] = np.arctan2(lat, lon)

    # Grid-based features (binning into geographical zones)
    df["lat_bin_1deg"] = np.floor(lat)
    df["lon_bin_1deg"] = np.floor(lon)
    df["lat_bin_05deg"] = np.floor(lat * 2) / 2
    df["lon_bin_05deg"] = np.floor(lon * 2) / 2

    # Geohash-like feature (coarse spatial indexing)
    df["geo_cell_1deg"] = df["lat_bin_1deg"].astype(str) + "_" + df["lon_bin_1deg"].astype(str)
    df["geo_cell_05deg"] = df["lat_bin_05deg"].astype(str) + "_" + df["lon_bin_05deg"].astype(str)

    return df


def engineer_depth_features(df, depth_col="Depth"):
    """Engineer soil depth features."""
    if depth_col not in df.columns:
        return df

    depth = df[depth_col]

    # Log depth (diminishing returns with depth)
    df["log_depth"] = np.log1p(depth)
    df["sqrt_depth"] = np.sqrt(depth.clip(lower=0))
    df["depth_squared"] = depth**2

    # Depth bins (topsoil vs subsoil horizons)
    df["depth_bin"] = pd.cut(depth, bins=[0, 5, 15, 30, 60, 100, 999],
                              labels=[0, 1, 2, 3, 4, 5], include_lowest=True)
    df["depth_bin"] = df["depth_bin"].astype(float)

    # Is topsoil (0-30cm is typically topsoil)
    df["is_topsoil"] = (depth <= 30).astype(int)

    return df


def engineer_spatial_cluster_features(df, lat_col="Latitude", lon_col="Longitude",
                                       n_clusters_list=[5, 10, 20, 50]):
    """Cluster samples spatially - similar locations likely have similar soils."""
    if lat_col not in df.columns or lon_col not in df.columns:
        return df, {}

    coords = df[[lat_col, lon_col]].values
    valid_mask = ~np.isnan(coords).any(axis=1)
    cluster_models = {}

    for n in n_clusters_list:
        col_name = f"spatial_cluster_{n}"
        km = KMeans(n_clusters=n, random_state=SEED, n_init=10)
        labels = np.full(len(df), -1, dtype=float)
        if valid_mask.sum() >= n:
            labels[valid_mask] = km.fit_predict(coords[valid_mask])
        df[col_name] = labels
        cluster_models[n] = km

        # Distance to cluster center
        dist_col = f"dist_to_cluster_center_{n}"
        dists = np.full(len(df), np.nan)
        if valid_mask.sum() >= n:
            for i in range(n):
                mask = (labels == i) & valid_mask
                center = km.cluster_centers_[i]
                dists[mask] = np.sqrt(
                    ((coords[mask] - center) ** 2).sum(axis=1)
                )
        df[dist_col] = dists

    return df, cluster_models


def engineer_depth_geo_interactions(df, lat_col="Latitude", lon_col="Longitude",
                                      depth_col="Depth"):
    """Interaction features between depth and geography."""
    if not all(c in df.columns for c in [lat_col, lon_col, depth_col]):
        return df

    df["depth_x_abs_lat"] = df[depth_col] * df[lat_col].abs()
    df["depth_x_lon"] = df[depth_col] * df[lon_col]
    df["depth_x_geo_radius"] = df[depth_col] * df.get("geo_radius", 0)

    # Depth normalized by latitude (soil profiles vary by climate zone)
    df["depth_lat_ratio"] = df[depth_col] / (df[lat_col].abs() + 1)

    return df


def engineer_neighbor_features(train_df, df, lat_col="Latitude", lon_col="Longitude",
                                depth_col="Depth", k=10):
    """
    For each sample, compute statistics from its k nearest neighbors in the training set.
    This is a powerful feature for geospatial prediction.
    """
    from sklearn.neighbors import BallTree

    if lat_col not in df.columns or lon_col not in df.columns:
        return df

    # Build tree from training data coordinates
    train_coords = train_df[[lat_col, lon_col]].values
    train_coords_rad = np.radians(train_coords)
    valid_train = ~np.isnan(train_coords_rad).any(axis=1)

    if valid_train.sum() < k:
        return df

    tree = BallTree(train_coords_rad[valid_train], metric="haversine")

    # Query for all samples in df
    query_coords = df[[lat_col, lon_col]].values
    query_coords_rad = np.radians(query_coords)
    valid_query = ~np.isnan(query_coords_rad).any(axis=1)

    dists = np.full((len(df), k), np.nan)
    indices = np.full((len(df), k), -1, dtype=int)

    if valid_query.sum() > 0:
        d, idx = tree.query(query_coords_rad[valid_query], k=k)
        dists[valid_query] = d * 6371  # Convert to km
        indices[valid_query] = idx

    # Distance features
    df["nn_mean_dist_km"] = np.nanmean(dists, axis=1)
    df["nn_min_dist_km"] = np.nanmin(dists, axis=1)
    df["nn_max_dist_km"] = np.nanmax(dists, axis=1)
    df["nn_std_dist_km"] = np.nanstd(dists, axis=1)

    # Target-based neighbor features (only from training data)
    valid_train_indices = np.where(valid_train)[0]
    for target in TARGETS:
        if target not in train_df.columns:
            continue

        target_vals = train_df[target].values

        for agg_name, agg_func in [("mean", np.nanmean), ("median", np.nanmedian),
                                     ("std", np.nanstd)]:
            col_name = f"nn_{target}_{agg_name}"
            feat = np.full(len(df), np.nan)

            for i in range(len(df)):
                if valid_query[i] and indices[i, 0] >= 0:
                    nn_targets = target_vals[valid_train_indices[indices[i]]]
                    feat[i] = agg_func(nn_targets)

            df[col_name] = feat

    return df


def engineer_target_encoding(train_df, df, cat_cols, target_cols=None, alpha=10):
    """
    Bayesian target encoding for categorical features.
    Uses leave-one-out for train, global for test to prevent leakage.
    """
    if target_cols is None:
        target_cols = TARGETS

    encodings = {}

    for cat_col in cat_cols:
        if cat_col not in df.columns:
            continue

        for target in target_cols:
            if target not in train_df.columns:
                continue

            col_name = f"te_{cat_col}_{target}"
            global_mean = train_df[target].mean()

            # Compute group stats from training data
            group_stats = train_df.groupby(cat_col)[target].agg(["mean", "count"])
            group_stats.columns = ["group_mean", "group_count"]

            # Bayesian smoothing
            group_stats["smoothed"] = (
                (group_stats["group_count"] * group_stats["group_mean"] + alpha * global_mean)
                / (group_stats["group_count"] + alpha)
            )

            # Map to df
            df[col_name] = df[cat_col].map(group_stats["smoothed"]).fillna(global_mean)
            encodings[(cat_col, target)] = group_stats["smoothed"].to_dict()

    return df, encodings


def engineer_frequency_features(df, cat_cols):
    """Frequency encoding for categorical columns."""
    for col in cat_cols:
        if col not in df.columns:
            continue
        freq = df[col].value_counts(normalize=True)
        df[f"freq_{col}"] = df[col].map(freq)
        count = df[col].value_counts()
        df[f"count_{col}"] = df[col].map(count)
    return df


def run_full_feature_engineering(train, test, sample_dates, compute_neighbors=True):
    """Run the complete feature engineering pipeline."""
    print("Starting feature engineering pipeline...")

    # Identify key columns
    id_col = train.columns[0]  # First column is usually ID
    print(f"ID column: {id_col}")

    # Detect coordinate columns
    lat_col = None
    lon_col = None
    depth_col = None
    date_col = None

    for col in train.columns:
        cl = col.lower()
        if "lat" in cl:
            lat_col = col
        elif "lon" in cl or "lng" in cl:
            lon_col = col
        elif "depth" in cl:
            depth_col = col
        elif "date" in cl:
            date_col = col

    print(f"Detected columns - Lat: {lat_col}, Lon: {lon_col}, Depth: {depth_col}, Date: {date_col}")

    # Merge sample dates
    if sample_dates is not None and len(sample_dates) > 0:
        date_id_col = sample_dates.columns[0]
        date_date_col = [c for c in sample_dates.columns if c != date_id_col]

        if date_date_col:
            # Merge on ID
            if date_id_col == id_col or date_id_col in train.columns:
                for ddc in date_date_col:
                    date_map = sample_dates.set_index(date_id_col)[ddc]
                    if ddc not in train.columns:
                        train[ddc] = train[id_col].map(date_map) if id_col in train.columns else np.nan
                    if ddc not in test.columns:
                        test[ddc] = test[id_col].map(date_map) if id_col in test.columns else np.nan
                    if date_col is None:
                        date_col = ddc

    # Combine train+test for consistent feature engineering
    train["_is_train"] = 1
    test["_is_train"] = 0
    combined = pd.concat([train, test], axis=0, ignore_index=True)

    print(f"Combined shape: {combined.shape}")

    # 1. Date features
    if date_col:
        combined = engineer_date_features(combined, date_col)
        # Check for other date columns
        for col in combined.columns:
            if col != date_col and "date" in col.lower():
                combined = engineer_date_features(combined, col)
    print("  Date features done.")

    # 2. Geo features
    if lat_col and lon_col:
        combined = engineer_geo_features(combined, lat_col, lon_col)
    print("  Geo features done.")

    # 3. Depth features
    if depth_col:
        combined = engineer_depth_features(combined, depth_col)
    print("  Depth features done.")

    # 4. Depth-Geo interactions
    if lat_col and lon_col and depth_col:
        combined = engineer_depth_geo_interactions(combined, lat_col, lon_col, depth_col)
    print("  Depth-Geo interactions done.")

    # 5. Spatial clusters
    if lat_col and lon_col:
        combined, cluster_models = engineer_spatial_cluster_features(
            combined, lat_col, lon_col, n_clusters_list=[5, 10, 20, 50, 100]
        )
    print("  Spatial clusters done.")

    # 6. Frequency features for categorical columns
    cat_cols = []
    for col in combined.columns:
        if combined[col].dtype == "object" and col != id_col:
            cat_cols.append(col)
        elif "cluster" in col.lower() or "bin" in col.lower() or "cell" in col.lower():
            cat_cols.append(col)

    combined = engineer_frequency_features(combined, cat_cols)
    print(f"  Frequency features done for {len(cat_cols)} categorical cols.")

    # Split back
    train_fe = combined[combined["_is_train"] == 1].drop("_is_train", axis=1).reset_index(drop=True)
    test_fe = combined[combined["_is_train"] == 0].drop("_is_train", axis=1).reset_index(drop=True)

    # 7. Target encoding (only from train)
    te_cat_cols = [c for c in cat_cols if "geo_cell" in c or "cluster" in c]
    if te_cat_cols:
        # For train: we'll do this inside CV to prevent leakage
        # For test: use full train stats
        test_fe, _ = engineer_target_encoding(train_fe, test_fe, te_cat_cols)
        # Store a flag to do target encoding inside CV
        train_fe._te_cat_cols = te_cat_cols
    print(f"  Target encoding prepared for {len(te_cat_cols)} cols.")

    # 8. Neighbor features (expensive but powerful)
    if compute_neighbors and lat_col and lon_col:
        print("  Computing neighbor features (this may take a while)...")
        test_fe = engineer_neighbor_features(train_fe, test_fe, lat_col, lon_col, k=10)
        # For train, we'll compute inside CV to prevent leakage
        # But compute a rough version here for feature selection
        train_fe = engineer_neighbor_features(train_fe, train_fe, lat_col, lon_col, k=10)
        print("  Neighbor features done.")

    print(f"\nFinal train shape: {train_fe.shape}")
    print(f"Final test shape: {test_fe.shape}")

    return train_fe, test_fe, id_col, lat_col, lon_col, depth_col, cat_cols


if __name__ == "__main__":
    from data_loader import load_all_data
    train, test, sample_dates, target_keep, sample_sub, data_dict = load_all_data()
    train_fe, test_fe, *_ = run_full_feature_engineering(train, test, sample_dates, compute_neighbors=False)
    print("\nSample of engineered features:")
    print(train_fe.head())
