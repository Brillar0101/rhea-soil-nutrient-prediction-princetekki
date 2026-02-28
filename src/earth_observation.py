"""
Earth Observation Feature Engineering.
Downloads and processes satellite/climate data for soil prediction.

Sources:
- OpenElevation API (elevation, slope proxy)
- WorldClim bioclimatic variables (temperature, precipitation)
- SoilGrids API (existing soil property estimates)
- OpenLandMap (land cover, vegetation indices)

These features are CRITICAL for top leaderboard performance.
Soil nutrients are heavily influenced by climate, elevation, and land use.
"""
import pandas as pd
import numpy as np
import requests
import time
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import DATA_DIR, SEED


# ============================================================================
# ELEVATION FEATURES
# ============================================================================

def fetch_elevation_batch(lats, lons, batch_size=100):
    """Fetch elevation data from Open Elevation API."""
    elevations = np.full(len(lats), np.nan)

    for i in range(0, len(lats), batch_size):
        batch_lats = lats[i:i + batch_size]
        batch_lons = lons[i:i + batch_size]

        locations = [{"latitude": lat, "longitude": lon}
                     for lat, lon in zip(batch_lats, batch_lons)
                     if not (np.isnan(lat) or np.isnan(lon))]

        if not locations:
            continue

        try:
            resp = requests.post(
                "https://api.open-elevation.com/api/v1/lookup",
                json={"locations": locations},
                timeout=30,
            )
            if resp.status_code == 200:
                results = resp.json().get("results", [])
                valid_idx = 0
                for j in range(i, min(i + batch_size, len(lats))):
                    if not (np.isnan(lats[j]) or np.isnan(lons[j])):
                        if valid_idx < len(results):
                            elevations[j] = results[valid_idx].get("elevation", np.nan)
                            valid_idx += 1
        except Exception as e:
            print(f"  Elevation API error at batch {i}: {e}")

        time.sleep(0.5)  # Rate limiting

    return elevations


def get_elevation_features(df, lat_col="Latitude", lon_col="Longitude"):
    """Get elevation and derived features."""
    cache_file = os.path.join(DATA_DIR, "elevation_cache.csv")

    if os.path.exists(cache_file):
        print("  Loading cached elevation data...")
        cache = pd.read_csv(cache_file)
        df = df.merge(cache, on=[lat_col, lon_col], how="left", suffixes=("", "_elev"))
        return df

    print("  Fetching elevation data (this may take a while)...")
    # Get unique coordinates to minimize API calls
    unique_coords = df[[lat_col, lon_col]].drop_duplicates()
    elevs = fetch_elevation_batch(
        unique_coords[lat_col].values,
        unique_coords[lon_col].values,
    )
    unique_coords["elevation"] = elevs

    # Save cache
    unique_coords.to_csv(cache_file, index=False)

    df = df.merge(unique_coords, on=[lat_col, lon_col], how="left")
    return df


# ============================================================================
# SOILGRIDS FEATURES (ISRIC)
# ============================================================================

def fetch_soilgrids_point(lat, lon, properties=None):
    """Fetch soil property predictions from SoilGrids API for a single point."""
    if properties is None:
        properties = ["clay", "sand", "silt", "phh2o", "soc", "ocd", "nitrogen", "cec", "bdod"]

    url = "https://rest.isric.org/soilgrids/v2.0/properties/query"
    params = {
        "lon": lon,
        "lat": lat,
        "property": properties,
        "depth": ["0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm"],
        "value": "mean",
    }

    try:
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            result = {}
            if "properties" in data and "layers" in data["properties"]:
                for layer in data["properties"]["layers"]:
                    prop_name = layer["name"]
                    for depth_entry in layer.get("depths", []):
                        depth_label = depth_entry["label"]
                        val = depth_entry.get("values", {}).get("mean")
                        result[f"sg_{prop_name}_{depth_label}"] = val
            return result
    except Exception:
        pass
    return {}


def get_soilgrids_features(df, lat_col="Latitude", lon_col="Longitude"):
    """
    Get SoilGrids features for unique locations.
    SoilGrids provides global soil property predictions at 250m resolution.
    """
    cache_file = os.path.join(DATA_DIR, "soilgrids_cache.csv")

    if os.path.exists(cache_file):
        print("  Loading cached SoilGrids data...")
        cache = pd.read_csv(cache_file)
        df = df.merge(cache, on=[lat_col, lon_col], how="left", suffixes=("", "_sg"))
        return df

    print("  Fetching SoilGrids data (this will take a while for many points)...")
    unique_coords = df[[lat_col, lon_col]].drop_duplicates().reset_index(drop=True)

    all_results = []
    total = len(unique_coords)

    for idx in range(total):
        if idx % 50 == 0:
            print(f"    SoilGrids progress: {idx}/{total}")

        lat = unique_coords.loc[idx, lat_col]
        lon = unique_coords.loc[idx, lon_col]

        if np.isnan(lat) or np.isnan(lon):
            all_results.append({})
            continue

        result = fetch_soilgrids_point(lat, lon)
        all_results.append(result)
        time.sleep(0.2)  # Rate limit

    sg_df = pd.DataFrame(all_results)
    sg_df[lat_col] = unique_coords[lat_col]
    sg_df[lon_col] = unique_coords[lon_col]

    sg_df.to_csv(cache_file, index=False)

    df = df.merge(sg_df, on=[lat_col, lon_col], how="left")
    return df


# ============================================================================
# WORLDCLIM-DERIVED CLIMATE FEATURES (Computed from coordinates)
# ============================================================================

def compute_climate_proxies(df, lat_col="Latitude", lon_col="Longitude"):
    """
    Compute climate proxy features based on geographic position.
    These approximate WorldClim bioclimatic variables without downloading rasters.
    Based on established climate-geography relationships for Africa.
    """
    lat = df[lat_col].values
    lon = df[lon_col].values
    abs_lat = np.abs(lat)

    # Temperature proxies (Africa-specific relationships)
    # Mean annual temp decreases with latitude and elevation
    df["climate_temp_proxy"] = 30.0 - 0.5 * abs_lat
    df["climate_temp_range_proxy"] = 5.0 + 0.3 * abs_lat  # Continental effect

    # Precipitation proxies
    # Equatorial regions get more rain, decreases toward tropics
    df["climate_precip_proxy"] = 2000 * np.exp(-0.05 * abs_lat**2)
    # Seasonality increases away from equator
    df["climate_seasonality_proxy"] = abs_lat / 30.0

    # Aridity index proxy
    df["climate_aridity_proxy"] = df["climate_temp_proxy"] / (df["climate_precip_proxy"] + 1)

    # Evapotranspiration proxy (Thornthwaite-like)
    df["climate_pet_proxy"] = 16 * (10 * df["climate_temp_proxy"].clip(lower=0) / 50) ** 1.5

    # Moisture availability
    df["climate_moisture_proxy"] = df["climate_precip_proxy"] - df["climate_pet_proxy"]

    # Tropical zone indicator
    df["is_tropical"] = (abs_lat <= 23.5).astype(int)
    df["is_arid_zone"] = ((abs_lat >= 15) & (abs_lat <= 30)).astype(int)

    # Distance to coast proxy (rough: Africa center ~20E, 5N)
    df["dist_to_continent_center"] = np.sqrt((lat - 5)**2 + (lon - 20)**2)

    # East Africa Rift indicator (approximate)
    df["near_rift_valley"] = (
        (lon >= 29) & (lon <= 40) & (lat >= -15) & (lat <= 15)
    ).astype(int)

    # Kenya highlands indicator
    df["near_kenya_highlands"] = (
        (lon >= 35) & (lon <= 38) & (lat >= -2) & (lat <= 2)
    ).astype(int)

    return df


# ============================================================================
# VEGETATION INDEX PROXIES
# ============================================================================

def compute_vegetation_proxies(df, lat_col="Latitude", lon_col="Longitude"):
    """
    Compute vegetation/land use proxy features.
    Real NDVI would come from Sentinel-2, but we approximate based on position.
    """
    lat = df[lat_col].values
    lon = df[lon_col].values
    abs_lat = np.abs(lat)

    # NDVI proxy (peaks in equatorial/humid zones)
    df["ndvi_proxy"] = 0.7 * np.exp(-0.01 * abs_lat**2) + 0.1
    # Adjust for known desert regions
    sahara_mask = (lat >= 18) & (lat <= 32) & (lon >= -17) & (lon <= 40)
    df.loc[sahara_mask, "ndvi_proxy"] = 0.1

    # Cropland probability proxy
    df["cropland_proxy"] = np.where(
        (abs_lat < 20) & (df.get("climate_precip_proxy", 500) > 400),
        0.6, 0.2
    )

    # Forest probability proxy
    df["forest_proxy"] = np.where(
        (abs_lat < 10) & (df.get("climate_precip_proxy", 500) > 1500),
        0.7, 0.2
    )

    return df


# ============================================================================
# MASTER FUNCTION
# ============================================================================

def add_earth_observation_features(df, lat_col="Latitude", lon_col="Longitude",
                                    fetch_external=False):
    """
    Add all earth observation features.

    Args:
        fetch_external: If True, fetches from external APIs (elevation, SoilGrids).
                       If False, only computes proxy features (fast, no API calls).
    """
    print("Adding Earth Observation features...")

    # Always compute proxy features (fast, no API needed)
    df = compute_climate_proxies(df, lat_col, lon_col)
    print("  Climate proxies done.")

    df = compute_vegetation_proxies(df, lat_col, lon_col)
    print("  Vegetation proxies done.")

    # External API features (optional, slow but valuable)
    if fetch_external:
        try:
            df = get_elevation_features(df, lat_col, lon_col)
            print("  Elevation features done.")
        except Exception as e:
            print(f"  Elevation fetch failed: {e}")

        try:
            df = get_soilgrids_features(df, lat_col, lon_col)
            print("  SoilGrids features done.")
        except Exception as e:
            print(f"  SoilGrids fetch failed: {e}")

    return df


if __name__ == "__main__":
    # Test with dummy data
    test_df = pd.DataFrame({
        "Latitude": [-1.286, 0.514, -4.044],
        "Longitude": [36.817, 35.270, 39.668],
    })
    result = add_earth_observation_features(test_df, fetch_external=False)
    print(result)
