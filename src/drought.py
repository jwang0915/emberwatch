"""Step 2b. Turn monthly weather into drought features for each cell and season.

Input:  data/raw/nasa_power_monthly.json
        Monthly NASA POWER values, 1999 to 2025, for 479 points about 50 km
        apart across BC. POWER's weather comes from the MERRA-2 reanalysis.
        Variables:
          T2M_MAX      mean daily maximum temperature, C
          PRECTOTCORR  precipitation, mm per day
          GWETROOT     root-zone soil wetness, 0 is bone dry and 1 is saturated
          GWETTOP      topsoil wetness, same scale
          SNODP        snow depth, cm

Output: data/processed/drought_features.parquet, one row per land cell per
        fire season, with two groups of columns:

  known by 31 December of the year before the season  (prefix pre_)
      pre_soil_fall      root-zone wetness in Sep and Oct of the year before
      pre_precip_summer  precipitation in Jun to Aug of the year before
      pre_precip_12m     precipitation over the 12 months to December
      pre_tmax_summer    maximum temperature in Jun to Aug of the year before

  known by 1 May of the season itself  (prefix may_)
      may_soil_apr       root-zone wetness in April
      may_topsoil_apr    topsoil wetness in April
      may_snow_mar       snow depth in March
      may_precip_winter  precipitation from October to April
      may_precip_24m     precipitation over the 24 months to April
      may_tmax_spring    maximum temperature in March and April

Every value is an anomaly: the difference from that point's own normal for
the same months. Normals use 1999 to 2019 only, so the test years 2022 to 2025
never leak into them. Anomalies matter because a wet coast and a dry interior
have very different raw numbers, and what drives an unusual fire year is being
drier than normal for that place.

Each 10 km cell takes an inverse-distance average of its 4 nearest points.
Province-wide averages of each feature are added too, prefix bc_, so the
model can learn how bad the whole season looks.
"""
import json

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from config import RAW, PROCESSED

NORMAL_YEARS = range(1999, 2020)
SEASONS = range(2001, 2026)
VARS = ["T2M_MAX", "PRECTOTCORR", "GWETROOT", "GWETTOP", "SNODP"]


def load_monthly():
    """Array (points, variables, years 1999-2025, months 1-12)."""
    raw = json.loads((RAW / "nasa_power_monthly.json").read_text())
    pts = np.array(raw["points"])
    years = list(range(1999, 2026))
    arr = np.full((len(pts), len(VARS), len(years), 12), np.nan)
    for p, rec in enumerate(raw["data"]):
        for v, name in enumerate(VARS):
            for key, val in rec[name].items():
                y, m = int(key[:4]), int(key[4:])
                if m <= 12 and val != -999 and y in years:
                    arr[p, v, years.index(y), m - 1] = val
    days = np.array([31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    arr[:, VARS.index("PRECTOTCORR")] *= days  # mm per day -> mm per month
    return pts, years, arr


def window(arr, years, var, season, months):
    """Mean over a list of (year offset, month) pairs, relative to `season`."""
    v = VARS.index(var)
    vals = [arr[:, v, years.index(season + dy), m - 1] for dy, m in months]
    with np.errstate(all="ignore"):
        import warnings
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(np.stack(vals), axis=0)


def total(arr, years, var, season, months):
    return window(arr, years, var, season, months) * len(months)


# (year offset from the season, month)
FALL_BEFORE = [(-1, 9), (-1, 10)]
SUMMER_BEFORE = [(-1, 6), (-1, 7), (-1, 8)]
LAST_12 = [(-1, m) for m in range(1, 13)]
APRIL = [(0, 4)]
MARCH = [(0, 3)]
WINTER = [(-1, 10), (-1, 11), (-1, 12), (0, 1), (0, 2), (0, 3), (0, 4)]
LAST_24 = [(-2, m) for m in range(5, 13)] + [(-1, m) for m in range(1, 13)] + [(0, m) for m in range(1, 5)]
SPRING = [(0, 3), (0, 4)]

FEATURES = {
    "pre_soil_fall":     ("GWETROOT", FALL_BEFORE, window),
    "pre_precip_summer": ("PRECTOTCORR", SUMMER_BEFORE, total),
    "pre_precip_12m":    ("PRECTOTCORR", LAST_12, total),
    "pre_tmax_summer":   ("T2M_MAX", SUMMER_BEFORE, window),
    "may_soil_apr":      ("GWETROOT", APRIL, window),
    "may_topsoil_apr":   ("GWETTOP", APRIL, window),
    "may_snow_mar":      ("SNODP", MARCH, window),
    "may_precip_winter": ("PRECTOTCORR", WINTER, total),
    "may_precip_24m":    ("PRECTOTCORR", LAST_24, total),
    "may_tmax_spring":   ("T2M_MAX", SPRING, window),
}


def main():
    pts, years, arr = load_monthly()
    grid = pd.read_parquet(PROCESSED / "grid.parquet")
    land = grid[grid["is_land"]]

    # raw value of every feature for every point and season
    raw = {name: np.stack([fn(arr, years, var, s, months) for s in SEASONS])
           for name, (var, months, fn) in FEATURES.items()}          # (seasons, points)

    # anomalies against 1999-2019 normals; precipitation as a percent of normal
    normal_idx = [i for i, s in enumerate(SEASONS) if s in NORMAL_YEARS]
    anom = {}
    for name, vals in raw.items():
        with np.errstate(all="ignore"):
            import warnings
            warnings.simplefilter("ignore", RuntimeWarning)
            normal = np.nanmean(vals[normal_idx], axis=0)
        if "precip" in name:
            anom[name] = 100 * (vals / np.maximum(normal, 1) - 1)
        elif "snow" in name:
            anom[name] = vals - normal
        else:
            anom[name] = vals - normal

    # inverse-distance weights from the 4 nearest points to each land cell
    from pyproj import Transformer
    to_albers = Transformer.from_crs(4326, 3005, always_xy=True)
    px, py = to_albers.transform(pts[:, 1], pts[:, 0])
    d, idx = cKDTree(np.c_[px, py]).query(land[["x", "y"]].to_numpy(), k=4)
    w = 1 / np.maximum(d, 1000) ** 2
    w /= w.sum(axis=1, keepdims=True)

    rows = []
    for si, s in enumerate(SEASONS):
        df = pd.DataFrame({"cell_id": land["cell_id"].to_numpy(), "fire_year": s})
        for name, vals in anom.items():
            v = vals[si][idx]                      # (cells, 4 nearest points)
            ok = ~np.isnan(v)                      # some coastal points have no soil or snow data
            ww = np.where(ok, w, 0)
            df[name] = (np.where(ok, v, 0) * ww).sum(axis=1) / np.maximum(ww.sum(axis=1), 1e-12)
            df.loc[~ok.any(axis=1), name] = np.nan
            df["bc_" + name] = float(np.nanmean(vals[si]))
        rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    out.to_parquet(PROCESSED / "drought_features.parquet", index=False)

    bc = out.groupby("fire_year")[[c for c in out.columns if c.startswith("bc_")]].first()
    print("province-wide anomalies by season:")
    print(bc[["bc_pre_soil_fall", "bc_may_soil_apr", "bc_may_precip_winter",
              "bc_may_snow_mar", "bc_may_tmax_spring"]].round(2).to_string())


if __name__ == "__main__":
    main()
