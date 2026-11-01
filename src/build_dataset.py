"""Step 1. Turn the raw BC data bundle into tidy tables on a 10 km grid.

Inputs:  data/raw/bc_wildfire_bundle.json  (made by fetch_data.py)
Outputs: data/processed/grid.parquet        one row per 10 km cell, static features
         data/processed/fires.parquet       one row per fire record, with its cell
         data/processed/burned.npy          burned fraction, shape (years, cells)
         data/processed/candidates.parquet  possible drone base sites
"""
import io
import json

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.ndimage import uniform_filter
from scipy.spatial import cKDTree

from config import BUNDLE, PROCESSED, GRID, NX, NY, MIN_LAND_FRAC, FIRST_YEAR, PLANNING_YEAR

TO_ALBERS = Transformer.from_crs(4326, 3005, always_xy=True)
TO_LONLAT = Transformer.from_crs(3005, 4326, always_xy=True)


def cell_of(x, y):
    """Return (row, col, cell_id) for Albers coordinates. -1 if outside grid."""
    col = np.floor((x - GRID["x0"]) / GRID["cell_m"]).astype(int)
    row = np.floor((GRID["y1"] - y) / GRID["cell_m"]).astype(int)
    ok = (col >= 0) & (col < NX) & (row >= 0) & (row < NY)
    cid = np.where(ok, row * NX + col, -1)
    return row, col, cid


def sparse_to_dense(pairs, n=NX * NY):
    a = np.zeros(n, dtype=np.float32)
    for i, v in pairs:
        a[i] = v
    return a


def parse_wkt_points(series):
    xy = series.str.extract(r"POINT \(([-\d\.]+) ([-\d\.]+)\)").astype(float)
    return xy[0].to_numpy(), xy[1].to_numpy()


def main():
    b = json.loads(BUNDLE.read_text())
    zones = b["meta"]["zones"]
    PROCESSED.mkdir(parents=True, exist_ok=True)

    # ---- grid skeleton ----------------------------------------------------
    rows, cols = np.divmod(np.arange(NX * NY), NX)
    cx = GRID["x0"] + (cols + 0.5) * GRID["cell_m"]
    cy = GRID["y1"] - (rows + 0.5) * GRID["cell_m"]
    lon, lat = TO_LONLAT.transform(cx, cy)
    grid = pd.DataFrame(dict(cell_id=np.arange(NX * NY), row=rows, col=cols,
                             x=cx, y=cy, lon=lon, lat=lat))

    # ---- biogeoclimatic zones: fraction of each cell in each zone ----------
    bec = np.zeros((NX * NY, len(zones)), dtype=np.float32)
    for cid, pairs in b["bec"]:
        for z, v in pairs:
            bec[cid, z] = v
    for z, name in enumerate(zones):
        grid[f"bec_{name}"] = bec[:, z]
    grid["land_frac"] = bec.sum(axis=1).clip(0, 1)
    grid["is_land"] = grid["land_frac"] >= MIN_LAND_FRAC
    grid["bec_major"] = np.where(grid["is_land"], np.array(zones)[bec.argmax(axis=1)], "")

    # ---- roads: share of 1 km pixels in the cell that contain a road -------
    roads = sparse_to_dense(b["roads_dra"]).reshape(NY, NX)
    grid["road_frac"] = roads.ravel()
    grid["road_frac_50km"] = uniform_filter(roads, size=5, mode="constant").ravel()

    # ---- distance to nearest town and its population -----------------------
    cities = pd.read_csv(io.StringIO(b["cities_csv"]))
    cities["x"], cities["y"] = parse_wkt_points(cities["GEOMETRY"])
    tree = cKDTree(cities[["x", "y"]].to_numpy())
    d, idx = tree.query(np.c_[cx, cy])
    grid["dist_town_km"] = d / 1000
    grid["log_town_pop"] = np.log1p(cities["POP_2000"].to_numpy()[idx])

    # ---- burned area by fire year (fuel already consumed) ------------------
    years = sorted(int(y) for y in b["burned"])
    burned = np.stack([sparse_to_dense(b["burned"][str(y)]) for y in years]).clip(0, 1)
    np.save(PROCESSED / "burned.npy", burned)
    (PROCESSED / "burned_years.json").write_text(json.dumps(years))

    grid.to_parquet(PROCESSED / "grid.parquet", index=False)

    # ---- fire records -----------------------------------------------------
    f = pd.read_csv(io.StringIO(b["fires_csv"]))
    f = f[f["FIRE_YEAR"] >= FIRST_YEAR].copy()
    f["x"], f["y"] = TO_ALBERS.transform(f["LONGITUDE"].to_numpy(), f["LATITUDE"].to_numpy())
    f["row"], f["col"], f["cell_id"] = cell_of(f["x"].to_numpy(), f["y"].to_numpy())
    f["date"] = pd.to_datetime(f["IGNITION_DATE"], errors="coerce")
    f["is_wildfire"] = f["FIRE_TYPE"].eq("Fire")
    # calls a detection drone would also be sent to: false alarms and small
    # nuisance fires still need someone to go and look
    f["is_call"] = f["FIRE_TYPE"].isin(["Fire", "Nuisance Fire", "Smoke Chase"])
    keep = ["FIRE_NUMBER", "FIRE_YEAR", "FIRE_TYPE", "FIRE_CAUSE", "RESPONSE_TYPE_DESC",
            "date", "LATITUDE", "LONGITUDE", "CURRENT_SIZE", "FIRE_CENTRE",
            "x", "y", "row", "col", "cell_id", "is_wildfire", "is_call"]
    f = f[keep].rename(columns=str.lower)
    f.to_parquet(PROCESSED / "fires.parquet", index=False)

    # ---- candidate base sites -------------------------------------------
    # BC Wildfire Service weather stations already have power, a cleared site
    # and a data link, which is most of what a drone dock needs. Towns are
    # added as a second option.
    st = pd.read_csv(io.StringIO(b["stations_csv"]))
    # a planner at the end of 2021 only knows about stations that existed then
    installed = pd.to_datetime(st["INSTALL_DATE"], errors="coerce")
    n_all = len(st)
    st = st[installed < f"{PLANNING_YEAR}-01-01"].copy()
    print(f"weather stations: {n_all}, of which {len(st)} were installed before {PLANNING_YEAR}")
    st["x"], st["y"] = parse_wkt_points(st["SHAPE"])
    st = pd.DataFrame(dict(name=st["STATION_NAME"].str.title(), kind="weather station",
                           x=st["x"], y=st["y"], lon=st["LONGITUDE"], lat=st["LATITUDE"]))
    ct = pd.DataFrame(dict(name=cities["NAME"], kind="town", x=cities["x"], y=cities["y"]))
    ct["lon"], ct["lat"] = TO_LONLAT.transform(ct["x"].to_numpy(), ct["y"].to_numpy())
    ct["pop"] = cities["POP_2000"]
    cand = pd.concat([st, ct], ignore_index=True)
    _, _, cid = cell_of(cand["x"].to_numpy(), cand["y"].to_numpy())
    cand = cand[(cid >= 0)].reset_index(drop=True)
    cand["cand_id"] = np.arange(len(cand))
    cand.to_parquet(PROCESSED / "candidates.parquet", index=False)

    # ---- summary ----------------------------------------------------------
    wf = f[f["is_wildfire"]]
    print(f"grid: {NX}x{NY} cells, {int(grid['is_land'].sum())} on land")
    print(f"fire records since {FIRST_YEAR}: {len(f):,}; wildfires: {len(wf):,}; "
          f"drone calls: {int(f['is_call'].sum()):,}")
    print(f"wildfires missing a date: {int(wf['date'].isna().sum()):,}; "
          f"outside grid: {int((wf['cell_id'] < 0).sum())}")
    print(f"candidate sites: {len(cand)} ({(cand.kind == 'weather station').sum()} stations)")
    print("burned km2 by year:", {y: int(burned[i].sum() * 100) for i, y in enumerate(years)})


if __name__ == "__main__":
    main()
