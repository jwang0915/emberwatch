"""Download monthly weather for 479 points across BC from NASA POWER.

    python fetch_weather.py

Writes data/raw/nasa_power_monthly.json, about 12 MB. Takes 5 to 10 minutes.
Needs data/processed/grid.parquet, so run build_dataset.py first.

NASA POWER, power.larc.nasa.gov, gives free monthly values for any point on
Earth. Its weather comes from the MERRA-2 reanalysis, about 50 km across,
which is why the points are about 50 km apart. No account is needed.

The prototype's copy was fetched with the same requests from a browser.
"""
import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
from pyproj import Transformer

from config import RAW, PROCESSED

PARAMS = "T2M_MAX,PRECTOTCORR,GWETROOT,GWETTOP,SNODP"
URL = ("https://power.larc.nasa.gov/api/temporal/monthly/point?parameters={p}&community=AG"
       "&longitude={lon}&latitude={lat}&start=1999&end=2025&format=JSON")


def points():
    """One point per 50 km block of land cells, at the mean centre of its land."""
    g = pd.read_parquet(PROCESSED / "grid.parquet")
    land = g[g["is_land"]].copy()
    land["bx"], land["by"] = land["col"] // 5, land["row"] // 5
    pts = land.groupby(["bx", "by"])[["x", "y"]].mean().reset_index()
    lon, lat = Transformer.from_crs(3005, 4326, always_xy=True).transform(pts["x"].to_numpy(), pts["y"].to_numpy())
    return [[round(float(a), 3), round(float(b), 3)] for a, b in zip(lat, lon)]


def main():
    pts = points()
    data = []
    for i, (lat, lon) in enumerate(pts):
        for attempt in range(4):
            try:
                r = requests.get(URL.format(p=PARAMS, lat=lat, lon=lon), timeout=120)
                r.raise_for_status()
                data.append(r.json()["properties"]["parameter"])
                break
            except Exception as e:
                if attempt == 3:
                    raise
                time.sleep(2 * (attempt + 1))
        if i % 50 == 0:
            print(f"{i}/{len(pts)}", flush=True)
    out = dict(meta=dict(source="NASA POWER monthly point API, power.larc.nasa.gov, MERRA-2 based",
                         fetched=datetime.now(timezone.utc).isoformat(), start=1999, end=2025,
                         parameters=PARAMS.split(",")),
               points=pts, data=data)
    (RAW / "nasa_power_monthly.json").write_text(json.dumps(out))
    print("wrote", RAW / "nasa_power_monthly.json")


if __name__ == "__main__":
    main()
