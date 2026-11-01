"""Step 5. Package results for the interactive map (map/index.html).

Reads everything the earlier steps wrote and bundles it into one JSON object
that is embedded inside map/template.html. The result is a single HTML file
that opens in any browser with no server.
"""
import json
from datetime import date

import geopandas as gpd
import numpy as np
import pandas as pd

from config import ROOT, RAW, PROCESSED, RESULTS, MAP_DIR, GRID, NX, NY, DRONE, TEST_YEARS, coverage_radius_km
from risk_model import LARGE_HA

X0, Y1 = GRID["x0"], GRID["y1"]


def to_uv(x, y):
    """Albers metres -> map units in km, origin at the grid's north-west corner."""
    return (np.asarray(x) - X0) / 1000, (Y1 - np.asarray(y)) / 1000


def outline_path():
    # BC outline from Natural Earth admin-1 boundaries, public domain
    bc = gpd.read_file(RAW / "bc_outline.geojson").to_crs(3005)
    geom = bc.geometry.iloc[0].simplify(1200)
    parts = []
    polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
    for p in polys:
        if p.area < 30e6:           # drop islets under 30 km2
            continue
        for ring in [p.exterior, *p.interiors]:
            u, v = to_uv(*np.array(ring.coords).T)
            parts.append("M" + "L".join(f"{a:.1f} {b:.1f}" for a, b in zip(u, v)) + "Z")
    return "".join(parts)


def r(v, n=4):
    return [round(float(x), n) for x in v]


def main():
    grid = pd.read_parquet(PROCESSED / "grid.parquet")
    surf = pd.read_parquet(PROCESSED / "risk_surface.parquet")
    cand = pd.read_parquet(PROCESSED / "candidates.parquet")
    fires = pd.read_parquet(PROCESSED / "fires.parquet")
    placements = json.loads((RESULTS / "placements.json").read_text())
    pres = pd.read_csv(RESULTS / "placement_results.csv")
    sim = pd.read_csv(RESULTS / "simulation_summary.csv")
    busiest = json.loads((RESULTS / "simulation_busiest_day.json").read_text())
    metrics = json.loads((RESULTS / "risk_metrics.json").read_text())
    imp = pd.read_csv(RESULTS / "feature_importance_all.csv")

    zones = sorted(z for z in grid["bec_major"].unique() if z)
    land = grid["is_land"].to_numpy()
    g = grid[land].merge(surf, on=["cell_id", "is_land"])
    cells = dict(id=g["cell_id"].astype(int).tolist(),
                 rm=r(g["risk_model"]), rh=r(g["risk_hist"]), rl=r(g["risk_model_large"], 5),
                 z=[zones.index(z) for z in g["bec_major"]])

    t = fires[fires["is_wildfire"] & fires["fire_year"].isin(TEST_YEARS)]
    fu, fv = to_uv(t["x"], t["y"])
    fire_data = dict(u=r(fu, 1), v=r(fv, 1), yr=t["fire_year"].astype(int).tolist(),
                     L=t["current_size"].ge(LARGE_HA).astype(int).tolist())

    cu, cv = to_uv(cand["x"], cand["y"])
    cand_data = [dict(n=n, k="station" if k == "weather station" else "town", u=round(float(a), 1), v=round(float(b), 1))
                 for n, k, a, b in zip(cand["name"], cand["kind"], cu, cv)]

    replay = {}
    for m, calls in busiest["replays"].items():
        rows = []
        for c in calls:
            u, v = to_uv(c["x"], c["y"])
            rows.append(dict(u=round(float(u), 1), v=round(float(v), 1), c=c["fire_cause"][0],
                             t=round(c["minute"], 1), s=c["status"],
                             fe=None if c["first_eyes"] is None else round(c["first_eyes"], 1),
                             w=round(c["waited"], 1), b=c["base"]))
        replay[m] = rows

    keep = ["strategy", "k", "covered_share", "covered_share_large", "median_distance_km"]
    sim_cols = ["scenario", "bases", "drones_per_base", "static_coverage", "within_target",
                "within_target_p5", "within_target_p95", "out_of_range", "grounded", "drones_busy",
                "median_first_eyes_min", "busiest_day_within_target", "busiest_day_calls",
                "flight_hours_per_drone_per_season"]
    data = dict(
        meta=dict(generated=str(date.today()), nx=NX, ny=NY, cell_km=GRID["cell_m"] / 1000,
                  width_km=(GRID["x1"] - GRID["x0"]) / 1000, height_km=(GRID["y1"] - GRID["y0"]) / 1000,
                  drone=DRONE, radius_km=coverage_radius_km(), test_years=TEST_YEARS,
                  large_ha=LARGE_HA, n_test_fires=len(t), n_test_large=int(t["current_size"].ge(LARGE_HA).sum())),
        outline=outline_path(), zones=zones, cells=cells, fires=fire_data, cand=cand_data,
        placements=placements,
        results=json.loads(pres[keep].round(4).to_json(orient="records")),
        sim=json.loads(sim[sim_cols].round(4).to_json(orient="records")),
        replay=dict(date=busiest["date"], bases=busiest["base_ids"], runs=replay),
        risk=dict(all=metrics["all"]["pooled"], large=metrics["large"]["pooled"],
                  plan_all=metrics["all"]["planning_surface_vs_2022_2025"],
                  plan_large=metrics["large"]["planning_surface_vs_2022_2025"],
                  importance=json.loads(imp.head(10)[["feature", "share"]].round(4).to_json(orient="records"))),
    )
    blob = json.dumps(data, separators=(",", ":"))
    MAP_DIR.mkdir(exist_ok=True)
    (MAP_DIR / "data.json").write_text(blob)

    tpl = (MAP_DIR / "template.html").read_text()
    body = tpl.replace("/*__DATA__*/null", blob)
    # artifact.html: page body only (the Claude artifact host adds <html>/<head>)
    (MAP_DIR / "artifact.html").write_text(body)
    # index.html: a complete file you can double-click to open locally
    head, rest = body.split("<!--/head-->", 1)
    (ROOT / "docs").mkdir(exist_ok=True)
    page = (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        + head + '</head>\n<body style="margin:0">\n' + rest + '\n</body>\n</html>\n')
    (MAP_DIR / "index.html").write_text(page)
    (ROOT / "docs" / "index.html").write_text(page)  # the GitHub Pages site
    print(f"map data: {len(blob) / 1e6:.2f} MB; cells {len(cells['id'])}, fires {len(t)}")


if __name__ == "__main__":
    main()
