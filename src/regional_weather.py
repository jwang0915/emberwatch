"""Step 2e. Use weather where it actually helps: regional monthly totals.

monthly_model.py showed that adding weather to the cell-by-cell model made it
slightly worse. With only a few dozen training months, the tree model learns
noise from the weather columns. A simpler model does better:

  1. For each of the six BC Wildfire Service fire centres and each month from
     May to September, predict the month's total wildfires with a Poisson
     regression:

        log(expected fires) = usual level for this centre and month
                              + b1 x last month's precipitation anomaly
                              + b2 x last month's temperature anomaly
                              + b3 x last month's root-zone soil wetness anomaly
                              + b4 x precipitation anomaly two months back
                              + b5 x fires so far this season versus usual

     Fitted on 2001-2021, so it has 630 centre-months to learn five weather
     numbers from.

  2. Share each centre's predicted total out to its 10 km cells using the
     history-only monthly model, M0, from monthly_model.py.

  3. Re-site 30 movable drone bases on the 1st of each month using this
     forecast and score them on that month's real fires.

Outputs: results/regional_weather_metrics.json
"""
import json
import warnings

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree
from sklearn.linear_model import PoissonRegressor

from config import RAW, PROCESSED, RESULTS, coverage_radius_km
from placement import solve_mclp, coverage_sets

warnings.simplefilter("ignore")
FIRE_CENTRES = {2: "Coastal", 3: "Northwest", 4: "Prince George", 5: "Kamloops", 6: "Southeast", 7: "Cariboo"}
MONTHS = [5, 6, 7, 8, 9]
K = 30
TERMS = {"climatology only": [],
         "weather": ["pr_a1", "tx_a1", "soil_a1", "pr_a2"],
         "weather and fires so far": ["pr_a1", "tx_a1", "soil_a1", "pr_a2", "sofar_a"]}


def centre_month_table(grid, wf):
    """One row per fire centre, year and month, with weather anomalies and fire counts."""
    raw = json.loads((RAW / "nasa_power_monthly.json").read_text())
    pts = np.array(raw["points"])
    px, py = Transformer.from_crs(4326, 3005, always_xy=True).transform(pts[:, 1], pts[:, 0])
    cc = wf.groupby("cell_id")["fire_centre"].agg(lambda s: s.mode().iat[0]).map(FIRE_CENTRES)
    land = grid[grid["is_land"]].copy()
    land["centre"] = land["cell_id"].map(cc)
    land = land.dropna(subset=["centre"])
    _, idx = cKDTree(land[["x", "y"]].to_numpy()).query(np.c_[px, py])
    point_centre = land["centre"].to_numpy()[idx]

    rows = []
    for p, rec in enumerate(raw["data"]):
        for y in range(1999, 2026):
            for m in range(1, 13):
                k = f"{y}{m:02d}"
                rows.append((point_centre[p], y, m, rec["PRECTOTCORR"][k], rec["T2M_MAX"][k], rec["GWETROOT"][k]))
    w = (pd.DataFrame(rows, columns=["centre", "y", "m", "pr", "tx", "soil"]).replace(-999, np.nan)
         .groupby(["centre", "y", "m"]).mean().reset_index())
    normal = w[w["y"] <= 2019].groupby(["centre", "m"])[["pr", "tx", "soil"]].mean().add_suffix("_n")
    w = w.merge(normal, on=["centre", "m"])
    w["pr_a"] = w["pr"] / w["pr_n"] - 1
    w["tx_a"] = w["tx"] - w["tx_n"]
    w["soil_a"] = w["soil"] - w["soil_n"]
    w = w.sort_values(["centre", "y", "m"])
    for c in ("pr_a", "tx_a", "soil_a"):
        w[c + "1"] = w.groupby("centre")[c].shift(1)
        w[c + "2"] = w.groupby("centre")[c].shift(2)

    cnt = (wf.assign(centre=wf["fire_centre"].map(FIRE_CENTRES), y=wf["date"].dt.year, m=wf["date"].dt.month)
           .groupby(["centre", "y", "m"]).size().rename("n").reset_index())
    d = w.merge(cnt, on=["centre", "y", "m"], how="left").fillna({"n": 0})
    d["cum"] = d[d["m"] >= 4].groupby(["centre", "y"])["n"].cumsum()
    d["sofar"] = d.groupby(["centre", "y"])["cum"].shift(1).fillna(0)
    d = d[d["m"].isin(MONTHS)].copy()
    usual = d[d["y"].between(2001, 2021)].groupby(["centre", "m"])["sofar"].mean().rename("sofar_n")
    d = d.join(usual, on=["centre", "m"])
    d["sofar_a"] = np.log1p(d["sofar"]) - np.log1p(d["sofar_n"])
    d["key"] = d["centre"] + "_" + d["m"].astype(str)
    return d


def design(df, terms, keys):
    X = pd.get_dummies(df["key"]).astype(float).reindex(columns=keys, fill_value=0)
    for c in terms:
        X[c] = df[c].fillna(0).to_numpy()
    return X


def main():
    grid = pd.read_parquet(PROCESSED / "grid.parquet")
    fires = pd.read_parquet(PROCESSED / "fires.parquet")
    wf = fires[fires["is_wildfire"] & fires["cell_id"].ge(0) & fires["date"].notna()]
    d = centre_month_table(grid, wf)
    tr, te = d[d["y"].between(2001, 2021)], d[d["y"].between(2022, 2025)].copy()
    keys = sorted(d["key"].unique())

    out = {"regional": {}}
    for name, terms in TERMS.items():
        mdl = PoissonRegressor(alpha=1e-4, max_iter=5000).fit(design(tr, terms, keys), tr["n"])
        te[name] = mdl.predict(design(te, terms, keys))
        err = {int(y): float(100 * (g[name] - g["n"]).abs().sum() / g["n"].sum()) for y, g in te.groupby("y")}
        out["regional"][name] = {
            "centre_month_error_pct": err,
            "mean_error_pct": float(np.mean(list(err.values()))),
            "totals": {int(y): [round(float(g[name].sum())), int(g["n"].sum())] for y, g in te.groupby("y")},
            "coefficients": dict(zip(terms, map(float, mdl.coef_[len(keys):]))),
            "2023_by_centre": {c: [round(float(g[name].sum())), int(g["n"].sum())]
                               for c, g in te[te["y"] == 2023].groupby("centre")},
        }
        print(f"{name:26s} error by year {({y: round(v, 1) for y, v in err.items()})} "
              f"mean {np.mean(list(err.values())):.1f}", flush=True)

    # combine: regional totals shared out to cells by the history-only monthly model
    cellp = pd.read_parquet(RESULTS / "monthly_test_predictions.parquet")
    best = "weather and fires so far"
    tot = te.set_index(["centre", "y", "m"])[best]
    m0_tot = cellp.groupby(["centre", "year", "month"])["M0"].transform("sum")
    target = [tot.get((c, y, m), np.nan) for c, y, m in zip(cellp["centre"], cellp["year"], cellp["month"])]
    scale = np.where(np.isnan(target), 1.0, np.array(target) / np.maximum(m0_tot, 1e-9))
    cellp["regional"] = cellp["M0"] * scale

    # movable bases
    R_KM = coverage_radius_km()
    cand = pd.read_parquet(PROCESSED / "candidates.parquet")
    cxy = cand[["x", "y"]].to_numpy()
    land = grid[grid["is_land"]]
    near = coverage_sets(cxy, land[["x", "y"]].to_numpy(), R_KM * 1000)
    fixed = json.loads((RESULTS / "placements.json").read_text())["model"][str(K)]
    prev = json.loads((RESULTS / "monthly_metrics.json").read_text())["movable"]["per_month"]
    reach = {"fixed": 0, "M0": 0, "regional": 0, "hindsight": 0}
    total, per_year = 0, {}
    for (Y, m), idx in cellp.groupby(["year", "month"]).indices.items():
        fm = wf[(wf["date"].dt.year == Y) & (wf["date"].dt.month == m)]
        fxy = fm[["x", "y"]].to_numpy()
        ch, _ = solve_mclp(cellp["regional"].to_numpy()[idx], near, len(cand), K)
        hit = int((cKDTree(cxy[ch]).query(fxy)[0] / 1000 <= R_KM).sum())
        old = next(r for r in prev[str(Y)] if r["month"] == m)
        row = {"fires": len(fm), "fixed": old["fixed"], "M0": old["M0"], "regional": hit, "hindsight": old["hindsight"]}
        per_year.setdefault(str(Y), []).append({"month": int(m), **row})
        total += len(fm)
        for k in reach:
            reach[k] += row[k]
        print(Y, m, row, flush=True)
    out["movable"] = {"share_reached": {k: v / total for k, v in reach.items()},
                      "per_year_share": {Y: {k: sum(r[k] for r in rows) / sum(r["fires"] for r in rows) for k in reach}
                                         for Y, rows in per_year.items()},
                      "per_month": per_year}
    (RESULTS / "regional_weather_metrics.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: round(v, 3) for k, v in out["movable"]["share_reached"].items()}))
    print({Y: {k: round(v, 3) for k, v in r.items()} for Y, r in out["movable"]["per_year_share"].items()})


if __name__ == "__main__":
    main()
