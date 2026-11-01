"""Step 2d. A month-ahead forecast that uses weather, for drones that can move.

What risk_model_drought.py found: conditions known by 1 May barely predict how bad a
BC fire season will be. Summer weather decides it. So this step forecasts one
month at a time instead. On the 1st of each month from May to September it
predicts how many wildfires will start in each 10 km cell that month, using
only what is known on that day:

  history   the same fire-history features as the yearly model, plus the
            cell's usual share of fires in this calendar month
  so far    fires that have already started this season, in the cell, within
            about 30 km, and across the province
  weather   last month's and the month before's anomalies in precipitation,
            maximum temperature and root-zone soil wetness, and precipitation
            since April, all from NASA POWER, as in drought.py

Three versions are trained to see what each part adds:
  M0 history only, M1 history + so far, M2 history + so far + weather.

Then 30 movable drone bases are re-sited on the 1st of each month using each
forecast, and scored on that month's real fires. This is compared with the
fixed 30-base network chosen at the end of 2021.

Train on 2010-2019, tune on 2020-2021, test on 2022-2025, as elsewhere.
Outputs: results/monthly_metrics.json, results/feature_importance_monthly.csv
"""
import json
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

from config import (RAW, PROCESSED, RESULTS, NX, NY, FIRST_YEAR, TRAIN_YEARS, VALID_YEARS,
                    TEST_YEARS, SEED, coverage_radius_km)
from risk_model import count_panel, features_for_year, poisson_dev, capture, LARGE_HA
from placement import solve_mclp, coverage_sets

MONTHS = [5, 6, 7, 8, 9]
K = 30
FIRE_CENTRES = {2: "Coastal", 3: "Northwest", 4: "Prince George", 5: "Kamloops", 6: "Southeast", 7: "Cariboo"}
warnings.simplefilter("ignore", RuntimeWarning)


def blur(v, s):
    return gaussian_filter(v.reshape(NY, NX), sigma=s, mode="constant").ravel()


def monthly_weather(land):
    """Dict (year, month) -> DataFrame of anomalies for every land cell."""
    raw = json.loads((RAW / "nasa_power_monthly.json").read_text())
    pts = np.array(raw["points"])
    names = {"PRECTOTCORR": "pr", "T2M_MAX": "tx", "GWETROOT": "soil"}
    years = range(1999, 2026)
    arr = {v: np.full((len(pts), len(years), 12), np.nan) for v in names.values()}
    for p, rec in enumerate(raw["data"]):
        for src, v in names.items():
            for key, val in rec[src].items():
                y, m = int(key[:4]), int(key[4:])
                if m <= 12 and val != -999:
                    arr[v][p, y - 1999, m - 1] = val
    normal = {v: np.nanmean(a[:, :21, :], axis=1) for v, a in arr.items()}      # 1999-2019
    anom = {"pr": 100 * (arr["pr"] / np.maximum(normal["pr"][:, None, :], 0.05) - 1),
            "tx": arr["tx"] - normal["tx"][:, None, :],
            "soil": arr["soil"] - normal["soil"][:, None, :]}
    to_albers = Transformer.from_crs(4326, 3005, always_xy=True)
    px, py = to_albers.transform(pts[:, 1], pts[:, 0])
    d, idx = cKDTree(np.c_[px, py]).query(land[["x", "y"]].to_numpy(), k=4)
    w = 1 / np.maximum(d, 1000) ** 2

    def to_cells(vals):
        v = vals[idx]
        ok = ~np.isnan(v)
        ww = np.where(ok, w, 0)
        out = (np.where(ok, v, 0) * ww).sum(axis=1) / np.maximum(ww.sum(axis=1), 1e-12)
        out[~ok.any(axis=1)] = np.nan
        return out

    return {(y, m): {k: to_cells(a[:, y - 1999, m - 1]) for k, a in anom.items()}
            for y in years for m in range(1, 13)}


def main():
    grid = pd.read_parquet(PROCESSED / "grid.parquet")
    fires = pd.read_parquet(PROCESSED / "fires.parquet")
    burned = np.load(PROCESSED / "burned.npy")
    burned_years = json.loads((PROCESSED / "burned_years.json").read_text())
    bec_cols = [c for c in grid.columns if c.startswith("bec_") and c != "bec_major"]
    landmask = grid["is_land"].to_numpy()
    land = grid[landmask]
    cells = land["cell_id"].to_numpy()

    wf = fires[fires["is_wildfire"] & fires["cell_id"].ge(0) & fires["date"].notna()].copy()
    wf["cal_y"], wf["cal_m"] = wf["date"].dt.year, wf["date"].dt.month
    years = list(range(FIRST_YEAR, max(TEST_YEARS) + 1))
    allc = count_panel(wf, years)
    large = count_panel(wf, years, wf["current_size"].ge(LARGE_HA))
    light = count_panel(wf, years, wf["fire_cause"].eq("Lightning"))
    person = count_panel(wf, years, wf["fire_cause"].eq("Person"))
    args = (years, allc, light, person, large, burned, burned_years, grid, bec_cols)

    # fire counts by calendar year, month and cell
    ym = np.zeros((len(years), 12, NX * NY), dtype=np.float32)
    for (y, m, c), n in wf[wf["cal_y"].isin(years)].groupby(["cal_y", "cal_m", "cell_id"]).size().items():
        ym[years.index(y), m - 1, c] = n

    wx = monthly_weather(land)
    print("weather ready", flush=True)

    def rows_for(Y):
        i = years.index(Y)
        base = features_for_year(Y, *args)[landmask].reset_index(drop=True)
        out = []
        for m in MONTHS:
            df = base.copy()
            df["month"] = m
            # usual number of fires in this cell in this month, from past years
            clim = ym[:i, m - 1].mean(axis=0)
            df["month_mean"] = clim[landmask]
            df["month_mean_blur2"] = blur(clim, 2)[landmask]
            # fires so far this season, Apr 1 to the end of last month
            so_far = ym[i, 3:m - 1].sum(axis=0) if m > 4 else np.zeros(NX * NY)
            last = ym[i, m - 2]
            df["sofar_cell"] = so_far[landmask]
            df["sofar_30km"] = blur(so_far, 3)[landmask]
            df["sofar_bc"] = so_far.sum()
            df["last_month_30km"] = blur(last, 3)[landmask]
            df["sofar_vs_usual"] = so_far.sum() / max(ym[:i, 3:m - 1].sum(axis=(1, 2)).mean(), 1) if m > 4 else 1.0
            # weather: last month, the month before, and precipitation since April
            for lag in (1, 2):
                yy, mm = (Y, m - lag) if m - lag >= 1 else (Y - 1, m - lag + 12)
                for k, v in wx[(Y, m - lag) if m - lag >= 1 else (Y - 1, m - lag + 12)].items():
                    df[f"wx_{k}_lag{lag}"] = v
            pr = [wx[(Y, mm)]["pr"] for mm in range(4, m)]
            df["wx_pr_since_apr"] = np.nanmean(np.stack(pr), axis=0)
            df["wx_bc_pr_lag1"] = np.nanmean(df["wx_pr_lag1"])
            df["wx_bc_tx_lag1"] = np.nanmean(df["wx_tx_lag1"])
            df["wx_bc_soil_lag1"] = np.nanmean(df["wx_soil_lag1"])
            df["n"] = ym[i, m - 1][landmask]
            df["year"] = Y
            out.append(df)
        return pd.concat(out, ignore_index=True)

    tr = pd.concat([rows_for(Y) for Y in TRAIN_YEARS], ignore_index=True)
    va = pd.concat([rows_for(Y) for Y in VALID_YEARS], ignore_index=True)
    te = pd.concat([rows_for(Y) for Y in TEST_YEARS], ignore_index=True)
    print("rows:", len(tr), len(va), len(te), flush=True)

    hist = [c for c in features_for_year(TRAIN_YEARS[0], *args).columns if c not in ("cell_id", "target_year")]
    hist += ["month", "month_mean", "month_mean_blur2"]
    sofar = ["sofar_cell", "sofar_30km", "sofar_bc", "last_month_30km", "sofar_vs_usual"]
    weather = [c for c in tr.columns if c.startswith("wx_")]
    versions = {"M0": hist, "M1": hist + sofar, "M2": hist + sofar + weather}

    cell_centre = wf.groupby("cell_id")["fire_centre"].agg(lambda s: s.mode().iat[0])
    te["centre"] = te["cell_id"].map(cell_centre).map(FIRE_CENTRES).fillna("Other")

    base_params = dict(objective="poisson", learning_rate=0.05, feature_fraction=0.8,
                       bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=SEED)
    preds, out = {}, {"versions": {}}
    for name, feats in versions.items():
        best = None
        for nl in (31, 63):
            for md in (50, 200):
                params = {**base_params, "num_leaves": nl, "min_data_in_leaf": md}
                mdl = lgb.train(params, lgb.Dataset(tr[feats], tr["n"]), num_boost_round=2000,
                                valid_sets=[lgb.Dataset(va[feats], va["n"])],
                                callbacks=[lgb.early_stopping(100, verbose=False)])
                dev = poisson_dev(mdl.predict(va[feats], num_iteration=mdl.best_iteration), va["n"].to_numpy())
                if best is None or dev < best[0]:
                    best = (dev, params, mdl.best_iteration)
        _, params, rounds = best
        trva = pd.concat([tr, va], ignore_index=True)
        mdl = lgb.train(params, lgb.Dataset(trva[feats], trva["n"]), num_boost_round=rounds)
        p = mdl.predict(te[feats])
        preds[name] = p
        a = te["n"].to_numpy()
        res = {"params": {k: params[k] for k in ("num_leaves", "min_data_in_leaf")}, "rounds": int(rounds),
               "pooled": {"poisson_deviance": poisson_dev(p, a)}, "per_year": {}}
        caps = []
        for (Y, m), idx in te.groupby(["year", "month"]).indices.items():
            caps.append(capture(p[idx], a[idx], 0.10))
        res["pooled"]["mean_monthly_capture_top10"] = float(np.mean(caps))
        for Y, idx in te.groupby("year").indices.items():
            byc = pd.DataFrame({"c": te["centre"].to_numpy()[idx], "m": te["month"].to_numpy()[idx],
                                "p": p[idx], "a": a[idx]})
            byc = byc[byc["c"] != "Other"].groupby(["m", "c"])[["p", "a"]].sum()
            res["per_year"][str(Y)] = dict(
                predicted_total=float(p[idx].sum()), actual_total=int(a[idx].sum()),
                centre_month_error_pct=float(100 * (byc["p"] - byc["a"]).abs().sum() / byc["a"].sum()),
                by_centre={c: [round(float(g["p"].sum())), int(g["a"].sum())] for c, g in byc.groupby(level="c")})
        out["versions"][name] = res
        print(name, "done", {k: round(v, 3) for k, v in res["pooled"].items()}, flush=True)
        if name == "M2":
            imp = pd.DataFrame({"feature": feats, "gain": mdl.feature_importance("gain")})
            imp["share"] = imp["gain"] / imp["gain"].sum()
            imp.sort_values("share", ascending=False).to_csv(RESULTS / "feature_importance_monthly.csv", index=False)
            res["share_of_gain"] = {"history": float(imp.loc[imp.feature.isin(hist), "share"].sum()),
                                    "so_far": float(imp.loc[imp.feature.isin(sofar), "share"].sum()),
                                    "weather": float(imp.loc[imp.feature.isin(weather), "share"].sum())}

    keep = te[["year", "month", "cell_id", "n", "centre"]].copy()
    for name, p in preds.items():
        keep[name] = p
    keep.to_parquet(RESULTS / "monthly_test_predictions.parquet", index=False)
    for n, v in out["versions"].items():
        print(n, v["pooled"], {Y: round(r["centre_month_error_pct"], 1) for Y, r in v["per_year"].items()}, flush=True)

    # movable bases, re-sited on the 1st of each month
    R_KM = coverage_radius_km()
    cand = pd.read_parquet(PROCESSED / "candidates.parquet")
    cxy = cand[["x", "y"]].to_numpy()
    near = coverage_sets(cxy, land[["x", "y"]].to_numpy(), R_KM * 1000)
    fixed = json.loads((RESULTS / "placements.json").read_text())["model"][str(K)]
    fixed_tree = cKDTree(cxy[fixed])
    reach = {k: 0 for k in ("fixed", "M0", "M1", "M2", "hindsight")}
    per_year = {}
    total = 0
    for (Y, m), idx in te.groupby(["year", "month"]).indices.items():
        fm = wf[(wf["cal_y"] == Y) & (wf["cal_m"] == m)]
        if len(fm) == 0:
            continue
        fxy = fm[["x", "y"]].to_numpy()
        hit = lambda tree: int((tree.query(fxy)[0] / 1000 <= R_KM).sum())
        row = {"fires": len(fm), "fixed": hit(fixed_tree)}
        for name in ("M0", "M1", "M2"):
            ch, _ = solve_mclp(preds[name][idx], near, len(cand), K)
            row[name] = hit(cKDTree(cxy[ch]))
        ch, _ = solve_mclp(te["n"].to_numpy()[idx].astype(float), near, len(cand), K)
        row["hindsight"] = hit(cKDTree(cxy[ch]))
        per_year.setdefault(str(Y), []).append({"month": int(m), **row})
        total += len(fm)
        for k in reach:
            reach[k] += row[k]
        print(Y, m, row, flush=True)
    out["movable"] = {"per_month": per_year,
                      "share_reached": {k: v / total for k, v in reach.items()},
                      "per_year_share": {Y: {k: sum(r[k] for r in rows) / sum(r["fires"] for r in rows)
                                             for k in ("fixed", "M0", "M1", "M2", "hindsight")}
                                         for Y, rows in per_year.items()}}
    (RESULTS / "monthly_metrics.json").write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps({k: out["movable"][k] for k in ("share_reached", "per_year_share")}, indent=1))
    for n, v in out["versions"].items():
        print(n, v["pooled"], {Y: round(r["centre_month_error_pct"], 1) for Y, r in v["per_year"].items()})
    print("2023 by centre, May-Sep [predicted, actual]:")
    for n, v in out["versions"].items():
        print(" ", n, v["per_year"]["2023"]["by_centre"])
    print("M2 share of gain:", out["versions"]["M2"].get("share_of_gain"))


if __name__ == "__main__":
    main()
