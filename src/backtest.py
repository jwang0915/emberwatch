"""Backtest: pretend it is the end of one fire season and plan for the next.

    python backtest.py 2023

For target year Y, everything is rebuilt using only information a planner
would have had on 31 December of Y-1:
  1. the risk model is retrained on seasons up to Y-1
  2. 30 drone bases are chosen from sites that existed before Y
  3. the real fires of year Y are then revealed and scored

The model settings, number of leaves and so on, are the ones tuned on
2020-2021 in risk_model.py, so use Y of 2022 or later to keep the test clean.

Output: results/backtest_<Y>.json
"""
import io
import json
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from config import PROCESSED, RESULTS, BUNDLE, DRONE, SEED, coverage_radius_km
from risk_model import count_panel, features_for_year, capture, LARGE_HA
from placement import solve_mclp, coverage_sets
from simulate import run_once

K = 30
REPS = 20
FIRE_CENTRES = {1: "Coastal", 2: "Coastal", 3: "Northwest", 4: "Prince George",
                5: "Kamloops", 6: "Southeast", 7: "Cariboo"}
NOTABLE = {"G80280": "Donnie Creek", "K21633": "Bush Creek East", "K52767": "McDougall Creek"}


def main(Y):
    R_KM = coverage_radius_km()
    grid = pd.read_parquet(PROCESSED / "grid.parquet")
    fires = pd.read_parquet(PROCESSED / "fires.parquet")
    burned = np.load(PROCESSED / "burned.npy")
    burned_years = json.loads((PROCESSED / "burned_years.json").read_text())
    metrics = json.loads((RESULTS / "risk_metrics.json").read_text())
    bec_cols = [c for c in grid.columns if c.startswith("bec_") and c != "bec_major"]
    land = grid["is_land"].to_numpy()

    wf = fires[fires["is_wildfire"]]
    years = list(range(2000, Y + 1))
    allc = count_panel(wf, years)
    large = count_panel(wf, years, wf["current_size"].ge(LARGE_HA))
    light = count_panel(wf, years, wf["fire_cause"].eq("Lightning"))
    person = count_panel(wf, years, wf["fire_cause"].eq("Person"))
    args = (years, allc, light, person, large, burned, burned_years, grid, bec_cols)

    # 1. retrain on seasons 2010 to Y-1, with the settings chosen earlier
    rows = []
    for t in range(2010, Y):
        ft = features_for_year(t, *args)
        ft["n"] = allc[years.index(t)]
        rows.append(ft[land])
    tr = pd.concat(rows, ignore_index=True)
    feats = metrics["features"]
    p = metrics["all"]["chosen_params"]
    params = dict(objective="poisson", learning_rate=0.03, feature_fraction=0.8, bagging_fraction=0.8,
                  bagging_freq=1, verbose=-1, seed=SEED, **p)
    model = lgb.train(params, lgb.Dataset(tr[feats], tr["n"]), num_boost_round=metrics["all"]["boosting_rounds"])

    plan = features_for_year(Y, *args)
    risk = np.zeros(len(grid))
    risk[land] = model.predict(plan.loc[land, feats])
    hist = allc[:years.index(Y)].mean(axis=0) * land
    actual = allc[years.index(Y)]

    # 2. choose 30 bases from sites that existed before Y
    b = json.loads(BUNDLE.read_text())
    st = pd.read_csv(io.StringIO(b["stations_csv"]))
    cand = pd.read_parquet(PROCESSED / "candidates.parquet")
    new_names = set(st.loc[pd.to_datetime(st["INSTALL_DATE"]) >= f"{Y}-01-01", "STATION_NAME"].str.title())
    ok = ~(cand["kind"].eq("weather station") & cand["name"].isin(new_names))
    cand = cand[ok].reset_index(drop=True)
    cxy = cand[["x", "y"]].to_numpy()
    near = coverage_sets(cxy, grid.loc[land, ["x", "y"]].to_numpy(), R_KM * 1000)
    placements = {}
    for name, w in [("model", risk[land]), ("history", hist[land]),
                    ("even", np.ones(land.sum())), ("hindsight", actual[land])]:
        placements[name], _ = solve_mclp(w, near, len(cand), K)

    # 3. reveal year Y
    fy = wf[wf["fire_year"].eq(Y) & wf["date"].notna()].reset_index(drop=True)
    fxy = fy[["x", "y"]].to_numpy()
    out = dict(year=Y, bases=K, radius_km=R_KM, n_fires=len(fy))

    # forecast: how many fires, where
    out["forecast_total"] = float(risk.sum())
    out["history_total"] = float(hist.sum())
    out["actual_total"] = int(actual.sum())
    fc = pd.DataFrame({"pred": risk, "act": actual})
    # assign each cell to the fire centre most of its past fires belong to
    cell_centre = wf.groupby("cell_id")["fire_centre"].agg(lambda s: s.mode().iat[0])
    fc["centre"] = grid["cell_id"].map(cell_centre).map(FIRE_CENTRES).fillna("unassigned")
    by = fc.groupby("centre")[["pred", "act"]].sum().round(0)
    out["by_fire_centre"] = by.astype(int).to_dict(orient="index")
    out["top10_capture"] = {"model": capture(risk[land], actual[land], 0.10),
                            "history": capture(hist[land], actual[land], 0.10)}

    # coverage of the real fires
    out["coverage"] = {}
    for name, ch in placements.items():
        d = cKDTree(cxy[ch]).query(fxy)[0] / 1000
        out["coverage"][name] = float((d <= R_KM).mean())

    # replay the season with the model's bases, one and two drones each
    calls = fy.copy()
    calls["day"] = (calls["date"] - pd.Timestamp(f"{Y}-01-01")).dt.days
    bxy = cxy[placements["model"]]
    out["replay"] = {}
    for m in (1, 2):
        within, logs = [], []
        for r in range(REPS):
            log, _ = run_once(calls, bxy, m, np.random.default_rng(SEED + r))
            within.append(float((log["first_eyes"] <= DRONE["target_min"]).mean()))
            logs.append(log)
        out["replay"][str(m)] = dict(within_target=float(np.mean(within)),
                                     low=float(np.min(within)), high=float(np.max(within)))
        if m == 1:
            logs1 = logs

    # the season's notable fires
    out["notable"] = []
    d_all = cKDTree(bxy).query(fxy)[0] / 1000
    for num, label in NOTABLE.items():
        idx = np.flatnonzero(fy["fire_number"].eq(num).to_numpy())
        if len(idx) == 0:
            continue
        i = idx[0]
        eyes = [float(l["first_eyes"].iat[i]) for l in logs1]
        out["notable"].append(dict(fire=label, number=num, date=str(fy["date"].iat[i].date()),
                                   size_ha=float(fy["current_size"].iat[i]),
                                   nearest_base_km=round(float(d_all[i]), 1),
                                   replays_reached_in_30_min=int(sum(e <= DRONE["target_min"] for e in eyes if e == e)),
                                   outcomes=dict(pd.Series([("late" if (l["status"].iat[i] == "reached" and l["first_eyes"].iat[i] > DRONE["target_min"]) else l["status"].iat[i]) for l in logs1]).value_counts()),
                                   replays=REPS))
    (RESULTS / f"backtest_{Y}.json").write_text(json.dumps(out, indent=2, default=int))
    print(json.dumps(out, indent=2, default=int))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2023)
