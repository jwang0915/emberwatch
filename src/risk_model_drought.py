"""Step 2c. Does drought data fix the forecast?

Trains three versions of the risk model with the same tuning and the same
train, validation and test years as risk_model.py, then compares them on the
2022 to 2025 seasons.

  base      fire history, roads, burns, ecosystem zones. The original model.
  winter    base plus drought features known by 31 December before the season.
            Useful for planning permanent bases a year ahead.
  spring    base plus everything known by 1 May: the winter features and this
            spring's soil moisture, snow, precipitation and temperature.
            Useful for deciding where to put movable drones each season.

It then asks the question that matters for the project: if 30 drone bases
could be moved every spring, using the spring forecast, how many more fires
would they reach than the fixed network chosen at the end of 2021?

Outputs: results/drought_metrics.json, results/feature_importance_spring.csv
"""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from config import (PROCESSED, RESULTS, FIRST_YEAR, TRAIN_YEARS, VALID_YEARS, TEST_YEARS, SEED,
                    coverage_radius_km)
from risk_model import count_panel, features_for_year, poisson_dev, capture, LARGE_HA
from placement import solve_mclp, coverage_sets

FIRE_CENTRES = {2: "Coastal", 3: "Northwest", 4: "Prince George", 5: "Kamloops", 6: "Southeast", 7: "Cariboo"}
K = 30


def tune_and_fit(tr, va, feats, target="n_all"):
    base = dict(objective="poisson", learning_rate=0.03, feature_fraction=0.8,
                bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=SEED)
    best = None
    for nl in (15, 31, 63):
        for md in (50, 200):
            for l2 in (0.0, 5.0):
                params = {**base, "num_leaves": nl, "min_data_in_leaf": md, "lambda_l2": l2}
                m = lgb.train(params, lgb.Dataset(tr[feats], tr[target]), num_boost_round=3000,
                              valid_sets=[lgb.Dataset(va[feats], va[target])],
                              callbacks=[lgb.early_stopping(100, verbose=False)])
                dev = poisson_dev(m.predict(va[feats], num_iteration=m.best_iteration), va[target].to_numpy())
                if best is None or dev < best[0]:
                    best = (dev, params, m.best_iteration)
    _, params, rounds = best
    trva = pd.concat([tr, va], ignore_index=True)
    model = lgb.train(params, lgb.Dataset(trva[feats], trva[target]), num_boost_round=rounds)
    return model, {k: params[k] for k in ("num_leaves", "min_data_in_leaf", "lambda_l2")}, int(rounds)


def main():
    grid = pd.read_parquet(PROCESSED / "grid.parquet")
    fires = pd.read_parquet(PROCESSED / "fires.parquet")
    burned = np.load(PROCESSED / "burned.npy")
    burned_years = json.loads((PROCESSED / "burned_years.json").read_text())
    drought = pd.read_parquet(PROCESSED / "drought_features.parquet")
    bec_cols = [c for c in grid.columns if c.startswith("bec_") and c != "bec_major"]
    land = grid["is_land"].to_numpy()

    wf = fires[fires["is_wildfire"]]
    years = list(range(FIRST_YEAR, max(TEST_YEARS) + 1))
    allc = count_panel(wf, years)
    large = count_panel(wf, years, wf["current_size"].ge(LARGE_HA))
    light = count_panel(wf, years, wf["fire_cause"].eq("Lightning"))
    person = count_panel(wf, years, wf["fire_cause"].eq("Person"))
    args = (years, allc, light, person, large, burned, burned_years, grid, bec_cols)

    def table(ts):
        parts = []
        for t in ts:
            ft = features_for_year(t, *args)[land]
            ft["n_all"] = allc[years.index(t)][land]
            parts.append(ft.merge(drought[drought["fire_year"] == t].drop(columns="fire_year"),
                                  on="cell_id", how="left"))
        return pd.concat(parts, ignore_index=True)

    tr, va, te = table(TRAIN_YEARS), table(VALID_YEARS), table(TEST_YEARS)
    base_feats = [c for c in features_for_year(TRAIN_YEARS[0], *args).columns if c not in ("cell_id", "target_year")]
    pre = [c for c in drought.columns if c.startswith(("pre_", "bc_pre_"))]
    may = [c for c in drought.columns if c.startswith(("may_", "bc_may_"))]
    versions = {"base": base_feats, "winter": base_feats + pre, "spring": base_feats + pre + may}

    # which fire centre each cell belongs to, from its past fires
    cell_centre = wf.groupby("cell_id")["fire_centre"].agg(lambda s: s.mode().iat[0])
    te["centre"] = te["cell_id"].map(cell_centre).map(FIRE_CENTRES).fillna("Other")

    out = {"versions": {}, "test_years": TEST_YEARS}
    preds = {}
    for name, feats in versions.items():
        model, params, rounds = tune_and_fit(tr, va, feats)
        p = model.predict(te[feats])
        preds[name] = p
        res = {"params": params, "rounds": rounds, "n_features": len(feats), "per_year": {}}
        for t, idx in te.groupby("target_year").indices.items():
            a = te["n_all"].to_numpy()[idx]
            byc = pd.DataFrame({"c": te["centre"].to_numpy()[idx], "p": p[idx], "a": a}).groupby("c").sum()
            byc = byc[byc.index != "Other"]
            res["per_year"][str(t)] = dict(
                predicted_total=float(p[idx].sum()), actual_total=int(a.sum()),
                total_error_pct=float(100 * (p[idx].sum() / a.sum() - 1)),
                poisson_deviance=poisson_dev(p[idx], a),
                capture_top10=capture(p[idx], a, 0.10),
                centre_error_pct=float(100 * (byc["p"] - byc["a"]).abs().sum() / byc["a"].sum()),
                by_centre={c: [round(float(r.p)), int(r.a)] for c, r in byc.iterrows()})
        a = te["n_all"].to_numpy()
        res["pooled"] = dict(poisson_deviance=poisson_dev(p, a), capture_top10=capture(p, a, 0.10))
        out["versions"][name] = res
        if name == "spring":
            imp = pd.DataFrame({"feature": feats, "gain": model.feature_importance("gain")})
            imp["share"] = imp["gain"] / imp["gain"].sum()
            imp.sort_values("share", ascending=False).to_csv(RESULTS / "feature_importance_spring.csv", index=False)
            dshare = imp.loc[imp["feature"].isin(pre + may), "share"].sum()
            res["drought_share_of_gain"] = float(dshare)

    # movable bases: re-site 30 bases each spring on that season's forecast
    R_KM = coverage_radius_km()
    cand = pd.read_parquet(PROCESSED / "candidates.parquet")
    cxy = cand[["x", "y"]].to_numpy()
    cells = te[te["target_year"] == TEST_YEARS[0]]["cell_id"].to_numpy()
    cell_xy = grid.set_index("cell_id").loc[cells, ["x", "y"]].to_numpy()
    near = coverage_sets(cxy, cell_xy, R_KM * 1000)
    fixed = json.loads((RESULTS / "placements.json").read_text())["model"][str(K)]
    out["movable"] = {}
    for t, idx in te.groupby("target_year").indices.items():
        fy = wf[wf["fire_year"].eq(t) & wf["cell_id"].ge(0)]
        fxy = fy[["x", "y"]].to_numpy()
        cov = lambda ch: float((cKDTree(cxy[ch]).query(fxy)[0] / 1000 <= R_KM).mean())
        row = {"fixed_2021_plan": cov(fixed)}
        for name in ("base", "spring"):
            ch, _ = solve_mclp(preds[name][idx], near, len(cand), K)
            row[f"moved_each_spring_{name}"] = cov(ch)
        ch, _ = solve_mclp(te["n_all"].to_numpy()[idx].astype(float), near, len(cand), K)
        row["hindsight"] = cov(ch)
        out["movable"][str(t)] = row
        print(t, {k: round(v, 3) for k, v in row.items()}, flush=True)

    (RESULTS / "drought_metrics.json").write_text(json.dumps(out, indent=2))

    print("\nforecast total vs actual, by season:")
    for t in TEST_YEARS:
        print(t, "actual", out["versions"]["base"]["per_year"][str(t)]["actual_total"],
              {n: round(v["per_year"][str(t)]["predicted_total"]) for n, v in out["versions"].items()})
    print("\nfire-centre error, % of actual fires misplaced or missed:")
    for t in TEST_YEARS:
        print(t, {n: round(v["per_year"][str(t)]["centre_error_pct"], 1) for n, v in out["versions"].items()})
    print("\npooled:", {n: {k: round(x, 3) for k, x in v["pooled"].items()} for n, v in out["versions"].items()})
    print("top10 by year:", {t: {n: round(v["per_year"][str(t)]["capture_top10"], 3) for n, v in out["versions"].items()} for t in TEST_YEARS})
    print("drought share of gain in spring model:", round(out["versions"]["spring"]["drought_share_of_gain"], 3))
    print("2023 by centre, [predicted, actual]:")
    for n, v in out["versions"].items():
        print(" ", n, v["per_year"]["2023"]["by_centre"])


if __name__ == "__main__":
    main()
