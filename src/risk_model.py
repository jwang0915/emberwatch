"""Step 2. Predict how many wildfires start in each 10 km cell in a fire year.

The model only sees information from years BEFORE the year it predicts, so it
can't cheat by looking at the answer.

Two targets are modelled with the same features:
  all    every wildfire (FIRE_TYPE == "Fire")
  large  wildfires whose final size reached LARGE_HA hectares or more

Compared against two simple baselines:
  hist10  average count in the cell over the previous 10 years
  smooth  the same average, blurred over neighbouring cells

Outputs: results/risk_metrics.json, results/feature_importance.csv,
         data/processed/risk_surface.parquet (the map used to place bases)
"""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score

from config import (PROCESSED, RESULTS, NX, NY, FIRST_YEAR, HISTORY_WINDOW,
                    TRAIN_YEARS, VALID_YEARS, TEST_YEARS, PLANNING_YEAR, SEED)

LARGE_HA = 10.0  # "large" = final size of at least 10 ha

STATIC = ["land_frac", "road_frac", "road_frac_50km", "dist_town_km", "log_town_pop"]


def count_panel(fires, years, mask=None):
    """Array (n_years, n_cells) of fire counts."""
    f = fires if mask is None else fires[mask]
    f = f[f["cell_id"] >= 0]
    out = np.zeros((len(years), NX * NY), dtype=np.float32)
    yi = {y: i for i, y in enumerate(years)}
    for (y, c), n in f.groupby(["fire_year", "cell_id"]).size().items():
        if y in yi:
            out[yi[y], c] = n
    return out


def blur(v, sigma):
    return gaussian_filter(v.reshape(NY, NX), sigma=sigma, mode="constant").ravel()


def features_for_year(t, years, allc, light, person, large, burned, burned_years, grid, bec_cols):
    """Feature table for target year t, using only years < t."""
    i = years.index(t)
    past = lambda arr, n: arr[max(0, i - n):i]
    f = pd.DataFrame({"cell_id": grid["cell_id"].to_numpy()})
    f["c_lag1"] = allc[i - 1]
    for n in (3, 5, HISTORY_WINDOW):
        f[f"c_mean{n}"] = past(allc, n).mean(axis=0)
    f["light_mean10"] = past(light, HISTORY_WINDOW).mean(axis=0)
    f["person_mean10"] = past(person, HISTORY_WINDOW).mean(axis=0)
    f["c_mean_all"] = allc[:i].mean(axis=0)
    f["large_mean_all"] = large[:i].mean(axis=0)
    f["large_mean_all_blur2"] = blur(f["large_mean_all"].to_numpy(), 2)
    for s in (1, 3):
        f[f"c_mean10_blur{s}"] = blur(f["c_mean10"].to_numpy(), s)
        f[f"light_mean10_blur{s}"] = blur(f["light_mean10"].to_numpy(), s)
    # years since the cell last had a fire (capped)
    had = past(allc, HISTORY_WINDOW) > 0
    last = np.where(had.any(axis=0), had.shape[0] - np.argmax(had[::-1], axis=0), HISTORY_WINDOW + 1)
    f["yrs_since_fire"] = last
    # fuel already burned in recent large fires
    bi = [j for j, y in enumerate(burned_years) if y < t]
    for n in (1, 5, 10):
        f[f"burned_last{n}"] = burned[bi[-n:]].sum(axis=0).clip(0, 1)
    f["burned_since2000"] = burned[bi].sum(axis=0).clip(0, 1)
    f["burned_last5_blur3"] = blur(f["burned_last5"].to_numpy(), 3)
    for c in STATIC + bec_cols:
        f[c] = grid[c].to_numpy()
    f["east_km"] = grid["x"].to_numpy() / 1000
    f["north_km"] = grid["y"].to_numpy() / 1000
    f["target_year"] = t
    return f


def capture(pred, actual, share):
    """Share of actual fires that fall in the top `share` of cells by prediction."""
    k = int(len(pred) * share)
    top = np.argsort(-pred)[:k]
    return float(actual[top].sum() / max(actual.sum(), 1))


def poisson_dev(pred, actual):
    p = np.clip(pred, 1e-6, None)
    term = np.zeros_like(p, dtype=float)
    pos = actual > 0
    term[pos] = actual[pos] * np.log(actual[pos] / p[pos])
    return float(2 * np.mean(term - (actual - p)))


def evaluate(pred, actual):
    return dict(poisson_deviance=poisson_dev(pred, actual),
                capture_top5=capture(pred, actual, 0.05),
                capture_top10=capture(pred, actual, 0.10),
                capture_top20=capture(pred, actual, 0.20),
                auc_any_fire=float(roc_auc_score(actual > 0, pred)))


def fit_target(name, target, tr, va, te, feats):
    """Tune, fit and test one Poisson model. Returns (model, metrics)."""
    hist_col = "c_mean10" if name == "all" else "large_mean_all"

    def smooth_pred(df, sigma):
        out = np.empty(len(df))
        for t, idx in df.groupby("target_year").indices.items():
            full = np.zeros(NX * NY)
            full[df["cell_id"].to_numpy()[idx]] = df[hist_col].to_numpy()[idx]
            out[idx] = blur(full, sigma)[df["cell_id"].to_numpy()[idx]]
        return out

    best_sigma = min((0.5, 1.0, 1.5, 2.0, 3.0),
                     key=lambda s: poisson_dev(smooth_pred(va, s), va[target].to_numpy()))

    base = dict(objective="poisson", learning_rate=0.03, feature_fraction=0.8,
                bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=SEED)
    search = [dict(num_leaves=nl, min_data_in_leaf=md, lambda_l2=l2)
              for nl in (15, 31, 63) for md in (50, 200) for l2 in (0.0, 5.0)]
    best = None
    for hp in search:
        params = {**base, **hp}
        m = lgb.train(params, lgb.Dataset(tr[feats], tr[target]), num_boost_round=3000,
                      valid_sets=[lgb.Dataset(va[feats], va[target])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        dev = poisson_dev(m.predict(va[feats], num_iteration=m.best_iteration), va[target].to_numpy())
        if best is None or dev < best[0]:
            best = (dev, params, m.best_iteration)
    _, params, n_rounds = best

    # refit on train + validation years with the chosen settings
    trva = pd.concat([tr, va], ignore_index=True)
    model = lgb.train(params, lgb.Dataset(trva[feats], trva[target]), num_boost_round=n_rounds)
    model.save_model(str(RESULTS / f"risk_model_{name}.txt"))

    metrics = {"chosen_params": {k: params[k] for k in ("num_leaves", "min_data_in_leaf", "lambda_l2")},
               "boosting_rounds": int(n_rounds), "blur_sigma_cells": best_sigma,
               "baseline_history_column": hist_col, "per_year": {}, "pooled": {}}
    preds = {"history": te[hist_col].to_numpy(),
             "history_smoothed": smooth_pred(te, best_sigma),
             "model": model.predict(te[feats])}
    actual = te[target].to_numpy()
    for pname, p in preds.items():
        metrics["pooled"][pname] = evaluate(p, actual)
        for t, idx in te.groupby("target_year").indices.items():
            metrics["per_year"].setdefault(str(t), {})[pname] = evaluate(p[idx], actual[idx])

    imp = pd.DataFrame({"feature": feats, "gain": model.feature_importance("gain")})
    imp["share"] = imp["gain"] / imp["gain"].sum()
    imp.sort_values("share", ascending=False).to_csv(RESULTS / f"feature_importance_{name}.csv", index=False)
    return model, metrics, imp


def main():
    grid = pd.read_parquet(PROCESSED / "grid.parquet")
    fires = pd.read_parquet(PROCESSED / "fires.parquet")
    burned = np.load(PROCESSED / "burned.npy")
    burned_years = json.loads((PROCESSED / "burned_years.json").read_text())
    bec_cols = [c for c in grid.columns if c.startswith("bec_") and c != "bec_major"]
    land = grid["is_land"].to_numpy()

    wf = fires[fires["is_wildfire"]]
    years = list(range(FIRST_YEAR, max(TEST_YEARS) + 1))
    counts = {"all": count_panel(wf, years),
              "large": count_panel(wf, years, wf["current_size"].ge(LARGE_HA))}
    light = count_panel(wf, years, wf["fire_cause"].eq("Lightning"))
    person = count_panel(wf, years, wf["fire_cause"].eq("Person"))
    feat_args = (years, counts["all"], light, person, counts["large"], burned, burned_years, grid, bec_cols)

    def table(ts):
        parts = []
        for t in ts:
            ft = features_for_year(t, *feat_args)
            for name, arr in counts.items():
                ft[f"n_{name}"] = arr[years.index(t)]
            parts.append(ft[land])
        return pd.concat(parts, ignore_index=True)

    tr, va, te = table(TRAIN_YEARS), table(VALID_YEARS), table(TEST_YEARS)
    feats = [c for c in tr.columns if c not in ("cell_id", "target_year", "n_all", "n_large")]

    # planning surface: features as a planner would have them at the end of 2021
    plan = features_for_year(PLANNING_YEAR, *feat_args)
    ip = years.index(PLANNING_YEAR)
    test_idx = [years.index(t) for t in TEST_YEARS]
    surf = pd.DataFrame({"cell_id": grid["cell_id"], "is_land": land,
                         "light_share_hist": np.divide(light[:ip].sum(0), np.maximum(counts["all"][:ip].sum(0), 1))})
    all_metrics = {"features": feats, "large_fire_threshold_ha": LARGE_HA}
    for name, arr in counts.items():
        model, m, imp = fit_target(name, f"n_{name}", tr, va, te, feats)
        risk = np.zeros(NX * NY)
        risk[land] = model.predict(plan.loc[land, feats])
        hist = arr[:ip].mean(axis=0) * land
        actual_test = arr[test_idx].sum(axis=0)
        sfx = "" if name == "all" else "_large"
        surf[f"risk_model{sfx}"] = risk         # expected fires per year
        surf[f"risk_hist{sfx}"] = hist          # 2000-2021 average per year
        surf[f"fires_test{sfx}"] = actual_test  # 2022-2025 actual, used only for scoring
        # how well does each surface, fixed at the end of 2021, rank 2022-2025?
        m["planning_surface_vs_2022_2025"] = {
            "model": evaluate(risk[land], actual_test[land]),
            "history_2000_2021": evaluate(hist[land], actual_test[land]),
        }
        all_metrics[name] = m
        print(f"\n=== target: {name} ===")
        print("chosen:", m["chosen_params"], "rounds:", m["boosting_rounds"], "blur sigma:", m["blur_sigma_cells"])
        print("one-year-ahead, 2022-2025 pooled:")
        print(pd.DataFrame(m["pooled"]).T.round(3).to_string())
        print("planning surface (fixed end of 2021) vs 2022-2025 total:")
        print(pd.DataFrame(m["planning_surface_vs_2022_2025"]).T[["capture_top10", "capture_top20", "auc_any_fire"]].round(3).to_string())
        print(imp.sort_values("share", ascending=False).head(8).round(3).to_string(index=False))

    surf.to_parquet(PROCESSED / "risk_surface.parquet", index=False)
    (RESULTS / "risk_metrics.json").write_text(json.dumps(all_metrics, indent=2))


if __name__ == "__main__":
    main()
