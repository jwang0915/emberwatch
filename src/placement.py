"""Step 3. Choose where to put k drone bases.

This is a Maximal Covering Location Problem (Church and ReVelle, 1974):

    maximise   sum_i  w_i * z_i                  (risk that gets covered)
    subject to z_i <= sum_{j near i} y_j        (cell i is covered only if a base is in range)
               sum_j y_j = k                     (exactly k bases)
               y_j in {0,1},  0 <= z_i <= 1

where i runs over 10 km land cells, j over candidate sites, "near" means within
the distance a drone covers inside the target response time, and w_i is how
much we care about cell i. Different choices of w_i give different strategies:

  model  expected fires from the risk model (known at end of 2021)
  hist   average fires per year 2000-2021
  model_large, hist_large, oracle_large
         the same, but counting only fires that grew to 10 ha or more
  area   every land cell counts the same, i.e. spread bases evenly
  towns  not optimised: put bases in the k biggest towns
  random not optimised: k random weather stations, averaged over 30 draws
  oracle optimised on the actual 2022-2025 fires. Impossible in real life,
         but it shows the best any placement could have done.

Every strategy is scored on the same things: the share of real 2022-2025
wildfires, and of real 2022-2025 large wildfires, that a drone could reach
within the target time.
"""
import json
import time

import numpy as np
import pandas as pd
import pulp
from scipy.spatial import cKDTree

from risk_model import LARGE_HA
from config import PROCESSED, RESULTS, DRONE, BASE_COUNTS, TEST_YEARS, SEED, coverage_radius_km

R_KM = coverage_radius_km()


def load():
    grid = pd.read_parquet(PROCESSED / "grid.parquet")
    surf = pd.read_parquet(PROCESSED / "risk_surface.parquet")
    cand = pd.read_parquet(PROCESSED / "candidates.parquet")
    fires = pd.read_parquet(PROCESSED / "fires.parquet")
    test = fires[fires["is_wildfire"] & fires["fire_year"].isin(TEST_YEARS)].reset_index(drop=True)
    test["is_large"] = test["current_size"].ge(LARGE_HA)
    return grid, surf, cand, test


def coverage_sets(cand_xy, cell_xy, r_m):
    """For each cell, the list of candidate sites within r_m metres."""
    tree = cKDTree(cand_xy)
    return tree.query_ball_point(cell_xy, r=r_m)


def solve_mclp(weights, near, n_cand, k, time_limit=120):
    """Exact MCLP with CBC. Returns chosen candidate ids and solver status."""
    idx = [i for i, w in enumerate(weights) if w > 0 and near[i]]
    prob = pulp.LpProblem("mclp", pulp.LpMaximize)
    y = [pulp.LpVariable(f"y{j}", cat="Binary") for j in range(n_cand)]
    z = {i: pulp.LpVariable(f"z{i}", lowBound=0, upBound=1) for i in idx}
    prob += pulp.lpSum(weights[i] * z[i] for i in idx)
    for i in idx:
        prob += z[i] <= pulp.lpSum(y[j] for j in near[i])
    prob += pulp.lpSum(y) == k
    status = prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit))
    chosen = [j for j in range(n_cand) if y[j].value() and y[j].value() > 0.5]
    return chosen, pulp.LpStatus[status]


def score(chosen, cand_xy, fire_xy, is_large):
    """How the real 2022-2025 wildfires would have been served by these bases."""
    d_km = cKDTree(cand_xy[chosen]).query(fire_xy)[0] / 1000
    minutes = DRONE["launch_min"] + d_km / DRONE["cruise_kmh"] * 60
    in_range = d_km <= DRONE["max_radius_km"]
    covered = d_km <= R_KM
    return dict(covered_share=float(covered.mean()),
                covered_share_large=float(covered[is_large].mean()),
                in_range_share=float(in_range.mean()),
                median_minutes_in_range=float(np.median(minutes[in_range])) if in_range.any() else None,
                median_distance_km=float(np.median(d_km)))


def main():
    grid, surf, cand, test = load()
    land = grid["is_land"].to_numpy()
    cell_xy = grid.loc[land, ["x", "y"]].to_numpy()
    cand_xy = cand[["x", "y"]].to_numpy()
    fire_xy = test[["x", "y"]].to_numpy()
    near = coverage_sets(cand_xy, cell_xy, R_KM * 1000)
    s = surf[land]

    weights = {
        "model": s["risk_model"].to_numpy(),
        "hist": s["risk_hist"].to_numpy(),
        "area": np.ones(land.sum()),
        "oracle": s["fires_test"].to_numpy(),
        "model_large": s["risk_model_large"].to_numpy(),
        "hist_large": s["risk_hist_large"].to_numpy(),
        "oracle_large": s["fires_test_large"].to_numpy(),
    }
    is_large = test["is_large"].to_numpy()
    rng = np.random.default_rng(SEED)
    stations = np.flatnonzero(cand["kind"].eq("weather station").to_numpy())
    towns_by_pop = cand[cand["kind"].eq("town")].sort_values("pop", ascending=False)["cand_id"].to_numpy()

    rows, placements = [], {}
    for k in BASE_COUNTS:
        for name, w in weights.items():
            t0 = time.time()
            chosen, status = solve_mclp(w, near, len(cand), k)
            r = score(chosen, cand_xy, fire_xy, is_large)
            rows.append(dict(strategy=name, k=k, solver=status, seconds=round(time.time() - t0, 1), **r))
            placements.setdefault(name, {})[k] = chosen
        chosen = list(towns_by_pop[:k])
        rows.append(dict(strategy="towns", k=k, solver="n/a", seconds=0, **score(chosen, cand_xy, fire_xy, is_large)))
        placements.setdefault("towns", {})[k] = [int(c) for c in chosen]
        draws = [score(list(rng.choice(stations, k, replace=False)), cand_xy, fire_xy, is_large) for _ in range(30)]
        rows.append(dict(strategy="random", k=k, solver="n/a", seconds=0,
                         **{m: float(np.mean([d[m] for d in draws])) for m in draws[0]}))
        print(f"k={k} done", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / "placement_results.csv", index=False)
    (RESULTS / "placements.json").write_text(json.dumps(
        {s: {str(k): [int(c) for c in v] for k, v in d.items()} for s, d in placements.items()}))
    table = res.pivot(index="k", columns="strategy", values="covered_share").round(3)
    print(f"\nShare of 2022-2025 wildfires reachable within {DRONE['target_min']:.0f} min "
          f"(radius {R_KM:.0f} km):")
    print(table[["oracle", "model", "hist", "area", "random", "towns", "model_large", "hist_large"]].to_string())
    tl = res.pivot(index="k", columns="strategy", values="covered_share_large").round(3)
    print(f"\nShare of 2022-2025 LARGE wildfires (>= {LARGE_HA:.0f} ha) reachable within the target time:")
    print(tl[["oracle_large", "model_large", "hist_large", "model", "hist", "area", "random", "towns"]].to_string())
    print("\nsolver status counts:", res["solver"].value_counts().to_dict())


if __name__ == "__main__":
    main()
