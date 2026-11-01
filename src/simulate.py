"""Step 4. Replay real fire seasons against a drone network.

The placement step asks "is each fire within range of a base?". That ignores
something important: on a bad lightning day dozens of fires start within a few
hours, and a base with one drone can only fly one mission at a time. This step
replays 2022-2025, call by call, and tracks every drone.

How one call is handled
  1. A fire is reported at some time on its ignition date.
  2. Bases within max_radius_km are considered. If none, the call is "out of range".
  3. With probability NO_FLY_PROB the weather keeps drones on the ground
     (thunderstorms, high wind, thick smoke). The call is "grounded".
  4. Otherwise, every drone at every base in range is checked. The one that can
     arrive first is sent. If it is busy, the call waits for it.
  5. That drone is then busy for: launch + fly out + observe + fly back + recharge.

Assumptions you should question (they are all in config.py or below):
  - Report times: the BC data has dates but not times, so the hour of each
    report is drawn at random from an afternoon-heavy distribution.
  - Weather grounding is random and independent of the fire. In reality the
    thunderstorms that start fires also ground drones, so this is optimistic.
  - Dispatch is greedy: each call gets the fastest drone at the moment it
    arrives. A smarter dispatcher might hold a drone back for a likely second call.

Outputs: results/simulation_summary.csv, results/simulation_busiest_day.json
"""
import json

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from config import PROCESSED, RESULTS, DRONE, TEST_YEARS, SEED

NO_FLY_PROB = 0.15
GIVE_UP_MIN = 180          # if no drone can get there within 3 hours, count as not reached
REPLICATIONS = 20
NETWORK_STRATEGY = "model"
NETWORK_SIZES = [20, 30, 40, 60]
DRONES_PER_BASE = [1, 2, 3]


def report_minutes(cause, rng):
    """Minute of the day each call comes in. An assumption: see module docstring."""
    mu = np.where(cause == "Lightning", 16.0, 14.0)
    sd = np.where(cause == "Lightning", 3.5, 4.0)
    return np.clip(rng.normal(mu, sd), 0, 23.99) * 60


def run_once(calls, base_xy, drones_per_base, rng):
    d = DRONE
    minutes = calls["day"].to_numpy() * 1440.0 + report_minutes(calls["fire_cause"].to_numpy(), rng)
    order = np.argsort(minutes)
    xy = calls[["x", "y"]].to_numpy()
    tree = cKDTree(base_xy)
    near = tree.query_ball_point(xy, r=d["max_radius_km"] * 1000)
    free_at = np.zeros((len(base_xy), drones_per_base))   # minute each drone is next free
    flight_min = np.zeros(len(base_xy))
    grounded = rng.random(len(calls)) < NO_FLY_PROB

    status = np.empty(len(calls), dtype=object)
    first_eyes = np.full(len(calls), np.nan)   # minutes from report to drone overhead
    waited = np.zeros(len(calls))
    base_used = np.full(len(calls), -1)
    for i in order:
        t = minutes[i]
        if not near[i]:
            status[i] = "out of range"
            continue
        if grounded[i]:
            status[i] = "grounded by weather"
            continue
        bases = np.array(near[i])
        dist_km = np.hypot(*(base_xy[bases] - xy[i]).T) / 1000
        fly = dist_km / d["cruise_kmh"] * 60
        start = np.maximum(t, free_at[bases])                   # (n_bases, drones)
        arrive = start + d["launch_min"] + fly[:, None]
        b, k = np.unravel_index(np.argmin(arrive), arrive.shape)
        if arrive[b, k] - t > GIVE_UP_MIN:
            status[i] = "drones busy"
            continue
        first_eyes[i] = arrive[b, k] - t
        waited[i] = start[b, k] - t
        busy_until = arrive[b, k] + d["observe_min"] + fly[b] + d["recharge_min"]
        free_at[bases[b], k] = busy_until
        flight_min[bases[b]] += 2 * fly[b] + d["observe_min"]
        base_used[i] = bases[b]
        status[i] = "reached"

    out = pd.DataFrame(dict(status=status, first_eyes=first_eyes, waited=waited,
                            base=base_used, minute=minutes))
    return out, flight_min


def summarize(calls, log, flight_min, n_bases, drones_per_base):
    within = log["first_eyes"] <= DRONE["target_min"]
    days = calls["date"].dt.normalize()
    busiest = days.value_counts().idxmax()
    on_busiest = (days == busiest).to_numpy()
    seasons = len(TEST_YEARS)
    return dict(
        calls=len(calls),
        within_target=float(within.mean()),
        reached_any_time=float((log["status"] == "reached").mean()),
        out_of_range=float((log["status"] == "out of range").mean()),
        grounded=float((log["status"] == "grounded by weather").mean()),
        drones_busy=float((log["status"] == "drones busy").mean()),
        delayed_by_queue=float((log["waited"] > 5).mean()),
        median_first_eyes_min=float(log["first_eyes"].median()),
        busiest_day=str(busiest.date()),
        busiest_day_calls=int(on_busiest.sum()),
        busiest_day_within_target=float(within[on_busiest].mean()),
        flight_hours_per_drone_per_season=float(flight_min.sum() / 60 / (n_bases * drones_per_base) / seasons),
    )


def main():
    cand = pd.read_parquet(PROCESSED / "candidates.parquet")
    placements = json.loads((RESULTS / "placements.json").read_text())
    fires = pd.read_parquet(PROCESSED / "fires.parquet")
    fires = fires[fires["fire_year"].isin(TEST_YEARS) & fires["date"].notna()].copy()
    fires["day"] = (fires["date"] - pd.Timestamp("2022-01-01")).dt.days
    scenarios = {
        "wildfires only": fires[fires["is_wildfire"]].reset_index(drop=True),
        # smoke reports that turn out to be nothing, and small nuisance fires,
        # still need someone to go and look. Drones could take that load.
        "all calls incl. smoke reports": fires[fires["is_call"]].reset_index(drop=True),
    }

    rows = []
    for scen, calls in scenarios.items():
        for k in NETWORK_SIZES:
            chosen = placements[NETWORK_STRATEGY][str(k)]
            base_xy = cand.loc[chosen, ["x", "y"]].to_numpy()
            static = float((cKDTree(base_xy).query(calls[["x", "y"]].to_numpy())[0] / 1000
                            <= (DRONE["target_min"] - DRONE["launch_min"]) / 60 * DRONE["cruise_kmh"]).mean())
            for m in DRONES_PER_BASE:
                reps = []
                for r in range(REPLICATIONS):
                    rng = np.random.default_rng(SEED + r)
                    log, fm = run_once(calls, base_xy, m, rng)
                    reps.append(summarize(calls, log, fm, k, m))
                rep = pd.DataFrame(reps)
                num = rep.select_dtypes("number")
                row = dict(scenario=scen, bases=k, drones_per_base=m, static_coverage=static,
                           busiest_day=rep["busiest_day"].iloc[0])
                for c in num.columns:
                    row[c] = float(num[c].mean())
                    if c in ("within_target", "busiest_day_within_target"):
                        row[c + "_p5"] = float(num[c].quantile(0.05))
                        row[c + "_p95"] = float(num[c].quantile(0.95))
                rows.append(row)
                print(f"{scen:32s} k={k:2d} m={m}  static={static:.3f}  "
                      f"sim within {DRONE['target_min']:.0f} min={row['within_target']:.3f}  "
                      f"busiest day={row['busiest_day_within_target']:.3f}  "
                      f"queue-delayed={row['delayed_by_queue']:.3f}", flush=True)

    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS / "simulation_summary.csv", index=False)

    # detailed replays of the busiest day, for the map
    calls = scenarios["wildfires only"]
    k = 30
    chosen = placements[NETWORK_STRATEGY][str(k)]
    base_xy = cand.loc[chosen, ["x", "y"]].to_numpy()
    day = calls["date"].dt.normalize().value_counts().idxmax()
    sel = (calls["date"].dt.normalize() == day).to_numpy()
    replays = {}
    for m in (1, 2):
        log, _ = run_once(calls, base_xy, m, np.random.default_rng(SEED))
        day_log = pd.concat([calls.loc[sel, ["fire_number", "fire_cause", "x", "y",
                                             "current_size"]].reset_index(drop=True),
                             log.loc[sel].reset_index(drop=True)], axis=1)
        day_log["minute"] = day_log["minute"] - (day - pd.Timestamp("2022-01-01")).days * 1440
        day_log["base"] = [int(chosen[b]) if b >= 0 else None for b in day_log["base"]]
        replays[str(m)] = json.loads(day_log.to_json(orient="records"))
        print(f"busiest day {day.date()}, {m} drone(s) per base:",
              day_log["status"].value_counts().to_dict(),
              "within target:", round(float((day_log["first_eyes"] <= DRONE["target_min"]).mean()), 3))
    (RESULTS / "simulation_busiest_day.json").write_text(json.dumps(dict(
        date=str(day.date()), bases=k, strategy=NETWORK_STRATEGY, base_ids=chosen,
        replays=replays)))

if __name__ == "__main__":
    main()
