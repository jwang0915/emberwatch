"""Shared settings for the whole pipeline.

Change numbers here, not inside the scripts, so every step stays consistent.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
MAP_DIR = ROOT / "map"

BUNDLE = RAW / "bc_wildfire_bundle.json"

# ---- Grid ------------------------------------------------------------------
# All geometry is in BC Albers, EPSG:3005, which is in metres.
# The grid is 164 x 140 cells of 10 km. Row 0 is the NORTH edge.
GRID = dict(x0=250_000, y0=350_000, x1=1_890_000, y1=1_750_000, cell_m=10_000)
NX = (GRID["x1"] - GRID["x0"]) // GRID["cell_m"]
NY = (GRID["y1"] - GRID["y0"]) // GRID["cell_m"]
MIN_LAND_FRAC = 0.05  # a cell counts as BC land if at least 5% of it is mapped BEC

# ---- Time split ------------------------------------------------------------
# BC fire years run April to March, so FIRE_YEAR is used, not calendar year.
FIRST_YEAR = 2000
HISTORY_WINDOW = 10           # years of history used for lag features
TRAIN_YEARS = list(range(2010, 2020))   # target years the model learns from
VALID_YEARS = [2020, 2021]              # used to pick model settings
TEST_YEARS = [2022, 2023, 2024, 2025]   # never touched until the end
PLANNING_YEAR = 2022  # bases are chosen with data up to the end of 2021

# ---- Drone assumptions -----------------------------------------------------
# These are assumptions, not specs of a real product. See README section
# "Drone assumptions" for where they come from and why they matter.
DRONE = dict(
    cruise_kmh=100.0,        # long-range VTOL fixed wing
    launch_min=3.0,          # dock opens, drone launches and climbs
    target_min=30.0,         # goal: first eyes on a new fire within 30 minutes
    max_radius_km=60.0,      # beyond this it can't go out, observe, and come back
    observe_min=15.0,        # time spent over the fire streaming video
    recharge_min=40.0,       # battery swap or charge after landing
)


def coverage_radius_km(d=DRONE):
    """Distance reachable within the target time, after launch overhead."""
    return (d["target_min"] - d["launch_min"]) / 60.0 * d["cruise_kmh"]


BASE_COUNTS = [5, 10, 15, 20, 25, 30, 40, 50, 60]
SEED = 42
