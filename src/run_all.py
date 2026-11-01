"""Run the whole pipeline after the data bundle exists.

    python src/fetch_data.py   # once, needs internet, a few minutes
    python src/run_all.py      # about 10 minutes on a laptop
"""
import time

import build_dataset
import risk_model
import placement
import simulate
import export_map

for step in (build_dataset, risk_model, placement, simulate, export_map):
    t0 = time.time()
    print(f"\n##### {step.__name__} #####", flush=True)
    step.main()
    print(f"({time.time() - t0:.0f} s)", flush=True)
