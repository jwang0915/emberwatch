"""Run the drought and weather steps, after run_all.py.

    python fetch_weather.py   # once, needs internet
    python run_weather.py     # about 15 minutes

  drought.py             drought features for each cell and season
  risk_model_drought.py  does drought known before the season fix the yearly forecast?
  monthly_model.py       month-ahead forecasts, and drones re-sited every month
  regional_weather.py    weather used for regional totals, the version that helps
"""
import time

import drought
import risk_model_drought
import monthly_model
import regional_weather

for step in (drought, risk_model_drought, monthly_model, regional_weather):
    t0 = time.time()
    print(f"\n##### {step.__name__} #####", flush=True)
    step.main()
    print(f"({time.time() - t0:.0f} s)", flush=True)
