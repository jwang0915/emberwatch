# Emberwatch

Machine learning and integer programming for placing wildfire detection drones in British Columbia.

By James Wang, West Point Grey Academy, Vancouver.

- **Interactive map and results:** https://jwang0915.github.io/emberwatch/, built from `docs/`
- **Research poster:** [docs/Emberwatch_poster.pdf](docs/Emberwatch_poster.pdf)

Emberwatch answers one question with real data: if British Columbia built a network of docked detection drones, where should the docks go, and how much would it help?

It has three parts.

1. **A fire-risk model.** Predicts how many wildfires will start in each 10 km cell next season, from past fires, roads, recent burns and ecosystem type.
2. **A placement optimizer.** Picks the best k sites for drone docks so that as much fire risk as possible is within a 30-minute flight.
3. **A season replay.** Sends a simulated drone to every real wildfire from 2022 to 2025, one call at a time, with busy drones, weather groundings and lightning storms.

Everything is planned with data up to the end of 2021, then scored against the fires that actually started in 2022 to 2025. That split is the core of the project: the model and the optimizer never see the years they are graded on.

## Results

Scored on the 7,192 wildfires BC recorded in the 2022 to 2025 seasons.

**Where the bases go matters a lot.** Share of real wildfires within 30 minutes of a base:

| Bases | Optimized on risk model | Optimized on history | Spread evenly | Random weather stations | Biggest towns | Hindsight ceiling |
|---|---|---|---|---|---|---|
| 10 | 23.8% | 26.2% | 8.2% | 11.0% | 7.9% | 26.9% |
| 20 | 41.0% | 40.9% | 19.8% | 20.8% | 11.2% | 42.9% |
| 30 | 52.8% | 53.0% | 29.9% | 28.8% | 16.7% | 55.2% |
| 60 | 76.7% | 76.5% | 53.3% | 48.3% | 33.7% | 78.5% |

The hindsight column is optimized on the 2022 to 2025 fires themselves. No real planner could do that, so it marks the ceiling. Optimizing on past fires gets within 2 to 3 points of it.

**The machine-learning model ranks cells better, but that doesn't change where bases go.** One season ahead, the model puts 47.4% of next season's fires in its top 10% of cells, against 44.5% for the average of the past 10 seasons. For fires that grew past 10 ha, the gap is wider: 31.1% against 25.9% for the historical average. But once each base covers a 45 km circle, those cell-level differences average out, and a plain 22-year history picks sites just as well. That is a real finding, not a failure. It tells a planner where the effort should go.

**Distance on a map overstates what a network delivers.** With 30 bases, 52.8% of fires are within range on paper. The replay, with one drone per base, gets a drone overhead within 30 minutes for 37.2%. The gap comes from fires beyond 60 km, a 15% weather grounding rate, and queues on lightning days. A second drone at each base lifts it to 42.4%.

**Capacity is set by a few storm days.** On 22 July 2024, 139 new wildfires started, almost all from lightning. A 30-base, one-drone network got a drone overhead within 30 minutes for about 16% of them. Across a whole season each drone flies only about 29 hours. So a network sized for average days is badly short on the days that matter.

**Large fires are harder.** Fires that grew past 10 ha tend to start in remote country. The 30-base network covers 33% of them. Optimizing for large fires instead lifts that to 41 to 44%, at the cost of covering fewer fires overall.

## Drought and weather

The 2023 test showed the model missing most of the northern fires, so I added weather data to see if it would fix that. Monthly temperature, precipitation, root-zone soil wetness, topsoil wetness and snow depth came from NASA POWER for 479 points about 50 km apart, from 1999 to 2025. Every value is turned into an anomaly against that point's own 1999 to 2019 normal, so the test years never leak into it.

**Drought known before the season did not help.** Adding drought features known by 31 December made the yearly forecast worse. Adding everything known by 1 May, including April soil moisture, March snow and winter precipitation, left it about the same: 47.9% of next season's fires in the top 10% of cells, against 47.4% without. The 2023 forecast was still 46% too low. Over 2001 to 2021, no spring measure had a consistent, sensible relationship with how many fires a season had. Precipitation in the same summer did, with a correlation of minus 0.80. In BC the season is decided by summer weather, which nobody knows in May.

**A month-ahead forecast helps a little.** On the 1st of each month from May to September, a simple regression predicts each fire centre's fires for that month from last month's weather and from how busy the season has been so far. A month with half its normal precipitation is followed by about 40% more fires. A season running at twice its usual pace so far points to about 24% more. This cut the error in fire centre monthly totals from 51.3% to 47.8% and was better in three of the four test years. For May to September 2023 it forecast 1,714 fires, against 1,485 from history alone and 2,119 in reality. The north is still badly under-forecast: Prince George got 350 against 648 real fires.

**Moving drones each month helps more than the weather does.** If the 30 bases could be moved on the 1st of each month, they would reach these shares of the 6,334 wildfires from May to September 2022 to 2025:

| Network | Within 30 minutes | 2023 only |
|---|---|---|
| Fixed network chosen at the end of 2021 | 51.2% | 42.5% |
| Moved monthly, forecast from history | 54.7% | 43.7% |
| Moved monthly, forecast adjusted for weather | 55.7% | 44.6% |
| Moved monthly with perfect hindsight | 72.3% | 64.7% |

Moving monthly beat the fixed network in 15 of 20 months. The weather adjustment added one more point, about 62 fires over four seasons, but it was better in 9 months and worse in 8, so that gain may be noise. Putting the weather into the cell-by-cell tree model instead made it slightly worse, because a few dozen training months are not enough for it to learn weather effects without overfitting.

So drought data did not fix the 2023 miss. The honest result is that pre-season drought does not predict BC fire seasons, month-ahead weather helps a little at the regional scale, and flexibility in where drones sit matters more than a better forecast.

## How it works

`src/config.py` holds every setting. Change numbers there, not inside the scripts.

**Grid.** BC is cut into 10 km cells in BC Albers, the province's standard equal-area projection. 10,649 cells are on land.

**Step 1, `build_dataset.py`.** Turns the raw download into tables: fire records with their cell, ecosystem zone shares, road density, distance to the nearest town, burned area by year, and candidate base sites. Candidate sites are the 255 BC Wildfire Service weather stations that existed before 2022, since they already have power and a data link, plus 166 towns.

**Step 2, `risk_model.py`.** A LightGBM gradient-boosted model with a Poisson loss, the standard choice for counts. Each row is one cell in one season, and every feature is built only from earlier seasons. Trained on 2010 to 2019, settings tuned on 2020 and 2021, tested on 2022 to 2025. Two versions: all wildfires, and fires that grew past 10 ha. Compared against a simple historical average and a smoothed version of it.

**Step 3, `placement.py`.** A Maximal Covering Location Problem, a classic operations research model from Church and ReVelle, 1974. Choose k sites to maximize the total risk within 45 km of a chosen site. Solved exactly with the CBC solver through PuLP. Every run in the results reached a proven optimum.

**Step 4, `simulate.py`.** Replays every 2022 to 2025 wildfire in time order. Each call goes to whichever drone can arrive first. A drone is busy for launch, flight out, 15 minutes over the fire, flight back and a 40-minute recharge. 15% of calls are grounded by weather. Each setting is replayed 20 times with different random report times and groundings. Results vary by about one percentage point between replays.

**Backtest, `backtest.py`.** Pretends it is 31 December of the year before a chosen season, retrains the model and picks 30 bases with only what was known then, then scores them on that season. `python backtest.py 2023` writes `results/backtest_2023.json`. In 2023 the model forecast 1,449 wildfires against 2,296 real ones, and missed most of the northern fires. Its 30 bases had 43% of the season's fires within 30 minutes, against 30% for even spacing and 54% with perfect hindsight.

**Step 5, `export_map.py`.** Bundles the results into `map/index.html` and `docs/index.html`, a single page you can open in any browser. `docs/` is what GitHub Pages serves.

## Drone assumptions

These describe a long-range fixed-wing drone that takes off vertically from a charging dock. They are not the specs of a real product, and they drive every result, so they are the first thing to test.

| Setting | Value |
|---|---|
| Cruise speed | 100 km/h |
| Launch and climb | 3 min |
| Target response | 30 min, which gives a 45 km radius |
| Longest mission | 60 km out |
| Time over the fire | 15 min |
| Recharge after landing | 40 min |
| Calls grounded by weather | 15% |

## Limitations

- The fire records have a date but no time. Report times are drawn at random, heavier in the afternoon.
- Weather grounding is random. In reality the thunderstorms that start fires also ground drones, so the replay is optimistic on lightning days.
- Candidate sites ignore airspace, terrain, road access, land ownership and Indigenous rights and title.
- Since 4 November 2025, Transport Canada allows flights beyond visual line of sight in lower-risk conditions only: uncontrolled airspace, below 122 m, away from aerodromes, with a Level 1 Complex pilot certificate. Other flights need a special certificate. The model does not check any of this.
- Some large fires were large on purpose. BC monitors some remote fires instead of fighting them.
- The road layer is today's road map, so the few roads built after 2021 leak into the 2022 plan. Everything else the planner uses is from 2021 or earlier.
- Cost and staffing are not modelled.
- The dispatcher is greedy. A smarter one might hold a drone back when a second call is likely nearby.

## Review and next steps

Prof. Jane Z. Wang of UBC, Prof. Jiangchuan Liu of SFU and Prof. Xiaotao Bi of UBC, former director of UBC's Clean Energy Research Centre, reviewed this work. A technical report for the BC Wildfire Service is in preparation with Prof. Bi.

Planned next:

- **Real drone specs.** Replace the assumed drone with a real long-range drone's range, speed and recharge time.
- **Correlated weather.** Ground drones on the days and places with lightning, not at random, and see how much the busiest-day numbers drop.
- **Cost per fire reached.** Add a price per dock and per drone, and find the cheapest network that hits a target.
- **Sensitivity.** Rerun with other cruise speeds and response targets to find which assumption changes the answer most.
- **The 2026 season.** When BC publishes it, it is a fresh test year nothing here has seen.

## How to run it

Python 3.10 or newer.

```
pip install -r requirements.txt
cd src
python fetch_data.py   # downloads the BC data, a few minutes
python run_all.py      # about 10 minutes
```

Then open `map/index.html` or `docs/index.html`.

For the drought and weather steps:

```
python fetch_weather.py   # NASA POWER monthly weather, 5 to 10 minutes
python run_weather.py     # about 15 minutes
```

`fetch_data.py` downloads the same layers the prototype used. The prototype's own copy was fetched with the same requests from a browser, so this script has not been run against the live server yet. If its printed totals differ from the numbers below, check the request settings first.

- 87,006 fire records since 2000, of which 43,724 are wildfires
- 280 weather stations and 166 towns
- Burned area of about 38,000 km² in 2023 and 13,500 km² in 2018, counted as 1 km pixels, which runs a little above official hectare totals

## Data

All from the BC Data Catalogue under the Open Government Licence, British Columbia, fetched 19 September 2026:

- BC Wildfire Fire Incident Locations, Historical
- BC Wildfire Fire Perimeters, Historical
- BC Wildfire Service Weather Stations
- BC Major Cities Points
- Biogeoclimatic Ecosystem Classification zones
- Digital Road Atlas

Monthly weather from NASA POWER, power.larc.nasa.gov, based on the MERRA-2 reanalysis, fetched 21 September 2026.

Province outline from Natural Earth admin-1 boundaries, public domain, simplified to 200 m and saved as `data/raw/bc_outline.geojson`.

## Files

```
src/            the pipeline, one script per step
data/raw/       the downloaded BC data bundle and the province outline
data/processed/ tables made by steps 1 and 2, not stored here; run_all.py rebuilds them
results/        metrics, chosen sites, simulation output, saved models
map/            template.html, and the finished index.html
docs/           the GitHub Pages site: index.html and the poster PDF
```
