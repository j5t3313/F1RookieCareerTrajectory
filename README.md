# F1 Rookie Performance Analysis

Statistical analysis of whether rookie-season teammate-relative metrics predict long-term F1 career outcomes. Covers all rookies debuting between 2019 and 2025 (16 drivers, 22 teammate pairings).

## Research Question

Do qualifying gaps, race pace gaps, and head-to-head win rates against teammates during a driver's rookie season correlate with career trajectory?

## Results Summary

Spearman rank correlations trend in the expected directions but largely fail to reach significance at n=12. Race pace gap shows the strongest association (rho = -0.57, p = 0.053). Sensitivity analysis reveals that removing rookie-versus-rookie pairings (Schumacher/Mazepin) shifts all primary correlations to significance, indicating the signal is dominated by benchmark quality rather than rookie talent.

Full writeup: [Substack link]

## Project Structure

```
.
├── main.py                 # Pipeline orchestrator (CLI entry point)
├── config.py               # Rookie cohort definitions, outcome coding, paths
├── data_collection.py      # FastF1 API collection with checkpointing
├── preprocessing.py        # Lap filtering, pairing aggregation, metric computation
├── analysis.py             # Correlations, bootstrap CIs, permutation tests, regression
├── visualization.py        # All plot generation
├── msc_vs_kmag.py          # Supplementary: Schumacher vs Magnussen 2022
├── requirements.txt
├── cache/                  # FastF1 session cache (not tracked in git)
├── data/                   # Raw parquet files (not tracked in git)
│   ├── qualifying_raw.parquet
│   ├── race_laps_raw.parquet
│   ├── race_results_raw.parquet
│   └── checkpoint.json
└── output/                 # Analysis results and figures
    ├── analysis_sample.csv
    ├── correlations.csv
    ├── bootstrap_results.csv
    ├── permutation_tests.csv
    ├── ordinal_regression.csv
    ├── sensitivity_analysis.csv
    └── *.png / *.pdf
```

## Pipeline

The pipeline runs in four stages, each executable independently:

```
python main.py full          # Run all stages sequentially
python main.py collect       # Stage 1: Collect data from FastF1 API
python main.py preprocess    # Stage 2: Filter laps, compute metrics
python main.py analyze       # Stage 3: Statistical analysis
python main.py visualize     # Stage 4: Generate plots
```

### Data Collection

Sources lap timing, qualifying results, and race results from the FastF1 API for all seasons 2019 through 2025. Implements conservative rate limiting (8 to 15 second randomized delays between sessions, 120 second delays between seasons) with exponential backoff retries and per-session checkpointing. Full collection takes approximately 8 to 12 hours. The checkpoint system (`data/checkpoint.json`) allows interruption and resumption without data loss.

### Preprocessing

Filters race laps to representative samples by removing: first laps of each stint, safety car and virtual safety car periods, pit in/out laps, laps flagged inaccurate by FastF1, and outliers outside the 5th to 95th percentile of remaining lap times. Requires a minimum of 10 clean laps per driver per race. Computes four metrics per pairing: median qualifying gap (%), median race pace gap (%), qualifying head-to-head win rate, and race finishing head-to-head win rate. Multi-teammate rookies are aggregated via race-count-weighted averages.

### Analysis

Spearman rank correlations and Kendall's tau between each metric and the ordinal outcome variable. Bootstrap confidence intervals (10,000 resamples). Permutation tests (10,000 shuffles). Single-predictor ordinal logistic regression with bootstrap standard errors. Sensitivity analysis across five sample specifications: full sample, excluding rookie-versus-rookie pairings, minimum 10 races, minimum 15 races, and a strict filter combining exclusions.

### Visualization

Generates scatter plots of rookie metrics against career outcomes, a correlation forest plot with bootstrap confidence intervals, sensitivity analysis comparison, per-pairing qualifying gap breakdown, focal rookie four-panel comparison, head-to-head strip plots, and a 2025 rookies in historical context overlay.

## Rookie Cohort

| Driver | Year | Team | Teammate(s) | Outcome |
|--------|------|------|-------------|---------|
| Lando Norris | 2019 | McLaren | Sainz | Race winner |
| George Russell | 2019 | Williams | Kubica | Race winner |
| Alexander Albon | 2019 | Toro Rosso / Red Bull | Kvyat, Verstappen | Podium finisher |
| Nicholas Latifi | 2020 | Williams | Russell | Short career |
| Yuki Tsunoda | 2021 | AlphaTauri | Gasly | Multi-season limited |
| Mick Schumacher | 2021 | Haas | Mazepin | Short career |
| Nikita Mazepin | 2021 | Haas | Schumacher | Sub-season |
| Zhou Guanyu | 2022 | Alfa Romeo | Bottas | Multi-season limited |
| Oscar Piastri | 2023 | McLaren | Norris | Race winner |
| Nyck de Vries | 2023 | AlphaTauri | Tsunoda | Sub-season |
| Logan Sargeant | 2023 | Williams | Albon | Short career |
| Jack Doohan | 2025 | Alpine | Gasly | Sub-season |
| Franco Colapinto | 2024 | Williams | Albon | Right-censored |
| Kimi Antonelli | 2025 | Mercedes | Russell | Right-censored |
| Oliver Bearman | 2025 | Haas | Ocon | Right-censored |
| Isack Hadjar | 2025 | Racing Bulls | Tsunoda, Lawson | Right-censored |
| Gabriel Bortoleto | 2025 | Sauber | Hulkenberg | Right-censored |

## Outcome Coding

1. Sub-season: Career shorter than one full season
2. Short career: One to two full seasons
3. Multi-season limited: Multiple seasons, no podiums
4. Established with podiums: Podium finisher
5. Race winner or above

2025 rookies with ongoing careers are treated as right-censored and excluded from correlation analysis.

## Requirements

Python 3.10 or higher. Install dependencies:

```
pip install -r requirements.txt
```

FastF1 3.0 or higher is required. Note that the FastF1 cache format is version-specific. Running the pipeline with different FastF1 versions against the same cache directory may cause errors.

## Data

Raw data is collected via the FastF1 API, which sources from F1's livetiming servers. The API is fan-maintained. The rate limiting in this pipeline is intentionally conservative to minimize load on the service.

Parquet files and the FastF1 cache are not tracked in git due to size. To reproduce, run `python main.py collect` (allow 8 to 12 hours) followed by subsequent pipeline stages.

## License

MIT
