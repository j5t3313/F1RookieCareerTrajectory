import pandas as pd
from pathlib import Path

DATA_DIR = './data'

quali_df = pd.read_parquet(Path(DATA_DIR) / 'qualifying_raw.parquet')
laps_df = pd.read_parquet(Path(DATA_DIR) / 'race_laps_raw.parquet')
results_df = pd.read_parquet(Path(DATA_DIR) / 'race_results_raw.parquet')

YEAR = 2022
MSC = 'MSC'
MAG = 'MAG'

q22 = quali_df[quali_df['year'] == YEAR]
rounds = sorted(q22['round'].unique())

quali_comps = []
for rd in rounds:
    rd_data = q22[q22['round'] == rd]
    msc = rd_data[rd_data['Abbreviation'] == MSC]
    mag = rd_data[rd_data['Abbreviation'] == MAG]
    if msc.empty or mag.empty:
        continue
    msc_t = msc['best_quali_time'].iloc[0]
    mag_t = mag['best_quali_time'].iloc[0]
    if pd.isna(msc_t) or pd.isna(mag_t):
        continue
    gap = (msc_t - mag_t) / mag_t * 100
    quali_comps.append({
        'round': rd, 'event': msc['event_name'].iloc[0],
        'gap_pct': gap, 'msc_ahead': msc_t < mag_t
    })

laps_22 = laps_df[laps_df['year'] == YEAR]

pace_comps = []
for rd in rounds:
    rd_laps = laps_22[laps_22['round'] == rd]
    if rd_laps.empty:
        continue
    event_name = rd_laps['event_name'].iloc[0]
    medians = {}
    for abbrev in [MSC, MAG]:
        dl = rd_laps[rd_laps['Driver'] == abbrev].copy()
        dl = dl[dl['LapNumber'] > 1]
        dl = dl[dl['TrackStatus'] == '1']
        dl = dl[dl['PitOutTime'].isna()]
        dl = dl[dl['PitInTime'].isna()]
        dl = dl[dl['IsAccurate'] == True]
        dl = dl[dl['lap_time_seconds'].notna()]
        if len(dl) > 0:
            q5 = dl['lap_time_seconds'].quantile(0.05)
            q95 = dl['lap_time_seconds'].quantile(0.95)
            dl = dl[(dl['lap_time_seconds'] >= q5) & (dl['lap_time_seconds'] <= q95)]
        if len(dl) >= 10:
            medians[abbrev] = dl['lap_time_seconds'].median()
    if MSC in medians and MAG in medians:
        gap = (medians[MSC] - medians[MAG]) / medians[MAG] * 100
        pace_comps.append({
            'round': rd, 'event': event_name, 'gap_pct': gap
        })

r22 = results_df[results_df['year'] == YEAR]

race_comps = []
for rd in rounds:
    rd_data = r22[r22['round'] == rd]
    msc = rd_data[rd_data['Abbreviation'] == MSC]
    mag = rd_data[rd_data['Abbreviation'] == MAG]
    if msc.empty or mag.empty:
        continue
    msc_pos = msc['Position'].iloc[0]
    mag_pos = mag['Position'].iloc[0]
    if pd.notna(msc_pos) and pd.notna(mag_pos):
        race_comps.append({
            'round': rd, 'event': rd_data['event_name'].iloc[0],
            'msc_pos': int(msc_pos), 'mag_pos': int(mag_pos),
            'msc_ahead': int(msc_pos) < int(mag_pos)
        })

print("=" * 60)
print("SCHUMACHER vs MAGNUSSEN — 2022 SEASON")
print("=" * 60)

qdf = pd.DataFrame(quali_comps)
print(f"\nQUALIFYING ({len(qdf)} sessions)")
print(f"  Median gap: {qdf['gap_pct'].median():+.3f}%")
print(f"  Mean gap:   {qdf['gap_pct'].mean():+.3f}%")
print(f"  H2H: MSC {qdf['msc_ahead'].sum()} - {(~qdf['msc_ahead']).sum()} MAG")
print(f"  H2H rate: {qdf['msc_ahead'].mean() * 100:.1f}%")

pdf = pd.DataFrame(pace_comps)
print(f"\nRACE PACE ({len(pdf)} races with sufficient laps)")
print(f"  Median gap: {pdf['gap_pct'].median():+.3f}%")
print(f"  Mean gap:   {pdf['gap_pct'].mean():+.3f}%")

rdf = pd.DataFrame(race_comps)
print(f"\nRACE FINISHING ({len(rdf)} races both classified)")
print(f"  H2H: MSC {rdf['msc_ahead'].sum()} - {(~rdf['msc_ahead']).sum()} MAG")
print(f"  H2H rate: {rdf['msc_ahead'].mean() * 100:.1f}%")

print(f"\n{'=' * 60}")
print("COMPARISON WITH 2021 ROOKIE SEASON")
print("=" * 60)
print(f"  2021 vs Mazepin:    Quali {-0.677:+.3f}% | Pace {-0.803:+.3f}% | Quali H2H 100.0% | Race H2H 75.0%")
print(f"  2022 vs Magnussen:  Quali {qdf['gap_pct'].median():+.3f}% | Pace {pdf['gap_pct'].median():+.3f}% | Quali H2H {qdf['msc_ahead'].mean()*100:.1f}% | Race H2H {rdf['msc_ahead'].mean()*100:.1f}%")