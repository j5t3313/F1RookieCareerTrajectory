import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional
from config import ROOKIE_COHORT, RookieEntry, TeammateAssignment, MINIMUM_RACE_THRESHOLD, DATA_DIR
from pathlib import Path


@dataclass
class QualifyingComparison:
    rookie: str
    teammate: str
    year: int
    round: int
    event_name: str
    rookie_time: float
    teammate_time: float
    gap_seconds: float
    gap_percent: float
    rookie_ahead: bool


@dataclass
class RacePaceComparison:
    rookie: str
    teammate: str
    year: int
    round: int
    event_name: str
    rookie_median_pace: float
    teammate_median_pace: float
    rookie_lap_count: int
    teammate_lap_count: int
    gap_seconds: float
    gap_percent: float


@dataclass
class RaceResultComparison:
    rookie: str
    teammate: str
    year: int
    round: int
    event_name: str
    rookie_position: Optional[int]
    teammate_position: Optional[int]
    rookie_points: float
    teammate_points: float
    rookie_ahead: Optional[bool]
    both_classified: bool


def get_teammate_for_round(entry: RookieEntry, round_number: int) -> Optional[TeammateAssignment]:
    for teammate in entry.teammates:
        if teammate.start_round <= round_number <= teammate.end_round:
            return teammate
    return None


def calculate_qualifying_comparison(
    quali_df: pd.DataFrame,
    rookie_abbrev: str,
    teammate_abbrev: str,
    year: int,
    round_number: int
) -> Optional[QualifyingComparison]:
    
    session_data = quali_df[(quali_df['year'] == year) & (quali_df['round'] == round_number)]
    
    rookie_row = session_data[session_data['Abbreviation'] == rookie_abbrev]
    teammate_row = session_data[session_data['Abbreviation'] == teammate_abbrev]
    
    if rookie_row.empty or teammate_row.empty:
        return None
    
    rookie_time = rookie_row['best_quali_time'].iloc[0]
    teammate_time = teammate_row['best_quali_time'].iloc[0]
    
    if pd.isna(rookie_time) or pd.isna(teammate_time):
        return None
    
    gap_seconds = rookie_time - teammate_time
    gap_percent = (gap_seconds / teammate_time) * 100
    
    return QualifyingComparison(
        rookie=rookie_abbrev,
        teammate=teammate_abbrev,
        year=year,
        round=round_number,
        event_name=rookie_row['event_name'].iloc[0],
        rookie_time=rookie_time,
        teammate_time=teammate_time,
        gap_seconds=gap_seconds,
        gap_percent=gap_percent,
        rookie_ahead=rookie_time < teammate_time
    )


def filter_representative_laps(laps_df: pd.DataFrame, driver: str, year: int, round_number: int) -> pd.DataFrame:
    driver_laps = laps_df[
        (laps_df['Driver'] == driver) &
        (laps_df['year'] == year) &
        (laps_df['round'] == round_number)
    ].copy()
    
    driver_laps = driver_laps[driver_laps['LapNumber'] > 1]
    driver_laps = driver_laps[driver_laps['TrackStatus'] == '1']
    driver_laps = driver_laps[driver_laps['PitOutTime'].isna()]
    driver_laps = driver_laps[driver_laps['PitInTime'].isna()]
    driver_laps = driver_laps[driver_laps['lap_time_seconds'].notna()]
    driver_laps = driver_laps[driver_laps['IsAccurate'] == True]
    
    if len(driver_laps) > 0:
        q1 = driver_laps['lap_time_seconds'].quantile(0.05)
        q99 = driver_laps['lap_time_seconds'].quantile(0.95)
        driver_laps = driver_laps[
            (driver_laps['lap_time_seconds'] >= q1) &
            (driver_laps['lap_time_seconds'] <= q99)
        ]
    
    return driver_laps


def calculate_race_pace_comparison(
    laps_df: pd.DataFrame,
    rookie_abbrev: str,
    teammate_abbrev: str,
    year: int,
    round_number: int,
    event_name: str
) -> Optional[RacePaceComparison]:
    
    rookie_laps = filter_representative_laps(laps_df, rookie_abbrev, year, round_number)
    teammate_laps = filter_representative_laps(laps_df, teammate_abbrev, year, round_number)
    
    if len(rookie_laps) < 10 or len(teammate_laps) < 10:
        return None
    
    rookie_median = rookie_laps['lap_time_seconds'].median()
    teammate_median = teammate_laps['lap_time_seconds'].median()
    
    gap_seconds = rookie_median - teammate_median
    gap_percent = (gap_seconds / teammate_median) * 100
    
    return RacePaceComparison(
        rookie=rookie_abbrev,
        teammate=teammate_abbrev,
        year=year,
        round=round_number,
        event_name=event_name,
        rookie_median_pace=rookie_median,
        teammate_median_pace=teammate_median,
        rookie_lap_count=len(rookie_laps),
        teammate_lap_count=len(teammate_laps),
        gap_seconds=gap_seconds,
        gap_percent=gap_percent
    )


def calculate_race_result_comparison(
    results_df: pd.DataFrame,
    rookie_abbrev: str,
    teammate_abbrev: str,
    year: int,
    round_number: int
) -> Optional[RaceResultComparison]:
    
    session_data = results_df[(results_df['year'] == year) & (results_df['round'] == round_number)]
    
    rookie_row = session_data[session_data['Abbreviation'] == rookie_abbrev]
    teammate_row = session_data[session_data['Abbreviation'] == teammate_abbrev]
    
    if rookie_row.empty or teammate_row.empty:
        return None
    
    rookie_pos = rookie_row['Position'].iloc[0]
    teammate_pos = teammate_row['Position'].iloc[0]
    rookie_points = rookie_row['Points'].iloc[0]
    teammate_points = teammate_row['Points'].iloc[0]
    
    rookie_pos = None if pd.isna(rookie_pos) else int(rookie_pos)
    teammate_pos = None if pd.isna(teammate_pos) else int(teammate_pos)
    
    both_classified = rookie_pos is not None and teammate_pos is not None
    rookie_ahead = rookie_pos < teammate_pos if both_classified else None
    
    return RaceResultComparison(
        rookie=rookie_abbrev,
        teammate=teammate_abbrev,
        year=year,
        round=round_number,
        event_name=rookie_row['event_name'].iloc[0],
        rookie_position=rookie_pos,
        teammate_position=teammate_pos,
        rookie_points=rookie_points if pd.notna(rookie_points) else 0,
        teammate_points=teammate_points if pd.notna(teammate_points) else 0,
        rookie_ahead=rookie_ahead,
        both_classified=both_classified
    )


def process_rookie_pairings(data: dict) -> dict:
    quali_df = data['qualifying']
    laps_df = data['race_laps']
    results_df = data['race_results']
    
    all_quali_comparisons = []
    all_pace_comparisons = []
    all_result_comparisons = []
    
    for entry in ROOKIE_COHORT:
        year = entry.year
        
        year_rounds = quali_df[quali_df['year'] == year]['round'].unique()
        
        for round_number in sorted(year_rounds):
            teammate = get_teammate_for_round(entry, round_number)
            if teammate is None:
                continue
            
            quali_comp = calculate_qualifying_comparison(
                quali_df, entry.abbreviation, teammate.abbreviation, year, round_number
            )
            if quali_comp:
                all_quali_comparisons.append(quali_comp)
            
            event_name = quali_df[
                (quali_df['year'] == year) & (quali_df['round'] == round_number)
            ]['event_name'].iloc[0] if len(quali_df[(quali_df['year'] == year) & (quali_df['round'] == round_number)]) > 0 else ""
            
            pace_comp = calculate_race_pace_comparison(
                laps_df, entry.abbreviation, teammate.abbreviation, year, round_number, event_name
            )
            if pace_comp:
                all_pace_comparisons.append(pace_comp)
            
            result_comp = calculate_race_result_comparison(
                results_df, entry.abbreviation, teammate.abbreviation, year, round_number
            )
            if result_comp:
                all_result_comparisons.append(result_comp)
    
    return {
        'qualifying': pd.DataFrame([vars(c) for c in all_quali_comparisons]),
        'race_pace': pd.DataFrame([vars(c) for c in all_pace_comparisons]),
        'race_results': pd.DataFrame([vars(c) for c in all_result_comparisons])
    }


@dataclass
class PairingSummary:
    rookie: str
    rookie_name: str
    teammate: str
    teammate_name: str
    year: int
    team: str
    race_count: int
    quali_gap_mean: float
    quali_gap_median: float
    quali_gap_std: float
    quali_h2h_wins: int
    quali_h2h_total: int
    quali_h2h_pct: float
    pace_gap_mean: Optional[float]
    pace_gap_median: Optional[float]
    pace_gap_std: Optional[float]
    pace_comparisons: int
    race_h2h_wins: int
    race_h2h_total: int
    race_h2h_pct: float
    points_total: float
    teammate_points_total: float
    outcome_level: Optional[int]
    flags: list


def aggregate_pairing_statistics(comparisons: dict) -> pd.DataFrame:
    quali_df = comparisons['qualifying']
    pace_df = comparisons['race_pace']
    results_df = comparisons['race_results']
    
    summaries = []
    
    for entry in ROOKIE_COHORT:
        for teammate in entry.teammates:
            pairing_quali = quali_df[
                (quali_df['rookie'] == entry.abbreviation) &
                (quali_df['teammate'] == teammate.abbreviation) &
                (quali_df['year'] == entry.year)
            ]
            
            pairing_pace = pace_df[
                (pace_df['rookie'] == entry.abbreviation) &
                (pace_df['teammate'] == teammate.abbreviation) &
                (pace_df['year'] == entry.year)
            ]
            
            pairing_results = results_df[
                (results_df['rookie'] == entry.abbreviation) &
                (results_df['teammate'] == teammate.abbreviation) &
                (results_df['year'] == entry.year)
            ]
            
            race_count = len(pairing_quali)
            
            if race_count == 0:
                continue
            
            classified_results = pairing_results[pairing_results['both_classified']]
            
            flags = entry.flags.copy()
            if race_count < MINIMUM_RACE_THRESHOLD:
                flags.append('sub_threshold_sample')
            
            summary = PairingSummary(
                rookie=entry.abbreviation,
                rookie_name=entry.name,
                teammate=teammate.abbreviation,
                teammate_name=teammate.name,
                year=entry.year,
                team=entry.team,
                race_count=race_count,
                quali_gap_mean=pairing_quali['gap_percent'].mean(),
                quali_gap_median=pairing_quali['gap_percent'].median(),
                quali_gap_std=pairing_quali['gap_percent'].std(),
                quali_h2h_wins=pairing_quali['rookie_ahead'].sum(),
                quali_h2h_total=len(pairing_quali),
                quali_h2h_pct=(pairing_quali['rookie_ahead'].sum() / len(pairing_quali)) * 100,
                pace_gap_mean=pairing_pace['gap_percent'].mean() if len(pairing_pace) > 0 else None,
                pace_gap_median=pairing_pace['gap_percent'].median() if len(pairing_pace) > 0 else None,
                pace_gap_std=pairing_pace['gap_percent'].std() if len(pairing_pace) > 0 else None,
                pace_comparisons=len(pairing_pace),
                race_h2h_wins=classified_results['rookie_ahead'].sum() if len(classified_results) > 0 else 0,
                race_h2h_total=len(classified_results),
                race_h2h_pct=(classified_results['rookie_ahead'].sum() / len(classified_results)) * 100 if len(classified_results) > 0 else 0,
                points_total=pairing_results['rookie_points'].sum(),
                teammate_points_total=pairing_results['teammate_points'].sum(),
                outcome_level=entry.outcome_level.value if entry.outcome_level else None,
                flags=flags
            )
            
            summaries.append(summary)
    
    return pd.DataFrame([vars(s) for s in summaries])


def aggregate_rookie_statistics(pairing_summaries: pd.DataFrame) -> pd.DataFrame:
    rookie_stats = []
    
    for entry in ROOKIE_COHORT:
        rookie_pairings = pairing_summaries[
            (pairing_summaries['rookie'] == entry.abbreviation) &
            (pairing_summaries['year'] == entry.year)
        ]
        
        if len(rookie_pairings) == 0:
            continue
        
        total_races = rookie_pairings['race_count'].sum()
        
        weights = rookie_pairings['race_count'] / total_races
        
        weighted_quali_gap = (rookie_pairings['quali_gap_median'] * weights).sum()
        weighted_pace_gap = None
        
        pace_valid = rookie_pairings[rookie_pairings['pace_gap_median'].notna()]
        if len(pace_valid) > 0:
            pace_weights = pace_valid['pace_comparisons'] / pace_valid['pace_comparisons'].sum()
            weighted_pace_gap = (pace_valid['pace_gap_median'] * pace_weights).sum()
        
        total_quali_h2h_wins = rookie_pairings['quali_h2h_wins'].sum()
        total_quali_h2h = rookie_pairings['quali_h2h_total'].sum()
        total_race_h2h_wins = rookie_pairings['race_h2h_wins'].sum()
        total_race_h2h = rookie_pairings['race_h2h_total'].sum()
        
        all_flags = []
        for flags in rookie_pairings['flags']:
            all_flags.extend(flags)
        unique_flags = list(set(all_flags))
        
        rookie_stats.append({
            'rookie': entry.abbreviation,
            'rookie_name': entry.name,
            'year': entry.year,
            'team': entry.team,
            'total_races': total_races,
            'num_teammates': len(rookie_pairings),
            'quali_gap_median_weighted': weighted_quali_gap,
            'pace_gap_median_weighted': weighted_pace_gap,
            'quali_h2h_wins': total_quali_h2h_wins,
            'quali_h2h_total': total_quali_h2h,
            'quali_h2h_pct': (total_quali_h2h_wins / total_quali_h2h) * 100 if total_quali_h2h > 0 else None,
            'race_h2h_wins': total_race_h2h_wins,
            'race_h2h_total': total_race_h2h,
            'race_h2h_pct': (total_race_h2h_wins / total_race_h2h) * 100 if total_race_h2h > 0 else None,
            'outcome_level': entry.outcome_level.value if entry.outcome_level else None,
            'flags': unique_flags
        })
    
    return pd.DataFrame(rookie_stats)


def save_processed_data(comparisons: dict, pairing_summaries: pd.DataFrame, rookie_summaries: pd.DataFrame):
    data_path = Path(DATA_DIR)
    data_path.mkdir(parents=True, exist_ok=True)
    
    comparisons['qualifying'].to_parquet(data_path / 'qualifying_comparisons.parquet', index=False)
    comparisons['race_pace'].to_parquet(data_path / 'race_pace_comparisons.parquet', index=False)
    comparisons['race_results'].to_parquet(data_path / 'race_results_comparisons.parquet', index=False)
    pairing_summaries.to_parquet(data_path / 'pairing_summaries.parquet', index=False)
    rookie_summaries.to_parquet(data_path / 'rookie_summaries.parquet', index=False)
    
    comparisons['qualifying'].to_csv(data_path / 'qualifying_comparisons.csv', index=False)
    comparisons['race_pace'].to_csv(data_path / 'race_pace_comparisons.csv', index=False)
    comparisons['race_results'].to_csv(data_path / 'race_results_comparisons.csv', index=False)
    pairing_summaries.to_csv(data_path / 'pairing_summaries.csv', index=False)
    rookie_summaries.to_csv(data_path / 'rookie_summaries.csv', index=False)


def load_processed_data() -> tuple:
    data_path = Path(DATA_DIR)
    
    comparisons = {
        'qualifying': pd.read_parquet(data_path / 'qualifying_comparisons.parquet'),
        'race_pace': pd.read_parquet(data_path / 'race_pace_comparisons.parquet'),
        'race_results': pd.read_parquet(data_path / 'race_results_comparisons.parquet')
    }
    pairing_summaries = pd.read_parquet(data_path / 'pairing_summaries.parquet')
    rookie_summaries = pd.read_parquet(data_path / 'rookie_summaries.parquet')
    
    return comparisons, pairing_summaries, rookie_summaries
