import fastf1
import pandas as pd
import time
import random
import json
from pathlib import Path
from datetime import datetime
from config import ROOKIE_COHORT, CACHE_DIR, DATA_DIR


SESSION_DELAY_MIN = 8
SESSION_DELAY_MAX = 15
YEAR_DELAY = 120
MAX_RETRIES = 5
RETRY_BASE_DELAY = 60


def initialize_cache():
    cache_path = Path(CACHE_DIR)
    cache_path.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_path))


def log_progress(message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


def random_delay(min_seconds: int, max_seconds: int):
    delay = random.uniform(min_seconds, max_seconds)
    time.sleep(delay)


def get_season_schedule(year: int) -> pd.DataFrame:
    schedule = fastf1.get_event_schedule(year)
    schedule = schedule[schedule['EventFormat'].notna()]
    return schedule


def load_session_with_retry(year: int, round_number: int, session_type: str):
    for attempt in range(MAX_RETRIES):
        try:
            session = fastf1.get_session(year, round_number, session_type)
            session.load()
            
            if session_type == 'R' and (not hasattr(session, 'laps') or session.laps.empty):
                raise Exception("Race session loaded but contains no lap data")
            if session_type == 'Q' and (not hasattr(session, 'results') or session.results.empty):
                raise Exception("Qualifying session loaded but contains no results")
            
            return session
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait_time = RETRY_BASE_DELAY * (2 ** attempt) + random.uniform(0, 30)
                log_progress(f"  Attempt {attempt + 1} failed: {str(e)[:80]}. Retrying in {wait_time:.0f}s...")
                time.sleep(wait_time)
            else:
                log_progress(f"  All {MAX_RETRIES} attempts failed for {year} R{round_number} {session_type}")
                return None
    return None


def extract_qualifying_results(session) -> pd.DataFrame:
    if session is None:
        return pd.DataFrame()
    
    try:
        results = session.results[['Abbreviation', 'TeamName', 'Q1', 'Q2', 'Q3', 'Position']].copy()
        
        for col in ['Q1', 'Q2', 'Q3']:
            results[f'{col}_seconds'] = results[col].apply(
                lambda x: x.total_seconds() if pd.notna(x) else None
            )
        
        results['best_quali_time'] = results[['Q1_seconds', 'Q2_seconds', 'Q3_seconds']].min(axis=1)
        results['event_name'] = session.event['EventName']
        results['year'] = session.event.year
        results['round'] = session.event['RoundNumber']
        
        return results
    except Exception:
        return pd.DataFrame()


def extract_race_laps(session) -> pd.DataFrame:
    if session is None:
        return pd.DataFrame()
    
    try:
        laps = session.laps[[
            'Driver', 'LapNumber', 'LapTime', 'Compound', 'TyreLife', 
            'Stint', 'TrackStatus', 'PitOutTime', 'PitInTime', 'IsAccurate'
        ]].copy()
        
        laps['lap_time_seconds'] = laps['LapTime'].apply(
            lambda x: x.total_seconds() if pd.notna(x) else None
        )
        laps['event_name'] = session.event['EventName']
        laps['year'] = session.event.year
        laps['round'] = session.event['RoundNumber']
        
        return laps
    except Exception:
        return pd.DataFrame()


def extract_race_results(session) -> pd.DataFrame:
    if session is None:
        return pd.DataFrame()
    
    try:
        results = session.results[[
            'Abbreviation', 'TeamName', 'Position', 'Points', 'Status', 'GridPosition'
        ]].copy()
        
        results['event_name'] = session.event['EventName']
        results['year'] = session.event.year
        results['round'] = session.event['RoundNumber']
        
        return results
    except Exception:
        return pd.DataFrame()


def load_checkpoint() -> dict:
    checkpoint_path = Path(DATA_DIR) / 'checkpoint.json'
    if checkpoint_path.exists():
        with open(checkpoint_path, 'r') as f:
            return json.load(f)
    return {'completed_sessions': []}


def save_checkpoint(completed_sessions: list):
    data_path = Path(DATA_DIR)
    data_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = data_path / 'checkpoint.json'
    with open(checkpoint_path, 'w') as f:
        json.dump({'completed_sessions': completed_sessions}, f)


def save_incremental_data(qualifying: list, race_laps: list, race_results: list):
    data_path = Path(DATA_DIR)
    data_path.mkdir(parents=True, exist_ok=True)
    
    if qualifying:
        df = pd.concat(qualifying, ignore_index=True)
        df.to_parquet(data_path / 'qualifying_raw_incremental.parquet', index=False)
    
    if race_laps:
        df = pd.concat(race_laps, ignore_index=True)
        df.to_parquet(data_path / 'race_laps_raw_incremental.parquet', index=False)
    
    if race_results:
        df = pd.concat(race_results, ignore_index=True)
        df.to_parquet(data_path / 'race_results_raw_incremental.parquet', index=False)


def collect_all_seasons() -> dict:
    years = sorted(set(entry.year for entry in ROOKIE_COHORT))
    
    checkpoint = load_checkpoint()
    completed = set(checkpoint['completed_sessions'])
    
    all_qualifying = []
    all_race_laps = []
    all_race_results = []
    
    data_path = Path(DATA_DIR)
    if (data_path / 'qualifying_raw_incremental.parquet').exists():
        all_qualifying.append(pd.read_parquet(data_path / 'qualifying_raw_incremental.parquet'))
    if (data_path / 'race_laps_raw_incremental.parquet').exists():
        all_race_laps.append(pd.read_parquet(data_path / 'race_laps_raw_incremental.parquet'))
    if (data_path / 'race_results_raw_incremental.parquet').exists():
        all_race_results.append(pd.read_parquet(data_path / 'race_results_raw_incremental.parquet'))
    
    total_years = len(years)
    
    for year_idx, year in enumerate(years):
        log_progress(f"Processing {year} season ({year_idx + 1}/{total_years})...")
        
        schedule = get_season_schedule(year)
        total_rounds = len(schedule)
        
        for round_idx, (_, event) in enumerate(schedule.iterrows()):
            round_num = event['RoundNumber']
            event_name = event['EventName']
            session_key = f"{year}_R{round_num}"
            
            if session_key in completed:
                log_progress(f"  Skipping {event_name} (already completed)")
                continue
            
            log_progress(f"  Round {round_num}/{total_rounds}: {event_name}")
            
            log_progress(f"    Loading qualifying...")
            quali_session = load_session_with_retry(year, round_num, 'Q')
            if quali_session is not None:
                quali_df = extract_qualifying_results(quali_session)
                if not quali_df.empty:
                    all_qualifying.append(quali_df)
                    log_progress(f"    Qualifying: {len(quali_df)} drivers")
            
            random_delay(SESSION_DELAY_MIN, SESSION_DELAY_MAX)
            
            log_progress(f"    Loading race...")
            race_session = load_session_with_retry(year, round_num, 'R')
            if race_session is not None:
                laps_df = extract_race_laps(race_session)
                if not laps_df.empty:
                    all_race_laps.append(laps_df)
                    log_progress(f"    Race laps: {len(laps_df)} laps")
                
                results_df = extract_race_results(race_session)
                if not results_df.empty:
                    all_race_results.append(results_df)
            
            completed.add(session_key)
            save_checkpoint(list(completed))
            save_incremental_data(all_qualifying, all_race_laps, all_race_results)
            
            if round_idx < total_rounds - 1:
                random_delay(SESSION_DELAY_MIN, SESSION_DELAY_MAX)
        
        if year_idx < total_years - 1:
            log_progress(f"Completed {year}. Waiting {YEAR_DELAY}s before next season...")
            time.sleep(YEAR_DELAY)
    
    log_progress("Data collection complete.")
    
    return {
        'qualifying': pd.concat(all_qualifying, ignore_index=True) if all_qualifying else pd.DataFrame(),
        'race_laps': pd.concat(all_race_laps, ignore_index=True) if all_race_laps else pd.DataFrame(),
        'race_results': pd.concat(all_race_results, ignore_index=True) if all_race_results else pd.DataFrame()
    }


def save_raw_data(data: dict):
    data_path = Path(DATA_DIR)
    data_path.mkdir(parents=True, exist_ok=True)
    
    data['qualifying'].to_parquet(data_path / 'qualifying_raw.parquet', index=False)
    data['race_laps'].to_parquet(data_path / 'race_laps_raw.parquet', index=False)
    data['race_results'].to_parquet(data_path / 'race_results_raw.parquet', index=False)


def load_raw_data() -> dict:
    data_path = Path(DATA_DIR)
    
    return {
        'qualifying': pd.read_parquet(data_path / 'qualifying_raw.parquet'),
        'race_laps': pd.read_parquet(data_path / 'race_laps_raw.parquet'),
        'race_results': pd.read_parquet(data_path / 'race_results_raw.parquet')
    }