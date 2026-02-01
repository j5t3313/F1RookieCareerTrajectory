import sys
import pandas as pd
from pathlib import Path

from config import DATA_DIR, OUTPUT_DIR
from data_collection import (
    initialize_cache,
    collect_all_seasons,
    save_raw_data,
    load_raw_data
)
from preprocessing import (
    process_rookie_pairings,
    aggregate_pairing_statistics,
    aggregate_rookie_statistics,
    save_processed_data,
    load_processed_data
)
from analysis import run_full_analysis, save_analysis_results
from visualization import generate_all_visualizations


def run_data_collection():
    print("Initializing FastF1 cache...")
    initialize_cache()
    
    print("Collecting season data from FastF1 API...")
    raw_data = collect_all_seasons()
    
    print("Saving raw data...")
    save_raw_data(raw_data)
    
    return raw_data


def run_preprocessing(raw_data: dict = None):
    if raw_data is None:
        print("Loading raw data from disk...")
        raw_data = load_raw_data()
    
    print("Processing rookie-teammate pairings...")
    comparisons = process_rookie_pairings(raw_data)
    
    print("Aggregating pairing statistics...")
    pairing_summaries = aggregate_pairing_statistics(comparisons)
    
    print("Aggregating rookie-level statistics...")
    rookie_summaries = aggregate_rookie_statistics(pairing_summaries)
    
    print("Saving processed data...")
    save_processed_data(comparisons, pairing_summaries, rookie_summaries)
    
    return comparisons, pairing_summaries, rookie_summaries


def run_analysis(rookie_summaries: pd.DataFrame = None):
    if rookie_summaries is None:
        print("Loading processed data...")
        _, _, rookie_summaries = load_processed_data()
    
    print("Running statistical analysis...")
    results = run_full_analysis(rookie_summaries)
    
    print("Saving analysis results...")
    save_analysis_results(results)
    
    return results


def run_visualization(
    rookie_summaries: pd.DataFrame = None,
    pairing_summaries: pd.DataFrame = None,
    analysis_results: dict = None
):
    if rookie_summaries is None or pairing_summaries is None:
        print("Loading processed data...")
        _, pairing_summaries, rookie_summaries = load_processed_data()
    
    if analysis_results is None:
        print("Loading analysis results...")
        output_path = Path(OUTPUT_DIR)
        analysis_results = {
            'bootstrap': pd.read_csv(output_path / 'bootstrap_results.csv'),
            'sensitivity': pd.read_csv(output_path / 'sensitivity_analysis.csv')
        }
    
    print("Generating visualizations...")
    generate_all_visualizations(rookie_summaries, pairing_summaries, analysis_results)


def run_full_pipeline():
    print("=" * 60)
    print("F1 ROOKIE PERFORMANCE ANALYSIS PIPELINE")
    print("=" * 60)
    
    print("\n[1/4] DATA COLLECTION")
    print("-" * 40)
    raw_data = run_data_collection()
    
    print("\n[2/4] PREPROCESSING")
    print("-" * 40)
    comparisons, pairing_summaries, rookie_summaries = run_preprocessing(raw_data)
    
    print("\n[3/4] STATISTICAL ANALYSIS")
    print("-" * 40)
    analysis_results = run_analysis(rookie_summaries)
    
    print("\n[4/4] VISUALIZATION")
    print("-" * 40)
    run_visualization(rookie_summaries, pairing_summaries, analysis_results)
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\nOutputs saved to: {OUTPUT_DIR}/")
    
    return {
        'comparisons': comparisons,
        'pairing_summaries': pairing_summaries,
        'rookie_summaries': rookie_summaries,
        'analysis_results': analysis_results
    }


if __name__ == '__main__':
    if len(sys.argv) > 1:
        stage = sys.argv[1].lower()
        
        if stage == 'collect':
            run_data_collection()
        elif stage == 'preprocess':
            run_preprocessing()
        elif stage == 'analyze':
            run_analysis()
        elif stage == 'visualize':
            run_visualization()
        elif stage == 'full':
            run_full_pipeline()
        else:
            print(f"Unknown stage: {stage}")
            print("Available stages: collect, preprocess, analyze, visualize, full")
            sys.exit(1)
    else:
        run_full_pipeline()
