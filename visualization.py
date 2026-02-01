import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba
from pathlib import Path
from config import OUTPUT_DIR


plt.rcParams['figure.facecolor'] = '#ffffff'
plt.rcParams['axes.facecolor'] = '#ffffff'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.labelcolor'] = '#333333'
plt.rcParams['text.color'] = '#333333'
plt.rcParams['xtick.color'] = '#333333'
plt.rcParams['ytick.color'] = '#333333'
plt.rcParams['grid.color'] = '#cccccc'
plt.rcParams['grid.alpha'] = 0.5
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False


TEAM_COLORS = {
    'McLaren': '#FF8000',
    'Williams': '#005AFF',
    'Toro Rosso / Red Bull': '#1E41FF',
    'Toro Rosso': '#469BFF',
    'Red Bull': '#1E41FF',
    'AlphaTauri': '#2B4562',
    'Racing Bulls': '#6692FF',
    'VCARB': '#6692FF',
    'Haas': '#B6BABD',
    'Alfa Romeo': '#900000',
    'Sauber': '#52E252',
    'Audi': '#52E252',
    'Mercedes': '#00D2BE',
    'Alpine': '#0090FF',
    'Ferrari': '#DC0000',
}


DRIVER_COLORS = {
    'NOR': '#FF8000',
    'SAI': '#FF9E3D',
    'RUS': '#00D2BE',
    'KUB': '#00A89D',
    'ALB': '#005AFF',
    'KVY': '#469BFF',
    'VER': '#1E41FF',
    'LAT': '#003F8A',
    'TSU': '#2B4562',
    'GAS': '#0090FF',
    'MSC': '#B6BABD',
    'MAZ': '#8A8D8F',
    'ZHO': '#900000',
    'BOT': '#C40000',
    'PIA': '#FF9E3D',
    'DEV': '#4A7A9E',
    'SAR': '#003F8A',
    'COL': '#0047AB',
    'ANT': '#00A89D',
    'BEA': '#D3D3D3',
    'OCO': '#0078C1',
    'DOO': '#0090FF',
    'HAD': '#6692FF',
    'LAW': '#4169E1',
    'BOR': '#52E252',
    'HUL': '#3CB371',
}


OUTCOME_COLORS = {
    1: '#d62728',
    2: '#ff7f0e',
    3: '#bcbd22',
    4: '#2ca02c',
    5: '#1f77b4'
}

OUTCOME_LABELS = {
    1: 'Sub-season',
    2: 'Short career (1-2 seasons)',
    3: 'Multi-season, limited success',
    4: 'Established with podiums',
    5: 'Race winner+'
}


def get_driver_color(driver_abbrev: str, team: str = None) -> str:
    if driver_abbrev in DRIVER_COLORS:
        return DRIVER_COLORS[driver_abbrev]
    if team and team in TEAM_COLORS:
        return TEAM_COLORS[team]
    return '#666666'


def plot_qualifying_gap_vs_outcome(rookie_summaries: pd.DataFrame, output_path: Path):
    df = rookie_summaries[rookie_summaries['outcome_level'].notna()].copy()
    
    fig, ax = plt.subplots(figsize=(9, 6))
    
    for _, row in df.iterrows():
        outcome = int(row['outcome_level'])
        color = OUTCOME_COLORS.get(outcome, '#666666')
        
        jitter = np.random.uniform(-0.12, 0.12)
        
        ax.scatter(
            row['quali_gap_median_weighted'],
            outcome + jitter,
            c=color,
            s=90,
            alpha=0.8,
            edgecolors='#333333',
            linewidths=0.5,
            zorder=3
        )
        
        ax.annotate(
            row['rookie'],
            (row['quali_gap_median_weighted'], outcome + jitter),
            xytext=(4, 2),
            textcoords='offset points',
            fontsize=8,
            color='#333333',
            alpha=0.9
        )
    
    ax.axvline(x=0, color='#333333', linestyle='--', alpha=0.4, linewidth=1)
    
    ax.set_xlabel('Median Qualifying Gap to Teammate (%)\n← Faster | Slower →', fontsize=10)
    ax.set_ylabel('Career Outcome Level', fontsize=10)
    ax.set_title('Rookie Qualifying Performance vs Career Outcome', fontsize=12, fontweight='bold', pad=12)
    
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels([OUTCOME_LABELS[i] for i in [1, 2, 3, 4, 5]], fontsize=9)
    
    ax.grid(True, axis='x', alpha=0.4, linestyle='-', linewidth=0.5)
    ax.set_xlim(df['quali_gap_median_weighted'].min() - 0.4, df['quali_gap_median_weighted'].max() + 0.4)
    ax.set_ylim(0.5, 5.5)
    
    plt.tight_layout()
    plt.savefig(output_path / 'qualifying_gap_vs_outcome.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'qualifying_gap_vs_outcome.pdf', bbox_inches='tight')
    plt.close()


def plot_race_pace_vs_outcome(rookie_summaries: pd.DataFrame, output_path: Path):
    df = rookie_summaries[
        (rookie_summaries['outcome_level'].notna()) &
        (rookie_summaries['pace_gap_median_weighted'].notna())
    ].copy()
    
    fig, ax = plt.subplots(figsize=(9, 6))
    
    for _, row in df.iterrows():
        outcome = int(row['outcome_level'])
        color = OUTCOME_COLORS.get(outcome, '#666666')
        
        jitter = np.random.uniform(-0.12, 0.12)
        
        ax.scatter(
            row['pace_gap_median_weighted'],
            outcome + jitter,
            c=color,
            s=90,
            alpha=0.8,
            edgecolors='#333333',
            linewidths=0.5,
            zorder=3
        )
        
        ax.annotate(
            row['rookie'],
            (row['pace_gap_median_weighted'], outcome + jitter),
            xytext=(4, 2),
            textcoords='offset points',
            fontsize=8,
            color='#333333',
            alpha=0.9
        )
    
    ax.axvline(x=0, color='#333333', linestyle='--', alpha=0.4, linewidth=1)
    
    ax.set_xlabel('Median Race Pace Gap to Teammate (%)\n← Faster | Slower →', fontsize=10)
    ax.set_ylabel('Career Outcome Level', fontsize=10)
    ax.set_title('Rookie Race Pace vs Career Outcome', fontsize=12, fontweight='bold', pad=12)
    
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels([OUTCOME_LABELS[i] for i in [1, 2, 3, 4, 5]], fontsize=9)
    
    ax.grid(True, axis='x', alpha=0.4, linestyle='-', linewidth=0.5)
    ax.set_xlim(df['pace_gap_median_weighted'].min() - 0.4, df['pace_gap_median_weighted'].max() + 0.4)
    ax.set_ylim(0.5, 5.5)
    
    plt.tight_layout()
    plt.savefig(output_path / 'race_pace_vs_outcome.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'race_pace_vs_outcome.pdf', bbox_inches='tight')
    plt.close()


def plot_h2h_vs_outcome(rookie_summaries: pd.DataFrame, output_path: Path):
    df = rookie_summaries[rookie_summaries['outcome_level'].notna()].copy()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for _, row in df.iterrows():
        outcome = int(row['outcome_level'])
        color = OUTCOME_COLORS.get(outcome, '#666666')
        jitter = np.random.uniform(-0.12, 0.12)
        
        axes[0].scatter(
            row['quali_h2h_pct'],
            outcome + jitter,
            c=color,
            s=70,
            alpha=0.8,
            edgecolors='#333333',
            linewidths=0.5,
            zorder=3
        )
        axes[0].annotate(
            row['rookie'],
            (row['quali_h2h_pct'], outcome + jitter),
            xytext=(3, 2),
            textcoords='offset points',
            fontsize=7,
            color='#333333',
            alpha=0.85
        )
        
        if pd.notna(row['race_h2h_pct']):
            axes[1].scatter(
                row['race_h2h_pct'],
                outcome + jitter,
                c=color,
                s=70,
                alpha=0.8,
                edgecolors='#333333',
                linewidths=0.5,
                zorder=3
            )
            axes[1].annotate(
                row['rookie'],
                (row['race_h2h_pct'], outcome + jitter),
                xytext=(3, 2),
                textcoords='offset points',
                fontsize=7,
                color='#333333',
                alpha=0.85
            )
    
    for ax in axes:
        ax.axvline(x=50, color='#333333', linestyle='--', alpha=0.4, linewidth=1)
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.set_yticklabels(['Sub-season', 'Short career', 'Multi-season', 'Podiums', 'Winner+'], fontsize=8)
        ax.grid(True, axis='x', alpha=0.4, linestyle='-', linewidth=0.5)
        ax.set_xlim(-5, 105)
        ax.set_ylim(0.5, 5.5)
    
    axes[0].set_xlabel('Qualifying Head-to-Head Win Rate (%)', fontsize=10)
    axes[0].set_ylabel('Career Outcome Level', fontsize=10)
    axes[0].set_title('Qualifying H2H', fontsize=11, fontweight='bold')
    
    axes[1].set_xlabel('Race Finishing Head-to-Head Win Rate (%)', fontsize=10)
    axes[1].set_ylabel('Career Outcome Level', fontsize=10)
    axes[1].set_title('Race Finishing H2H', fontsize=11, fontweight='bold')
    
    fig.suptitle('Head-to-Head Performance vs Career Outcome', fontsize=12, fontweight='bold', y=1.01)
    
    plt.tight_layout()
    plt.savefig(output_path / 'h2h_vs_outcome.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'h2h_vs_outcome.pdf', bbox_inches='tight')
    plt.close()


def plot_correlation_forest(bootstrap_results: pd.DataFrame, output_path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    
    predictor_labels = {
        'quali_gap_median_weighted': 'Qualifying Gap (%)',
        'pace_gap_median_weighted': 'Race Pace Gap (%)',
        'quali_h2h_pct': 'Qualifying H2H Win Rate',
        'race_h2h_pct': 'Race H2H Win Rate'
    }
    
    y_positions = range(len(bootstrap_results))
    
    for i, (_, row) in enumerate(bootstrap_results.iterrows()):
        color = '#1f77b4' if row['ci_lower'] > 0 or row['ci_upper'] < 0 else '#666666'
        
        ax.errorbar(
            row['estimate'],
            i,
            xerr=[[row['estimate'] - row['ci_lower']], [row['ci_upper'] - row['estimate']]],
            fmt='o',
            color=color,
            ecolor=color,
            capsize=4,
            capthick=1.5,
            markersize=8,
            elinewidth=1.5,
            zorder=3
        )
        
        ax.annotate(
            f"ρ = {row['estimate']:.2f} [{row['ci_lower']:.2f}, {row['ci_upper']:.2f}]",
            (max(row['ci_upper'] + 0.08, 0.15), i),
            fontsize=9,
            color='#333333',
            va='center'
        )
    
    ax.axvline(x=0, color='#333333', linestyle='-', alpha=0.6, linewidth=1)
    
    ax.set_yticks(list(y_positions))
    ax.set_yticklabels([predictor_labels.get(p, p) for p in bootstrap_results['predictor']])
    
    ax.set_xlabel('Spearman Correlation (ρ) with Career Outcome', fontsize=10)
    ax.set_title('Correlation Estimates with 95% Bootstrap Confidence Intervals', fontsize=11, fontweight='bold', pad=12)
    
    ax.grid(True, axis='x', alpha=0.4, linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path / 'correlation_forest.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'correlation_forest.pdf', bbox_inches='tight')
    plt.close()


def plot_sensitivity_analysis(sensitivity_results: pd.DataFrame, output_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    
    scenarios = sensitivity_results['scenario'].values
    y_positions = range(len(scenarios))
    
    scenario_labels = {
        'full_sample': 'Full Sample',
        'exclude_rookie_vs_rookie': 'Excl. Rookie vs Rookie',
        'minimum_10_races': 'Min. 10 Races',
        'minimum_15_races': 'Min. 15 Races',
        'strict_filter': 'Strict Filter'
    }
    
    metrics = [
        ('quali_gap_correlation', 'quali_gap_ci_lower', 'quali_gap_ci_upper', 'Qualifying Gap'),
        ('pace_gap_correlation', 'pace_gap_ci_lower', 'pace_gap_ci_upper', 'Race Pace Gap'),
        ('quali_h2h_correlation', 'quali_h2h_ci_lower', 'quali_h2h_ci_upper', 'Qualifying H2H')
    ]
    
    for ax, (col, ci_lower, ci_upper, title) in zip(axes, metrics):
        for i, (_, row) in enumerate(sensitivity_results.iterrows()):
            if pd.notna(row[col]):
                lower_err = row[col] - row[ci_lower] if pd.notna(row[ci_lower]) else 0
                upper_err = row[ci_upper] - row[col] if pd.notna(row[ci_upper]) else 0
                
                sig = (row[ci_lower] > 0 or row[ci_upper] < 0) if pd.notna(row[ci_lower]) and pd.notna(row[ci_upper]) else False
                color = '#1f77b4' if sig else '#666666'
                
                ax.errorbar(
                    row[col],
                    i,
                    xerr=[[lower_err], [upper_err]],
                    fmt='o',
                    color=color,
                    ecolor=color,
                    capsize=3,
                    capthick=1,
                    markersize=6,
                    elinewidth=1,
                    zorder=3
                )
        
        ax.axvline(x=0, color='#333333', linestyle='-', alpha=0.5, linewidth=1)
        ax.set_yticks(list(y_positions))
        ax.set_yticklabels([scenario_labels.get(s, s) for s in scenarios], fontsize=9)
        ax.set_xlabel('Spearman ρ', fontsize=10)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.4, linestyle='-', linewidth=0.5)
    
    fig.suptitle('Sensitivity Analysis: Correlation Stability Across Sample Specifications', fontsize=11, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path / 'sensitivity_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'sensitivity_analysis.pdf', bbox_inches='tight')
    plt.close()


def plot_rookie_comparison_panel(pairing_summaries: pd.DataFrame, output_path: Path):
    focal_rookies = ['NOR', 'RUS', 'ALB', 'PIA']
    
    focal_data = pairing_summaries[pairing_summaries['rookie'].isin(focal_rookies)].copy()
    
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes.flatten()
    
    for ax, rookie in zip(axes, focal_rookies):
        rookie_data = focal_data[focal_data['rookie'] == rookie]
        
        if len(rookie_data) == 0:
            continue
        
        rookie_name = rookie_data['rookie_name'].iloc[0]
        year = rookie_data['year'].iloc[0]
        team = rookie_data['team'].iloc[0]
        driver_color = get_driver_color(rookie, team)
        
        metrics = ['quali_gap_median', 'pace_gap_median', 'quali_h2h_pct', 'race_h2h_pct']
        metric_labels = ['Quali Gap\n(%)', 'Pace Gap\n(%)', 'Quali H2H\n(%)', 'Race H2H\n(%)']
        
        x_positions = range(len(metrics))
        
        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            values = []
            for _, row in rookie_data.iterrows():
                if pd.notna(row.get(metric)):
                    values.append(row[metric])
            
            if values:
                value = np.mean(values)
                
                if 'h2h' in metric:
                    normalized = (value - 50) / 50
                else:
                    normalized = -value / 0.5
                
                bar_color = driver_color if normalized >= 0 else to_rgba(driver_color, 0.5)
                
                ax.bar(
                    i,
                    normalized,
                    color=bar_color,
                    edgecolor='#333333',
                    linewidth=0.5,
                    width=0.6
                )
                
                ax.annotate(
                    f'{value:.1f}',
                    (i, normalized),
                    ha='center',
                    va='bottom' if normalized >= 0 else 'top',
                    fontsize=9,
                    color='#333333'
                )
        
        ax.axhline(y=0, color='#333333', linestyle='-', alpha=0.5, linewidth=1)
        ax.set_xticks(list(x_positions))
        ax.set_xticklabels(metric_labels, fontsize=9)
        ax.set_title(f'{rookie_name} ({year})', fontsize=11, fontweight='bold', color=driver_color)
        ax.set_ylabel('Normalized Performance\n(positive = better)', fontsize=9)
        ax.grid(True, axis='y', alpha=0.4, linestyle='-', linewidth=0.5)
        
        teammate_info = ', '.join([f"vs {row['teammate_name']}" for _, row in rookie_data.iterrows()])
        ax.annotate(
            teammate_info,
            (0.5, -0.12),
            xycoords='axes fraction',
            ha='center',
            fontsize=8,
            color='#666666',
            style='italic'
        )
    
    fig.suptitle('Focal Rookies: Teammate-Relative Performance', fontsize=12, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig(output_path / 'focal_rookie_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'focal_rookie_comparison.pdf', bbox_inches='tight')
    plt.close()


def plot_2025_rookie_predictions(rookie_summaries: pd.DataFrame, analysis_results: dict, output_path: Path):
    rookies_2025 = rookie_summaries[
        (rookie_summaries['year'] == 2025) |
        ((rookie_summaries['rookie'] == 'COL') & (rookie_summaries['year'] == 2024))
    ].copy()
    
    historical = rookie_summaries[rookie_summaries['outcome_level'].notna()].copy()
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for _, row in historical.iterrows():
        outcome = int(row['outcome_level'])
        ax.scatter(
            row['quali_gap_median_weighted'],
            row['quali_h2h_pct'],
            c=OUTCOME_COLORS.get(outcome, '#888888'),
            s=60,
            alpha=0.6,
            edgecolors='#333333',
            linewidths=0.3,
            zorder=2
        )
        ax.annotate(
            row['rookie'],
            (row['quali_gap_median_weighted'], row['quali_h2h_pct']),
            xytext=(3, 3),
            textcoords='offset points',
            fontsize=7,
            color='#666666',
            alpha=0.8
        )
    
    for _, row in rookies_2025.iterrows():
        ax.scatter(
            row['quali_gap_median_weighted'],
            row['quali_h2h_pct'],
            c='#222222',
            s=140,
            alpha=0.95,
            edgecolors='#333333',
            linewidths=1.5,
            marker='D',
            zorder=4
        )
        ax.annotate(
            row['rookie_name'],
            (row['quali_gap_median_weighted'], row['quali_h2h_pct']),
            xytext=(6, 6),
            textcoords='offset points',
            fontsize=9,
            color='#333333',
            fontweight='bold'
        )
    
    ax.axvline(x=0, color='#333333', linestyle='--', alpha=0.3, linewidth=1)
    ax.axhline(y=50, color='#333333', linestyle='--', alpha=0.3, linewidth=1)
    
    ax.set_xlabel('Median Qualifying Gap to Teammate (%)\n← Faster | Slower →', fontsize=10)
    ax.set_ylabel('Qualifying Head-to-Head Win Rate (%)', fontsize=10)
    ax.set_title('2025 Rookies in Historical Context', fontsize=12, fontweight='bold', pad=12)
    
    legend_elements = [
        mpatches.Patch(color=OUTCOME_COLORS[5], label='Race Winner+', alpha=0.7),
        mpatches.Patch(color=OUTCOME_COLORS[4], label='Podium Finisher', alpha=0.7),
        mpatches.Patch(color=OUTCOME_COLORS[3], label='Multi-season Limited', alpha=0.7),
        mpatches.Patch(color=OUTCOME_COLORS[2], label='Short Career', alpha=0.7),
        mpatches.Patch(color=OUTCOME_COLORS[1], label='Sub-season', alpha=0.7),
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='#333333', 
                   markeredgecolor='#333333', markersize=8, label='2025 Rookies')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.95)
    
    ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path / '2025_rookies_context.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / '2025_rookies_context.pdf', bbox_inches='tight')
    plt.close()


def plot_pairing_detail(pairing_summaries: pd.DataFrame, output_path: Path):
    df = pairing_summaries.copy()
    df = df.sort_values(['year', 'rookie'])
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    y_positions = range(len(df))
    
    for i, (_, row) in enumerate(df.iterrows()):
        driver_color = get_driver_color(row['rookie'], row['team'])
        
        ax.barh(
            i,
            row['quali_gap_median'],
            color=driver_color,
            edgecolor='#333333',
            linewidth=0.5,
            alpha=0.8,
            height=0.7
        )
    
    ax.axvline(x=0, color='#333333', linestyle='-', alpha=0.6, linewidth=1)
    
    ax.set_yticks(list(y_positions))
    labels = [f"{row['rookie']} vs {row['teammate']} ({row['year']})" for _, row in df.iterrows()]
    ax.set_yticklabels(labels, fontsize=8)
    
    ax.set_xlabel('Median Qualifying Gap (%)\n← Faster | Slower →', fontsize=10)
    ax.set_title('Rookie-Teammate Qualifying Gaps by Pairing', fontsize=12, fontweight='bold', pad=12)
    
    ax.grid(True, axis='x', alpha=0.4, linestyle='-', linewidth=0.5)
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_path / 'pairing_detail.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'pairing_detail.pdf', bbox_inches='tight')
    plt.close()


def generate_all_visualizations(
    rookie_summaries: pd.DataFrame,
    pairing_summaries: pd.DataFrame,
    analysis_results: dict
):
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    
    plot_qualifying_gap_vs_outcome(rookie_summaries, output_path)
    plot_race_pace_vs_outcome(rookie_summaries, output_path)
    plot_h2h_vs_outcome(rookie_summaries, output_path)
    
    if 'bootstrap' in analysis_results and len(analysis_results['bootstrap']) > 0:
        plot_correlation_forest(analysis_results['bootstrap'], output_path)
    
    if 'sensitivity' in analysis_results and len(analysis_results['sensitivity']) > 0:
        plot_sensitivity_analysis(analysis_results['sensitivity'], output_path)
    
    plot_rookie_comparison_panel(pairing_summaries, output_path)
    plot_2025_rookie_predictions(rookie_summaries, analysis_results, output_path)
    plot_pairing_detail(pairing_summaries, output_path)