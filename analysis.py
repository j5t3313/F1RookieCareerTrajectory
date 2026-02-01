import pandas as pd
import numpy as np
from scipy import stats
from scipy.special import expit
from typing import Optional
from dataclasses import dataclass
from pathlib import Path
from config import OUTPUT_DIR


@dataclass
class BootstrapResult:
    estimate: float
    se: float
    ci_lower: float
    ci_upper: float
    n_bootstrap: int


@dataclass
class CorrelationResult:
    predictor: str
    outcome: str
    spearman_rho: float
    spearman_p: float
    kendall_tau: float
    kendall_p: float
    n: int


@dataclass
class PermutationTestResult:
    observed_statistic: float
    p_value: float
    n_permutations: int
    null_distribution_mean: float
    null_distribution_std: float


@dataclass
class OrdinalRegressionResult:
    predictor: str
    coefficient: float
    se: float
    ci_lower: float
    ci_upper: float
    odds_ratio: float
    odds_ratio_ci_lower: float
    odds_ratio_ci_upper: float


def filter_analysis_sample(
    rookie_summaries: pd.DataFrame,
    exclude_rookie_vs_rookie: bool = False,
    exclude_right_censored: bool = True,
    minimum_races: Optional[int] = None
) -> pd.DataFrame:
    
    df = rookie_summaries.copy()
    
    if exclude_right_censored:
        df = df[df['outcome_level'].notna()]
    
    if exclude_rookie_vs_rookie:
        df = df[~df['flags'].apply(lambda x: 'rookie_vs_rookie' in x if isinstance(x, list) else False)]
    
    if minimum_races is not None:
        df = df[df['total_races'] >= minimum_races]
    
    return df


def calculate_correlation(
    df: pd.DataFrame,
    predictor_col: str,
    outcome_col: str = 'outcome_level'
) -> CorrelationResult:
    
    valid_df = df[[predictor_col, outcome_col]].dropna()
    
    if len(valid_df) < 3:
        return None
    
    x = valid_df[predictor_col].values
    y = valid_df[outcome_col].values
    
    spearman_rho, spearman_p = stats.spearmanr(x, y)
    kendall_tau, kendall_p = stats.kendalltau(x, y)
    
    return CorrelationResult(
        predictor=predictor_col,
        outcome=outcome_col,
        spearman_rho=spearman_rho,
        spearman_p=spearman_p,
        kendall_tau=kendall_tau,
        kendall_p=kendall_p,
        n=len(valid_df)
    )


def bootstrap_correlation(
    df: pd.DataFrame,
    predictor_col: str,
    outcome_col: str = 'outcome_level',
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    random_state: int = 42
) -> BootstrapResult:
    
    valid_df = df[[predictor_col, outcome_col]].dropna()
    
    if len(valid_df) < 3:
        return None
    
    rng = np.random.default_rng(random_state)
    
    bootstrap_correlations = []
    n = len(valid_df)
    
    for _ in range(n_bootstrap):
        indices = rng.choice(n, size=n, replace=True)
        sample = valid_df.iloc[indices]
        
        rho, _ = stats.spearmanr(sample[predictor_col], sample[outcome_col])
        if not np.isnan(rho):
            bootstrap_correlations.append(rho)
    
    bootstrap_correlations = np.array(bootstrap_correlations)
    
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_correlations, alpha / 2 * 100)
    ci_upper = np.percentile(bootstrap_correlations, (1 - alpha / 2) * 100)
    
    observed_rho, _ = stats.spearmanr(valid_df[predictor_col], valid_df[outcome_col])
    
    return BootstrapResult(
        estimate=observed_rho,
        se=np.std(bootstrap_correlations),
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n_bootstrap=len(bootstrap_correlations)
    )


def permutation_test_correlation(
    df: pd.DataFrame,
    predictor_col: str,
    outcome_col: str = 'outcome_level',
    n_permutations: int = 10000,
    random_state: int = 42
) -> PermutationTestResult:
    
    valid_df = df[[predictor_col, outcome_col]].dropna()
    
    if len(valid_df) < 3:
        return None
    
    rng = np.random.default_rng(random_state)
    
    observed_rho, _ = stats.spearmanr(valid_df[predictor_col], valid_df[outcome_col])
    
    x = valid_df[predictor_col].values
    y = valid_df[outcome_col].values
    
    null_correlations = []
    
    for _ in range(n_permutations):
        y_permuted = rng.permutation(y)
        rho, _ = stats.spearmanr(x, y_permuted)
        if not np.isnan(rho):
            null_correlations.append(rho)
    
    null_correlations = np.array(null_correlations)
    
    p_value = np.mean(np.abs(null_correlations) >= np.abs(observed_rho))
    
    return PermutationTestResult(
        observed_statistic=observed_rho,
        p_value=p_value,
        n_permutations=len(null_correlations),
        null_distribution_mean=np.mean(null_correlations),
        null_distribution_std=np.std(null_correlations)
    )


def ordinal_logistic_regression_single_predictor(
    df: pd.DataFrame,
    predictor_col: str,
    outcome_col: str = 'outcome_level',
    n_bootstrap: int = 2000,
    confidence_level: float = 0.95,
    random_state: int = 42
) -> OrdinalRegressionResult:
    
    valid_df = df[[predictor_col, outcome_col]].dropna()
    
    if len(valid_df) < 5:
        return None
    
    x = valid_df[predictor_col].values
    y = valid_df[outcome_col].values.astype(int)
    
    unique_outcomes = np.sort(np.unique(y))
    
    def neg_log_likelihood(params, x, y, unique_outcomes):
        n_thresholds = len(unique_outcomes) - 1
        thresholds = params[:n_thresholds]
        beta = params[n_thresholds]
        
        if not np.all(np.diff(thresholds) > 0):
            return 1e10
        
        linear_pred = x * beta
        
        log_lik = 0
        for i, yi in enumerate(y):
            outcome_idx = np.where(unique_outcomes == yi)[0][0]
            
            if outcome_idx == 0:
                prob = expit(thresholds[0] - linear_pred[i])
            elif outcome_idx == len(unique_outcomes) - 1:
                prob = 1 - expit(thresholds[-1] - linear_pred[i])
            else:
                prob_upper = expit(thresholds[outcome_idx] - linear_pred[i])
                prob_lower = expit(thresholds[outcome_idx - 1] - linear_pred[i])
                prob = prob_upper - prob_lower
            
            prob = np.clip(prob, 1e-10, 1 - 1e-10)
            log_lik += np.log(prob)
        
        return -log_lik
    
    from scipy.optimize import minimize
    
    n_thresholds = len(unique_outcomes) - 1
    initial_thresholds = np.linspace(-2, 2, n_thresholds)
    initial_beta = 0.0
    initial_params = np.concatenate([initial_thresholds, [initial_beta]])
    
    result = minimize(
        neg_log_likelihood,
        initial_params,
        args=(x, y, unique_outcomes),
        method='Nelder-Mead',
        options={'maxiter': 5000}
    )
    
    beta_estimate = result.x[-1]
    
    rng = np.random.default_rng(random_state)
    bootstrap_betas = []
    n = len(valid_df)
    
    for _ in range(n_bootstrap):
        indices = rng.choice(n, size=n, replace=True)
        x_boot = x[indices]
        y_boot = y[indices]
        
        if len(np.unique(y_boot)) < 2:
            continue
        
        try:
            boot_result = minimize(
                neg_log_likelihood,
                initial_params,
                args=(x_boot, y_boot, unique_outcomes),
                method='Nelder-Mead',
                options={'maxiter': 2000}
            )
            if boot_result.success or boot_result.fun < 1e9:
                bootstrap_betas.append(boot_result.x[-1])
        except:
            continue
    
    bootstrap_betas = np.array(bootstrap_betas)
    
    if len(bootstrap_betas) < 100:
        return None
    
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_betas, alpha / 2 * 100)
    ci_upper = np.percentile(bootstrap_betas, (1 - alpha / 2) * 100)
    se = np.std(bootstrap_betas)
    
    odds_ratio = np.exp(beta_estimate)
    or_ci_lower = np.exp(ci_lower)
    or_ci_upper = np.exp(ci_upper)
    
    return OrdinalRegressionResult(
        predictor=predictor_col,
        coefficient=beta_estimate,
        se=se,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        odds_ratio=odds_ratio,
        odds_ratio_ci_lower=or_ci_lower,
        odds_ratio_ci_upper=or_ci_upper
    )


@dataclass
class SensitivityAnalysisResult:
    scenario: str
    n_observations: int
    quali_gap_correlation: Optional[float]
    quali_gap_ci_lower: Optional[float]
    quali_gap_ci_upper: Optional[float]
    quali_gap_p_value: Optional[float]
    pace_gap_correlation: Optional[float]
    pace_gap_ci_lower: Optional[float]
    pace_gap_ci_upper: Optional[float]
    pace_gap_p_value: Optional[float]
    quali_h2h_correlation: Optional[float]
    quali_h2h_ci_lower: Optional[float]
    quali_h2h_ci_upper: Optional[float]
    quali_h2h_p_value: Optional[float]


def run_sensitivity_analysis(rookie_summaries: pd.DataFrame) -> pd.DataFrame:
    scenarios = [
        {
            'name': 'full_sample',
            'exclude_rookie_vs_rookie': False,
            'minimum_races': None
        },
        {
            'name': 'exclude_rookie_vs_rookie',
            'exclude_rookie_vs_rookie': True,
            'minimum_races': None
        },
        {
            'name': 'minimum_10_races',
            'exclude_rookie_vs_rookie': False,
            'minimum_races': 10
        },
        {
            'name': 'minimum_15_races',
            'exclude_rookie_vs_rookie': False,
            'minimum_races': 15
        },
        {
            'name': 'strict_filter',
            'exclude_rookie_vs_rookie': True,
            'minimum_races': 10
        }
    ]
    
    results = []
    
    for scenario in scenarios:
        df = filter_analysis_sample(
            rookie_summaries,
            exclude_rookie_vs_rookie=scenario['exclude_rookie_vs_rookie'],
            exclude_right_censored=True,
            minimum_races=scenario['minimum_races']
        )
        
        quali_gap_boot = bootstrap_correlation(df, 'quali_gap_median_weighted')
        quali_gap_perm = permutation_test_correlation(df, 'quali_gap_median_weighted')
        
        pace_gap_boot = bootstrap_correlation(df, 'pace_gap_median_weighted')
        pace_gap_perm = permutation_test_correlation(df, 'pace_gap_median_weighted')
        
        quali_h2h_boot = bootstrap_correlation(df, 'quali_h2h_pct')
        quali_h2h_perm = permutation_test_correlation(df, 'quali_h2h_pct')
        
        result = SensitivityAnalysisResult(
            scenario=scenario['name'],
            n_observations=len(df),
            quali_gap_correlation=quali_gap_boot.estimate if quali_gap_boot else None,
            quali_gap_ci_lower=quali_gap_boot.ci_lower if quali_gap_boot else None,
            quali_gap_ci_upper=quali_gap_boot.ci_upper if quali_gap_boot else None,
            quali_gap_p_value=quali_gap_perm.p_value if quali_gap_perm else None,
            pace_gap_correlation=pace_gap_boot.estimate if pace_gap_boot else None,
            pace_gap_ci_lower=pace_gap_boot.ci_lower if pace_gap_boot else None,
            pace_gap_ci_upper=pace_gap_boot.ci_upper if pace_gap_boot else None,
            pace_gap_p_value=pace_gap_perm.p_value if pace_gap_perm else None,
            quali_h2h_correlation=quali_h2h_boot.estimate if quali_h2h_boot else None,
            quali_h2h_ci_lower=quali_h2h_boot.ci_lower if quali_h2h_boot else None,
            quali_h2h_ci_upper=quali_h2h_boot.ci_upper if quali_h2h_boot else None,
            quali_h2h_p_value=quali_h2h_perm.p_value if quali_h2h_perm else None
        )
        
        results.append(result)
    
    return pd.DataFrame([vars(r) for r in results])


def run_full_analysis(rookie_summaries: pd.DataFrame) -> dict:
    analysis_df = filter_analysis_sample(
        rookie_summaries,
        exclude_rookie_vs_rookie=False,
        exclude_right_censored=True
    )
    
    predictors = [
        'quali_gap_median_weighted',
        'pace_gap_median_weighted', 
        'quali_h2h_pct',
        'race_h2h_pct'
    ]
    
    correlation_results = []
    bootstrap_results = []
    permutation_results = []
    regression_results = []
    
    for predictor in predictors:
        corr = calculate_correlation(analysis_df, predictor)
        if corr:
            correlation_results.append(vars(corr))
        
        boot = bootstrap_correlation(analysis_df, predictor)
        if boot:
            bootstrap_results.append({
                'predictor': predictor,
                **vars(boot)
            })
        
        perm = permutation_test_correlation(analysis_df, predictor)
        if perm:
            permutation_results.append({
                'predictor': predictor,
                **vars(perm)
            })
        
        reg = ordinal_logistic_regression_single_predictor(analysis_df, predictor)
        if reg:
            regression_results.append(vars(reg))
    
    sensitivity_results = run_sensitivity_analysis(rookie_summaries)
    
    return {
        'correlations': pd.DataFrame(correlation_results),
        'bootstrap': pd.DataFrame(bootstrap_results),
        'permutation_tests': pd.DataFrame(permutation_results),
        'ordinal_regression': pd.DataFrame(regression_results),
        'sensitivity': sensitivity_results,
        'analysis_sample': analysis_df
    }


def save_analysis_results(results: dict):
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results['correlations'].to_csv(output_path / 'correlations.csv', index=False)
    results['bootstrap'].to_csv(output_path / 'bootstrap_results.csv', index=False)
    results['permutation_tests'].to_csv(output_path / 'permutation_tests.csv', index=False)
    results['ordinal_regression'].to_csv(output_path / 'ordinal_regression.csv', index=False)
    results['sensitivity'].to_csv(output_path / 'sensitivity_analysis.csv', index=False)
    results['analysis_sample'].to_csv(output_path / 'analysis_sample.csv', index=False)
