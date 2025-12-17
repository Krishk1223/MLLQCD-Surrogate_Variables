import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import logging
import time
import sys
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

"""
Preprocessing STEP 2: Sanity check on processed CSV data files (before averaging over time sources)
"""

def logging_setup(enable_logging=False, log_level=logging.INFO):
    """Setup logging configuration."""
    if not enable_logging:
        logging.disable(logging.CRITICAL)
        return None
    
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    log_path = project_root / "logs"
    log_path.mkdir(exist_ok=True)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_file = log_path / f"data_sanity_check_{timestamp}.log"
    
    logging.basicConfig(level=log_level,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        handlers=[
                            logging.FileHandler(log_file),
                            logging.StreamHandler()
                        ],
                        force=True)

    logger = logging.getLogger(__name__)
    logger.info(f"Sanity check logging enabled. Saving log file to {log_file}")
    return logger

class DataSanityChecker:
    def __init__(self, processed_path, results_path, min_T=10, enable_logging=False):
        """Initialize the DataSanityChecker with paths and parameters."""
        self.processed_path = Path(processed_path)
        self.results_path = Path(results_path)
        self.results_path.mkdir(exist_ok=True)
        self.enable_logging = enable_logging
        self.logger = logging_setup(enable_logging)
        self.min_T = int(min_T)
        
        # Data storage
        self.csv_files = []
        self.data_dictionary = {}
        self.numeric_arrays = {}
        self.results = {}

    #I/O, filtering and loading:
    def find_csv_files(self, pattern="*.csv"):
        """Find all CSV files matching the pattern in the processed data path."""
        if not self.processed_path.exists():
            raise FileNotFoundError(f"Processed data path {self.processed_path} does not exist.")
        self.csv_files = list(self.processed_path.rglob(pattern))
        return self.csv_files

    def dataframe_filtering(self, df, name):
        """Apply any necessary filtering to the dataframe such as header removal, config id removal etc."""
        # Check if DataFrame is empty
        if df.empty:
            if self.logger:
                self.logger.warning(f"Empty DataFrame found for {name}")
            return df, []
            
        first_row = df.iloc[0]
        is_header_row = False

        #check for keywords in first row from known header keywords
        header_keywords = ['config_id', 'τ_1']
        if any(str(first_row[col]).lower().strip() in hk.lower() for col in df.columns for hk in header_keywords):
            if self.logger:
                self.logger.info(f"Header row detected in {name}, removing first row.")
            df = df.iloc[1:].reset_index(drop=True)
        
        #numeric columns:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        removal_cols = []

        if 'config_id' in numeric_cols:
            removal_cols.append('config_id')
        
        #if you have a header and tau_0 then remove tau_0 too.
        tau_zero = 'τ_0'
        for col in numeric_cols:
            if str(col).lower().strip() == tau_zero.lower():
                removal_cols.append(col)
                if self.logger:
                    self.logger.info(f"τ_0 column detected in {name}, removing τ_0 column.")
        
        for col in removal_cols:
            if col in numeric_cols:
                numeric_cols.remove(col)
                if self.logger:
                    self.logger.info(f"Removing column {col} from numeric columns in {name}.")
        
        return df, numeric_cols
    
    def data_loading(self, require_numeric=True, drop_id=True):
        """Load CSV files into data dictionary and numeric arrays."""
        if not self.csv_files:
            self.find_csv_files()
        
        for file in self.csv_files:
            df = pd.read_csv(file)
            name = file.stem
            self.data_dictionary[name] = df

            filtered_df, numeric_cols = self.dataframe_filtering(df, name)
            
            # Check if filtered dataframe is empty
            if filtered_df.empty or len(filtered_df) == 0:
                if self.logger:
                    self.logger.warning(f"No data rows found in {name} after filtering. File may contain only headers.")
                continue
            
            if require_numeric and len(numeric_cols) == 0:
                if self.logger:
                    self.logger.warning(f"No numeric columns found in {name}. Skipping.")
                continue
            
            self.numeric_arrays[name] = filtered_df[numeric_cols].values.astype(float)
        
        return self.data_dictionary


    #Basic sanity checks:
    def shape_consistency_check(self):
        """Check if all numeric arrays have consistent shapes."""
        summary = {
            'ensembles' : {},
            'same_T': True,
            'common_T': None,
            'errors': []
        }
        T_vals = []
        for name, array in self.numeric_arrays.items():
            if array.ndim != 2:
                summary['errors'].append(f"Data for {name} is not a 2D array.")
                continue
            N, T = array.shape # Num samples and Time extent
            summary['ensembles'][name] = {'Number of Configs': int(N), 'Time Extent': int(T)}
            T_vals.append(T)
            if T < self.min_T:
                summary['errors'].append(f"Data for {name} has T={T} < {self.min_T} (minimum Time extent).")
            
        if len(T_vals) == 0:
            if self.logger:
                self.logger.warning("No valid data arrays found for shape consistency check.")
            summary['errors'].append("No valid data arrays found for shape consistency check.")
            self.results['shape_consistency'] = summary
            return summary
        
        common_T = max(set(T_vals), key=T_vals.count) # most common time extent value
        summary['common_T'] = common_T
        summary['same_T'] = all(t == common_T for t in T_vals)
        
        self.results['shape_consistency'] = summary
        return summary
        
    def finite_checks(self, threshold=0.05):
        """ Checks for infinite, NaN and non finite entries in numeric arrays."""
        report = {}
        for name, array in self.numeric_arrays.items():
            N, T = array.shape
            is_finite = np.isfinite(array)
            per_config_finite = is_finite.sum(axis=1) # Checks finite entries per row/config (axis=1)
            per_config_nonfinite = T - per_config_finite
            nonfinite_fraction = per_config_nonfinite / float(T)

            # Identify which configs have non-finite entries above required threshold
            problematic_config_id = np.where(nonfinite_fraction > threshold)[0].tolist()

            #overall per time slice missing counts:
            nonfinite_entries_per_tau = (N - is_finite.sum(axis=0)).astype(int)
            nonfinite_entries_per_tau_fraction = (nonfinite_entries_per_tau.astype(float)/float(N)).tolist()
            
            # report storing
            ensemble_stats = {
                'total_configs': int(N),
                'Time extent': int(T),
                'total non-finite entries': int(np.sum(~is_finite)),
                'per time slice missing counts': nonfinite_entries_per_tau.tolist(), 
                'per time slice missing as a fraction of total configs': nonfinite_entries_per_tau_fraction,
                'number of problematic configs': int(len(problematic_config_id)),
                'problematic_config_ids': problematic_config_id
            }
            report[name] = ensemble_stats
            
        self.results['finite_checks'] = report
        return report

    def negative_checks(self):
        """ Checks for negative entries in the numeric arrays."""
        report = {}
        for name, array in self.numeric_arrays.items():
            N, T = array.shape
            first_negative_config = np.full(N, T, dtype=int) # default T means no -ve found
            neg_mask = array < 0
            first_negative_config = np.where(neg_mask.any(axis=1), neg_mask.argmax(axis=1), T)

            neg_counts_per_tau = np.sum(array < 0, axis=0)

            if np.any(neg_counts_per_tau > 0):
                first_negative_tau = int(np.argmax(neg_counts_per_tau > 0))
            else:
                first_negative_tau = T # No negatives at all
            
            #Reccomendations:
            cutoff = T
            tau_max_reccomended = min(cutoff, first_negative_tau)

            report[name] = {
                'total_configs' : int(N),
                'Time extent' : int(T),
                'first negative entry per config' : first_negative_config.tolist(),
                'negative counts per time slice' : neg_counts_per_tau.tolist(),
                'first negative time slice across configs' : int(first_negative_tau),
                'maximum allowed cuttoff' : int(cutoff),
                'recommended maximum time cuttoff' : int(tau_max_reccomended)
            }

        self.results['negative_checks'] = report
        return report

    def tau_symmetry(self, relative_tolerance=0.03, passing_threshold=0.8):
        """Check for tau symmetry in the numeric arrays."""
        report = {}
        for name, array in self.numeric_arrays.items():
            N, T = array.shape
            symmetry_results = {
                'total_configs': int(N),
                'Time extent': int(T),
                'symmetric_pair_counts': 0,
                'asymmetric_pair_counts': 0,
                'symmetry_violations_per_config': [],
                'symmetry_score': 0.0,
                'midpoint_consistency': True,
                'asymmetric_pairs_details': [],
                'relative_difference_statistics': {}
            }
            
            config_violations = []
            total_pairs = T//2
            all_relative_differences = []
            
            for i in range(N):
                correlator = array[i]
                violations = 0

                # Handle both even and odd T correctly
                forward_vals = correlator[:total_pairs]
                backward_vals = correlator[T-total_pairs:T][::-1]  # Get last total_pairs elements and reverse

                absolute_differences = np.abs(forward_vals - backward_vals)
                max_values = np.maximum(np.abs(forward_vals), np.abs(backward_vals))
                relative_differences = np.where(max_values > 0, absolute_differences / max_values, 0.0)
                all_relative_differences.extend(relative_differences)

                median_relative_diff = np.median(relative_differences)
                fraction_passing = np.mean(relative_differences <= relative_tolerance)
                symmetric = (median_relative_diff <= relative_tolerance) and (fraction_passing >= passing_threshold)
                
                # Count violations for this config
                violations = int(np.sum(relative_differences > relative_tolerance))

                if not symmetric and len(symmetry_results['asymmetric_pairs_details']) < 10:
                    bad_indicies = np.where(relative_differences > relative_tolerance)[0]
                    for idx in bad_indicies[:min(10 - len(symmetry_results['asymmetric_pairs_details']), len(bad_indicies))]:
                        symmetry_results['asymmetric_pairs_details'].append({
                            'config': int(i),
                            'tau_forward': int(idx),
                            'tau_backward': int(T- idx - 1),
                            'forward_value': float(forward_vals[idx]),
                            'backward_value': float(backward_vals[idx]),
                            'absolute_difference': float(absolute_differences[idx]),
                            'relative_difference': float(relative_differences[idx]),
                            'config_median_relative_difference': float(median_relative_diff),
                            'config_fraction_passing': float(fraction_passing)
                        })

                config_violations.append(violations)

            # Relative difference statistics
            if all_relative_differences:
                symmetry_results['relative_difference_statistics'] = {
                    'mean_relative_diff': float(np.mean(all_relative_differences)),
                    'median_relative_diff': float(np.median(all_relative_differences)),
                    'std_relative_diff': float(np.std(all_relative_differences)),
                    'max_relative_diff': float(np.max(all_relative_differences)),
                    'fraction_passing_tolerance': float(np.mean(np.array(all_relative_differences) <= relative_tolerance))
                }

            # Overall stats:
            total_possible_pairs = N * total_pairs
            symmetry_results['symmetric_pair_counts'] = total_possible_pairs - sum(config_violations)
            symmetry_results['asymmetric_pair_counts'] = sum(config_violations)
            symmetry_results['symmetry_violations_per_config'] = config_violations
            symmetry_results['symmetry_score'] = float(1.0 - (sum(config_violations) / total_possible_pairs))

            # Midpoint consistency
            if T % 2 == 1:  # odd T
                midpoint_values = array[:, T//2]
                midpoint_std = np.std(midpoint_values)
                midpoint_mean = np.mean(midpoint_values)

                symmetry_results['midpoint_consistency'] = {
                    'midpoint_tau': T//2 + 1,
                    'midpoint_mean': float(midpoint_mean),
                    'midpoint_standard_deviation': float(midpoint_std)
                }
            
            # Summary stats:
            symmetry_results['summary'] = {
                'total_possible_pairs': int(total_possible_pairs),
                'symmetric_pairs': int(symmetry_results['symmetric_pair_counts']),
                'asymmetric_pairs': int(symmetry_results['asymmetric_pair_counts']),
                'symmetry_score': float(symmetry_results['symmetry_score']),
                'configs_with_violations': int(sum(1 for v in config_violations if v > 0)),
                'configs_without_violations': int(sum(1 for v in config_violations if v == 0)),
                'max_violations_in_a_config': int(max(config_violations)),
                'mean_violations_per_config': float(np.mean(config_violations))
            }

            report[name] = symmetry_results
            if self.logger:
                score = symmetry_results['symmetry_score']
                self.logger.info(f"Tau symmetry score for {name}: {score:.4f}")
        
        self.results['tau_symmetry'] = report
        return report

    def correlation_decay_check(self, decay_threshold=0.2, mean_decay_threshold=0.2):
        """Check if correlators show expected exponential decay pattern."""
        report = {}

        for name, array in self.numeric_arrays.items():
            N, T = array.shape

            #Ensemble mean decay check
            first_quarter = T // 4
            mean_corr = array.mean(axis=0)

            mean_increases = 0
            for t in range(1, first_quarter):
                if mean_corr[t] > mean_corr[t - 1]:
                    mean_increases += 1

            mean_total_comparisons = max(first_quarter - 1, 1)
            mean_increase_fraction = mean_increases / mean_total_comparisons

            mean_decay_issue = (mean_increase_fraction > mean_decay_threshold)

            decay_issues = []

            # Config-level decay check
            for config in range(N):
                correlator = array[config]
                increases = 0
                for t in range(1, first_quarter):
                    if correlator[t] > correlator[t - 1]:
                        increases += 1

                total_comparisons = first_quarter - 1
                increase_fraction = increases / total_comparisons if total_comparisons > 0 else 0

                # Flag configs above threshold
                if increase_fraction > decay_threshold:
                    decay_issues.append({
                        'config': config,
                        'increases_in_first_quarter': increases,
                        'total_comparisons': total_comparisons,
                        'increase_fraction': float(increase_fraction),
                        'threshold': decay_threshold
                    })

            # Compile report
            report[name] = {
                'total_configs': N,
                'time_extent': T,
                'first_quarter_length': first_quarter,
                'ensemble_mean_decay_check': {
                    'mean_increases': int(mean_increases),
                    'mean_total_comparisons': int(mean_total_comparisons),
                    'mean_increase_fraction': float(mean_increase_fraction),
                    'mean_threshold': mean_decay_threshold,
                    'mean_decay_issue': mean_decay_issue
                },
                'decay_threshold': decay_threshold,
                'configs_with_decay_issues': len(decay_issues),
                'decay_issue_details': decay_issues[:10],  # truncated
                'decay_pass_rate': (N - len(decay_issues)) / N,
                'summary': {
                    'total_configs_checked': N,
                    'configs_passing_decay': N - len(decay_issues),
                    'configs_failing_decay': len(decay_issues),
                    'overall_pass_rate': float((N - len(decay_issues)) / N),
                    'ensemble_mean_decay_issue': mean_decay_issue
                }
            }
            if self.logger:
                pass_rate = (N - len(decay_issues)) / N
                self.logger.info(
                    f"Decay check for {name}: {len(decay_issues)}/{N} configs failing "
                    f"(pass rate: {pass_rate:.3f}), mean_decay_issue={mean_decay_issue}"
                )

        self.results['decay_check'] = report
        return report

    #Writing results to txt file:
    def generate_report(self, filename="sanity_check_report.txt"):
        """Generate a concise summary report of all sanity checks."""
        report_path = self.results_path / filename
        
        with open(report_path, 'w') as f:
            f.write("DATA SANITY CHECK SUMMARY\n")
            f.write("="*50 + "\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Datasets: {len(self.numeric_arrays)}\n\n")

            # Quick Status Overview
            self._write_quick_summary(f)
            
            # Dataset Details
            self._write_dataset_details(f)
            
            # Critical Issues Summary
            self._write_issues_only(f)
            
        if self.logger:
            self.logger.info(f"Concise report saved to {report_path}")
        
        return report_path

    def _write_quick_summary(self, f):
        """Write a quick pass/fail summary."""
        f.write("QUICK STATUS:\n")
        f.write("-" * 20 + "\n")
        
        # Count issues
        total_issues = 0
        status_checks = [
            ('Shape Consistency', 'shape_consistency', lambda r: not r.get('same_T', True)),
            ('Finite Values', 'finite_checks', lambda r: any(d.get('has_issues', False) for d in r.values())),
            ('Negative Values', 'negative_checks', lambda r: any(d.get('has_negatives', False) for d in r.values())),
            ('Tau Symmetry', 'tau_symmetry', lambda r: any(d.get('symmetry_score', 1.0) < 0.85 for d in r.values())),
            ('Decay Pattern', 'decay_check', lambda r: any(not d.get('passes_decay_check', True) for d in r.values()))
        ]
        
        for check_name, result_key, has_issues_func in status_checks:
            if result_key in self.results:
                has_issues = has_issues_func(self.results[result_key])
                status = "FAIL" if has_issues else "PASS"
                f.write(f"{check_name:20} {status}\n")
                if has_issues:
                    total_issues += 1
            else:
                f.write(f"{check_name:20} SKIPPED\n")
        
        f.write(f"\nTotal Issues: {total_issues}\n")
        f.write("="*50 + "\n\n")

    def _write_dataset_details(self, f):
        """Write key details for each dataset."""
        f.write("DATASET ANALYSIS:\n")
        f.write("-" * 30 + "\n")
        
        for name in sorted(self.numeric_arrays.keys()):
            f.write(f"\n{name}:\n")
            
            # Basic info
            array = self.numeric_arrays[name]
            f.write(f"  Shape: {array.shape} | ")
            
            # Determine if cutoff is needed
            T = array.shape[1]
            rec_trunc = None
            truncation_reason = None
            
            # Check for negative values
            if 'negative_checks' in self.results and name in self.results['negative_checks']:
                neg_data = self.results['negative_checks'][name]
                first_neg_tau = neg_data.get('first_negative_tau', T)
                if first_neg_tau < T:
                    rec_trunc = first_neg_tau - 1
                    truncation_reason = f"first negative at τ={first_neg_tau}"
            
            # Write truncation recommendation (only if needed)
            if rec_trunc is not None:
                f.write(f"Recommended truncation: τ≤{rec_trunc} ({truncation_reason})\n")
            else:
                f.write("Recommended truncation: None\n")
            
            # Symmetry status
            if 'tau_symmetry' in self.results and name in self.results['tau_symmetry']:
                sym_data = self.results['tau_symmetry'][name]
                symmetry_score = sym_data.get('symmetry_score', 0.0)
                is_symmetric = symmetry_score >= 0.85
                sym_status = "Symmetric" if is_symmetric else "Asymmetric"
                
                if 'relative_difference_statistics' in sym_data:
                    avg_diff = sym_data['relative_difference_statistics'].get('mean_relative_diff', 0.0)
                    f.write(f"  Symmetry: {sym_status} (score: {symmetry_score:.3f}, avg diff: {avg_diff:.3f})\n")
                else:
                    f.write(f"  Symmetry: {sym_status} (score: {symmetry_score:.3f})\n")
            
            # Data quality
            issues = []
            if 'finite_checks' in self.results and name in self.results['finite_checks']:
                if self.results['finite_checks'][name].get('number of problematic configs', 0) > 0:
                    issues.append("non-finite values")
            
            if 'decay_check' in self.results and name in self.results['decay_check']:
                if not self.results['decay_check'][name]['summary'].get('overall_pass_rate', 1.0) > 0.95:
                    issues.append("poor decay")
            
            if issues:
                f.write(f"  Issues: {', '.join(issues)}\n")
            else:
                f.write("  Quality: Good\n")
    
        f.write("\n")

    def _write_issues_only(self, f):
        """Write critical issues summary."""
        f.write("CRITICAL ISSUES:\n")
        f.write("-" * 20 + "\n")
        
        critical_issues = []
        
        # Shape inconsistency
        if 'shape_consistency' in self.results:
            result = self.results['shape_consistency']
            if not result.get('same_T', True):
                critical_issues.append(f"Shape inconsistency detected (Common T={result.get('common_T')})")
        
        # Count datasets with issues
        datasets_with_negatives = 0
        datasets_with_asymmetry = 0
        
        if 'negative_checks' in self.results:
            datasets_with_negatives = sum(1 for data in self.results['negative_checks'].values() 
                                        if data.get('has_negatives', False))
        
        if 'tau_symmetry' in self.results:
            datasets_with_asymmetry = sum(1 for data in self.results['tau_symmetry'].values() 
                                        if data.get('symmetry_score', 1.0) < 0.85)
        
        if datasets_with_negatives > 0:
            critical_issues.append(f"{datasets_with_negatives} datasets have negative correlators")
        
        if datasets_with_asymmetry > 0:
            critical_issues.append(f"{datasets_with_asymmetry} datasets violate tau symmetry")
        
        if critical_issues:
            for issue in critical_issues:
                f.write(f"• {issue}\n")
        else:
            f.write("No critical issues detected.\n")
        
        f.write("\n")

    def _write_executive_summary(self, f):
        """Write executive summary section."""
        total_datasets = len(self.numeric_arrays)
        issues = []
        
        # Check for major issues
        if 'shape_consistency' in self.results:
            if not self.results['shape_consistency'].get('same_T', True):
                issues.append("Inconsistent time extents across datasets")
            if self.results['shape_consistency'].get('errors'):
                issues.append("Shape validation errors detected")
        
        if 'finite_checks' in self.results:
            for name, data in self.results['finite_checks'].items():
                if data.get('number of problematic configs', 0) > 0:
                    issues.append(f"Non-finite values in {name}")
        
        if 'decay_check' in self.results:
            for name, data in self.results['decay_check'].items():
                if data.get('summary', {}).get('ensemble_mean_decay_issue', False):
                    issues.append(f"Ensemble mean decay issues in {name}")
        
        # Write summary
        f.write(f"Total datasets analyzed: {total_datasets}\n")
        f.write(f"Major issues found: {len(issues)}\n")
        
        if issues:
            f.write("\nCritical Issues:\n")
            for i, issue in enumerate(issues, 1):
                f.write(f"  {i}. {issue}\n")
        else:
            f.write("\nNo critical issues detected\n")

    def _write_shape_consistency(self, f):
        """Write shape consistency section."""
        results = self.results['shape_consistency']
        
        f.write(f"Same time extent: {'PASS' if results.get('same_T', False) else 'FAIL'}\n")
        if results.get('common_T'):
            f.write(f"Common time extent: {results['common_T']}\n")
        
        f.write("\nDataset Details:\n")
        for name, details in results.get('ensembles', {}).items():
            f.write(f"  {name}: {details['Number of Configs']} configs × {details['Time Extent']} time points\n")
        
        if results.get('errors'):
            f.write("\nErrors:\n")
            for error in results['errors']:
                f.write(f"  • {error}\n")

    def _write_finite_checks(self, f):
        """Write finite value checks section."""
        results = self.results['finite_checks']
        
        for name, data in results.items():
            f.write(f"\nDataset: {name}\n")
            f.write(f"  Total configs: {data['total_configs']}\n")
            f.write(f"  Time extent: {data['Time extent']}\n")
            f.write(f"  Total non-finite entries: {data['total nonfinite entries']}\n")
            f.write(f"  Problematic configs: {data['number of problematic configs']}\n")
            
            if data['number of problematic configs'] > 0:
                f.write(f"  Problematic config IDs: {data['problematic_config_ids'][:10]}")
                if len(data['problematic_config_ids']) > 10:
                    f.write(f" ... (and {len(data['problematic_config_ids']) - 10} more)")
                f.write("\n")

    def _write_negative_checks(self, f):
        """Write negative value checks section."""
        results = self.results['negative_checks']
        
        for name, data in results.items():
            f.write(f"\nDataset: {name}\n")
            f.write(f"  Time extent: {data['Time extent']}\n")
            f.write(f"  First negative time slice: τ_{data['first negative time slice across configs']}\n")
            f.write(f"  Recommended max cutoff: τ_{data['recommended maximum time cuttoff']}\n")
            
            # Count configs with negatives
            neg_configs = sum(1 for x in data['first negative entry per config'] if x < data['Time extent'])
            f.write(f"  Configs with negatives: {neg_configs}/{data['total_configs']}\n")

    def _write_tau_symmetry(self, f):
        """Write tau symmetry section."""
        results = self.results['tau_symmetry']
        
        for name, data in results.items():
            f.write(f"\nDataset: {name}\n")
            f.write(f"  Symmetry score: {data['symmetry_score']:.4f}\n")
            f.write(f"  Configs with perfect symmetry: {data['summary']['configs_without_violations']}/{data['total_configs']}\n")
            f.write(f"  Configs with violations: {data['summary']['configs_with_violations']}\n")
            f.write(f"  Max violations per config: {data['summary']['max_violations_in_a_config']}\n")
            
            if 'midpoint_consistency' in data and isinstance(data['midpoint_consistency'], dict):
                mid = data['midpoint_consistency']
                f.write(f"  Midpoint τ_{mid['midpoint_tau']}: mean={mid['midpoint_mean']:.6e}, std={mid['midpoint_standard_deviation']:.6e}\n")

    def _write_decay_check(self, f):
        """Write decay check section."""
        results = self.results['decay_check']
        
        for name, data in results.items():
            f.write(f"\nDataset: {name}\n")
            f.write(f"  Config-level pass rate: {data['decay_pass_rate']:.3f}\n")
            f.write(f"  Configs failing decay: {data['configs_with_decay_issues']}/{data['total_configs']}\n")
            
            ensemble = data['ensemble_mean_decay_check']
            f.write(f"  Ensemble mean decay issue: {'YES' if ensemble['mean_decay_issue'] else 'NO'}\n")
            f.write(f"  Mean increase fraction: {ensemble['mean_increase_fraction']:.3f} (threshold: {ensemble['mean_threshold']})\n")


    # Run all checks and generate report:
    def run_all_checks(self):
        """Run all sanity checks and generate report."""
        # Run all checks
        self.shape_consistency_check()
        self.finite_checks()
        self.negative_checks()
        self.tau_symmetry()
        self.correlation_decay_check()
        
        # Generate report
        report_path = self.generate_report()
        
        return report_path
    
    def run_sanity_check_pipeline(self):
        """Run the full sanity check pipeline."""
        self.find_csv_files()
        self.data_loading()
        return self.run_all_checks()

def sanity_check(enable_logging=False):
    """Main function"""
    # Setup paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    processed_path = project_root / "data" / "processed" / "unaveraged_data"
    results_path = project_root / "results" / "sanity_check"
    
    checker = DataSanityChecker(processed_path, results_path, enable_logging=enable_logging)
    csv_files = checker.find_csv_files()
    if not csv_files:
        print(f"No CSV files found in {processed_path}")
        return False
        
    print(f"Found {len(csv_files)} CSV files")
    checker.data_loading()
    
    if not checker.numeric_arrays:
        print("No valid numeric data found in any CSV files.")
        print("All files appear to contain only headers with no data rows.")
        print("Please check your data files and ensure they contain actual correlator data.")
        return False
        
    print(f"Loaded {len(checker.numeric_arrays)} datasets with valid data")
    report_path = checker.run_all_checks()
    print(f"Sanity check completed. Report saved to: {report_path}")
    return True

if __name__ == "__main__":
    # Check for logging flag
    enable_logs = False
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg.lower() in ['--log', '--verbose', '-l', '-v']:
                enable_logs = True
                break
    
    sanity_check(enable_logging=enable_logs)