import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

class CorrelatorAnalysis:
    def __init__(self, processed_data_path, jackknife_errors_path, results_path, show_error_bars=False):
        """
        Initialize truth analysis for 2pt correlators using averaged data and jackknife errors
        
        Args:
            processed_data_path: Path to averaged CSV files (data/processed/averaged_data/)
            jackknife_errors_path: Path to jackknife error files (data/processed/jackknife_errors/)
            results_path: Path to save analysis results
            show_error_bars: Whether to show error bars on plots (default: False)
        """
        self.processed_data_path = Path(processed_data_path)
        self.jackknife_errors_path = Path(jackknife_errors_path)
        self.results_path = Path(results_path)
        self.results_path.mkdir(parents=True, exist_ok=True)
        self.show_error_bars = show_error_bars
        
        # Data storage
        self.ensemble_data = {}
        self.ensemble_errors = {}
        self.ensemble_stats = {}
        
        # Filtering for 2pt correlators only
        self.two_pt_pattern = None  # Will be set based on file naming

    def load_averaged_data(self):
        """Load averaged 2pt correlator CSV files and corresponding jackknife errors"""
        csv_files = list(self.processed_data_path.glob("*_averaged.csv"))
        if not csv_files:
            raise ValueError(f"No averaged CSV files found in {self.processed_data_path}")
        
        two_pt_files = [f for f in csv_files if self._is_2pt_correlator(f)] or csv_files
        
        for csv_file in two_pt_files:
            df = pd.read_csv(csv_file)
            ensemble_name = csv_file.stem.replace('_averaged', '')
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if 'config_id' in numeric_cols:
                numeric_cols = numeric_cols.drop('config_id')
            
            self.ensemble_data[ensemble_name] = df[numeric_cols].values[0]
            
            error_file = self.jackknife_errors_path / f"{ensemble_name}_jackknife_error.npy"
            if error_file.exists():
                self.ensemble_errors[ensemble_name] = np.load(error_file)
            else:
                self.ensemble_errors[ensemble_name] = 0.01 * np.abs(self.ensemble_data[ensemble_name])
        
        return len(self.ensemble_data) > 0

    def _is_2pt_correlator(self, filepath):
        """Check if file contains 2pt correlator data based on naming convention"""
        filename = filepath.name.lower()
        # Common patterns for 2pt correlators
        two_pt_indicators = ['2pt', 'twopt', 'pion', 'nucleon', 'baryon', 'meson']
        # Exclude 3pt correlator patterns
        three_pt_indicators = ['3pt', 'threept', 'form', 'gA']
        
        has_2pt = any(indicator in filename for indicator in two_pt_indicators)
        has_3pt = any(indicator in filename for indicator in three_pt_indicators)
        
        return has_2pt and not has_3pt

    def prepare_analysis_data(self):
        """Prepare data and errors for analysis"""
        for ensemble_name in self.ensemble_data.keys():
            self.ensemble_stats[ensemble_name] = {
                'mean': self.ensemble_data[ensemble_name],
                'err': self.ensemble_errors[ensemble_name],
                'n_timeslices': len(self.ensemble_data[ensemble_name])
            }

    def plot_log_correlator(self, ensemble_name=None, save_plot=True):
        """Plot log correlator vs Euclidean time"""
        ensembles = [ensemble_name] if ensemble_name else list(self.ensemble_stats.keys())
        fig, ax = plt.subplots(figsize=(12, 8))
        colors = plt.cm.tab10(np.linspace(0, 1, len(ensembles)))
        
        for i, ens_name in enumerate(ensembles):
            stats = self.ensemble_stats[ens_name]
            mean_corr, err_corr = stats['mean'], stats['err']
            time_slices = np.arange(len(mean_corr))
            
            positive_mask = mean_corr > 0
            if not np.any(positive_mask):
                continue
            
            log_corr = np.log10(mean_corr[positive_mask])
            time_pos = time_slices[positive_mask]
            
            if self.show_error_bars:
                log_err = err_corr[positive_mask] / mean_corr[positive_mask]
                ax.errorbar(time_pos, log_corr, yerr=log_err, fmt='o-', label=ens_name, 
                           capsize=3, markersize=4, color=colors[i], alpha=0.8)
            else:
                ax.plot(time_pos, log_corr, 'o-', label=ens_name, markersize=4, color=colors[i])
        
        ax.set_xlabel('Euclidean Time τ')
        ax.set_ylabel('C(τ) (log 10 scale)')
        ax.set_title('Correlator vs Euclidean Time')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        if save_plot:
            filename = f"log_correlator_{ensemble_name}.png" if ensemble_name else "log_correlator_all.png"
            plt.savefig(self.results_path / filename, dpi=300, bbox_inches='tight')
        
        plt.close()

    def compute_effective_mass(self, ensemble_name):
        """Compute effective mass for a given ensemble with jackknife error propagation (logarithmic method)"""
        stats = self.ensemble_stats[ensemble_name]
        mean_corr = stats['mean']
        err_corr = stats['err']
        
        n_times = len(mean_corr)
        
        # Effective mass: m_eff(t) = ln(C(t)/C(t+1))
        m_eff = np.full(n_times - 1, np.nan)
        m_eff_err = np.full(n_times - 1, np.nan)
        
        for t in range(n_times - 1):
            C_t = mean_corr[t]
            C_t1 = mean_corr[t + 1]
            err_t = err_corr[t]
            err_t1 = err_corr[t + 1]
            
            if C_t > 0 and C_t1 > 0:
                m_eff[t] = np.log(C_t / C_t1)
                
                # Jackknife error propagation for log(C_t/C_t1)
                rel_err_t = err_t / C_t
                rel_err_t1 = err_t1 / C_t1
                m_eff_err[t] = np.sqrt(rel_err_t**2 + rel_err_t1**2)
        
        return m_eff, m_eff_err

    def _get_optimal_window_size(self, ensemble_name):
        """Determine optimal window size based on meson type"""
        name_lower = ensemble_name.lower()
        
        # Light mesons (K, pion) - need larger windows due to longer plateaus
        if any(indicator in name_lower for indicator in ['k_', 'pion', 'kaon']):
            return 7
        # Heavy mesons (D, B) - need smaller windows due to shorter plateaus  
        elif any(indicator in name_lower for indicator in ['d_', 'b_', 'charm', 'beauty']):
            return 4
        # Default for unknown mesons
        else:
            return 5

    def find_plateau_mass(self, ensemble_name, window_size=None, t_min=None, t_max=None, min_snr=5.0):
        """Find plateau mass using chi-squared optimization with adaptive window sizes"""
        
        # Use adaptive window size if not specified
        if window_size is None:
            window_size = self._get_optimal_window_size(ensemble_name)
        
        m_eff, m_eff_err = self.compute_effective_mass(ensemble_name)
        
        # Get original correlator data to check for negative values
        stats = self.ensemble_stats[ensemble_name]
        correlator_data = stats['mean']
        
        # Find first negative correlator value to limit search range
        negative_indices = np.where(correlator_data < 0)[0]
        first_negative_t = negative_indices[0] if len(negative_indices) > 0 else len(correlator_data)
        
        # Get finite data points
        finite_mask = np.isfinite(m_eff) & np.isfinite(m_eff_err) & (m_eff_err > 0)
        if not np.any(finite_mask):
            return {'mass': np.nan, 'error': np.nan, 't_range': None, 'chi2_reduced': np.inf, 'chi2_raw': np.inf, 'variance_score': np.inf}
        
        finite_times = np.arange(len(m_eff))[finite_mask]
        finite_m_eff = m_eff[finite_mask]
        finite_m_err = m_eff_err[finite_mask]
        
        # Auto-determine search range if not provided
        if t_min is None:
            t_min = max(2, finite_times[0])  # Start after thermalization
        if t_max is None:
            # Limit to before negative correlators appear, with some safety margin
            if first_negative_t <= 3:  # Very early negative correlators - use all available data
                safe_t_max = finite_times[-1]
            else:
                safe_t_max = max(first_negative_t - 2, t_min + window_size)
            t_max = min(finite_times[-1], len(m_eff) - 2, safe_t_max)  # End before noise dominates
        
        # Find available time indices within search range
        search_mask = (finite_times >= t_min) & (finite_times <= t_max)
        search_times = finite_times[search_mask]
        search_m_eff = finite_m_eff[search_mask]
        search_m_err = finite_m_err[search_mask]
        
        if len(search_times) < window_size:
            return {'mass': np.nan, 'error': np.nan, 't_range': None, 'chi2_reduced': np.inf, 'chi2_raw': np.inf, 'variance_score': np.inf}
        
        # Filter for positive correlators and adequate S/N ratio
        signal_to_noise = np.abs(correlator_data) / stats['err']
        
        # First try with normal S/N constraint
        valid_mask = (search_times < len(correlator_data) - 1) & (correlator_data[search_times] > 0) & (signal_to_noise[search_times] >= min_snr)
        
        # If too few points, relax S/N constraint for noisy data
        if np.sum(valid_mask) < window_size:
            relaxed_snr = max(1.0, min_snr / 2)  # Relax S/N to at least 2.5 or down to 1.0
            valid_mask = (search_times < len(correlator_data) - 1) & (correlator_data[search_times] > 0) & (signal_to_noise[search_times] >= relaxed_snr)
            
        search_times = search_times[valid_mask]
        search_m_eff = search_m_eff[valid_mask]
        search_m_err = search_m_err[valid_mask]
        
        if len(search_times) < window_size:
            return {'mass': np.nan, 'error': np.nan, 't_range': None, 'chi2_reduced': np.inf, 'chi2_raw': np.inf, 'avg_snr': 0}
        
        # Find window with lowest chi-squared
        best_window = {'chi2_reduced': np.inf}
        
        for start_idx in range(len(search_times) - window_size + 1):
            window_m_eff = search_m_eff[start_idx:start_idx + window_size]
            window_m_err = search_m_err[start_idx:start_idx + window_size]
            window_times = search_times[start_idx:start_idx + window_size]
            
            # Skip windows with negative effective masses, but be more lenient for noisy data
            if np.any(window_m_eff <= 0):
                # For very noisy data, allow some negative masses if most are positive
                if np.sum(window_m_eff > 0) < len(window_m_eff) * 0.6:  # At least 60% must be positive
                    continue
            
            # Calculate chi-squared
            weights = 1.0 / (window_m_err**2)
            weighted_mean = np.sum(weights * window_m_eff) / np.sum(weights)
            chi2_raw = np.sum(((window_m_eff - weighted_mean) / window_m_err)**2)
            chi2_reduced = chi2_raw / (len(window_m_eff) - 1)
            
            avg_snr = np.mean(signal_to_noise[window_times])
            
            # Prefer lower chi-squared, with small bias for earlier times and higher S/N
            score = chi2_reduced + 0.001 * np.mean(window_times) - 0.001 * avg_snr
            
            if score < best_window['chi2_reduced']:
                best_window = {
                    'chi2_reduced': chi2_reduced,
                    'chi2_raw': chi2_raw,
                    'weighted_mean': weighted_mean,
                    'avg_snr': avg_snr,
                    'start_idx': start_idx
                }
        
        # Check if valid window found
        if 'start_idx' not in best_window:
            return {'mass': np.nan, 'error': np.nan, 't_range': None, 'chi2_reduced': np.inf, 'chi2_raw': np.inf, 'avg_snr': 0}
        
        # Extract best window
        start_idx = best_window['start_idx']
        plateau_times = search_times[start_idx:start_idx + window_size]
        plateau_m_err = search_m_err[start_idx:start_idx + window_size]
        
        plateau_mass = best_window['weighted_mean']
        plateau_error = 1.0 / np.sqrt(np.sum(1.0 / plateau_m_err**2))
        chi2_raw = best_window['chi2_raw']
        chi2_reduced = best_window['chi2_reduced']
        
        return {
            'mass': plateau_mass,
            'error': plateau_error,
            't_range': (plateau_times[0], plateau_times[-1]),
            'chi2_reduced': chi2_reduced,
            'chi2_raw': chi2_raw,
            'avg_snr': best_window['avg_snr'],
            'n_points': window_size,
            'first_negative_t': first_negative_t if len(negative_indices) > 0 else None
        }

    def plot_effective_mass(self, ensemble_name=None, save_plot=True):
        """Plot effective mass with plateau visualization"""
        ensembles = [ensemble_name] if ensemble_name else list(self.ensemble_stats.keys())
        fig, ax = plt.subplots(figsize=(12, 8))
        colors = plt.cm.tab10(np.linspace(0, 1, len(ensembles)))
        
        for i, ens_name in enumerate(ensembles):
            m_eff, m_eff_err = self.compute_effective_mass(ens_name)
            plateau_info = self.find_plateau_mass(ens_name)
            
            time_slices = np.arange(len(m_eff))
            finite_mask = np.isfinite(m_eff) & (np.isfinite(m_eff_err) if self.show_error_bars else True)
            
            color = colors[i]
            if self.show_error_bars:
                ax.errorbar(time_slices[finite_mask], m_eff[finite_mask], yerr=m_eff_err[finite_mask],
                           fmt='o-', label=ens_name, capsize=3, markersize=4, color=color, alpha=0.8)
            else:
                ax.plot(time_slices[finite_mask], m_eff[finite_mask], 'o-', label=ens_name, markersize=4, color=color)
            
            # Plot plateau
            if plateau_info['t_range'] and not np.isnan(plateau_info['mass']):
                t_start, t_end = plateau_info['t_range']
                plateau_mass, plateau_error = plateau_info['mass'], plateau_info['error']
                
                ax.axvspan(t_start, t_end, alpha=0.2, color=color)
                plateau_x = np.linspace(t_start, t_end, 50)
                ax.plot(plateau_x, np.full_like(plateau_x, plateau_mass), '--', color=color, linewidth=2)
                ax.fill_between(plateau_x, plateau_mass - plateau_error, plateau_mass + plateau_error, alpha=0.15, color=color)
                
                mid_x = (t_start + t_end) / 2
                ax.annotate(f'$m$ = {plateau_mass:.4f}({int(plateau_error*10000)})\nχ²/dof = {plateau_info["chi2_reduced"]:.2f}',
                           xy=(mid_x, plateau_mass), xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3), fontsize=9)
        
        ax.set_xlabel('Euclidean Time τ')
        ax.set_ylabel('$m_{eff}(τ)$')
        ax.set_title('Effective Mass with Plateau Analysis')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        if save_plot:
            filename = f"effective_mass_{ensemble_name}.png" if ensemble_name else "effective_mass_all.png"
            plt.savefig(self.results_path / filename, dpi=300, bbox_inches='tight')
        
        plt.close()

    def generate_summary_report(self):
        """Generate tabular summary report of plateau mass analysis"""
        
        # Collect data for table
        table_data = []
        for ensemble_name in self.ensemble_stats.keys():
            stats = self.ensemble_stats[ensemble_name]
            plateau_info = self.find_plateau_mass(ensemble_name)
            
            # Calculate average S/N
            signal_to_noise = stats['mean'] / stats['err']
            avg_snr = np.mean(signal_to_noise[np.isfinite(signal_to_noise)])
            
            # Find first negative correlator
            neg_indices = np.where(stats['mean'] < 0)[0]
            first_negative = neg_indices[0] if len(neg_indices) > 0 else None
            
            # Calculate correlator and error ranges
            correlator_min, correlator_max = stats['mean'].min(), stats['mean'].max()
            error_min, error_max = stats['err'].min(), stats['err'].max()
            
            if plateau_info['t_range'] and not np.isnan(plateau_info['mass']):
                t_start, t_end = plateau_info['t_range']
                table_data.append({
                    'Ensemble': ensemble_name,
                    'Plateau Mass ± Error': f"{plateau_info['mass']:.4f} ± {plateau_info['error']:.4f}",
                    'τ Range': f"{t_start}-{t_end}",
                    'χ²/dof': f"{plateau_info['chi2_reduced']:.2f}",
                    'S/N': f"{plateau_info['avg_snr']:.1f}",
                    'Window': plateau_info['n_points'],
                    'Correlator Range': f"{correlator_min:.2e} to {correlator_max:.2e}",
                    'Error Range': f"{error_min:.2e} to {error_max:.2e}",
                    'First Neg': first_negative if first_negative else 'None'
                })
            else:
                table_data.append({
                    'Ensemble': ensemble_name,
                    'Plateau Mass ± Error': 'No plateau found',
                    'τ Range': 'N/A',
                    'χ²/dof': 'N/A',
                    'S/N': f"{avg_snr:.1f}",
                    'Window': 'N/A',
                    'Correlator Range': f"{correlator_min:.2e} to {correlator_max:.2e}",
                    'Error Range': f"{error_min:.2e} to {error_max:.2e}",
                    'First Neg': first_negative if first_negative else 'None'
                })
        
        # Create formatted table
        report_lines = [
            "=" * 170,
            "2PT CORRELATOR PLATEAU MASS ANALYSIS RESULTS",
            "=" * 170,
            f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Ensembles: {len(self.ensemble_data)}",
            "",
            "| {:<18} | {:<22} | {:<22} | {:<8} | {:<12} | {:<6} | {:<18} | {:<18} | {:<9} |".format(
                'Ensemble', 'Plateau Mass ± Error', 'Plateau Window τ Range', 'χ²/dof', 'Average S/N', 'Window', 'Correlator Range', 'Error Range', 'First Neg'),
            "|" + "-" * 20 + "|" + "-" * 24 + "|" + "-" * 24 + "|" + "-" * 10 + "|" + "-" * 14 + "|" + "-" * 8 + "|" + "-" * 20 + "|" + "-" * 20 + "|" + "-" * 11 + "|"
        ]
        
        for row in table_data:
            report_lines.append(
                "| {:<18} | {:<22} | {:<22} | {:<8} | {:<12} | {:<6} | {:<18} | {:<18} | {:<9} |".format(
                    row['Ensemble'][:18], 
                    row['Plateau Mass ± Error'][:22],
                    row['τ Range'], 
                    row['χ²/dof'], 
                    row['S/N'], 
                    row['Window'], 
                    row['Correlator Range'][:18],
                    row['Error Range'][:18],
                    row['First Neg']
                )
            )
        
        report_lines.extend([
            "|" + "-" * 20 + "|" + "-" * 24 + "|" + "-" * 24 + "|" + "-" * 10 + "|" + "-" * 14 + "|" + "-" * 8 + "|" + "-" * 20 + "|" + "-" * 20 + "|" + "-" * 11 + "|",
            "",
            "=" * 170
        ])
        
        # Save report
        report_text = "\n".join(report_lines)
        report_file = self.results_path / "truth_analysis_table.txt"
        
        with open(report_file, 'w') as f:
            f.write(report_text)
        
        print(f"Analysis table saved to: {report_file}")
        return report_text

    def run_full_analysis(self):
        """Run complete truth analysis"""
        if not self.load_averaged_data():
            print("No data loaded")
            return False
        
        self.prepare_analysis_data()
        
        for ensemble_name in self.ensemble_stats.keys():
            self.plot_log_correlator(ensemble_name)
            self.plot_effective_mass(ensemble_name)
            
            plateau_info = self.find_plateau_mass(ensemble_name)
            if plateau_info['t_range'] and not np.isnan(plateau_info['mass']):
                t_start, t_end = plateau_info['t_range']
                print(f"{ensemble_name}: m = {plateau_info['mass']:.4f} ± {plateau_info['error']:.4f}, "
                      f"τ={t_start}-{t_end}, χ²/dof = {plateau_info['chi2_reduced']:.2f}, S/N = {plateau_info['avg_snr']:.1f}")
        
        if len(self.ensemble_stats) > 1:
            self.plot_log_correlator()
            self.plot_effective_mass()
        
        self.generate_summary_report()
        return True


def main():
    """Main function"""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    
    averaged_data_path = project_root / "data" / "processed" / "averaged_data"
    jackknife_errors_path = project_root / "data" / "processed" / "jackknife_errors"
    results_path = project_root / "results" / "truth_analysis"
    
    show_error_bars = '--error-bars' in sys.argv or '--errors' in sys.argv or '-e' in sys.argv
    
    print("2PT CORRELATOR TRUTH ANALYSIS")
    print(f"Found {len(list(averaged_data_path.glob('*_averaged.csv')))} datasets")
    
    analyzer = CorrelatorAnalysis(averaged_data_path, jackknife_errors_path, results_path, show_error_bars)
    return analyzer.run_full_analysis()


if __name__ == "__main__":
    main()
