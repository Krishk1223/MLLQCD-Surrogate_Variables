import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

class TruthAnalysis:
    def __init__(self, processed_data_path, results_path, show_error_bars=False):
        """
        Initialize truth analysis for 2pt correlators
        
        Args:
            processed_data_path: Path to processed CSV files
            results_path: Path to save analysis results
            show_error_bars: Whether to show error bars on plots (default: False)
        """
        self.processed_data_path = Path(processed_data_path)
        self.results_path = Path(results_path)
        self.results_path.mkdir(parents=True, exist_ok=True)
        self.show_error_bars = show_error_bars
        
        # Data storage
        self.ensemble_data = {}
        self.ensemble_stats = {}
        
        # Filtering for 2pt correlators only
        self.two_pt_pattern = None  # Will be set based on file naming

    def load_2pt_data(self):
        """Load only 2pt correlator CSV files"""
        csv_files = list(self.processed_data_path.glob("*.csv"))
        
        if not csv_files:
            raise ValueError(f"No CSV files found in {self.processed_data_path}")
        
        # Filter for 2pt correlators (assuming naming convention)
        two_pt_files = []
        for csv_file in csv_files:
            # Check if file contains 2pt correlator data
            # This assumes naming like "ensemble_2pt_..." or similar
            if self._is_2pt_correlator(csv_file):
                two_pt_files.append(csv_file)
        
        if not two_pt_files:
            print("Warning: No 2pt correlator files identified. Processing all CSV files.")
            two_pt_files = csv_files
        
        for csv_file in two_pt_files:
            try:
                df = pd.read_csv(csv_file)
                ensemble_name = csv_file.stem
                
                # Get numeric columns (time slices)
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if 'config_id' in numeric_cols:
                    numeric_cols = numeric_cols.drop('config_id')
                
                if len(numeric_cols) == 0:
                    print(f"No time slice data found in {csv_file.name}, skipping")
                    continue
                
                self.ensemble_data[ensemble_name] = df[numeric_cols].values
                print(f"Loaded {ensemble_name}: {len(df)} configs, {len(numeric_cols)} time slices")
                
            except Exception as e:
                print(f"Failed to load {csv_file.name}: {e}")
        
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

    def compute_ensemble_averages(self):
        """Compute ensemble averages and errors for each time slice"""
        for ensemble_name, data in self.ensemble_data.items():
            # data shape: [configs, time_slices]
            ensemble_mean = np.mean(data, axis=0)
            ensemble_std = np.std(data, axis=0)
            ensemble_err = ensemble_std / np.sqrt(data.shape[0])  # Standard error
            
            # Store statistics
            self.ensemble_stats[ensemble_name] = {
                'mean': ensemble_mean,
                'std': ensemble_std,
                'err': ensemble_err,
                'n_configs': data.shape[0],
                'n_timeslices': data.shape[1]
            }
            
            print(f"Computed averages for {ensemble_name}: {data.shape[0]} configs, {data.shape[1]} time slices")

    def plot_log_correlator(self, ensemble_name=None, save_plot=True):
        """Plot log correlator vs Euclidean time with optional error bars"""
        if ensemble_name is None:
            # Plot all ensembles
            ensembles_to_plot = list(self.ensemble_stats.keys())
        else:
            ensembles_to_plot = [ensemble_name]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for ens_name in ensembles_to_plot:
            stats = self.ensemble_stats[ens_name]
            mean_corr = stats['mean']
            err_corr = stats['err']
            
            # Create time array
            time_slices = np.arange(len(mean_corr))
            
            # Only plot positive correlator values (for log)
            positive_mask = mean_corr > 0
            if not np.any(positive_mask):
                print(f"Warning: No positive correlator values found for {ens_name}")
                continue
            
            # Apply log transformation
            log_corr = np.log(mean_corr[positive_mask])
            time_pos = time_slices[positive_mask]
            
            # Plot with or without error bars
            if self.show_error_bars:
                log_err = err_corr[positive_mask] / mean_corr[positive_mask]  # Propagated error
                ax.errorbar(time_pos, log_corr, yerr=log_err, 
                           fmt='o-', label=ens_name, capsize=3, markersize=4)
            else:
                ax.plot(time_pos, log_corr, 'o-', label=ens_name, markersize=4)
        
        ax.set_xlabel('Euclidean Time t')
        ax.set_ylabel('log C(t)')
        ax.set_title('Log Correlator vs Euclidean Time')
        ax.grid(True, alpha=0.3)
        
        if len(ensembles_to_plot) > 1:
            ax.legend()
        
        if save_plot:
            if ensemble_name:
                filename = f"log_correlator_{ensemble_name}.png"
            else:
                filename = "log_correlator_all.png"
            plt.savefig(self.results_path / filename, dpi=300, bbox_inches='tight')
            print(f"Saved log correlator plot: {filename}")
        
        plt.show()

    def compute_effective_mass(self, ensemble_name):
        """Compute effective mass for a given ensemble"""
        stats = self.ensemble_stats[ensemble_name]
        mean_corr = stats['mean']
        err_corr = stats['err']
        
        # Effective mass: m_eff(t) = ln(C(t)/C(t+1))
        # Only compute where both C(t) and C(t+1) are positive
        n_times = len(mean_corr)
        m_eff = np.full(n_times - 1, np.nan)
        m_eff_err = np.full(n_times - 1, np.nan)
        
        for t in range(n_times - 1):
            C_t = mean_corr[t]
            C_t1 = mean_corr[t + 1]
            err_t = err_corr[t]
            err_t1 = err_corr[t + 1]
            
            if C_t > 0 and C_t1 > 0:
                m_eff[t] = np.log(C_t / C_t1)
                
                # Error propagation for log(C_t/C_t1)
                rel_err_t = err_t / C_t
                rel_err_t1 = err_t1 / C_t1
                m_eff_err[t] = np.sqrt(rel_err_t**2 + rel_err_t1**2)
        
        return m_eff, m_eff_err

    def plot_effective_mass(self, ensemble_name=None, save_plot=True):
        """Plot effective mass vs Euclidean time with optional error bars"""
        if ensemble_name is None:
            # Plot all ensembles
            ensembles_to_plot = list(self.ensemble_stats.keys())
        else:
            ensembles_to_plot = [ensemble_name]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for ens_name in ensembles_to_plot:
            m_eff, m_eff_err = self.compute_effective_mass(ens_name)
            
            # Create time array (one less than correlator since we use ratios)
            time_slices = np.arange(len(m_eff))
            
            # Only plot finite values
            if self.show_error_bars:
                finite_mask = np.isfinite(m_eff) & np.isfinite(m_eff_err)
            else:
                finite_mask = np.isfinite(m_eff)
            
            if not np.any(finite_mask):
                print(f"Warning: No finite effective mass values found for {ens_name}")
                continue
            
            # Plot with or without error bars
            if self.show_error_bars:
                ax.errorbar(time_slices[finite_mask], m_eff[finite_mask], 
                           yerr=m_eff_err[finite_mask],
                           fmt='o-', label=ens_name, capsize=3, markersize=4)
            else:
                ax.plot(time_slices[finite_mask], m_eff[finite_mask], 
                       'o-', label=ens_name, markersize=4)
        
        ax.set_xlabel('Euclidean Time t')
        ax.set_ylabel('m_eff(t)')
        ax.set_title('Effective Mass vs Euclidean Time')
        ax.grid(True, alpha=0.3)
        
        if len(ensembles_to_plot) > 1:
            ax.legend()
        
        if save_plot:
            if ensemble_name:
                filename = f"effective_mass_{ensemble_name}.png"
            else:
                filename = "effective_mass_all.png"
            plt.savefig(self.results_path / filename, dpi=300, bbox_inches='tight')
            print(f"Saved effective mass plot: {filename}")
        
        plt.show()

    def generate_summary_report(self):
        """Generate summary report of the analysis"""
        report_lines = [
            "=" * 60,
            "2PT CORRELATOR TRUTH ANALYSIS REPORT",
            "=" * 60,
            f"Analysis of {len(self.ensemble_data)} ensembles",
            ""
        ]
        
        for ensemble_name, stats in self.ensemble_stats.items():
            report_lines.extend([
                f"{ensemble_name}:",
                f"  Configurations: {stats['n_configs']}",
                f"  Time slices: {stats['n_timeslices']}",
                f"  Mean correlator range: {stats['mean'].min():.2e} to {stats['mean'].max():.2e}",
                f"  Statistical errors: {stats['err'].min():.2e} to {stats['err'].max():.2e}",
                ""
            ])
        
        report_lines.extend([
            "Generated Plots:",
            "  log_correlator_*.png - Log correlator vs time",
            "  effective_mass_*.png - Effective mass vs time",
            "=" * 60
        ])
        
        # Save report
        report_text = "\n".join(report_lines)
        report_file = self.results_path / "truth_analysis_report.txt"
        
        with open(report_file, 'w') as f:
            f.write(report_text)
        
        print(f"Analysis report saved to: {report_file}")
        return report_text

    def run_full_analysis(self):
        """Run complete truth analysis"""
        print("Starting 2pt correlator truth analysis...")
        
        try:
            # Load 2pt data
            if not self.load_2pt_data():
                print("No data loaded. Exiting.")
                return False
            
            # Compute ensemble averages
            self.compute_ensemble_averages()
            
            # Generate plots for each ensemble
            for ensemble_name in self.ensemble_stats.keys():
                print(f"\nAnalyzing {ensemble_name}...")
                self.plot_log_correlator(ensemble_name)
                self.plot_effective_mass(ensemble_name)
            
            # Generate combined plots if multiple ensembles
            if len(self.ensemble_stats) > 1:
                print("\nGenerating combined plots...")
                self.plot_log_correlator()  # All ensembles
                self.plot_effective_mass()  # All ensembles
            
            # Generate summary report
            self.generate_summary_report()
            
            print("Truth analysis completed successfully!")
            return True
            
        except Exception as e:
            print(f"Truth analysis failed: {e}")
            return False


def main():
    """Main function"""
    # Setup paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    
    processed_data_path = project_root / "data" / "processed"
    results_path = project_root / "results" / "truth_analysis"
    
    # Check for error bar flag
    show_error_bars = False
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg.lower() in ['--error-bars', '--errors', '-e']:
                show_error_bars = True
                break
    
    print("2PT CORRELATOR TRUTH ANALYSIS")
    print("=" * 40)
    print(f"Data path: {processed_data_path}")
    print(f"Results path: {results_path}")
    print(f"Error bars: {'Enabled' if show_error_bars else 'Disabled'}")
    print()
    
    # Check if processed data exists
    if not processed_data_path.exists():
        print("Processed data directory not found!")
        print("Please run convert_data.py first to process your raw data.")
        return False
    
    csv_files = list(processed_data_path.glob("*.csv"))
    if not csv_files:
        print("No CSV files found in processed directory!")
        return False
    
    # Initialize and run analysis
    analyzer = TruthAnalysis(processed_data_path, results_path, show_error_bars)
    success = analyzer.run_full_analysis()
    
    if success:
        print(f"\nResults saved to: {results_path}")
    
    return success


if __name__ == "__main__":
    main()
