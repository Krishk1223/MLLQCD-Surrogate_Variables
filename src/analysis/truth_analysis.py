"""
Correlator Analysis Classes for Lattice QCD 2pt Function
STEP 1: Preliminary Analysis and Baseline Estimates for Bayesian Fitting
Goals:
1. Preliminary analysis: Quick sliding-window plateau estimates (baseline for fitting)
2. Bayesian analysis: Proper lsqfit/gvar fitting with oscillating states (future)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from abc import ABC, abstractmethod
import sys
import warnings
warnings.filterwarnings('ignore')


class CorrelatorAnalysisBase(ABC):
    """Base class for correlator analysis with common methods."""
    
    def __init__(self, results_path, show_error_bars=False):
        self.results_path = Path(results_path)
        self.results_path.mkdir(parents=True, exist_ok=True)
        self.show_error_bars = show_error_bars
        self.stats = {}  # {name: {'mean': array, 'err': array, 'samples': array}}
    
    def compute_effective_mass(self, name):
        """Compute effective mass: m_eff(t) = ln(C(t)/C(t+1))
        
        Note: This is a simple log-ratio estimator. For staggered fermions with
        oscillating (-1)^t terms, proper fitting with gvar/lsqfit is needed.
        """
        if name not in self.stats:
            return np.array([np.nan]), np.array([np.nan])
        
        mean, err = self.stats[name]['mean'], self.stats[name]['err']
        n = len(mean) - 1
        m_eff = np.full(n, np.nan)
        m_eff_err = np.full(n, np.nan)
        
        for t in range(n):
            if mean[t] > 0 and mean[t + 1] > 0:
                m_eff[t] = np.log(mean[t] / mean[t + 1])
                m_eff_err[t] = np.sqrt((err[t]/mean[t])**2 + (err[t+1]/mean[t+1])**2)
        
        return m_eff, m_eff_err
    
    def find_plateau_mass(self, name, min_t=5, max_t=12, window_size=4, use_odd_only=True):
        """Find plateau by locating the most stable window of effective mass values.
        
        This is a PRELIMINARY method using sliding windows to estimate:
        - Approximate plateau mass
        - Suggested t_min and t_max for proper Bayesian fitting
        
        For publication-quality results, use BayesianCorrelatorFitter with
        gvar/lsqfit to properly handle correlations and oscillating states.
        
        Args:
            name: Key in self.stats
            min_t, max_t: Time range to search
            window_size: Number of points to consider for stability
            use_odd_only: If True, use only odd timeslices (for staggered fermions)
        
        Returns:
            dict with 'mass', 'error', 't_range', 'chi2_reduced', 'stability'
        """
        m_eff, m_eff_err = self.compute_effective_mass(name)
        if name not in self.stats:
            return {'mass': np.nan, 'error': np.nan, 't_range': None, 'chi2_reduced': np.inf}
        
        max_t = min(max_t, len(m_eff))
        times = np.arange(min_t, max_t, 2) if use_odd_only else np.arange(min_t, max_t)
        
        m_vals = np.array([m_eff[t] for t in times if t < len(m_eff)])
        m_errs = np.array([m_eff_err[t] for t in times if t < len(m_eff)])
        times = np.array([t for t in times if t < len(m_eff)])
        
        mask = np.isfinite(m_vals) & np.isfinite(m_errs) & (m_errs > 0)
        if np.sum(mask) < window_size:
            return {'mass': np.nan, 'error': np.nan, 't_range': None, 'chi2_reduced': np.inf}
        
        m_vals, m_errs, times = m_vals[mask], m_errs[mask], times[mask]
        
        # Find most stable window (smallest std dev)
        best_std = np.inf
        best_start = 0
        
        for i in range(len(m_vals) - window_size + 1):
            window_vals = m_vals[i:i + window_size]
            window_std = np.std(window_vals)
            if window_std < best_std:
                best_std = window_std
                best_start = i
        
        # Use the most stable window
        plateau_vals = m_vals[best_start:best_start + window_size]
        plateau_errs = m_errs[best_start:best_start + window_size]
        plateau_times = times[best_start:best_start + window_size]
        
        # Weighted average within plateau
        weights = 1.0 / plateau_errs**2
        w_mean = np.sum(weights * plateau_vals) / np.sum(weights)
        w_err = 1.0 / np.sqrt(np.sum(weights))
        chi2 = np.sum(((plateau_vals - w_mean) / plateau_errs)**2) / max(1, len(plateau_vals) - 1)
        
        return {
            'mass': w_mean,
            'error': w_err,
            't_min': int(plateau_times[0]),
            't_max': int(plateau_times[-1]),
            't_range': (int(plateau_times[0]), int(plateau_times[-1])),
            'chi2_reduced': chi2,
            'stability': best_std  # Lower = more stable plateau
        }
    
    def _plot_log_correlator_data(self, ax, name, style, color, label=None):
        """Plot log correlator for a single dataset."""
        if name not in self.stats:
            return
        mean, err = self.stats[name]['mean'], self.stats[name]['err']
        t = np.arange(len(mean))
        mask = mean > 0
        
        if self.show_error_bars:
            ax.errorbar(t[mask], np.log10(mean[mask]), yerr=err[mask]/(mean[mask]*np.log(10)),
                       fmt=style, label=label or name, capsize=3, color=color)
        else:
            ax.plot(t[mask], np.log10(mean[mask]), style, label=label or name, color=color)

    @abstractmethod
    def load_data(self):
        """Load data"""
        pass
    
    @abstractmethod
    def run_analysis(self):
        """Run full analysis"""
        pass


class TruthAnalysis2pt(CorrelatorAnalysisBase):
    """Preliminary analysis of truth data from averaged CSV files.
    
    Generates baseline estimates for plateau mass, t_min, t_max which
    can be used as starting points for Bayesian fitting with lsqfit/gvar.
    """
    
    def __init__(self, processed_data_path, jackknife_errors_path, jackknife_samples_path, results_path, show_error_bars=False):
        super().__init__(results_path, show_error_bars)
        self.processed_data_path = Path(processed_data_path)
        self.jackknife_errors_path = Path(jackknife_errors_path)
        self.jackknife_samples_path = Path(jackknife_samples_path)
    
    def load_data(self):
        """Load averaged 2pt correlator CSV files and jackknife errors."""
        csv_files = list(self.processed_data_path.glob("*_averaged.csv"))
        if not csv_files:
            raise ValueError(f"No averaged CSV files found in {self.processed_data_path}")
        
        # Filter for 2pt correlators
        two_pt_files = [f for f in csv_files if self._is_2pt_correlator(f)] or csv_files
        
        for csv_file in two_pt_files:
            ensemble_name = csv_file.stem.replace('_averaged', '')
            df = pd.read_csv(csv_file)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if 'config_id' in numeric_cols:
                numeric_cols = numeric_cols.drop('config_id')
            
            mean = df[numeric_cols].values[0]
            
            # Load jackknife errors or estimate
            error_file = self.jackknife_errors_path / f"{ensemble_name}_jackknife_error.npy"
            err = np.load(error_file) if error_file.exists() else np.zeros_like(mean)
            
            # Load jackknife samples
            samples_file = self.jackknife_samples_path / f"{ensemble_name}_jackknife_samples.npy"
            samples = np.load(samples_file) if samples_file.exists() else None
            
            self.stats[ensemble_name] = {
                'mean': mean, 
                'err': err, 
                'n_timeslices': len(mean), 
                'samples': samples
            }
        
        return len(self.stats) > 0

    def _is_2pt_correlator(self, filepath):
        """Check if file contains 2pt correlator data."""
        name = filepath.name.lower()
        has_2pt = any(x in name for x in ['2pt', 'twopt', 'pion', 'nucleon', 'baryon', 'meson'])
        has_3pt = any(x in name for x in ['3pt', 'threept', 'form', 'gA'])
        return has_2pt and not has_3pt
    
    def plot_log_correlator(self, ensemble_name=None):
        """Plot log correlator vs time."""
        ensembles = [ensemble_name] if ensemble_name else list(self.stats.keys())
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for i, name in enumerate(ensembles):
            self._plot_log_correlator_data(ax, name, 'o-', f'C{i}')
        
        ax.set(xlabel='τ', ylabel='log₁₀ C(τ)', title='Correlator')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.savefig(self.results_path / f"log_correlator_{ensemble_name or 'all'}.png", dpi=150)
        plt.close()
        
    def generate_preliminary_report(self):
        """Generate preliminary analysis report with baseline estimates.
        
        This provides starting values for proper Bayesian fitting:
        - Approximate plateau mass (initial guess for ground state mass)
        - Suggested t_min, t_max (fit range for lsqfit)
        - Stability metric (lower = more reliable estimate)
        """
        lines = [
            "=" * 100,
            "PRELIMINARY 2PT CORRELATOR ANALYSIS",
            "=" * 100,
            "",
            "NOTE: These are APPROXIMATE values from sliding-window analysis.",
            "Use these as starting points for Bayesian fitting with lsqfit/gvar.",
            "Proper fits will account for excited states and (-1)^t oscillations.",
            "",
            "-" * 100,
            f"{'Ensemble':<40} {'Mass':>12} {'Error':>12} {'t_min':>8} {'t_max':>8} {'χ²/dof':>10} {'Stability':>12}",
            "-" * 100,
        ]
        
        # Store results for later use
        preliminary_results = {}
        
        for name in sorted(self.stats.keys()):
            plateau = self.find_plateau_mass(name)
            preliminary_results[name] = plateau
            
            if plateau['t_range'] and not np.isnan(plateau['mass']):
                lines.append(
                    f"{name:<40} {plateau['mass']:>12.6f} {plateau['error']:>12.6f} "
                    f"{plateau['t_min']:>8d} {plateau['t_max']:>8d} "
                    f"{plateau['chi2_reduced']:>10.3f} {plateau['stability']:>12.6f}"
                )
            else:
                lines.append(f"{name:<40} {'No plateau found':<60}")
        
        lines.extend([
            "",
            "-" * 100,
            "",
            "RECOMMENDED NEXT STEPS:",
            "  1. Use t_min, t_max as initial fit range for lsqfit",
            "  2. Use 'Mass' as initial guess for ground state mass prior",
            "  3. Include oscillating state: C(t) = A*exp(-m*t) + (-1)^t * Ao*exp(-mo*t)",
            "  4. Iterate on fit range based on fit quality (Q-value, chi2/dof)",
            "",
            "=" * 100
        ])
        
        report = "\n".join(lines)
        
        # Save report
        with open(self.results_path / "preliminary_analysis.txt", 'w') as f:
            f.write(report)
        
        # Also save as npz file for easy loading in physics analysis:
        np.savez(
            self.results_path / "preliminary_analysis_data.npz",
            **{name: np.array([
                plateau['mass'], 
                plateau['error'], 
                plateau.get('t_min', np.nan), 
                plateau.get('t_max', np.nan),
                plateau['chi2_reduced'],
                plateau.get('stability', np.nan)
            ]) for name, plateau in preliminary_results.items()}
        )
        
        print(report)
        return preliminary_results
    
    def run_analysis(self):
        """Run preliminary truth analysis."""
        if not self.load_data():
            print("No data loaded")
            return False
        
        # Generate log correlator plots (useful for visual inspection)
        for name in self.stats:
            self.plot_log_correlator(name)
        
        if len(self.stats) > 1:
            self.plot_log_correlator()
        
        # Generate preliminary report with baseline estimates
        results = self.generate_preliminary_report()
        
        print(f"\nPlots and report saved to: {self.results_path}")
        return results


class MLCorrelatorAnalysis2pt(CorrelatorAnalysisBase):
    """Compare ML predictions to truth data."""
    
    def __init__(self, experiment_folder, truth_correlator_name=None, show_error_bars=False):
        self.project_root = Path(__file__).parent.parent.parent
        results_path = self.project_root / "results" / experiment_folder / "ml_analysis"
        super().__init__(results_path, show_error_bars)
        
        self.experiment_path = self.project_root / "results" / experiment_folder
        self.experiment_name = experiment_folder
        self.truth_correlator_name = truth_correlator_name
        
        self.predictions = None
        self.targets = None
    
    def load_data(self):
        """Load ML predictions, apply inverse transform, and load actual truth data."""
        pred_path = self.experiment_path / "correlator_predictions.npy"
        target_path = self.experiment_path / "test_targets.npy"
        
        if not pred_path.exists() or not target_path.exists():
            raise FileNotFoundError(f"Missing predictions or targets in {self.experiment_path}")
        
        self.predictions = np.load(pred_path).squeeze()
        self.targets = np.load(target_path).squeeze()
        print(f"Loaded: predictions {self.predictions.shape}, targets {self.targets.shape}")
        
        # Apply inverse transform
        self._apply_inverse_transform()
        
        # Compute statistics
        for name, data in [('ml', self.predictions), ('test_targets', self.targets)]:
            self.stats[name] = {
                'mean': np.mean(data, axis=0),
                'err': np.std(data, axis=0) / np.sqrt(len(data)),
                'n_configs': data.shape[0],
                'n_timeslices': data.shape[1]
            }
        
        # Load actual truth data (qsq0 from CSV)
        self._load_actual_truth_data()
        return True
    
    def _apply_inverse_transform(self):
        """Apply inverse transform (sinh or exp) based on scalers."""
        exp_name = self.experiment_path.parent.name if "bias_" in str(self.experiment_path) else self.experiment_path.name
        scalers_paths = [
            self.project_root / "models" / exp_name / "cnn_model" / "scalers.npz",
            self.project_root / "models" / exp_name / "transformer_model" / "scalers.npz",
        ]
        
        scalers_path = next((p for p in scalers_paths if p.exists()), None)
        
        if scalers_path:
            scalers = np.load(scalers_path)
            if 'target_signs' in scalers:
                signs = scalers['target_signs']
                self.predictions = np.exp(self.predictions) * signs
                self.targets = np.exp(self.targets) * signs
                print(f"Applied exp inverse transform with signs")
                return
            elif 'arcsinh_target_scale' in scalers:
                # Legacy support for old arcsinh transform
                scale = float(scalers['arcsinh_target_scale'])
                self.predictions = np.sinh(self.predictions) * scale
                self.targets = np.sinh(self.targets) * scale
                print(f"Applied sinh inverse transform (scale={scale:.4f})")
                return
        
        # Fallback to exp
        self.predictions = np.exp(self.predictions)
        self.targets = np.exp(self.targets)
        print("Applied exp inverse transform")
    
    def _load_actual_truth_data(self):
        """Load actual truth data from averaged CSV (qsq0 correlator)."""
        averaged_path = self.project_root / "data" / "processed" / "averaged_data"
        jackknife_path = self.project_root / "data" / "processed" / "jackknife_errors"
        
        if not averaged_path.exists():
            print(f"Warning: No averaged data found at {averaged_path}")
            return
        
        csv_files = list(averaged_path.glob("*_averaged.csv"))
        if not csv_files:
            return
        
        # Find target file
        target_file = None
        if self.truth_correlator_name:
            target_file = next((f for f in csv_files if self.truth_correlator_name in f.stem), None)
        if not target_file:
            target_file = next((f for f in csv_files if '2pt_K_fine_qsq0' in f.stem), None)
        if not target_file:
            target_file = csv_files[0]
        
        # Load data (no header in these files)
        df = pd.read_csv(target_file, header=None)
        data = df.values
        
        mean = np.mean(data, axis=0)
        err = np.std(data, axis=0) / np.sqrt(len(data))
        
        # Check for jackknife errors
        ensemble_name = target_file.stem.replace('_averaged', '')
        error_file = jackknife_path / f"{ensemble_name}_jackknife_error.npy"
        if error_file.exists():
            err = np.load(error_file)
        
        self.stats['truth'] = {
            'mean': mean, 'err': err,
            'name': ensemble_name, 'n_samples': len(data)
        }
        print(f"Loaded truth data: {ensemble_name} (shape: {data.shape})")
    
    def plot_log_correlator(self):
        """Plot log correlator: ML vs Truth."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        self._plot_log_correlator_data(ax, 'ml', 'o-', 'blue', 'ML')
        if 'truth' in self.stats:
            self._plot_log_correlator_data(ax, 'truth', 's-', 'green', 'Truth')
        
        ax.set(xlabel='τ', ylabel='log₁₀ C(τ)', title='Correlator Comparison', xlim=(-0.5, 25))
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.savefig(self.results_path / "log_correlator_comparison.png", dpi=150)
        plt.close()
    
    def generate_preliminary_report(self):
        """Generate preliminary comparison report."""
        p_ml = self.find_plateau_mass('ml')
        p_test = self.find_plateau_mass('test_targets')
        p_truth = self.find_plateau_mass('truth') if 'truth' in self.stats else None
        
        # Correlator metrics
        mse = np.mean((self.stats['ml']['mean'] - self.stats['test_targets']['mean'])**2)
        mae = np.mean(np.abs(self.stats['ml']['mean'] - self.stats['test_targets']['mean']))
        
        lines = [
            "=" * 100,
            f"PRELIMINARY ML CORRELATOR ANALYSIS - {self.experiment_name}",
            "=" * 100,
            "",
            "NOTE: Plateau masses are APPROXIMATE (sliding-window method).",
            "Use these as starting points for proper Bayesian fitting.",
            "",
            f"Configs: {self.stats['ml']['n_configs']}, Timeslices: {self.stats['ml']['n_timeslices']}",
            "",
            "-" * 100,
            f"{'Source':<20} {'Mass':>12} {'Error':>12} {'t_min':>8} {'t_max':>8} {'χ²/dof':>10}",
            "-" * 100,
        ]
        
        for name, p in [('ML Prediction', p_ml), ('Test Targets', p_test), ('Actual Truth', p_truth)]:
            if p and p['t_range'] and not np.isnan(p['mass']):
                lines.append(
                    f"{name:<20} {p['mass']:>12.6f} {p['error']:>12.6f} "
                    f"{p['t_min']:>8d} {p['t_max']:>8d} {p['chi2_reduced']:>10.3f}"
                )
            elif p:
                lines.append(f"{name:<20} {'No plateau found':<50}")
        
        # Comparisons
        lines.extend(["", "-" * 100, "COMPARISONS:", "-" * 100])
        
        def compare(name1, p1, name2, p2):
            if p1 and p2 and not np.isnan(p1['mass']) and not np.isnan(p2['mass']):
                diff = p1['mass'] - p2['mass']
                rel = (diff / p2['mass']) * 100
                sigma = abs(diff) / np.sqrt(p1['error']**2 + p2['error']**2)
                return f"  {name1} vs {name2}: Δm = {diff:+.6f} ({rel:+.2f}%), {sigma:.2f}σ tension"
            return None
        
        if c := compare('ML', p_ml, 'Test Targets', p_test): lines.append(c)
        if c := compare('ML', p_ml, 'Truth', p_truth): lines.append(c)
        if c := compare('Test Targets', p_test, 'Truth', p_truth): lines.append(c)
        
        lines.extend([
            "",
            "-" * 100,
            "CORRELATOR METRICS (ML vs Test Targets):",
            "-" * 100,
            f"  MSE: {mse:.6e}",
            f"  MAE: {mae:.6e}",
            "",
            "=" * 100
        ])
        
        report = "\n".join(lines)
        with open(self.results_path / "preliminary_analysis.txt", 'w') as f:
            f.write(report)
        print(report)
        return report
    
    def run_analysis(self):
        """Run preliminary ML analysis."""
        self.load_data()
        print("\nGenerating comparison plots.")
        self.plot_log_correlator()
        print(f"Plots saved to: {self.results_path}")
        self.generate_preliminary_report()
        return True


class MLCorrelatorAnalysis3pt(CorrelatorAnalysisBase):
    """Placeholder for 3pt correlator analysis."""
    
    def load_data(self):
        raise NotImplementedError("3pt analysis not yet implemented")
    
    def run_analysis(self):
        raise NotImplementedError("3pt analysis not yet implemented")


def main():
    """Main entry point."""
    project_root = Path(__file__).parent.parent.parent
    
    if '--ml' in sys.argv:
        idx = sys.argv.index('--ml')
        experiment = sys.argv[idx + 1] if len(sys.argv) > idx + 1 else "Two_pt_ML_Kaon_qmax_to_qsq0_Experiment"
        show_errors = '--error-bars' in sys.argv or '-e' in sys.argv
        truth_name = None
        if '--truth' in sys.argv:
            truth_idx = sys.argv.index('--truth')
            truth_name = sys.argv[truth_idx + 1] if len(sys.argv) > truth_idx + 1 else None
        
        print(f"Running Preliminary ML Analysis: {experiment}")
        return MLCorrelatorAnalysis2pt(experiment, truth_name, show_errors).run_analysis()
    
    # Truth analysis mode
    averaged_path = project_root / "data" / "processed" / "averaged_data"
    jackknife_errors_path = project_root / "data" / "processed" / "jackknife_errors"
    jackknife_samples_path = project_root / "data" / "processed" / "jackknife_samples"
    results_path = project_root / "results" / "truth_analysis"
    show_errors = '--error-bars' in sys.argv or '-e' in sys.argv
    
    print("PRELIMINARY 2PT CORRELATOR ANALYSIS")
    print(f"Found {len(list(averaged_path.glob('*_averaged.csv')))} datasets")
    
    return TruthAnalysis2pt(
        averaged_path, 
        jackknife_errors_path, 
        jackknife_samples_path,
        results_path, 
        show_errors
    ).run_analysis()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
