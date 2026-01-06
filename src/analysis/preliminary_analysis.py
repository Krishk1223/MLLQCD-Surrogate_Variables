"""
Preliminary Analysis for Lattice QCD 2pt Correlators
Returns baseline estimates (plateau mass, t_min, t_max) for Bayesian fitting.
Finds out values to use as a baseline for E0 (ground state/plateau mass)
will also provide an estimate for fitting in physics_analysis.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys


class PreliminaryAnalysis2pt:
    """Sliding-window plateau estimates for 2pt correlators."""
    
    def __init__(self, processed_data_path, jackknife_errors_path, results_path):
        self.processed_data_path = Path(processed_data_path)
        self.jackknife_errors_path = Path(jackknife_errors_path)
        self.results_path = Path(results_path)
        self.results_path.mkdir(parents=True, exist_ok=True)
        self.stats = {}
    
    def load_data(self):
        """Load averaged 2pt correlator CSVs and jackknife errors."""
        csv_files = list(self.processed_data_path.glob("*_averaged.csv"))
        if not csv_files:
            raise ValueError(f"No averaged CSV files in {self.processed_data_path}")
        
        for csv_file in csv_files:
            name = csv_file.stem.replace('_averaged', '')
            df = pd.read_csv(csv_file)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if 'config_id' in numeric_cols:
                numeric_cols = numeric_cols.drop('config_id')
            
            mean = df[numeric_cols].values[0]
            error_file = self.jackknife_errors_path / f"{name}_jackknife_error.npy"
            err = np.load(error_file) if error_file.exists() else np.zeros_like(mean)
            
            self.stats[name] = {'mean': mean, 'err': err}
        
        return len(self.stats) > 0
    
    def compute_effective_mass(self, name):
        """Compute m_eff(t) = ln(C(t)/C(t+1))."""
        if name not in self.stats:
            return np.array([np.nan]), np.array([np.nan])
        
        mean, err = self.stats[name]['mean'], self.stats[name]['err']
        n = len(mean) - 1
        m_eff, m_eff_err = np.full(n, np.nan), np.full(n, np.nan)
        
        for t in range(n):
            if mean[t] > 0 and mean[t + 1] > 0:
                m_eff[t] = np.log(mean[t] / mean[t + 1])
                m_eff_err[t] = np.sqrt((err[t]/mean[t])**2 + (err[t+1]/mean[t+1])**2)
        
        return m_eff, m_eff_err
    
    def find_plateau(self, name, min_t=5, max_t=35, window_size=4, use_odd_only=True):
        """Find plateau via sliding window, minimizing chi2.
        
        Args:
            min_t: Minimum timeslice to consider
            max_t: Maximum timeslice (hard cutoff, no-go region above this)
            window_size: Number of points in plateau window
            use_odd_only: Use only odd timeslices (staggered fermions)
        """
        m_eff, m_eff_err = self.compute_effective_mass(name)
        if name not in self.stats:
            return {'mass': np.nan, 'error': np.nan, 't_min': np.nan, 't_max': np.nan, 'chi2': np.inf, 'stability': np.nan}
        
        # Hard cutoff at max_t (no-go region)
        max_t = min(max_t, len(m_eff), 35)
        times = np.arange(min_t, max_t, 2) if use_odd_only else np.arange(min_t, max_t)
        
        m_vals = np.array([m_eff[t] for t in times if t < len(m_eff)])
        m_errs = np.array([m_eff_err[t] for t in times if t < len(m_eff)])
        times = np.array([t for t in times if t < len(m_eff)])
        
        mask = np.isfinite(m_vals) & np.isfinite(m_errs) & (m_errs > 0)
        if np.sum(mask) < window_size:
            return {'mass': np.nan, 'error': np.nan, 't_min': np.nan, 't_max': np.nan, 'chi2': np.inf, 'stability': np.nan}
        
        m_vals, m_errs, times = m_vals[mask], m_errs[mask], times[mask]
        
        # Find window with minimum chi2
        best_chi2, best_start = np.inf, 0
        for i in range(len(m_vals) - window_size + 1):
            window_vals = m_vals[i:i + window_size]
            window_errs = m_errs[i:i + window_size]
            weights = 1.0 / window_errs**2
            w_mean = np.sum(weights * window_vals) / np.sum(weights)
            chi2 = np.sum(((window_vals - w_mean) / window_errs)**2) / max(1, window_size - 1)
            if chi2 < best_chi2:
                best_chi2, best_start = chi2, i
        
        plateau_vals = m_vals[best_start:best_start + window_size]
        plateau_errs = m_errs[best_start:best_start + window_size]
        plateau_times = times[best_start:best_start + window_size]
        
        weights = 1.0 / plateau_errs**2
        E0 = np.sum(weights * plateau_vals) / np.sum(weights)
        E0_error = 1.0 / np.sqrt(np.sum(weights))
        stability = np.std(plateau_vals)
        
        t_min = float(plateau_times[0])
        t_max = float(plateau_times[-1])

        E0_interval = 0.2 #assume a hardcoded 0.2 interval for E0 to give some leeway to fitter
        logE0 = np.log(E0)
        logE0_error = E0_error / E0
        logE0_interval = 0.1 #hardcoded interval for logE0

        #difference in ground and 1st excited state energies assume E1 is about 0.1 greater than E0:
        delta_E1 = 0.1
        log_delta_E1 = np.log(delta_E1)
        log_delta_E1_interval = np.log(2) #hardcoded interval for delta E1 from Lepage et al.

        #a0 calculation use plateau mass values across range of tmin to tmax to get an a0 estimate.
        a0_vals = []
        for t in range(int(t_min), int(t_max)+1):
            C_t = self.stats[name]['mean'][t]
            a0_t = C_t * np.exp(E0 * t)
            a0_vals.append(a0_t)
        a0_vals = np.array(a0_vals)
        a0 = np.mean(a0_vals)
        a0_error = np.std(a0_vals) / np.sqrt(len(a0_vals))
        a0_interval = 0.1 #hardcoded interval for a0 with extra leeway
        #ln fits incase user wishes to fit in log space for a0:
        loga0 = np.log(a0)
        loga0_error = a0_error / a0
        loga0_interval = np.log(2) #Lepage et al suggests loga0 interval of about ln(2)

        #a1 estimate (set a1 as a bit less than a0 with wider interval):
        a1_estimate = a0 * 0.8 #set a1 to be 80% of a0 as a rough estimate
        loga1 = np.log(a1_estimate)
        loga1_interval = np.log(5) #very wide interval for loga1

        return {'E0': E0, 
                'E0_interval': E0_interval,
                'E0_error': E0_error, 
                'lnE0': logE0,
                'lnE0_interval': logE0_interval,
                'lnE0_error': logE0_error,
                'ln_delta_E1': log_delta_E1,
                'ln_delta_E1_interval': log_delta_E1_interval,
                'a0': a0,
                'a0_error': a0_error,
                'lna0': loga0,
                'lna0_interval': loga0_interval,
                'a1_estimate': a1_estimate,
                'lna1': loga1,
                'lna1_interval': loga1_interval,
                't_min': t_min, 
                't_max': t_max,
                'chi2': best_chi2, 
                'stability': stability
                }
    
    def run(self, save=False):
        """Run analysis and return results.
        
        Args:
            save: If True, save results to npz file. Default False.
        
        Returns:
            dict: Plateau results for each ensemble.
        """
        if not self.load_data():
            print("No data loaded")
            return {}
        
        results = {name: self.find_plateau(name) for name in self.stats}
        
        if save:
            # Save full dict for each ensemble
            save_dict = {}
            for name, p in results.items():
                for key, val in p.items():
                    save_dict[f"{name}_{key}"] = val
            np.savez(self.results_path / "preliminary_analysis_data.npz", **save_dict)
            print(f"Saved {len(results)} ensembles to: {self.results_path / 'preliminary_analysis_data.npz'}")
        
        return results

class PreliminaryMLAnalysis2pt:
    """Preliminary analysis comparing ML predictions to truth."""
    
    def __init__(self, experiment_folder, truth_correlator_name=None):
        self.project_root = Path(__file__).parent.parent.parent
        self.experiment_path = self.project_root / "results" / experiment_folder
        self.results_path = self.experiment_path / "ml_analysis"
        self.results_path.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_folder
        self.truth_correlator_name = truth_correlator_name
        self.stats = {}
    
    def load_data(self):
        """Load ML predictions and truth data."""
        pred_path = self.experiment_path / "correlator_predictions.npy"
        target_path = self.experiment_path / "test_targets.npy"
        
        if not pred_path.exists() or not target_path.exists():
            raise FileNotFoundError(f"Missing predictions/targets in {self.experiment_path}")
        
        predictions = np.load(pred_path).squeeze()
        targets = np.load(target_path).squeeze()
        predictions, targets = self._apply_inverse_transform(predictions, targets)
        
        for name, data in [('ml', predictions), ('test_targets', targets)]:
            self.stats[name] = {'mean': np.mean(data, axis=0), 'err': np.std(data, axis=0) / np.sqrt(len(data)), 'n_configs': data.shape[0]}
        
        self._load_truth_data()
        return True
    
    def _apply_inverse_transform(self, predictions, targets):
        """Apply inverse transform based on scalers."""
        scalers_paths = [
            self.project_root / "models" / self.experiment_name / "cnn_model" / "scalers.npz",
            self.project_root / "models" / self.experiment_name / "transformer_model" / "scalers.npz",
            self.project_root / "models" / self.experiment_name / "gbr_model" / "scalers.npz",
            self.project_root / "models" / self.experiment_name / "mlp_model" / "scalers.npz",
        ]
        
        for path in scalers_paths:
            if path.exists():
                scalers = np.load(path)
                if 'target_signs' in scalers:
                    signs = scalers['target_signs']
                    return np.exp(predictions) * signs, np.exp(targets) * signs
                elif 'arcsinh_target_scale' in scalers:
                    # Legacy support for old arcsinh transform
                    scale = float(scalers['arcsinh_target_scale'])
                    return np.sinh(predictions) * scale, np.sinh(targets) * scale
        
        return np.exp(predictions), np.exp(targets)
    
    def _load_truth_data(self):
        """Load actual truth data from averaged CSV."""
        averaged_path = self.project_root / "data" / "processed" / "averaged_data"
        jackknife_path = self.project_root / "data" / "processed" / "jackknife_errors"
        
        if not averaged_path.exists():
            return
        
        csv_files = list(averaged_path.glob("*_averaged.csv"))
        target_file = None
        
        if self.truth_correlator_name:
            target_file = next((f for f in csv_files if self.truth_correlator_name in f.stem), None)
        if not target_file:
            target_file = next((f for f in csv_files if '2pt_K_fine_qsq0' in f.stem), None)
        if not target_file and csv_files:
            target_file = csv_files[0]
        if not target_file:
            return
        
        df = pd.read_csv(target_file, header=None)
        data = df.values
        name = target_file.stem.replace('_averaged', '')
        
        error_file = jackknife_path / f"{name}_jackknife_error.npy"
        err = np.load(error_file) if error_file.exists() else np.std(data, axis=0) / np.sqrt(len(data))
        
        self.stats['truth'] = {'mean': np.mean(data, axis=0), 'err': err, 'name': name}
    
    def compute_effective_mass(self, name):
        """Compute m_eff(t) = ln(C(t)/C(t+1))."""
        if name not in self.stats:
            return np.array([np.nan]), np.array([np.nan])
        
        mean, err = self.stats[name]['mean'], self.stats[name]['err']
        n = len(mean) - 1
        m_eff, m_eff_err = np.full(n, np.nan), np.full(n, np.nan)
        
        for t in range(n):
            if mean[t] > 0 and mean[t + 1] > 0:
                m_eff[t] = np.log(mean[t] / mean[t + 1])
                m_eff_err[t] = np.sqrt((err[t]/mean[t])**2 + (err[t+1]/mean[t+1])**2)
        
        return m_eff, m_eff_err
    
    def find_plateau(self, name, min_t=5, max_t=35, window_size=4, use_odd_only=True):
        """Find plateau via sliding window, minimizing chi2."""
        m_eff, m_eff_err = self.compute_effective_mass(name)
        if name not in self.stats:
            return {'E0': np.nan, 'E0_error': np.nan, 't_min': np.nan, 't_max': np.nan, 'chi2': np.inf, 'stability': np.nan}
        
        # Hard cutoff at max_t (no-go region)
        max_t = min(max_t, len(m_eff), 35)
        times = np.arange(min_t, max_t, 2) if use_odd_only else np.arange(min_t, max_t)
        
        m_vals = np.array([m_eff[t] for t in times if t < len(m_eff)])
        m_errs = np.array([m_eff_err[t] for t in times if t < len(m_eff)])
        times = np.array([t for t in times if t < len(m_eff)])
        
        mask = np.isfinite(m_vals) & np.isfinite(m_errs) & (m_errs > 0)
        if np.sum(mask) < window_size:
            return {'E0': np.nan, 'E0_error': np.nan, 't_min': np.nan, 't_max': np.nan, 'chi2': np.inf, 'stability': np.nan}
        
        m_vals, m_errs, times = m_vals[mask], m_errs[mask], times[mask]
        
        # Find window with minimum chi2
        best_chi2, best_start = np.inf, 0
        for i in range(len(m_vals) - window_size + 1):
            window_vals = m_vals[i:i + window_size]
            window_errs = m_errs[i:i + window_size]
            weights = 1.0 / window_errs**2
            w_mean = np.sum(weights * window_vals) / np.sum(weights)
            chi2 = np.sum(((window_vals - w_mean) / window_errs)**2) / max(1, window_size - 1)
            if chi2 < best_chi2:
                best_chi2, best_start = chi2, i
        
        plateau_vals = m_vals[best_start:best_start + window_size]
        plateau_errs = m_errs[best_start:best_start + window_size]
        plateau_times = times[best_start:best_start + window_size]
        
        weights = 1.0 / plateau_errs**2
        E0 = np.sum(weights * plateau_vals) / np.sum(weights)
        E0_error = 1.0 / np.sqrt(np.sum(weights))
        stability = np.std(plateau_vals)
        
        t_min = float(plateau_times[0])
        t_max = float(plateau_times[-1])

        E0_interval = 0.2
        logE0 = np.log(E0)
        logE0_error = E0_error / E0
        logE0_interval = 0.1

        # Difference in ground and 1st excited state energies
        delta_E1 = 0.1
        log_delta_E1 = np.log(delta_E1)
        log_delta_E1_interval = np.log(2)

        # a0 calculation from correlator values in plateau region
        a0_vals = []
        for t in range(int(t_min), int(t_max)+1):
            if t < len(self.stats[name]['mean']):
                C_t = self.stats[name]['mean'][t]
                a0_t = C_t * np.exp(E0 * t)
                a0_vals.append(a0_t)
        a0_vals = np.array(a0_vals) if a0_vals else np.array([np.nan])
        a0 = np.mean(a0_vals)
        a0_error = np.std(a0_vals) / np.sqrt(len(a0_vals)) if len(a0_vals) > 1 else 0.0
        a0_interval = 0.1
        loga0 = np.log(a0)
        loga0_error = a0_error / a0 if a0 > 0 else np.nan
        loga0_interval = np.log(2)

        # a1 estimate
        a1_estimate = a0 * 0.8
        loga1 = np.log(a1_estimate)
        loga1_interval = np.log(5)

        return {'E0': E0, 
                'E0_interval': E0_interval,
                'E0_error': E0_error, 
                'lnE0': logE0,
                'lnE0_interval': logE0_interval,
                'lnE0_error': logE0_error,
                'ln_delta_E1': log_delta_E1,
                'ln_delta_E1_interval': log_delta_E1_interval,
                'a0': a0,
                'a0_error': a0_error,
                'lna0': loga0,
                'lna0_interval': loga0_interval,
                'a1_estimate': a1_estimate,
                'lna1': loga1,
                'lna1_interval': loga1_interval,
                't_min': t_min, 
                't_max': t_max,
                'chi2': best_chi2, 
                'stability': stability
                }
    
    def run(self, save=False):
        """Run analysis and return results.
        
        Args:
            save: If True, save results to npz file. Default False.
        
        Returns:
            dict: Plateau results and metrics.
        """
        self.load_data()
        
        results = {name: self.find_plateau(name) for name in self.stats}
        mse = np.mean((self.stats['ml']['mean'] - self.stats['test_targets']['mean'])**2)
        mae = np.mean(np.abs(self.stats['ml']['mean'] - self.stats['test_targets']['mean']))
        results['metrics'] = {'mse': mse, 'mae': mae}
        
        if save:
            # Save full dict for each source
            save_dict = {'mse': mse, 'mae': mae}
            for name in ['ml', 'test_targets', 'truth']:
                if name in results and name != 'metrics':
                    for key, val in results[name].items():
                        save_dict[f"{name}_{key}"] = val
            np.savez(self.results_path / "preliminary_analysis_data.npz", **save_dict)
            print(f"Saved to: {self.results_path / 'preliminary_analysis_data.npz'}")
        
        return results

def main():
    """CLI entry point - saves results by default."""
    project_root = Path(__file__).parent.parent.parent
    save = '--save' in sys.argv or '--no-save' not in sys.argv  # Save by default for CLI
    
    if '--ml' in sys.argv:
        idx = sys.argv.index('--ml')
        experiment = sys.argv[idx + 1] if len(sys.argv) > idx + 1 else "Two_pt_ML_Kaon_qmax_to_qsq0_Experiment"
        truth_name = sys.argv[sys.argv.index('--truth') + 1] if '--truth' in sys.argv else None
        return PreliminaryMLAnalysis2pt(experiment, truth_name).run(save=save)
    
    averaged_path = project_root / "data" / "processed" / "averaged_data"
    jackknife_path = project_root / "data" / "processed" / "jackknife_errors"
    results_path = project_root / "results" / "truth_analysis"
    
    return PreliminaryAnalysis2pt(averaged_path, jackknife_path, results_path).run(save=save)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
