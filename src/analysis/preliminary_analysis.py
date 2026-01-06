"""Preliminary analysis for 2pt correlators - plateau estimates for Bayesian priors."""

import numpy as np
from pathlib import Path


def effective_mass(mean: np.ndarray, err: np.ndarray, staggered: bool = True):
    """Compute effective mass. For staggered: m_eff = 0.5*ln(C(t)/C(t+2))."""
    step = 2 if staggered else 1
    scale = 0.5 if staggered else 1.0
    n = len(mean) - step
    
    m_eff, m_err, times = [], [], []
    for t in range(n):
        c_t, c_next = np.abs(mean[t]), np.abs(mean[t + step])
        if c_t > 1e-20 and c_next > 1e-20:
            m = scale * np.log(c_t / c_next)
            rel_err = scale * np.sqrt((err[t]/c_t)**2 + (err[t+step]/c_next)**2)
            if np.isfinite(m) and np.isfinite(rel_err) and m > 0:
                m_eff.append(m)
                m_err.append(rel_err)
                times.append(t)
    return np.array(times), np.array(m_eff), np.array(m_err)


def find_plateau(mean: np.ndarray, err: np.ndarray, min_t: int = 5, max_t: int = 35,
                 window: int = 4, staggered: bool = True) -> dict:
    """Find plateau via sliding window chi² minimization."""
    t, m_eff, m_err = effective_mass(mean, err, staggered)
    
    # Filter to range
    mask = (t >= min_t) & (t <= max_t) & np.isfinite(m_eff) & (m_err > 0)
    if np.sum(mask) < window:
        return {'E0': 0.8, 'lnE0': np.log(0.8), 'a0': 0.1, 'lna0': np.log(0.1), 't_min': 10, 't_max': 30}
    
    t, m_eff, m_err = t[mask], m_eff[mask], m_err[mask]
    
    # Find minimum chi² window
    best_chi2, best_i = np.inf, 0
    for i in range(len(m_eff) - window + 1):
        w, e = m_eff[i:i+window], m_err[i:i+window]
        wt = 1 / e**2
        wm = np.sum(wt * w) / np.sum(wt)
        chi2 = np.sum(((w - wm) / e)**2) / max(1, window - 1)
        if chi2 < best_chi2:
            best_chi2, best_i = chi2, i
    
    # Extract plateau values
    pv, pe, pt = m_eff[best_i:best_i+window], m_err[best_i:best_i+window], t[best_i:best_i+window]
    wt = 1 / pe**2
    E0 = np.sum(wt * pv) / np.sum(wt)
    
    # Estimate amplitude from C(t) ~ a²*exp(-E*t)
    t_mid = int(np.mean(pt))
    C_t = np.abs(mean[t_mid]) if t_mid < len(mean) else 0.1
    a0 = np.sqrt(C_t * np.exp(E0 * t_mid)) if C_t > 0 else 0.1
    
    return {
        'E0': E0, 'lnE0': np.log(max(E0, 0.01)),
        'a0': a0, 'lna0': np.log(max(a0, 0.001)),
        'lnao0': np.log(max(a0 * 0.5, 0.001)), 'lnEo0': np.log(max(E0, 0.01)),
        't_min': int(pt[0]), 't_max': int(pt[-1]), 'chi2': best_chi2
    }


class PreliminaryAnalysis2pt:
    """Plateau estimates from averaged correlator data."""
    
    def __init__(self, data_path: Path, errors_path: Path, results_path: Path):
        self.data_path, self.errors_path = Path(data_path), Path(errors_path)
        self.results_path = Path(results_path)
        self.results_path.mkdir(parents=True, exist_ok=True)
        self.stats = {}
    
    def load_data(self) -> bool:
        import pandas as pd
        for csv in self.data_path.glob("*_averaged.csv"):
            name = csv.stem.replace('_averaged', '')
            df = pd.read_csv(csv)
            cols = df.select_dtypes(include=[np.number]).columns
            if 'config_id' in cols:
                cols = cols.drop('config_id')
            mean = df[cols].values[0]
            err_file = self.errors_path / f"{name}_jackknife_error.npy"
            err = np.load(err_file) if err_file.exists() else np.zeros_like(mean)
            self.stats[name] = {'mean': mean, 'err': err}
        return len(self.stats) > 0
    
    def run(self, save: bool = False) -> dict:
        if not self.load_data():
            return {}
        results = {name: find_plateau(s['mean'], s['err']) for name, s in self.stats.items()}
        if save:
            save_dict = {f"{n}_{k}": v for n, p in results.items() for k, v in p.items()}
            np.savez(self.results_path / "preliminary_analysis.npz", **save_dict)
        return results


class PreliminaryMLAnalysis2pt:
    """Compare ML predictions to truth correlators."""
    
    def __init__(self, experiment: str):
        self.root = Path(__file__).parent.parent.parent
        self.exp_path = self.root / "results" / experiment
        self.stats = {}
    
    def load_data(self) -> bool:
        pred = self.exp_path / "correlator_predictions.npy"
        targ = self.exp_path / "test_targets.npy"
        if not pred.exists() or not targ.exists():
            return False
        
        predictions, targets = np.load(pred).squeeze(), np.load(targ).squeeze()
        for name, data in [('ml', predictions), ('truth', targets)]:
            self.stats[name] = {
                'mean': np.mean(data, axis=0),
                'err': np.std(data, axis=0) / np.sqrt(len(data))
            }
        return True
    
    def run(self) -> dict:
        if not self.load_data():
            return {}
        return {name: find_plateau(s['mean'], s['err']) for name, s in self.stats.items()}


if __name__ == "__main__":
    import sys
    root = Path(__file__).parent.parent.parent
    PreliminaryAnalysis2pt(
        root / "data/processed/averaged_data",
        root / "data/processed/jackknife_errors",
        root / "results/truth_analysis"
    ).run(save='--save' in sys.argv)
